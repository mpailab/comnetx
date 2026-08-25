"""Compare cut and no-cut parent-quotient hierarchy initialization.

This research audit leaves the production launcher unchanged.  Both variants
run the same Python Leiden call at level zero.  The production-compatible
variant then carries forward an adjacency cut by each stored partition; the
parent-quotient variant always rebuilds ``P A_0 P^T`` from the original
bootstrap adjacency and the immediately preceding level's blocks.

Example for the local dyn_pubmed copy::

    PYTHONPATH=.:src python scripts/paper/audit_initial_hierarchy_variants.py \
        --dataset dyn_pubmed --dataset-root /workspace/datasets \
        --batch-strategy 999:10 --force-undirected \
        --output /tmp/dyn_pubmed_initial_hierarchy.json

On a measurement host with the TGC datasets configured, pass both datasets
with ``--paths-config``::

    PYTHONPATH=.:src python scripts/paper/audit_initial_hierarchy_variants.py \
        --dataset dyn_pubmed --dataset arxivmath \
        --paths-config datasets-info/paths/cn69.json \
        --batch-strategy 999:10 --force-undirected
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import sparse  # noqa: E402
from baselines.leiden import leidenalg_partition  # noqa: E402
from metrics import Metrics  # noqa: E402
from optimizer import Optimizer  # noqa: E402
from scripts.paper.audit_adapter_invariants import (  # noqa: E402
    hierarchy_refinement_report,
    refinement_report,
)
from scripts.paper.audit_stream_invariants import load_dataset  # noqa: E402


def canonical_partition(labels: torch.Tensor) -> torch.Tensor:
    """Represent each block by its minimum vertex for label-invariant checks."""
    labels = labels.detach().cpu().to(torch.long)
    _, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    vertices = torch.arange(labels.numel(), dtype=torch.long)
    representatives = torch.full(
        (int(inverse.max().item()) + 1,),
        labels.numel(),
        dtype=torch.long,
    )
    representatives.scatter_reduce_(
        0, inverse, vertices, reduce="amin", include_self=True
    )
    return representatives[inverse]


def partition_hash(labels: torch.Tensor) -> str:
    """Return a stable equivalence-relation hash, independent of label IDs."""
    canonical = canonical_partition(labels).numpy()
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def quotient_diagnostics(quotient: torch.Tensor) -> dict[str, Any]:
    quotient = quotient.coalesce()
    indices = quotient.indices()
    values = quotient.values()
    nonzero = values != 0
    off_diagonal = nonzero & (indices[0] != indices[1])
    return {
        "vertices": int(quotient.size(0)),
        "stored_entries": int(quotient._nnz()),
        "nonzero_entries": int(nonzero.sum().item()),
        "nonzero_off_diagonal_entries": int(off_diagonal.sum().item()),
        "absolute_off_diagonal_weight": float(
            values[off_diagonal].abs().sum().item()
        ),
    }


def build_initial_hierarchy(
    adjacency: torch.Tensor,
    *,
    depth: int,
    resolution: float,
    carry_cut: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Build current-cut or original-adjacency parent quotients."""
    if depth < 1:
        raise ValueError("depth must be positive")
    original = adjacency.float().coalesce()
    first = leidenalg_partition(
        original, resolution=resolution
    ).to(original.device)
    layers = [first]
    diagnostics = [
        {
            "level": 0,
            "atom_source": "original_vertices",
            "blocks": int(torch.unique(first).numel()),
            "partition_hash": partition_hash(first),
        }
    ]
    working = original.clone()
    vertices = torch.arange(original.size(0), device=original.device)
    all_vertices = torch.ones(
        original.size(0), dtype=torch.bool, device=original.device
    )

    for level in range(1, depth):
        previous = layers[-1]
        old_labels, inverse = torch.unique(
            previous, sorted=True, return_inverse=True
        )
        pattern = sparse.tensor(
            torch.stack((inverse, vertices)),
            (old_labels.numel(), original.size(0)),
            original.dtype,
        )
        quotient_source = working if carry_cut else original
        quotient = Optimizer.aggregate(quotient_source, pattern)
        quotient_labels = leidenalg_partition(
            quotient, resolution=resolution
        ).to(original.device)
        restored = old_labels[quotient_labels[inverse]]
        layers.append(restored)
        diagnostics.append(
            {
                "level": level,
                "atom_source": f"level_{level - 1}_blocks",
                "blocks": int(torch.unique(restored).numel()),
                "strictly_coarser_than_parent": (
                    torch.unique(restored).numel()
                    < torch.unique(previous).numel()
                ),
                "partition_hash": partition_hash(restored),
                "quotient": quotient_diagnostics(quotient),
                "adjacency_source": (
                    "carried_cut_adjacency"
                    if carry_cut
                    else "original_bootstrap_adjacency"
                ),
            }
        )
        if carry_cut:
            working = Optimizer.cut_by_partition(
                working, all_vertices, restored
            )

    return torch.stack(layers), diagnostics


def equivalent_partitions(
    first: torch.Tensor,
    second: torch.Tensor,
) -> bool:
    return bool(
        refinement_report(first, second)["refines"]
        and refinement_report(second, first)["refines"]
    )


def hierarchy_summary(
    hierarchy: torch.Tensor,
    diagnostics: list[dict[str, Any]],
    original: torch.Tensor,
    *,
    resolution: float,
    directed: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "block_counts_fine_to_coarse": [
            int(torch.unique(row).numel()) for row in hierarchy
        ],
        "strict_coarsening_by_transition": [
            int(torch.unique(hierarchy[level]).numel())
            < int(torch.unique(hierarchy[level - 1]).numel())
            for level in range(1, hierarchy.size(0))
        ],
        "modularity_by_level": [
            float(
                Metrics.modularity(
                    original,
                    hierarchy[level],
                    gamma=resolution,
                    directed=directed,
                )
            )
            for level in range(hierarchy.size(0))
        ],
        "refinement": hierarchy_refinement_report(hierarchy),
        "elapsed_seconds": elapsed_seconds,
        "levels": diagnostics,
    }


def audit_adjacency(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    first_batch: torch.Tensor,
    directed: bool,
    source: str,
) -> dict[str, Any]:
    original = first_batch.float().coalesce()

    start = time.perf_counter()
    current, current_diagnostics = build_initial_hierarchy(
        first_batch.clone(),
        depth=args.depth,
        resolution=args.resolution,
        carry_cut=True,
    )
    current_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parent, parent_diagnostics = build_initial_hierarchy(
        first_batch.clone(),
        depth=args.depth,
        resolution=args.resolution,
        carry_cut=False,
    )
    parent_seconds = time.perf_counter() - start

    row0_equivalent = equivalent_partitions(current[0], parent[0])
    row0_exact = torch.equal(current[0], parent[0])
    return {
        "dataset": dataset_name,
        "source": source,
        "nodes": int(original.size(0)),
        "bootstrap_nonzero_entries": int(original._nnz()),
        "directed": bool(directed),
        "depth": args.depth,
        "current_cut": hierarchy_summary(
            current,
            current_diagnostics,
            original,
            resolution=args.resolution,
            directed=directed,
            elapsed_seconds=current_seconds,
        ),
        "no_cut_parent_quotient": hierarchy_summary(
            parent,
            parent_diagnostics,
            original,
            resolution=args.resolution,
            directed=directed,
            elapsed_seconds=parent_seconds,
        ),
        "row0_check": {
            "exact_label_vector_match": row0_exact,
            "partition_equivalent": row0_equivalent,
            "current_hash": partition_hash(current[0]),
            "no_cut_hash": partition_hash(parent[0]),
            "current_modularity": float(
                Metrics.modularity(
                    original,
                    current[0],
                    gamma=args.resolution,
                    directed=directed,
                )
            ),
            "no_cut_modularity": float(
                Metrics.modularity(
                    original,
                    parent[0],
                    gamma=args.resolution,
                    directed=directed,
                )
            ),
        },
        "same_partition_by_level": [
            equivalent_partitions(current[level], parent[level])
            for level in range(args.depth)
        ],
    }


def audit_dataset(args: argparse.Namespace, dataset_name: str) -> dict[str, Any]:
    dataset = load_dataset(
        name=dataset_name,
        batch_strategy=args.batch_strategy,
        paths_config=args.paths_config,
        dataset_root=args.dataset_root,
        force_undirected=args.force_undirected,
    )
    first_batch = (
        dataset.adj[0] if dataset.adj.dim() == 3 else dataset.adj
    )
    return audit_adjacency(
        args,
        dataset_name=dataset.name,
        first_batch=first_batch,
        directed=dataset.is_directed,
        source="configured Dataset loader",
    )


def raw_tgc_bootstrap(
    dataset_name: str,
    edge_list: Path,
    batch_strategy: str,
    *,
    force_undirected: bool,
) -> tuple[torch.Tensor, bool]:
    """Reconstruct a p:n bootstrap directly from a public TGC edge list."""
    if ":" not in batch_strategy:
        raise ValueError("raw TGC reconstruction requires a p:n strategy")
    p_string, _ = batch_strategy.split(":", 1)
    p = int(p_string)
    batches = p + 1

    metadata_path = PROJECT_PATH / "datasets-info" / "json" / "tgc.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if dataset_name not in metadata:
        raise ValueError(f"unknown TGC dataset metadata: {dataset_name}")
    nodes = int(metadata[dataset_name]["n"])
    directed = metadata[dataset_name]["d"] == "directed"

    raw = np.loadtxt(edge_list, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError("raw TGC edges must contain source, target, timestamp")
    sources = torch.from_numpy(raw[:, 0].astype(np.int64, copy=False))
    targets = torch.from_numpy(raw[:, 1].astype(np.int64, copy=False))
    timestamps = torch.from_numpy(raw[:, 2])
    order = torch.argsort(timestamps)
    sources = sources[order]
    targets = targets[order]
    edge_count = sources.numel()
    batch_indices = torch.arange(edge_count) * batches // edge_count
    bootstrap = batch_indices < p
    indices = torch.stack((sources[bootstrap], targets[bootstrap]))
    values = torch.ones(indices.size(1), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices, values, (nodes, nodes)
    ).coalesce()
    if force_undirected and directed:
        adjacency = (adjacency + adjacency.transpose(0, 1)).coalesce()
        directed = False
    return adjacency, directed


def parse_raw_edge_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--raw-tgc-edge-list must be NAME=PATH")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise ValueError("--raw-tgc-edge-list must be NAME=PATH")
    return name, Path(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append")
    parser.add_argument(
        "--raw-tgc-edge-list",
        action="append",
        metavar="NAME=PATH",
        help=(
            "reconstruct a TGC p:n bootstrap from a public raw temporal "
            "edge list; node count and direction come from tgc.json"
        ),
    )
    parser.add_argument("--batch-strategy", default="999:10")
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument("--paths-config")
    roots.add_argument("--dataset-root")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--force-undirected", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset and not args.raw_tgc_edge_list:
        raise ValueError("provide --dataset or --raw-tgc-edge-list")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        datasets = {
            name: audit_dataset(args, name) for name in (args.dataset or [])
        }
        for specification in args.raw_tgc_edge_list or []:
            name, path = parse_raw_edge_spec(specification)
            adjacency, directed = raw_tgc_bootstrap(
                name,
                path,
                args.batch_strategy,
                force_undirected=args.force_undirected,
            )
            report_name = f"{name}-sym" if not directed else name
            datasets[report_name] = audit_adjacency(
                args,
                dataset_name=report_name,
                first_batch=adjacency,
                directed=directed,
                source=f"raw TGC edge reconstruction: {path}",
            )
    report = {
        "schema": "comnetx_initial_hierarchy_variant_audit_v1",
        "status": "research audit; production initialization is unchanged",
        "batch_strategy": args.batch_strategy,
        "force_undirected": args.force_undirected,
        "resolution": args.resolution,
        "datasets": datasets,
    }
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        write_json(args.output, report)
        compact = {
            name: {
                "current_blocks": row["current_cut"][
                    "block_counts_fine_to_coarse"
                ],
                "parent_blocks": row["no_cut_parent_quotient"][
                    "block_counts_fine_to_coarse"
                ],
                "row0_check": row["row0_check"],
                "same_partition_by_level": row[
                    "same_partition_by_level"
                ],
            }
            for name, row in datasets.items()
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
