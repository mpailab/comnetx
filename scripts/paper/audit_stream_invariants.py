"""Audit hierarchy invariants on an actual ComNetX smart-mode stream.

The optimizer is executed without modifying or monkey-patching its semantics.
Immediately before every ``Optimizer.run`` call, this utility computes the
same per-level closure masks ``U_l`` from the pre-update hierarchy.  It then
records:

* refinement between every adjacent fine-to-coarse stored level;
* labels shared across each actual ``U_l`` boundary, before and after the run;
* numeric hierarchy entries changed outside ``U_l``;
* affected, radius-expanded, closure, block, and changed-entry counts.

Example for the dev-container copy of dyn_pubmed::

    PYTHONPATH=.:src python scripts/paper/audit_stream_invariants.py \
        --dataset dyn_pubmed --dataset-root /workspace/datasets \
        --batch-strategy 999:10 --force-undirected --output /tmp/audit.json

For ``p:n`` streams, the audit uses the same launcher bootstrap helper,
representation-aware dataset name, parent-quotient cache schema, and campaign
cache as the measured production run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from datasets import Dataset  # noqa: E402
from optimizer import Optimizer  # noqa: E402
from scripts.paper.audit_adapter_invariants import (  # noqa: E402
    hierarchy_refinement_report,
)


DEFAULT_MAX_WITNESSES = 5
DEFAULT_MAX_WITNESS_VERTICES = 12


def iter_adjacency_batches(adj: torch.Tensor) -> Iterable[torch.Tensor]:
    if adj.dim() == 2:
        yield adj
    elif adj.dim() == 3:
        for batch_index in range(adj.size(0)):
            yield adj[batch_index]
    else:
        raise ValueError(f"unsupported adjacency rank: {adj.dim()}")


def active_nodes_mask(
    batch: torch.Tensor,
    nodes_num: int,
    device: torch.device,
) -> torch.Tensor:
    """Return all endpoints represented by the first adjacency batch."""
    if batch.is_sparse:
        batch = batch.coalesce()
        active_nodes = batch.indices().unique()
        result = torch.zeros(nodes_num, dtype=torch.bool, device=device)
        result[active_nodes.to(device)] = True
        return result
    return ((batch != 0).any(dim=0) | (batch != 0).any(dim=1)).to(device)


def compute_update_scopes(
    communities: torch.Tensor,
    expanded_nodes_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the exact pre-run ``U_l`` masks used by ``Optimizer.run``."""
    if communities.dim() != 2:
        raise ValueError("communities must have shape [levels, vertices]")
    expanded_nodes_mask = expanded_nodes_mask.to(
        device=communities.device,
        dtype=torch.bool,
    )
    if expanded_nodes_mask.numel() != communities.size(1):
        raise ValueError("expanded_nodes_mask length must match communities")

    nodes = torch.nonzero(expanded_nodes_mask, as_tuple=True)[0]
    scopes = torch.zeros_like(communities, dtype=torch.bool)
    if nodes.numel() == 0:
        return scopes
    for level in range(communities.size(0)):
        touched_labels = communities[level].index_select(0, nodes)
        scopes[level] = torch.isin(
            communities[level],
            torch.unique(touched_labels),
        )
    return scopes


def compact_refinement_report(
    communities: torch.Tensor,
    *,
    max_witnesses: int = DEFAULT_MAX_WITNESSES,
    max_witness_vertices: int = DEFAULT_MAX_WITNESS_VERTICES,
) -> dict[str, Any]:
    """Compact a full adjacent-level report for per-update JSON storage."""
    report = hierarchy_refinement_report(communities)
    adjacent = []
    for item in report["adjacent"]:
        violations = item["violations"]
        adjacent.append(
            {
                "fine_level": item["fine_level"],
                "coarse_level": item["coarse_level"],
                "refines": item["refines"],
                "fine_blocks": item["fine_blocks"],
                "coarse_blocks": item["coarse_blocks"],
                "violating_fine_blocks": len(violations),
                "vertices_in_violating_blocks": sum(
                    len(violation["vertices"]) for violation in violations
                ),
                "witnesses": [
                    {
                        "fine_label": violation["fine_label"],
                        "coarse_labels": violation["coarse_labels"],
                        "vertices": violation["vertices"][:max_witness_vertices],
                        "block_size": len(violation["vertices"]),
                    }
                    for violation in violations[:max_witnesses]
                ],
            }
        )
    return {
        "storage_order": report["storage_order"],
        "all_adjacent_refine": report["all_adjacent_refine"],
        "adjacent": adjacent,
    }


def build_python_leiden_hierarchy(
    adjacency: torch.Tensor,
    *,
    depth: int,
    resolution: float,
    device: torch.device,
) -> torch.Tensor:
    """Dependency-light Python-Leiden form of the production bootstrap.

    Research prototypes import this helper directly.  Keeping the wrapper
    preserves their explicit Python-Leiden baseline in lightweight
    environments without the optional DGC wheel.  It follows the production
    no-cut parent-quotient and canonical-projection semantics; the Stage-1
    stream audit itself uses the exact cached launcher helper below.
    """
    import sparse
    from baselines.leiden import leidenalg_partition

    if depth < 1:
        raise ValueError("depth must be at least one")
    original = adjacency.to(device).float()
    if not original.is_sparse:
        original = original.to_sparse_coo()
    original = original.coalesce()
    vertices = torch.arange(original.size(0), device=device)
    first = Optimizer.canonicalize_partition(
        leidenalg_partition(original, resolution=resolution).to(device)
    )
    layers = [first]
    for _ in range(1, depth):
        _, inverse = torch.unique(
            layers[-1], sorted=True, return_inverse=True
        )
        group_count = int(inverse.max().item()) + 1
        pattern = sparse.tensor(
            torch.stack((inverse, vertices)),
            (group_count, original.size(0)),
            original.dtype,
        )
        quotient = Optimizer.aggregate(original, pattern)
        proposal = leidenalg_partition(
            quotient, resolution=resolution
        ).to(device)
        layers.append(
            Optimizer.canonicalize_partition(proposal[inverse], vertices)
        )
    return torch.stack(layers)


def compact_scope_overlap_report(
    labels: torch.Tensor,
    scope: torch.Tensor,
    *,
    max_witnesses: int = DEFAULT_MAX_WITNESSES,
) -> dict[str, Any]:
    """Summarize label namespace overlap without storing every vertex."""
    labels_cpu = labels.detach().cpu()
    scope_cpu = scope.detach().cpu().to(torch.bool)
    if labels_cpu.dim() != 1 or scope_cpu.shape != labels_cpu.shape:
        raise ValueError("labels and scope must be equal-length vectors")
    inside_labels = torch.unique(labels_cpu[scope_cpu], sorted=True)
    outside_labels = torch.unique(labels_cpu[~scope_cpu], sorted=True)
    if inside_labels.numel() and outside_labels.numel():
        shared = inside_labels[torch.isin(inside_labels, outside_labels)]
    else:
        shared = inside_labels[:0]
    if shared.numel():
        inside_entries = int(
            (scope_cpu & torch.isin(labels_cpu, shared)).sum().item()
        )
        outside_entries = int(
            ((~scope_cpu) & torch.isin(labels_cpu, shared)).sum().item()
        )
    else:
        inside_entries = outside_entries = 0

    witnesses = []
    for label in shared[:max_witnesses]:
        inside_vertex = torch.nonzero(
            scope_cpu & (labels_cpu == label), as_tuple=True
        )[0][0]
        outside_vertex = torch.nonzero(
            (~scope_cpu) & (labels_cpu == label), as_tuple=True
        )[0][0]
        witnesses.append(
            {
                "label": label.item(),
                "inside_vertex": int(inside_vertex.item()),
                "outside_vertex": int(outside_vertex.item()),
            }
        )

    return {
        "has_shared_labels": bool(shared.numel()),
        "shared_label_count": int(shared.numel()),
        "shared_labels_sample": shared[:max_witnesses].tolist(),
        "inside_entries_with_shared_label": inside_entries,
        "outside_entries_with_shared_label": outside_entries,
        "witnesses": witnesses,
    }


def outside_entry_write_report(
    before: torch.Tensor,
    after: torch.Tensor,
    scope: torch.Tensor,
    *,
    max_witnesses: int = DEFAULT_MAX_WITNESSES,
) -> dict[str, Any]:
    """Report numeric label writes inside and outside one update scope."""
    before = before.detach().cpu()
    after = after.detach().cpu()
    scope = scope.detach().cpu().to(torch.bool)
    if before.shape != after.shape or before.shape != scope.shape:
        raise ValueError("before, after, and scope must have identical shapes")

    changed = before != after
    outside_vertices = torch.nonzero(changed & (~scope), as_tuple=True)[0]
    inside_vertices = torch.nonzero(changed & scope, as_tuple=True)[0]
    return {
        "inside_changed_entries": int(inside_vertices.numel()),
        "outside_changed_entries": int(outside_vertices.numel()),
        "outside_vertices_sample": outside_vertices[:max_witnesses].tolist(),
    }


def _level_audit(
    before: torch.Tensor,
    after: torch.Tensor,
    scopes: torch.Tensor,
) -> list[dict[str, Any]]:
    levels = []
    for level in range(before.size(0)):
        scope = scopes[level]
        levels.append(
            {
                "level": level,
                "scope_vertices": int(scope.sum().item()),
                "scope_complement_vertices": int((~scope).sum().item()),
                "blocks_before": int(torch.unique(before[level]).numel()),
                "blocks_after": int(torch.unique(after[level]).numel()),
                "scope_label_overlap_before": compact_scope_overlap_report(
                    before[level], scope
                ),
                "scope_label_overlap_after": compact_scope_overlap_report(
                    after[level], scope
                ),
                "entry_writes": outside_entry_write_report(
                    before[level], after[level], scope
                ),
            }
        )
    return levels


def audit_optimizer_update(
    optimizer: Optimizer,
    directly_affected: torch.Tensor,
    *,
    radius: int,
) -> dict[str, Any]:
    """Run one unmodified smart update and return its structural audit."""
    directly_affected = directly_affected.to(
        device=optimizer.runtime_device(),
        dtype=torch.bool,
    )
    expanded = optimizer.neighborhood(
        optimizer.runtime_adj(),
        directly_affected,
        step=radius,
    )
    before = optimizer.coms.detach().clone()
    scopes = compute_update_scopes(before, expanded)
    refinement_before = compact_refinement_report(before)
    backend_calls_before = optimizer.local_algorithm_calls

    optimizer.run(expanded)

    after = optimizer.coms.detach().clone()
    levels = _level_audit(before, after, scopes)
    return {
        "directly_affected_vertices": int(directly_affected.sum().item()),
        "radius_expanded_vertices": int(expanded.sum().item()),
        "backend_calls": optimizer.local_algorithm_calls - backend_calls_before,
        "reported_modularity": optimizer.modularity(
            gamma=optimizer.resolution,
        ),
        "refinement_before": refinement_before,
        "refinement_after": compact_refinement_report(after),
        "levels": levels,
    }


def load_dataset(
    *,
    name: str,
    batch_strategy: str,
    paths_config: str | None,
    dataset_root: str | None,
    force_undirected: bool,
) -> Dataset:
    """Load through ``Dataset`` while allowing an explicit parent root."""
    constructor_config = paths_config
    if dataset_root is not None and constructor_config is None:
        constructor_config = str(PROJECT_PATH / "datasets-info" / "paths" / "astra.json")
    dataset = Dataset(name, constructor_config)
    if dataset_root is not None:
        dataset.dataset_root = Path(dataset_root)
    dataset.load(batches_strategy=batch_strategy)
    if force_undirected and dataset.is_directed:
        dataset._force_undirected()
        dataset.name = f"{name}-sym"
    return dataset


def summarize_updates(updates: list[dict[str, Any]]) -> dict[str, Any]:
    def indices_where(predicate) -> list[int]:
        return [
            int(update["update_index"])
            for update in updates
            if predicate(update)
        ]

    post_refinement_failures = indices_where(
        lambda update: not update["refinement_after"]["all_adjacent_refine"]
    )
    return {
        "update_indexing": "zero_based",
        "updates_audited": len(updates),
        "pre_update_refinement_failure_updates": indices_where(
            lambda update: not update["refinement_before"]["all_adjacent_refine"]
        ),
        "post_update_refinement_failure_updates": post_refinement_failures,
        "first_post_update_refinement_failure": post_refinement_failures[0]
        if post_refinement_failures
        else None,
        "scope_collision_before_updates": indices_where(
            lambda update: any(
                level["scope_label_overlap_before"]["has_shared_labels"]
                for level in update["levels"]
            )
        ),
        "scope_collision_after_updates": indices_where(
            lambda update: any(
                level["scope_label_overlap_after"]["has_shared_labels"]
                for level in update["levels"]
            )
        ),
        "outside_entry_write_updates": indices_where(
            lambda update: any(
                level["entry_writes"]["outside_changed_entries"] > 0
                for level in update["levels"]
            )
        ),
        "final_reported_modularity": float(updates[-1]["reported_modularity"])
        if updates
        else None,
        "maximum_scope_vertices_by_level": [
            max(
                int(update["levels"][level]["scope_vertices"])
                for update in updates
            )
            for level in range(len(updates[0]["levels"]))
        ]
        if updates
        else [],
    }


def run_stream_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(
        name=args.dataset,
        batch_strategy=args.batch_strategy,
        paths_config=args.paths_config,
        dataset_root=args.dataset_root,
        force_undirected=args.force_undirected,
    )
    batches = iter(iter_adjacency_batches(dataset.adj))
    try:
        first_batch = next(batches)
    except StopIteration as exc:
        raise ValueError("dataset contains no adjacency batches") from exc

    optimizer = Optimizer(
        first_batch,
        dataset.features,
        subcoms_depth=args.depth,
        method="leidenalg",
        use_gpu=args.use_gpu,
        aggregation_mode=args.aggregation_mode,
        resolution=args.resolution,
    )
    strategy_has_bootstrap = ":" in str(args.batch_strategy)
    if strategy_has_bootstrap:
        # Measurement runs require the production wheel; import the launcher
        # only on the bootstrap path so audit helpers remain testable without it.
        from launcher import (
            INITIAL_HIERARCHY_CACHE_SCHEMA,
            _compute_launch_initial_partition,
        )

        init_batch_number = str(args.batch_strategy).split(":", 1)[0]
        optimizer.set_communities(
            _compute_launch_initial_partition(
                adj_matrix=first_batch,
                dataset_name=dataset.name,
                init_batch_number=init_batch_number,
                cache_dir=args.cache_dir,
                subcoms_depth=args.depth,
                device=optimizer.runtime_device(),
                verbose=int(args.verbose),
                resolution=args.resolution,
            )
        )
        hierarchy_cache_schema = INITIAL_HIERARCHY_CACHE_SCHEMA
    else:
        init_batch_number = None
        hierarchy_cache_schema = None

    initial_refinement = compact_refinement_report(optimizer.coms)
    updates: list[dict[str, Any]] = []

    if not strategy_has_bootstrap:
        first_affected = active_nodes_mask(
            first_batch, optimizer.nodes_num, optimizer.runtime_device()
        )
        first_report = audit_optimizer_update(
            optimizer, first_affected, radius=args.radius
        )
        first_report.update({"batch_index": 0, "update_index": 0})
        updates.append(first_report)

    for batch_index, batch in enumerate(batches, start=1):
        if args.max_updates is not None and len(updates) >= args.max_updates:
            break
        directly_affected = optimizer.update_adj(batch, return_mask=True)
        report = audit_optimizer_update(
            optimizer,
            directly_affected,
            radius=args.radius,
        )
        report.update(
            {
                "batch_index": batch_index,
                "update_index": len(updates),
                "batch_nonzero_entries": int(batch.coalesce()._nnz())
                if batch.is_sparse
                else int(torch.count_nonzero(batch).item()),
            }
        )
        updates.append(report)
        if args.verbose:
            print(
                f"audited update {report['update_index'] + 1}: "
                f"B={report['directly_affected_vertices']}, "
                f"expanded={report['radius_expanded_vertices']}, "
                f"post_refines={report['refinement_after']['all_adjacent_refine']}",
                file=sys.stderr,
            )

    return {
        "schema": "comnetx_stream_invariant_audit_v1",
        "dataset": dataset.name,
        "batch_strategy": str(args.batch_strategy),
        "nodes": int(optimizer.nodes_num),
        "method": "leidenalg",
        "mode": "smart",
        "scope_definition": (
            "For each level, U_l is computed before Optimizer.run as the union "
            "of complete pre-update label blocks intersecting the radius-expanded "
            "affected set; this matches the implementation's ext_mask_work."
        ),
        "depth": args.depth,
        "radius": args.radius,
        "resolution": args.resolution,
        "aggregation_mode": args.aggregation_mode,
        "force_undirected": args.force_undirected,
        "device": str(optimizer.runtime_device()),
        "bootstrap": {
            "strategy_has_bootstrap_batch": strategy_has_bootstrap,
            "implementation": "launcher._compute_launch_initial_partition"
            if strategy_has_bootstrap
            else "optimizer_default_singletons",
            "initial_batch_number": init_batch_number,
            "hierarchy_cache_schema": hierarchy_cache_schema
            if args.depth > 1
            else None,
            "campaign_cache_enabled": args.cache_dir is not None,
            "note": (
                "The audit and production launcher use the same Leiden bootstrap "
                "helper and parent-quotient cache entry."
            ),
        },
        "initial_refinement": initial_refinement,
        "updates": updates,
        "summary": summarize_updates(updates),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-strategy", default="999:10")
    root_group = parser.add_mutually_exclusive_group()
    root_group.add_argument("--paths-config")
    root_group.add_argument(
        "--dataset-root",
        help="parent directory containing the named dataset directory",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--aggregation-mode", default="sum")
    parser.add_argument(
        "--cache-dir",
        help="campaign-local bootstrap cache shared with scripts/launch.py",
    )
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--force-undirected", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_stream_audit(args)
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        write_json(args.output, report)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
