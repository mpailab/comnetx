"""Compute targeted cut metrics for final Leiden partitions.

This reviewer-follow-up script reruns only selected Leiden configurations,
keeps the final partition, and evaluates cut-based structural metrics on the
final accumulated graph. Results are written after every completed run so a
partial measurement remains usable if the process is interrupted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# The lightweight dev container used for paper editing may not include the
# optional dynamic_graphs_communities package. This targeted script only uses
# Optimizer-backed Leiden modes, but launcher imports baselines.dgc at module
# import time. Provide a narrow stub so the unused dynamic-backend path remains
# unavailable without blocking this measurement.
try:
    import baselines.dgc  # noqa: F401
except ImportError as exc:
    if "dynamic_graphs_communities" not in str(exc):
        raise
    dgc_stub = types.ModuleType("baselines.dgc")

    def _create_leiden_unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise ImportError(
            "dynamic_graphs_communities is unavailable in this environment"
        )

    dgc_stub.create_leiden = _create_leiden_unavailable
    sys.modules["baselines.dgc"] = dgc_stub

from datasets import Dataset  # noqa: E402
from launcher import (  # noqa: E402
    _build_launch_config,
    _compute_full_adj,
    _iter_adjacency_batches,
    _metrics_modularity,
    _run_optimizer_modes,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_cpu_coo(adj: torch.Tensor) -> torch.Tensor:
    if not adj.is_sparse:
        adj = adj.to_sparse_coo()
    return adj.detach().cpu().coalesce()


def partition_cut_metrics(adj: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Return multiway conductance and normalized-cut metrics.

    Definitions use the weighted final adjacency as represented by the
    experiment. For the symmetrized graphs used in the paper, each undirected
    edge is represented by two directed sparse entries; this scaling cancels in
    the reported ratios.
    """
    adj = _to_cpu_coo(adj)
    labels = labels.detach().cpu().long()
    row, col = adj.indices()
    weight = adj.values().to(dtype=torch.float64)
    n = int(adj.size(0))

    if labels.numel() != n:
        raise ValueError(f"labels length {labels.numel()} does not match graph size {n}")

    total_volume = float(weight.sum().item())
    if total_volume <= 0.0:
        return {
            "num_nodes": n,
            "num_clusters": int(torch.unique(labels).numel()),
            "total_volume": total_volume,
            "cross_weight": 0.0,
            "ncut": 0.0,
            "conductance_mean": 0.0,
            "conductance_median": 0.0,
            "conductance_max": 0.0,
            "conductance_volume_weighted": 0.0,
            "nonzero_volume_clusters": 0,
        }

    clusters, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    num_clusters = int(clusters.numel())

    degree = torch.zeros(n, dtype=torch.float64)
    degree.scatter_add_(0, row, weight)

    volume = torch.zeros(num_clusters, dtype=torch.float64)
    volume.scatter_add_(0, inverse, degree)

    crossing = inverse[row] != inverse[col]
    cut = torch.zeros(num_clusters, dtype=torch.float64)
    if torch.any(crossing):
        cut.scatter_add_(0, inverse[row][crossing], weight[crossing])

    valid_volume = volume > 0
    ncut_terms = torch.zeros_like(volume)
    ncut_terms[valid_volume] = cut[valid_volume] / volume[valid_volume]
    ncut = float(ncut_terms.sum().item())

    complement_volume = total_volume - volume
    conductance_denom = torch.minimum(volume, complement_volume)
    valid_conductance = conductance_denom > 0
    conductance_terms = torch.zeros_like(volume)
    conductance_terms[valid_conductance] = (
        cut[valid_conductance] / conductance_denom[valid_conductance]
    )
    valid_terms = conductance_terms[valid_conductance]
    if valid_terms.numel() == 0:
        conductance_mean = conductance_median = conductance_max = 0.0
    else:
        conductance_mean = float(valid_terms.mean().item())
        conductance_median = float(valid_terms.median().item())
        conductance_max = float(valid_terms.max().item())

    if torch.any(valid_conductance):
        weighted = (conductance_terms[valid_conductance] * volume[valid_conductance]).sum()
        conductance_volume_weighted = float((weighted / volume[valid_conductance].sum()).item())
    else:
        conductance_volume_weighted = 0.0

    return {
        "definition": (
            "ncut=sum_C cut(C,V-C)/vol(C); conductance(C)="
            "cut(C,V-C)/min(vol(C),vol(V-C)); reported conductance_mean is "
            "the unweighted mean over nontrivial positive-volume clusters"
        ),
        "num_nodes": n,
        "num_clusters": num_clusters,
        "nonzero_volume_clusters": int(valid_volume.sum().item()),
        "total_volume": total_volume,
        "cross_weight": float(cut.sum().item()),
        "ncut": ncut,
        "conductance_mean": conductance_mean,
        "conductance_median": conductance_median,
        "conductance_max": conductance_max,
        "conductance_volume_weighted": conductance_volume_weighted,
    }


def write_output(path: Path, records: list[dict[str, Any]], started_at: str) -> None:
    payload = {
        "group": "leiden_cut_metrics_targeted",
        "description": (
            "Targeted reviewer-follow-up cut metrics for final Leiden partitions "
            "on key 999:10 graph streams."
        ),
        "generated_at": utc_now(),
        "started_at": started_at,
        "record_count": len(records),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_one(args: argparse.Namespace, dataset_name: str, mode: str) -> dict[str, Any]:
    ds = Dataset(dataset_name, paths_config=args.paths_config)
    ds.load(
        batches_strategy=args.batch,
        feature_mode="dataset",
    )
    if args.force_undirected and ds.is_directed:
        ds._force_undirected()
        # Cache/result identity must encode the graph representation just as
        # scripts/launch.py does; otherwise a directed and symmetrized run can
        # silently share an initial-partition cache file.
        ds.name = f"{dataset_name}-sym"

    config = _build_launch_config(
        ds=ds,
        batches_strategy=args.batch,
        underlying_static_method="leidenalg",
        baseline_iter=None,
        mode=mode,
        smart_subcoms_depth=args.smart_depth,
        smart_neighborhood_step=args.smart_radius,
        verbose=args.verbose,
        use_gpu=args.use_gpu,
        aggregation_mode="sum",
        cache_dir=args.cache_dir,
        ground_truth_metrics=True,
        resolution=args.resolution,
    )

    started = time.perf_counter()
    results, labels = _run_optimizer_modes(ds, _iter_adjacency_batches(ds.adj), config)
    elapsed = time.perf_counter() - started

    final_adj = _compute_full_adj(ds.adj)
    cut_metrics = partition_cut_metrics(final_adj, labels)
    final_modularity = _metrics_modularity(
        final_adj,
        labels,
        args.resolution,
        ds.is_directed,
    )

    record = {
        "dataset": dataset_name,
        "batch": args.batch,
        "method": "leidenalg",
        "mode": mode,
        "force_undirected": bool(args.force_undirected),
        "resolution": float(args.resolution),
        "smart_depth": args.smart_depth if mode == "smart" else None,
        "smart_radius": args.smart_radius if mode == "smart" else None,
        "total_time": float(sum(item.get("time", 0.0) for item in results)),
        "wall_time": float(elapsed),
        "updates": len(results),
        "final_modularity": float(final_modularity),
        "cut_metrics": cut_metrics,
    }
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dyn_pubmed", "arxivmath"],
    )
    parser.add_argument("--modes", nargs="+", default=["naive", "smart"])
    parser.add_argument("--batch", default="999:10")
    parser.add_argument("--smart-depth", type=int, default=3)
    parser.add_argument("--smart-radius", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--cache-dir", default="tmp/icdm_cut_metrics_cache")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/icdm-2026-1/measurements/leiden_cut_metrics_targeted.json"),
    )
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--force-undirected", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = str(cache_dir)

    started_at = utc_now()
    records: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        for mode in args.modes:
            print(f"[{utc_now()}] START dataset={dataset_name} mode={mode}", flush=True)
            record = run_one(args, dataset_name, mode)
            records.append(record)
            write_output(args.out, records, started_at)
            cm = record["cut_metrics"]
            print(
                f"[{utc_now()}] DONE dataset={dataset_name} mode={mode} "
                f"Q={record['final_modularity']:.6f} "
                f"conductance_mean={cm['conductance_mean']:.6f} "
                f"ncut={cm['ncut']:.6f}",
                flush=True,
            )
    write_output(args.out, records, started_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
