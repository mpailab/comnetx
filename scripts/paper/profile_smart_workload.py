"""Profile ComNetX smart-mode workload, timing breakdown, and memory.

The regular launcher reports end-to-end per-update quality and runtime. For the
paper we also need evidence that the runtime reduction is caused by small local
workloads rather than hidden full-graph work. This script mirrors the smart-mode
Optimizer loop and records:

* directly affected and radius-expanded vertices;
* post-closure vertices at every hierarchy level;
* contracted backend nodes and edges;
* time spent in update, radius expansion, closure, reset, aggregation, backend,
  projection, and cut phases;
* CPU RSS and CUDA peak memory.

It writes a custom JSON format understood by ``collect_results_registry.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch  # noqa: E402

import sparse  # noqa: E402
from datasets import Dataset  # noqa: E402
from launcher import _compute_full_adj, _compute_launch_initial_partition, _iter_adjacency_batches  # noqa: E402
from metrics import Metrics, calculate_ground_truth_metrics  # noqa: E402
from optimizer import Optimizer  # noqa: E402


FEATURE_METHODS = {"magi", "dmon", "mfc", "dese", "s2cag"}
ITER_DEFAULTS = {"flmig": 1, "s2cag": 10, "dmon": 10, "magi": 10, "mfc": 100, "lago": 1}


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now(device: torch.device) -> float:
    sync_if_needed(device)
    return time.perf_counter()


def elapsed(start: float, device: torch.device) -> float:
    sync_if_needed(device)
    return time.perf_counter() - start


def current_rss_mb() -> float:
    # Linux reports kilobytes, macOS bytes. The cn69 runs are Linux, but keep a
    # conservative fallback so local dry runs remain interpretable.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10_000_000_000:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def cuda_memory(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "cuda_allocated_mb": None,
            "cuda_reserved_mb": None,
            "cuda_peak_allocated_mb": None,
            "cuda_peak_reserved_mb": None,
        }
    sync_if_needed(device)
    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
        "cuda_reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
    }


def algorithm_name(
    method: str,
    baseline_iter: int | None,
    depth: int,
    radius: int,
    aggregation_mode: str,
    use_gpu: bool,
    feature_mode: str,
) -> str:
    prefix = f"{method}-i:{baseline_iter}" if baseline_iter is not None else method
    device = "gpu" if use_gpu else "cpu"
    name = f"{prefix}-L:{depth}-r:{radius}"
    if method in FEATURE_METHODS:
        agg = "norm" if aggregation_mode in {"normalized", "norm"} else aggregation_mode
        name = f"{name}-agg:{agg}"
    name = f"{name}-{device}"
    if method in FEATURE_METHODS:
        name = f"{name}-feat:{feature_mode}"
    return name


def nnz(tensor: torch.Tensor) -> int:
    if tensor.is_sparse:
        return int(tensor.coalesce()._nnz())
    return int(torch.count_nonzero(tensor).item())


def profile_smart_update(
    opt: Optimizer,
    affected_nodes_mask: torch.Tensor,
    method: str,
    aggregation_mode: str,
    radius: int,
    directed: bool,
    variant: str = "full",
) -> tuple[dict[str, Any], float]:
    device = opt.runtime_device()
    timings: dict[str, float] = {}
    row: dict[str, Any] = {
        "affected_vertices": int(affected_nodes_mask.sum().item()),
        "variant": variant,
    }

    start_total = now(device)

    start = now(device)
    expanded_nodes_mask = opt.neighborhood(
        opt.runtime_adj(),
        affected_nodes_mask,
        step=radius,
        is_symmetric=not directed,
    )
    timings["radius_time"] = elapsed(start, device)
    row["radius_vertices"] = int(expanded_nodes_mask.sum().item())

    start = now(device)
    compute_device = opt.device
    needs_features = opt._local_algorithm_requires_features()
    coms_work = opt.coms
    adj_base = opt.adj
    features_work = opt.features if needs_features else None
    nodes_mask_work = (
        expanded_nodes_mask
        if expanded_nodes_mask.device == compute_device
        else expanded_nodes_mask.to(compute_device)
    )
    nodes = torch.nonzero(nodes_mask_work, as_tuple=True)[0]

    ext_mask_work = torch.zeros_like(coms_work, dtype=torch.bool)
    if variant == "no_closure":
        for level in range(opt.subcoms_depth):
            ext_mask_work[level] = nodes_mask_work
    else:
        for level in range(opt.subcoms_depth):
            touched = coms_work[level].index_select(0, nodes)
            ext_mask_work[level] = torch.isin(coms_work[level], torch.unique(touched))

    coms_work[-1, nodes_mask_work] = nodes
    for level in range(opt.subcoms_depth - 2, -1, -1):
        level_ext_mask = ext_mask_work[level]
        coms_work[level, level_ext_mask] = coms_work[level + 1, level_ext_mask]
    timings["closure_time"] = elapsed(start, device)

    row["closure_vertices_by_level"] = [
        int(ext_mask_work[level].sum().item()) for level in range(opt.subcoms_depth)
    ]

    start = now(device)
    affected_nodes_lvl0 = torch.nonzero(ext_mask_work[0], as_tuple=True)[0]
    adj_work = sparse.reset_matrix(adj_base, affected_nodes_lvl0)
    timings["reset_time"] = elapsed(start, device)

    level_rows = []
    backend_total = 0.0
    aggregation_total = 0.0
    projection_total = 0.0
    cut_total = 0.0
    backend_conversion_total = 0.0

    for level in range(opt.subcoms_depth):
        level_row: dict[str, Any] = {"level": level}

        start = now(device)
        level_ext_mask = ext_mask_work[level]
        coms = coms_work[level, level_ext_mask]
        ext_nodes = torch.nonzero(level_ext_mask, as_tuple=True)[0]
        old_idx, inverse, counts = torch.unique(
            coms,
            sorted=True,
            return_counts=True,
            return_inverse=True,
        )
        contracted_nodes = int(old_idx.size(0))
        timings[f"level_{level}_prepare_time"] = elapsed(start, device)

        if contracted_nodes == 0:
            level_rows.append(level_row)
            continue

        start = now(device)
        if variant == "no_contraction":
            local_idx = torch.arange(ext_nodes.size(0), device=compute_device)
            aggr_idx = torch.stack((local_idx, ext_nodes))
            aggr_adj_ptn = sparse.tensor(
                aggr_idx,
                (ext_nodes.size(0), opt.nodes_num),
                adj_work.dtype,
            )
            aggr_adj = opt.aggregate(adj_work, aggr_adj_ptn)
            del aggr_adj_ptn

            contracted_nodes = int(ext_nodes.size(0))
            aggr_features = features_work.index_select(0, ext_nodes) if needs_features else None
        else:
            aggr_idx = torch.stack((inverse, ext_nodes))
            aggr_adj_ptn = sparse.tensor(
                aggr_idx,
                (contracted_nodes, opt.nodes_num),
                adj_work.dtype,
            )
            aggr_adj = opt.aggregate(adj_work, aggr_adj_ptn)
            del aggr_adj_ptn

            aggr_features = None
            if needs_features:
                ext_features = features_work.index_select(0, ext_nodes)
                aggr_features = torch.zeros(
                    (contracted_nodes, ext_features.size(1)),
                    dtype=ext_features.dtype,
                    device=ext_features.device,
                )
                aggr_features.index_add_(0, inverse, ext_features)
                if opt.aggregation_mode == "normalized":
                    aggr_features /= counts.to(dtype=ext_features.dtype).unsqueeze(1)
        aggregation_time = elapsed(start, device)
        aggregation_total += aggregation_time

        start = now(device)
        coms = opt.local_algorithm(aggr_adj, aggr_features, level > 0).to(
            device=compute_device,
            dtype=torch.long,
        )
        backend_time = elapsed(start, device)
        backend_total += backend_time
        backend_conversion_total += float((opt.last_timing_info or {}).get("conversion_time", 0.0))

        start = now(device)
        if variant == "no_contraction":
            label_ids, label_inverse = torch.unique(
                coms,
                sorted=True,
                return_inverse=True,
            )
            representatives = torch.empty(
                label_ids.size(0),
                dtype=ext_nodes.dtype,
                device=compute_device,
            )
            for label_pos in range(label_ids.size(0)):
                first_local = torch.nonzero(label_inverse == label_pos, as_tuple=True)[0][0]
                representatives[label_pos] = ext_nodes[first_local]
            new_coms = representatives[label_inverse]
        else:
            new_coms = old_idx[coms[inverse]]
        coms_work[level, level_ext_mask] = new_coms
        projection_time = elapsed(start, device)
        projection_total += projection_time

        start = now(device)
        adj_work = opt.cut_by_partition(adj_work, level_ext_mask, coms_work[level])
        cut_time = elapsed(start, device)
        cut_total += cut_time

        level_row.update(
            {
                "closure_vertices": int(level_ext_mask.sum().item()),
                "contracted_nodes": contracted_nodes,
                "contracted_edges": nnz(aggr_adj),
                "aggregation_time": aggregation_time,
                "backend_time": backend_time,
                "projection_time": projection_time,
                "cut_time": cut_time,
            }
        )
        level_rows.append(level_row)

    timings["aggregation_time"] = aggregation_total
    timings["backend_time"] = backend_total
    timings["backend_conversion_time"] = backend_conversion_total
    timings["projection_time"] = projection_total
    timings["cut_time"] = cut_total
    timings["total_profiled_time"] = elapsed(start_total, device)

    row.update(timings)
    row["levels"] = level_rows
    row["contracted_nodes_by_level"] = [
        int(item.get("contracted_nodes", 0)) for item in level_rows
    ]
    row["contracted_edges_by_level"] = [
        int(item.get("contracted_edges", 0)) for item in level_rows
    ]
    row["modularity"] = Metrics.modularity(opt.adj, opt.coms[0], directed=directed)
    row["aggregation_mode"] = aggregation_mode
    row.update(cuda_memory(device))
    row["rss_max_mb"] = current_rss_mb()
    return row, timings["total_profiled_time"]


def load_dataset(
    dataset: str,
    batch_strategy: str,
    paths_config: str | None,
    feature_mode: str,
    random_feature_dim: int,
    random_feature_seed: int,
    force_undirected: bool,
):
    ds = Dataset(dataset, paths_config)
    ds.load(
        batches_strategy=batch_strategy,
        feature_mode=feature_mode,
        random_feat_dim=random_feature_dim,
        random_feat_seed=random_feature_seed,
    )
    if force_undirected and ds.is_directed:
        ds._force_undirected()
        ds.name = f"{dataset}-sym"
    return ds


def profile_run(
    args,
    dataset: str,
    batch_strategy: str,
    method: str,
    feature_mode: str,
    variant: str,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    baseline_iter = args.baseline_iter if args.baseline_iter is not None else ITER_DEFAULTS.get(method)
    ds = load_dataset(
        dataset,
        batch_strategy,
        args.paths_config,
        feature_mode,
        args.random_feature_dim,
        args.random_feature_seed,
        args.force_undirected,
    )

    batches = list(_iter_adjacency_batches(ds.adj))
    use_gpu = bool(args.use_gpu)
    alg = algorithm_name(
        method,
        baseline_iter,
        args.smart_depth,
        args.smart_radius,
        args.aggregation_mode,
        use_gpu,
        feature_mode,
    )

    rows = []
    opt = None
    device_name = "unknown"
    profile = {
        "algorithm": alg,
        "variant": variant,
        "method": method,
        "mode": "smart",
        "dataset": ds.name,
        "base_dataset": dataset,
        "batch_strategy": str(batch_strategy),
        "feature_mode": feature_mode if method in FEATURE_METHODS else None,
        "baseline_iter": baseline_iter,
        "smart_depth": args.smart_depth,
        "smart_radius": args.smart_radius,
        "aggregation_mode": args.aggregation_mode,
        "device": device_name,
        "use_gpu": use_gpu,
        "updates_profiled": 0,
        "total_profiled_time": 0.0,
        "peak_cuda_allocated_mb": None,
        "peak_rss_mb": None,
        "metrics": {},
        "rows": rows,
        "incomplete": True,
        "updated_at": datetime.now().isoformat(),
    }

    def refresh_profile(metrics: dict[str, Any] | None = None, completed: bool = False) -> dict[str, Any]:
        peak_cuda = max(
            [row.get("cuda_peak_allocated_mb") for row in rows if row.get("cuda_peak_allocated_mb") is not None],
            default=None,
        )
        peak_rss = max([float(row.get("rss_max_mb", 0.0)) for row in rows], default=None)
        profile.update(
            {
                "device": device_name,
                "updates_profiled": len(rows),
                "total_profiled_time": sum(
                    float(row.get("total_profiled_time", 0.0))
                    + float(row.get("update_time", 0.0))
                    for row in rows
                ),
                "peak_cuda_allocated_mb": peak_cuda,
                "peak_rss_mb": peak_rss,
                "incomplete": not completed,
                "updated_at": datetime.now().isoformat(),
            }
        )
        if metrics is not None:
            profile["metrics"] = metrics
        return profile

    if checkpoint is not None:
        checkpoint(refresh_profile())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for batch_idx, batch in enumerate(batches):
        if args.max_updates is not None and len(rows) >= args.max_updates:
            break

        if batch_idx == 0:
            opt = Optimizer(
                batch,
                getattr(ds, "features", None),
                subcoms_depth=args.smart_depth,
                method=method,
                baseline_iter=baseline_iter,
                verbose=args.verbose,
                use_gpu=use_gpu,
                aggregation_mode=args.aggregation_mode,
            )
            device_name = str(opt.runtime_device())

            if ":" in str(batch_strategy):
                init_batch_number = str(batch_strategy).split(":", 1)[0]
                init_partition = _compute_launch_initial_partition(
                    adj_matrix=batch,
                    dataset_name=ds.name,
                    init_batch_number=init_batch_number,
                    cache_dir=args.cache_dir,
                    subcoms_depth=opt.subcoms_depth,
                    device=opt.runtime_device(),
                    verbose=args.verbose,
                )
                if init_partition.dim() == 1:
                    init_partition = init_partition.unsqueeze(0)
                opt.set_communities(communities=init_partition)
                continue

            affected_nodes_mask = torch.zeros(opt.nodes_num, dtype=torch.bool, device=opt.runtime_device())
            active_nodes = batch.coalesce().indices().unique() if batch.is_sparse else torch.nonzero(batch, as_tuple=True)[0].unique()
            affected_nodes_mask[active_nodes.to(opt.runtime_device())] = True
        else:
            start = now(opt.runtime_device())
            affected_nodes_mask = opt.update_adj(batch, return_mask=True)
            update_time = elapsed(start, opt.runtime_device())

        if batch_idx == 0 and ":" not in str(batch_strategy):
            update_time = 0.0

        row, _profiled_time = profile_smart_update(
            opt,
            affected_nodes_mask,
            method,
            args.aggregation_mode,
            args.smart_radius,
            ds.is_directed,
            variant,
        )
        row["batch_idx"] = batch_idx
        row["update_time"] = update_time
        rows.append(row)
        if checkpoint is not None:
            checkpoint(refresh_profile())

    full_adj = _compute_full_adj(ds.adj)
    final_partition = opt.coms[0].detach().cpu()
    final_modularity = Metrics.modularity(full_adj, final_partition, directed=ds.is_directed)
    metrics = {"Final modularity": final_modularity}
    if args.ground_truth_metrics and getattr(ds, "label", None) is not None:
        metrics.update(calculate_ground_truth_metrics(ds.label, final_partition))
        metrics["Labels modularity"] = Metrics.modularity(full_adj, ds.label, directed=ds.is_directed)

    return refresh_profile(metrics, completed=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--batches", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--feature-modes", nargs="+", default=["dataset"])
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="results/paper_icdm")
    parser.add_argument("--name", default=None)
    parser.add_argument("--smart-depth", type=int, default=3)
    parser.add_argument("--smart-radius", type=int, default=1)
    parser.add_argument("--aggregation-mode", default="norm")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full"],
        choices=["full", "no_closure", "no_contraction"],
        help="Smart-mode variants to profile for direct closure/contraction ablation.",
    )
    parser.add_argument("--baseline-iter", type=int, default=None)
    parser.add_argument("--random-feature-dim", type=int, default=64)
    parser.add_argument("--random-feature-seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int, default=20)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--force-undirected", action="store_true")
    parser.add_argument("--ground-truth-metrics", action="store_true")
    parser.add_argument("--catch-errors", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    output_name = args.name or f"smart_workload_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir)
    errors = []
    profiles = []
    active_profile = None
    interrupted_by_signal = None
    output_path = output_dir / f"{output_name}.json"

    def payload(status: str) -> dict[str, Any]:
        current_profiles = list(profiles)
        if active_profile is not None:
            current_profiles.append(active_profile)
        result = {
            "generated_at": datetime.now().isoformat(),
            "profile_schema": "comnetx_smart_workload_v1",
            "status": status,
            "checkpointed": True,
            "parameters": vars(args),
            "profiles": current_profiles,
            "errors": errors,
        }
        if interrupted_by_signal is not None:
            result["interrupted_by_signal"] = interrupted_by_signal
        return result

    def checkpoint(status: str = "running") -> None:
        write_json(output_path, payload(status))
        if errors:
            write_json(output_dir / f"errors_{output_name}.json", errors)

    def profile_checkpoint(profile: dict[str, Any]) -> None:
        nonlocal active_profile
        active_profile = profile
        checkpoint("running")

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal interrupted_by_signal
        interrupted_by_signal = signal.Signals(signum).name
        checkpoint("interrupted")
        raise SystemExit(f"Interrupted by {interrupted_by_signal}; checkpoint was written to {output_path}.")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    checkpoint("started")

    for dataset in args.datasets:
        for batch_strategy in args.batches:
            for method in args.methods:
                feature_modes = args.feature_modes if method in FEATURE_METHODS else ["dataset"]
                for feature_mode in feature_modes:
                    for variant in args.variants:
                        try:
                            profile = profile_run(
                                args,
                                dataset,
                                batch_strategy,
                                method,
                                feature_mode,
                                variant,
                                checkpoint=profile_checkpoint,
                            )
                        except Exception as exc:
                            if active_profile is not None and active_profile.get("rows"):
                                active_profile["failed"] = True
                                active_profile["error"] = str(exc)
                                profiles.append(active_profile)
                                active_profile = None
                            if not args.catch_errors:
                                raise
                            errors.append(
                                {
                                    "dataset": dataset,
                                    "batch_strategy": str(batch_strategy),
                                    "method": method,
                                    "feature_mode": feature_mode,
                                    "variant": variant,
                                    "error": str(exc),
                                }
                            )
                            print(f"Error on {dataset} {batch_strategy} {method} {feature_mode} {variant}: {exc}")
                            checkpoint("running")
                            continue
                        active_profile = None
                        profiles.append(profile)
                        checkpoint("running")

    checkpoint("completed")
    print(json.dumps({"output": str(output_path), "profiles": len(profiles), "errors": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
