"""Profile ComNetX smart-mode workload, timing breakdown, and memory.

The regular launcher reports end-to-end per-update quality and runtime. For the
paper we also need evidence that the runtime reduction is caused by small local
workloads rather than hidden full-graph work. This script calls the production
``Optimizer.run`` implementation with its instrumentation enabled and records:

* directly affected and radius-expanded vertices;
* post-closure vertices at every hierarchy level;
* contracted backend nodes and edges;
* per-level restricted boundary-objective certificates;
* identity-atom ranking-certificate comparisons;
* time spent in update, radius expansion, closure, restriction, aggregation,
  backend, projection, and certificate phases;
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

from datasets import Dataset  # noqa: E402
from metrics import Metrics, calculate_ground_truth_metrics  # noqa: E402
from optimizer import Optimizer  # noqa: E402


FEATURE_METHODS = {"magi", "dmon", "mfc", "s2cag"}
ITER_DEFAULTS = {"flmig": 1, "s2cag": 10, "dmon": 10, "magi": 10, "mfc": 100}


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
    # Linux reports kilobytes, macOS bytes. Keep a conservative fallback so
    # local dry runs remain interpretable.
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


def profile_smart_update(
    opt: Optimizer,
    affected_nodes_mask: torch.Tensor,
    radius: int,
    directed: bool,
    variant: str = "full",
) -> tuple[dict[str, Any], float]:
    """Run one instrumented production update under a registered ablation.

    ``full`` is exactly the production path. ``no_closure`` changes only the
    scope to the radius-expanded vertices. ``no_contraction`` changes only the
    level-zero atoms to original-vertex singletons; every higher level remains
    a parent quotient of the updated preceding level.
    """
    run_options = {
        "full": {
            "closure_enabled": True,
            "base_atom_policy": "hierarchical",
            "mechanism_label": "repaired full method",
        },
        "no_closure": {
            "closure_enabled": False,
            "base_atom_policy": "hierarchical",
            "mechanism_label": "radius-only scope",
        },
        "no_contraction": {
            "closure_enabled": True,
            "base_atom_policy": "singleton",
            "mechanism_label": "vertex-level base atoms",
        },
    }
    if variant not in run_options:
        raise ValueError(f"unsupported smart-profile variant: {variant}")

    device = opt.runtime_device()
    options = run_options[variant]
    row: dict[str, Any] = {
        "affected_vertices": int(affected_nodes_mask.sum().item()),
        "variant": variant,
        "mechanism_label": options["mechanism_label"],
    }

    start = now(device)
    expanded_nodes_mask = opt.neighborhood(
        opt.runtime_adj(),
        affected_nodes_mask,
        step=radius,
    )
    radius_time = elapsed(start, device)
    row["radius_time"] = radius_time
    row["radius_vertices"] = int(expanded_nodes_mask.sum().item())

    opt.run(
        expanded_nodes_mask,
        closure_enabled=bool(options["closure_enabled"]),
        base_atom_policy=str(options["base_atom_policy"]),
        collect_profile=True,
    )
    production_profile = opt.last_run_profile
    if not isinstance(production_profile, dict):
        raise RuntimeError("Optimizer did not return the requested run profile")
    row.update(production_profile)
    optimizer_time = float(production_profile["total_profiled_time"])
    row["optimizer_time"] = optimizer_time
    row["total_profiled_time"] = radius_time + optimizer_time
    backend_conversion = float(production_profile["backend_conversion_time"])
    row["principal_profiled_time"] = max(
        0.0,
        row["total_profiled_time"] - backend_conversion,
    )
    row["timing_accounting"] = (
        "certificate_time is excluded once from total_profiled_time; "
        "backend_conversion_time is a diagnostic subcomponent of backend_time, "
        "included once in total_profiled_time, and subtracted once only for "
        "principal_profiled_time"
    )
    level_rows = production_profile["levels"]
    row["closure_vertices_by_level"] = [
        int(item["closure_vertices"]) for item in level_rows
    ]
    row["contracted_nodes_by_level"] = [
        int(item.get("contracted_nodes", 0)) for item in level_rows
    ]
    row["contracted_edges_by_level"] = [
        int(item.get("contracted_edges", 0)) for item in level_rows
    ]
    row["boundary_certificates_by_level"] = [
        item.get("boundary_certificate") for item in level_rows
    ]
    row["identity_ranking_certificates_by_level"] = [
        item.get("ranking_certificate") for item in level_rows
    ]
    row["modularity"] = Metrics.modularity(opt.adj, opt.coms[0], directed=directed)
    row.update(cuda_memory(device))
    row["rss_max_mb"] = current_rss_mb()
    return row, row["total_profiled_time"]


def hierarchy_is_nested(communities: torch.Tensor) -> bool:
    """Return whether every stored fine block refines the next coarse row."""
    if communities.dim() != 2:
        return False
    nodes = communities.size(1)
    if communities.numel() and (
        int(communities.min().item()) < 0
        or int(communities.max().item()) >= nodes
    ):
        return False
    for level in range(communities.size(0) - 1):
        fine = communities[level]
        coarse = communities[level + 1]
        if not torch.equal(coarse, coarse.index_select(0, fine)):
            return False
    return True


def isolated_scope_invariants(
    before: torch.Tensor,
    after: torch.Tensor,
    scope: torch.Tensor,
) -> dict[str, Any]:
    """Audit a discarded radius-only step without relying on label identity.

    Radius-only execution canonicalizes each complete hierarchy row.  Numeric
    representatives outside the scope may consequently change even though the
    partition induced on those vertices is unchanged.  Compare that partition
    relation explicitly and retain numeric writes only as a diagnostic.
    """
    collisions = []
    outside_partition_preserved = []
    outside_numeric_writes = []
    outside = ~scope
    outside_vertices = torch.nonzero(outside, as_tuple=False).flatten()
    for level in range(after.size(0)):
        inside_labels = torch.unique(after[level, scope])
        outside_labels = torch.unique(after[level, outside])
        collisions.append(
            int(torch.isin(inside_labels, outside_labels).sum().item())
            if inside_labels.numel() and outside_labels.numel()
            else 0
        )
        before_outside = before[level, outside]
        after_outside = after[level, outside]
        before_partition = Optimizer.canonicalize_partition(
            before_outside,
            outside_vertices,
        )
        after_partition = Optimizer.canonicalize_partition(
            after_outside,
            outside_vertices,
        )
        outside_partition_preserved.append(
            bool(torch.equal(before_partition, after_partition))
        )
        outside_numeric_writes.append(
            int(torch.count_nonzero(before_outside != after_outside).item())
        )
    return {
        "nested_before": hierarchy_is_nested(before),
        "nested_after": hierarchy_is_nested(after),
        "scope_label_collisions_by_level": collisions,
        "outside_partition_preserved_by_level": outside_partition_preserved,
        "outside_numeric_writes_by_level": outside_numeric_writes,
    }


def clone_optimizer_state(opt: Optimizer) -> Optimizer:
    """Clone labels while sharing read-only adjacency/features for one step."""
    features = None if opt.synthetic_features else opt.features
    cloned = Optimizer(
        opt.adj,
        features,
        communities=opt.coms.detach().clone(),
        subcoms_depth=opt.subcoms_depth,
        method=opt.method,
        baseline_iter=opt.baseline_iter,
        verbose=opt.verbose,
        use_gpu=opt.runtime_device().type == "cuda",
        aggregation_mode=opt.aggregation_mode,
        resolution=opt.resolution,
    )
    return cloned


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
    # The production measurement image provides the optional DGC wheel used by
    # launcher bootstrap. Keep this import on the actual run path so the
    # instrumentation helper remains unit-testable in the lightweight image.
    from launcher import (
        _compute_full_adj,
        _compute_launch_initial_partition,
        _iter_adjacency_batches,
    )

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
        "state_policy": (
            "isolated_one_step_from_production_pre_update_state"
            if variant == "no_closure"
            else "persistent_variant_trajectory"
        ),
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
        "total_principal_profiled_time": 0.0,
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
                "total_principal_profiled_time": sum(
                    float(row.get("principal_profiled_time", 0.0))
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

    last_evaluated_partition = None
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

        run_optimizer = opt
        persistent_before = None
        expanded_for_audit = None
        if variant == "no_closure":
            persistent_before = opt.coms.detach().clone()
            run_optimizer = clone_optimizer_state(opt)
            expanded_for_audit = run_optimizer.neighborhood(
                run_optimizer.runtime_adj(),
                affected_nodes_mask,
                step=args.smart_radius,
            )
        row, _profiled_time = profile_smart_update(
            run_optimizer,
            affected_nodes_mask,
            args.smart_radius,
            ds.is_directed,
            variant,
        )
        if variant == "no_closure":
            assert persistent_before is not None
            assert expanded_for_audit is not None
            row["state_policy"] = profile["state_policy"]
            row["persistent_state_unchanged_by_control"] = torch.equal(
                opt.coms,
                persistent_before,
            )
            row["isolated_invariants"] = isolated_scope_invariants(
                persistent_before,
                run_optimizer.coms,
                expanded_for_audit,
            )
            last_evaluated_partition = run_optimizer.coms[0].detach().cpu()

            # Advance only the valid production trajectory. The radius-only
            # result above is discarded and can never contaminate batch t+1.
            production_scope = opt.neighborhood(
                opt.runtime_adj(),
                affected_nodes_mask,
                step=args.smart_radius,
            )
            opt.run(production_scope)
            row["production_post_advance_nested"] = hierarchy_is_nested(opt.coms)
        else:
            row["state_policy"] = profile["state_policy"]
            row["persistent_state_unchanged_by_control"] = None
            last_evaluated_partition = opt.coms[0].detach().cpu()
        row["batch_idx"] = batch_idx
        row["update_time"] = update_time
        rows.append(row)
        if checkpoint is not None:
            checkpoint(refresh_profile())

    full_adj = _compute_full_adj(ds.adj)
    if opt is None or last_evaluated_partition is None:
        raise RuntimeError("profile run produced no evaluated updates")
    final_partition = last_evaluated_partition
    final_modularity = Metrics.modularity(full_adj, final_partition, directed=ds.is_directed)
    metrics = {"Final modularity": final_modularity}
    if variant == "no_closure":
        metrics["interpretation"] = (
            "final one-step radius-only counterfactual from the production "
            "pre-update state; not a persistent multibatch trajectory"
        )
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
            "profile_schema": "comnetx_smart_workload_v2",
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
