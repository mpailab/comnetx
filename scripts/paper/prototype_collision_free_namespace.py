"""Isolate label-namespace effects in the hierarchical update procedure.

This is a research-only prototype.  It does not modify or monkey-patch the
production :class:`optimizer.Optimizer`.  The prototype subclasses it and
changes exactly two namespace-related operations in ``Optimizer.run``:

* provisional groups receive fresh labels that cannot occur outside the
  current update scope; and
* every backend partition is projected with fresh labels, rather than by
  reusing entries of ``old_idx``.

The order of backend calls, closure masks, contractions, graph cuts, feature
aggregation, and backend itself are unchanged.  In particular, this script
does not join or repair adjacent hierarchy levels.  It therefore separates
numeric label collisions from independently recomputed level partitions.

The default command compares the production and prototype procedures on the
local dyn_pubmed ``999:10`` stream::

    PYTHONPATH=.:src python scripts/paper/prototype_collision_free_namespace.py \
        --dataset-root /workspace/datasets --force-undirected \
        --output /tmp/dyn_pubmed_namespace_prototype.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Type

import torch


PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_PATH / "src"
for import_path in (PROJECT_PATH, SRC_PATH):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import sparse  # noqa: E402
from optimizer import Optimizer  # noqa: E402
from scripts.paper.audit_stream_invariants import (  # noqa: E402
    _level_audit,
    build_python_leiden_hierarchy,
    compact_refinement_report,
    compute_update_scopes,
    iter_adjacency_batches,
    load_dataset,
    summarize_updates,
)


class CollisionFreeNamespaceOptimizer(Optimizer):
    """Optimizer prototype whose updated labels use a fresh global namespace."""

    def _start_label_allocation(self) -> None:
        if self.coms.numel() == 0:
            self._next_fresh_label = 0
        else:
            self._next_fresh_label = int(self.coms.max().item()) + 1
        self.last_namespace_stats: dict[str, Any] = {
            "fresh_provisional_labels": 0,
            "fresh_projected_labels": 0,
            "raw_provisional_collision_labels": 0,
            "raw_projection_collision_labels": 0,
            "by_level": [],
        }

    def _fresh_labels(self, count: int, *, device: torch.device) -> torch.Tensor:
        start = self._next_fresh_label
        stop = start + count
        self._next_fresh_label = stop
        return torch.arange(start, stop, dtype=torch.long, device=device)

    @staticmethod
    def _collision_label_count(
        candidate_labels: torch.Tensor,
        outside_labels: torch.Tensor,
    ) -> int:
        candidate_labels = torch.unique(candidate_labels)
        outside_labels = torch.unique(outside_labels)
        if candidate_labels.numel() == 0 or outside_labels.numel() == 0:
            return 0
        return int(torch.isin(candidate_labels, outside_labels).sum().item())

    def _fresh_equivalent_partition(
        self,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Return an equivalence-identical partition with all-new labels."""
        unique_labels, inverse = torch.unique(
            labels,
            sorted=True,
            return_inverse=True,
        )
        fresh = self._fresh_labels(
            int(unique_labels.numel()),
            device=labels.device,
        )
        return fresh[inverse], int(unique_labels.numel())

    def run(self, nodes_mask: torch.Tensor) -> None:
        """Run the production update flow with namespace-only corrections."""
        compute_device = self.device
        needs_features = self._local_algorithm_requires_features()
        coms_work = self.coms
        adj_base = self.adj
        features_work = self.features if needs_features else None
        nodes_mask_work = nodes_mask.to(device=compute_device, dtype=torch.bool)
        nodes = torch.nonzero(nodes_mask_work, as_tuple=True)[0]
        if nodes.numel() == 0:
            self.last_namespace_stats = {
                "fresh_provisional_labels": 0,
                "fresh_projected_labels": 0,
                "raw_provisional_collision_labels": 0,
                "raw_projection_collision_labels": 0,
                "by_level": [],
            }
            return

        ext_mask_work = torch.zeros_like(coms_work, dtype=torch.bool)
        for level in range(self.subcoms_depth):
            touched = coms_work[level].index_select(0, nodes)
            ext_mask_work[level] = torch.isin(
                coms_work[level],
                torch.unique(touched),
            )

        self._start_label_allocation()
        level_stats = [
            {
                "level": level,
                "fresh_provisional_labels": 0,
                "fresh_projected_labels": 0,
                "raw_provisional_collision_labels": 0,
                "raw_projection_collision_labels": 0,
            }
            for level in range(self.subcoms_depth)
        ]

        # Production uses vertex indices here.  Preserve singleton semantics,
        # but allocate labels above the entire current hierarchy namespace.
        last_level = self.subcoms_depth - 1
        raw_last_labels = nodes
        outside_last_labels = coms_work[last_level, ~nodes_mask_work]
        raw_collisions = self._collision_label_count(
            raw_last_labels,
            outside_last_labels,
        )
        fresh_singletons = self._fresh_labels(
            int(nodes.numel()),
            device=compute_device,
        )
        coms_work[last_level, nodes_mask_work] = fresh_singletons
        level_stats[last_level]["fresh_provisional_labels"] += int(nodes.numel())
        level_stats[last_level]["raw_provisional_collision_labels"] += raw_collisions

        # Production copies the next-level numeric labels into this level.
        # Copy only their equivalence relation and give it a fresh namespace.
        for level in range(self.subcoms_depth - 2, -1, -1):
            level_scope = ext_mask_work[level]
            raw_labels = coms_work[level + 1, level_scope]
            raw_collisions = self._collision_label_count(
                raw_labels,
                coms_work[level, ~level_scope],
            )
            provisional, allocated = self._fresh_equivalent_partition(raw_labels)
            coms_work[level, level_scope] = provisional
            level_stats[level]["fresh_provisional_labels"] += allocated
            level_stats[level]["raw_provisional_collision_labels"] += raw_collisions

        affected_nodes_level0 = torch.nonzero(
            ext_mask_work[0],
            as_tuple=True,
        )[0]
        adj_work = sparse.reset_matrix(adj_base, affected_nodes_level0)

        for level in range(self.subcoms_depth):
            level_scope = ext_mask_work[level]
            scoped_labels = coms_work[level, level_scope]
            ext_nodes = torch.nonzero(level_scope, as_tuple=True)[0]
            old_idx, inverse, counts = torch.unique(
                scoped_labels,
                sorted=True,
                return_counts=True,
                return_inverse=True,
            )
            groups = int(old_idx.numel())
            if groups == 0:
                continue

            aggregation_indices = torch.stack((inverse, ext_nodes))
            aggregation_pattern = sparse.tensor(
                aggregation_indices,
                (groups, self.nodes_num),
                adj_work.dtype,
            )
            aggregated_adjacency = self.aggregate(
                adj_work,
                aggregation_pattern,
            )
            del aggregation_pattern

            aggregated_features = None
            if needs_features:
                ext_features = features_work.index_select(0, ext_nodes)
                aggregated_features = torch.zeros(
                    (groups, ext_features.size(1)),
                    dtype=ext_features.dtype,
                    device=ext_features.device,
                )
                aggregated_features.index_add_(0, inverse, ext_features)
                if self.aggregation_mode == "normalized":
                    aggregated_features /= counts.to(
                        dtype=ext_features.dtype
                    ).unsqueeze(1)

            backend_labels = self.local_algorithm(
                aggregated_adjacency,
                aggregated_features,
                level > 0,
            ).to(device=compute_device, dtype=torch.long)

            # ``old_idx[backend_labels]`` has the right equivalence relation,
            # but may reuse a label still present outside this pre-update scope.
            raw_projection = old_idx[backend_labels[inverse]]
            raw_collisions = self._collision_label_count(
                raw_projection,
                coms_work[level, ~level_scope],
            )
            fresh_backend, allocated = self._fresh_equivalent_partition(
                backend_labels
            )
            coms_work[level, level_scope] = fresh_backend[inverse]
            level_stats[level]["fresh_projected_labels"] += allocated
            level_stats[level]["raw_projection_collision_labels"] += raw_collisions

            adj_work = self.cut_by_partition(
                adj_work,
                level_scope,
                coms_work[level],
            )

        self.last_namespace_stats = {
            "fresh_provisional_labels": sum(
                item["fresh_provisional_labels"] for item in level_stats
            ),
            "fresh_projected_labels": sum(
                item["fresh_projected_labels"] for item in level_stats
            ),
            "raw_provisional_collision_labels": sum(
                item["raw_provisional_collision_labels"] for item in level_stats
            ),
            "raw_projection_collision_labels": sum(
                item["raw_projection_collision_labels"] for item in level_stats
            ),
            "by_level": level_stats,
        }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_audit_update(
    optimizer: Optimizer,
    directly_affected: torch.Tensor,
    *,
    radius: int,
) -> dict[str, Any]:
    """Audit one update while timing only algorithmic phases."""
    device = optimizer.runtime_device()
    directly_affected = directly_affected.to(device=device, dtype=torch.bool)

    _synchronize(device)
    start = time.perf_counter()
    expanded = optimizer.neighborhood(
        optimizer.runtime_adj(),
        directly_affected,
        step=radius,
    )
    _synchronize(device)
    neighborhood_seconds = time.perf_counter() - start

    before = optimizer.coms.detach().clone()
    scopes = compute_update_scopes(before, expanded)
    refinement_before = compact_refinement_report(before)
    backend_calls_before = optimizer.local_algorithm_calls

    _synchronize(device)
    start = time.perf_counter()
    optimizer.run(expanded)
    _synchronize(device)
    optimizer_run_seconds = time.perf_counter() - start

    after = optimizer.coms.detach().clone()
    report = {
        "directly_affected_vertices": int(directly_affected.sum().item()),
        "radius_expanded_vertices": int(expanded.sum().item()),
        "backend_calls": optimizer.local_algorithm_calls - backend_calls_before,
        "reported_modularity": optimizer.modularity(gamma=optimizer.resolution),
        "refinement_before": refinement_before,
        "refinement_after": compact_refinement_report(after),
        "levels": _level_audit(before, after, scopes),
        "timing": {
            "neighborhood_seconds": neighborhood_seconds,
            "optimizer_run_seconds": optimizer_run_seconds,
        },
    }
    if isinstance(optimizer, CollisionFreeNamespaceOptimizer):
        report["namespace"] = optimizer.last_namespace_stats
    return report


def _timing_summary(updates: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "adjacency_update_seconds": sum(
            update["timing"]["adjacency_update_seconds"] for update in updates
        ),
        "neighborhood_seconds": sum(
            update["timing"]["neighborhood_seconds"] for update in updates
        ),
        "optimizer_run_seconds": sum(
            update["timing"]["optimizer_run_seconds"] for update in updates
        ),
        "algorithm_total_seconds": sum(
            sum(update["timing"].values()) for update in updates
        ),
    }


def _namespace_summary(updates: list[dict[str, Any]]) -> dict[str, Any] | None:
    namespace_updates = [update.get("namespace") for update in updates]
    if not any(namespace_updates):
        return None
    keys = (
        "fresh_provisional_labels",
        "fresh_projected_labels",
        "raw_provisional_collision_labels",
        "raw_projection_collision_labels",
    )
    return {
        key: sum(int(item[key]) for item in namespace_updates if item)
        for key in keys
    }


def run_variant(
    *,
    optimizer_type: Type[Optimizer],
    first_batch: torch.Tensor,
    update_batches: Iterable[torch.Tensor],
    features: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one fresh optimizer instance from an identical stream snapshot."""
    # Keep the initialization semantics of audit_stream_invariants.py while
    # isolating each variant from sparse in-place operations in the other run.
    initial_adjacency = first_batch.clone()
    optimizer = optimizer_type(
        initial_adjacency,
        features,
        subcoms_depth=args.depth,
        method="leidenalg",
        use_gpu=args.use_gpu,
        aggregation_mode=args.aggregation_mode,
        resolution=args.resolution,
    )

    bootstrap_start = time.perf_counter()
    hierarchy = build_python_leiden_hierarchy(
        initial_adjacency,
        depth=args.depth,
        resolution=args.resolution,
        device=optimizer.runtime_device(),
    )
    optimizer.set_communities(hierarchy)
    bootstrap_seconds = time.perf_counter() - bootstrap_start

    updates: list[dict[str, Any]] = []
    stream_start = time.perf_counter()
    for update_index, batch in enumerate(update_batches):
        if args.max_updates is not None and update_index >= args.max_updates:
            break
        device = optimizer.runtime_device()
        _synchronize(device)
        update_start = time.perf_counter()
        directly_affected = optimizer.update_adj(batch, return_mask=True)
        _synchronize(device)
        adjacency_update_seconds = time.perf_counter() - update_start
        report = timed_audit_update(
            optimizer,
            directly_affected,
            radius=args.radius,
        )
        report.update(
            {
                "batch_index": update_index + 1,
                "update_index": update_index,
                "batch_nonzero_entries": int(batch.coalesce()._nnz())
                if batch.is_sparse
                else int(torch.count_nonzero(batch).item()),
            }
        )
        report["timing"]["adjacency_update_seconds"] = adjacency_update_seconds
        updates.append(report)
    audited_stream_wall_seconds = time.perf_counter() - stream_start

    summary = summarize_updates(updates)
    summary["timing"] = {
        **_timing_summary(updates),
        "bootstrap_seconds": bootstrap_seconds,
        "audited_stream_wall_seconds": audited_stream_wall_seconds,
    }
    namespace = _namespace_summary(updates)
    if namespace is not None:
        summary["namespace"] = namespace
    return {
        "initial_refinement": compact_refinement_report(hierarchy),
        "updates": updates,
        "summary": summary,
    }


def _median_timing(repetitions: list[dict[str, Any]]) -> dict[str, float]:
    timing_keys = repetitions[0]["summary"]["timing"].keys()
    return {
        key: statistics.median(
            repetition["summary"]["timing"][key]
            for repetition in repetitions
        )
        for key in timing_keys
    }


def _compact_repetition(run: dict[str, Any]) -> dict[str, Any]:
    return {"summary": run["summary"]}


def _structural_totals(run: dict[str, Any]) -> dict[str, int]:
    updates = run["updates"]
    return {
        "shared_boundary_labels": sum(
            level["scope_label_overlap_after"]["shared_label_count"]
            for update in updates
            for level in update["levels"]
        ),
        "violating_fine_blocks": sum(
            adjacent["violating_fine_blocks"]
            for update in updates
            for adjacent in update["refinement_after"]["adjacent"]
        ),
        "vertices_in_violating_blocks": sum(
            adjacent["vertices_in_violating_blocks"]
            for update in updates
            for adjacent in update["refinement_after"]["adjacent"]
        ),
        "outside_entry_writes": sum(
            level["entry_writes"]["outside_changed_entries"]
            for update in updates
            for level in update["levels"]
        ),
    }


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(
        name=args.dataset,
        batch_strategy=args.batch_strategy,
        paths_config=args.paths_config,
        dataset_root=args.dataset_root,
        force_undirected=args.force_undirected,
    )
    batches = list(iter_adjacency_batches(dataset.adj))
    if len(batches) < 2:
        raise ValueError("the bootstrap strategy requires at least two batches")
    first_batch = batches[0]
    update_batches = batches[1:]

    variants: dict[str, tuple[Type[Optimizer], list[dict[str, Any]]]] = {
        "current": (Optimizer, []),
        "collision_free_namespace": (CollisionFreeNamespaceOptimizer, []),
    }
    # Alternate order on successive repetitions to reduce warm-cache bias.
    names = list(variants)
    for repetition in range(args.repetitions):
        order = names if repetition % 2 == 0 else list(reversed(names))
        for name in order:
            optimizer_type, runs = variants[name]
            run = run_variant(
                optimizer_type=optimizer_type,
                first_batch=first_batch,
                update_batches=update_batches,
                features=dataset.features,
                args=args,
            )
            runs.append(run)
            print(
                f"{name} repetition {repetition + 1}/{args.repetitions}: "
                f"Q={run['summary']['final_reported_modularity']:.10f}, "
                f"run={run['summary']['timing']['optimizer_run_seconds']:.3f}s, "
                f"refinement_failures="
                f"{len(run['summary']['post_update_refinement_failure_updates'])}, "
                f"collision_updates="
                f"{len(run['summary']['scope_collision_after_updates'])}",
                file=sys.stderr,
            )

    variant_payload: dict[str, Any] = {}
    for name, (_, runs) in variants.items():
        variant_payload[name] = {
            "representative_run": runs[0],
            "repetitions": [_compact_repetition(run) for run in runs],
            "median_timing": _median_timing(runs),
        }

    current = variant_payload["current"]["representative_run"]["summary"]
    prototype = variant_payload["collision_free_namespace"][
        "representative_run"
    ]["summary"]
    current_time = variant_payload["current"]["median_timing"][
        "optimizer_run_seconds"
    ]
    prototype_time = variant_payload["collision_free_namespace"][
        "median_timing"
    ]["optimizer_run_seconds"]
    paired_runtime_ratios = [
        prototype_run["summary"]["timing"]["optimizer_run_seconds"]
        / current_run["summary"]["timing"]["optimizer_run_seconds"]
        for current_run, prototype_run in zip(
            variants["current"][1],
            variants["collision_free_namespace"][1],
        )
    ]
    current_totals = _structural_totals(
        variant_payload["current"]["representative_run"]
    )
    prototype_totals = _structural_totals(
        variant_payload["collision_free_namespace"]["representative_run"]
    )
    comparison = {
        "reference_modularity": args.reference_modularity,
        "current_modularity": current["final_reported_modularity"],
        "prototype_modularity": prototype["final_reported_modularity"],
        "prototype_minus_current_modularity": (
            prototype["final_reported_modularity"]
            - current["final_reported_modularity"]
        ),
        "current_post_update_refinement_failures": current[
            "post_update_refinement_failure_updates"
        ],
        "prototype_post_update_refinement_failures": prototype[
            "post_update_refinement_failure_updates"
        ],
        "current_scope_collision_updates": current[
            "scope_collision_after_updates"
        ],
        "prototype_scope_collision_updates": prototype[
            "scope_collision_after_updates"
        ],
        "current_outside_entry_write_updates": current[
            "outside_entry_write_updates"
        ],
        "prototype_outside_entry_write_updates": prototype[
            "outside_entry_write_updates"
        ],
        "current_structural_totals": current_totals,
        "prototype_structural_totals": prototype_totals,
        "median_optimizer_run_seconds_current": current_time,
        "median_optimizer_run_seconds_prototype": prototype_time,
        "prototype_to_current_runtime_ratio": prototype_time / current_time,
        "median_paired_optimizer_run_ratio": statistics.median(
            paired_runtime_ratios
        ),
        "paired_optimizer_run_ratios": paired_runtime_ratios,
    }
    return {
        "schema": "comnetx_collision_free_namespace_prototype_v1",
        "research_scope": (
            "Fresh collision-free provisional and projected labels only; no "
            "cross-level hierarchy join or repair."
        ),
        "dataset": dataset.name,
        "batch_strategy": args.batch_strategy,
        "nodes": int(first_batch.size(0)),
        "depth": args.depth,
        "radius": args.radius,
        "resolution": args.resolution,
        "force_undirected": args.force_undirected,
        "environment": {
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "device": "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu",
        },
        "repetitions": args.repetitions,
        "variants": variant_payload,
        "comparison": comparison,
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
    parser.add_argument("--dataset", default="dyn_pubmed")
    parser.add_argument("--batch-strategy", default="999:10")
    root_group = parser.add_mutually_exclusive_group()
    root_group.add_argument("--paths-config")
    root_group.add_argument("--dataset-root")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--aggregation-mode", default="sum")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--force-undirected", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--reference-modularity",
        type=float,
        default=0.7567715644836426,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        report = run_comparison(args)
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        write_json(args.output, report)
        print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
