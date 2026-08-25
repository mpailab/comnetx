"""Research prototype for collision-free, nested hierarchy updates.

This research script does **not** modify :mod:`src.optimizer`.  It retains two
historical experimental variants for comparison with the repaired production
``Optimizer.run``:

* every provisional or projected block receives a label that has never been
  used by this optimizer instance;
* cross-level propagation preserves only the source partition structure, not
  its numeric label namespace;
* a join-based repair is retained as a negative control; and
* the recommended parent-quotient variant uses updated level ``l - 1`` blocks
  as level-``l`` atoms and independently rebuilds the quotient from
  ``A_t[U_l]``, without carrying a destructively cut adjacency.

The command-line audit runs the repaired production optimizer and both
historical variants on the same stream and initial Python-Leiden hierarchy.
Its timing scope matches the article's smart-mode clock: radius expansion plus
``run``, less backend format conversion; adjacency-update time is excluded.

Example (inside the development container)::

    PYTHONPATH=.:src python scripts/paper/prototype_hierarchy_repair.py \
        --dataset dyn_pubmed --dataset-root /workspace/datasets \
        --batch-strategy 999:10 --force-undirected \
        --output /tmp/dyn_pubmed_hierarchy_repair.json --verbose

This is an evaluation artifact, not a proposed production implementation.
The union--find join deliberately favors transparency over GPU efficiency and
is not the recommended implementation path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch


PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import sparse  # noqa: E402
from optimizer import Optimizer  # noqa: E402
from scripts.paper.audit_stream_invariants import (  # noqa: E402
    build_python_leiden_hierarchy,
    compact_refinement_report,
    compute_update_scopes,
    load_dataset,
    outside_entry_write_report,
)


CURRENT_REFERENCE_MODULARITY = 0.7567715645


def partition_block_count(labels: torch.Tensor) -> int:
    """Return the number of blocks represented by one label vector."""
    return int(torch.unique(labels).numel())


def boundary_collision_report(
    labels: torch.Tensor,
    scope: torch.Tensor,
    *,
    sample_size: int = 8,
) -> dict[str, Any]:
    """Report numeric labels occurring on both sides of ``scope``."""
    labels_cpu = labels.detach().cpu()
    scope_cpu = scope.detach().cpu().to(torch.bool)
    inside = torch.unique(labels_cpu[scope_cpu], sorted=True)
    outside = torch.unique(labels_cpu[~scope_cpu], sorted=True)
    if inside.numel() and outside.numel():
        shared = inside[torch.isin(inside, outside)]
    else:
        shared = inside[:0]
    return {
        "has_collisions": bool(shared.numel()),
        "collision_count": int(shared.numel()),
        "sample": shared[:sample_size].tolist(),
    }


class HierarchyRepairPrototypeOptimizer(Optimizer):
    """Experimental ``Optimizer`` with explicit hierarchy repair.

    The class is kept outside ``src`` so that measurements cannot silently
    alter production behavior or previously reported results.
    """

    def _begin_fresh_label_run(self) -> None:
        current_next = (
            int(self.coms.max().item()) + 1 if self.coms.numel() else 0
        )
        previous_next = getattr(self, "_prototype_next_fresh_label", 0)
        self._prototype_next_fresh_label = max(previous_next, current_next)

    def _allocate_fresh_labels(
        self,
        count: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Allocate labels never previously issued by this instance."""
        if count < 0:
            raise ValueError("fresh-label count cannot be negative")
        start = self._prototype_next_fresh_label
        stop = start + count
        self._prototype_next_fresh_label = stop
        return torch.arange(start, stop, dtype=torch.long, device=device)

    def _fresh_partition_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Preserve a partition while moving it to a fresh namespace."""
        if labels.dim() != 1:
            raise ValueError("partition labels must be a vector")
        if labels.numel() == 0:
            return labels.to(dtype=torch.long)
        _, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        fresh = self._allocate_fresh_labels(
            int(inverse.max().item()) + 1,
            device=labels.device,
        )
        return fresh[inverse]

    @staticmethod
    def _partition_join_inverse(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        """Return compact labels for the finest common coarsening.

        Blocks of ``first`` and ``second`` form the two vertex sets of a
        bipartite graph; every original vertex contributes one edge.  Connected
        components of that graph are exactly the equivalence classes generated
        by the union of the two partition relations, i.e. their lattice join.

        The prototype uses a CPU union--find because the goal is to establish
        semantics and measure their cost before choosing a production kernel.
        """
        if first.dim() != 1 or second.dim() != 1 or first.shape != second.shape:
            raise ValueError("join inputs must be equal-length label vectors")
        if first.numel() == 0:
            return first.to(dtype=torch.long)

        _, first_inverse = torch.unique(
            first, sorted=True, return_inverse=True
        )
        _, second_inverse = torch.unique(
            second, sorted=True, return_inverse=True
        )
        first_blocks = int(first_inverse.max().item()) + 1
        second_blocks = int(second_inverse.max().item()) + 1
        parent = list(range(first_blocks + second_blocks))
        rank = [0] * len(parent)

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if rank[left_root] < rank[right_root]:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            if rank[left_root] == rank[right_root]:
                rank[left_root] += 1

        first_cpu = first_inverse.detach().cpu().tolist()
        second_cpu = second_inverse.detach().cpu().tolist()
        for first_block, second_block in zip(first_cpu, second_cpu):
            union(first_block, first_blocks + second_block)

        roots = torch.tensor(
            [find(first_block) for first_block in first_cpu],
            dtype=torch.long,
            device=first.device,
        )
        _, join_inverse = torch.unique(roots, sorted=True, return_inverse=True)
        return join_inverse

    def _fresh_partition_join(
        self,
        proposal: torch.Tensor,
        updated_finer: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Join two partitions and assign collision-free result labels."""
        join_inverse = self._partition_join_inverse(proposal, updated_finer)
        proposal_blocks = partition_block_count(proposal)
        finer_blocks = partition_block_count(updated_finer)
        join_blocks = partition_block_count(join_inverse)

        # If no proposal blocks are merged, its already-fresh namespace is a
        # valid representation of the join and need not be replaced again.
        if join_blocks == proposal_blocks:
            joined = proposal
        else:
            joined = self._fresh_partition_labels(join_inverse)
        return joined, {
            "proposal_blocks": proposal_blocks,
            "updated_finer_blocks": finer_blocks,
            "join_blocks": join_blocks,
            "proposal_blocks_merged_by_join": proposal_blocks - join_blocks,
        }

    def run(self, nodes_mask: torch.Tensor) -> None:
        """Mirror ``Optimizer.run`` with collision-free hierarchy repair."""
        compute_device = self.device
        needs_features = self._local_algorithm_requires_features()

        coms_work = self.coms
        adj_base = self.adj
        features_work = self.features if needs_features else None
        nodes_mask_work = (
            nodes_mask
            if nodes_mask.device == compute_device
            else nodes_mask.to(compute_device)
        )
        nodes = torch.nonzero(nodes_mask_work, as_tuple=True)[0]
        if nodes.numel() == 0:
            self.prototype_last_run = {
                "affected_vertices": 0,
                "fresh_labels_allocated": 0,
                "levels": [],
            }
            return

        self._begin_fresh_label_run()
        fresh_start = self._prototype_next_fresh_label

        ext_mask_work = torch.zeros_like(coms_work, dtype=torch.bool)
        for level in range(self.subcoms_depth):
            touched = coms_work[level].index_select(0, nodes)
            ext_mask_work[level] = torch.isin(
                coms_work[level], torch.unique(touched)
            )

        # Production assigns vertex IDs here.  The prototype preserves the
        # intended singleton partition but uses labels outside every current
        # and historical namespace.
        coms_work[-1, nodes_mask_work] = self._allocate_fresh_labels(
            int(nodes.numel()), device=compute_device
        )

        propagation_reports: list[dict[str, Any]] = []
        for level in range(self.subcoms_depth - 2, -1, -1):
            level_scope = ext_mask_work[level]
            propagated = self._fresh_partition_labels(
                coms_work[level + 1, level_scope]
            )
            coms_work[level, level_scope] = propagated
            propagation_reports.append(
                {
                    "target_level": level,
                    "scope_vertices": int(level_scope.sum().item()),
                    "propagated_blocks": partition_block_count(propagated),
                    "boundary_collisions_after": boundary_collision_report(
                        coms_work[level], level_scope
                    )["collision_count"],
                }
            )

        affected_nodes_level0 = torch.nonzero(
            ext_mask_work[0], as_tuple=True
        )[0]
        adj_work = sparse.reset_matrix(adj_base, affected_nodes_level0)

        level_reports: list[dict[str, Any]] = []
        for level in range(self.subcoms_depth):
            level_scope = ext_mask_work[level]
            provisional = coms_work[level, level_scope]
            ext_nodes = torch.nonzero(level_scope, as_tuple=True)[0]
            _, inverse, counts = torch.unique(
                provisional,
                sorted=True,
                return_counts=True,
                return_inverse=True,
            )
            group_count = int(counts.numel())
            if group_count == 0:
                continue

            aggregation_indices = torch.stack((inverse, ext_nodes))
            aggregation_pattern = sparse.tensor(
                aggregation_indices,
                (group_count, self.nodes_num),
                adj_work.dtype,
            )
            aggregated_adjacency = self.aggregate(
                adj_work, aggregation_pattern
            )
            del aggregation_pattern

            aggregated_features = None
            if needs_features:
                ext_features = features_work.index_select(0, ext_nodes)
                aggregated_features = torch.zeros(
                    (group_count, ext_features.size(1)),
                    dtype=ext_features.dtype,
                    device=ext_features.device,
                )
                aggregated_features.index_add_(0, inverse, ext_features)
                if self.aggregation_mode == "normalized":
                    aggregated_features /= counts.to(
                        dtype=ext_features.dtype
                    ).unsqueeze(1)

            backend_groups = self.local_algorithm(
                aggregated_adjacency,
                aggregated_features,
                level > 0,
            ).to(device=compute_device, dtype=torch.long)
            if backend_groups.numel() != group_count:
                raise ValueError(
                    "backend returned "
                    f"{backend_groups.numel()} labels for {group_count} groups"
                )

            # Restore the backend partition structure to original vertices,
            # but never recycle the provisional labels as production does.
            proposal = self._fresh_partition_labels(
                backend_groups[inverse]
            )
            join_report = None
            if level == 0:
                updated = proposal
            else:
                finer_on_scope = coms_work[level - 1, level_scope]
                finer_boundary = boundary_collision_report(
                    coms_work[level - 1], level_scope
                )
                if finer_boundary["has_collisions"]:
                    raise RuntimeError(
                        "updated finer blocks cross the current update scope; "
                        "a scope-local join cannot guarantee global refinement"
                    )
                updated, join_report = self._fresh_partition_join(
                    proposal, finer_on_scope
                )

            coms_work[level, level_scope] = updated
            post_projection_collisions = boundary_collision_report(
                coms_work[level], level_scope
            )
            adj_work = self.cut_by_partition(
                adj_work, level_scope, coms_work[level]
            )
            level_reports.append(
                {
                    "level": level,
                    "scope_vertices": int(level_scope.sum().item()),
                    "contracted_vertices": group_count,
                    "backend_blocks": partition_block_count(backend_groups),
                    "stored_blocks_on_scope": partition_block_count(updated),
                    "boundary_collisions_after_projection": (
                        post_projection_collisions["collision_count"]
                    ),
                    "join": join_report,
                }
            )

        self.prototype_last_run = {
            "affected_vertices": int(nodes.numel()),
            "fresh_labels_allocated": (
                self._prototype_next_fresh_label - fresh_start
            ),
            "strategy": "proposal_join_negative_control",
            "propagation": propagation_reports,
            "levels": level_reports,
        }


class ParentQuotientPrototypeOptimizer(HierarchyRepairPrototypeOptimizer):
    """Recommended prototype: rebuild every higher level as a parent quotient.

    Level zero keeps the adapter's existing provisional-atom semantics.  At
    every level ``l > 0``, however, the atoms are exactly the blocks of the
    already updated level ``l - 1`` restricted to ``U_l``.  The level-local
    adjacency is rebuilt independently from ``A_t[U_l]``; no cut adjacency is
    carried from an earlier backend call.  Consequently, every projected
    backend partition is a coarsening of level ``l - 1`` by construction.
    """

    def run(self, nodes_mask: torch.Tensor) -> None:
        compute_device = self.device
        needs_features = self._local_algorithm_requires_features()
        coms_work = self.coms
        adj_base = self.adj
        features_work = self.features if needs_features else None
        nodes_mask_work = nodes_mask.to(
            device=compute_device, dtype=torch.bool
        )
        nodes = torch.nonzero(nodes_mask_work, as_tuple=True)[0]
        if nodes.numel() == 0:
            self.prototype_last_run = {
                "affected_vertices": 0,
                "fresh_labels_allocated": 0,
                "strategy": "parent_quotient",
                "levels": [],
            }
            return

        self._begin_fresh_label_run()
        fresh_start = self._prototype_next_fresh_label

        ext_mask_work = torch.zeros_like(coms_work, dtype=torch.bool)
        for level in range(self.subcoms_depth):
            touched = coms_work[level].index_select(0, nodes)
            ext_mask_work[level] = torch.isin(
                coms_work[level], torch.unique(touched)
            )

        coms_work[-1, nodes_mask_work] = self._allocate_fresh_labels(
            int(nodes.numel()), device=compute_device
        )
        propagation_reports: list[dict[str, Any]] = []
        for level in range(self.subcoms_depth - 2, -1, -1):
            level_scope = ext_mask_work[level]
            propagated = self._fresh_partition_labels(
                coms_work[level + 1, level_scope]
            )
            coms_work[level, level_scope] = propagated
            propagation_reports.append(
                {
                    "target_level": level,
                    "scope_vertices": int(level_scope.sum().item()),
                    "propagated_blocks": partition_block_count(propagated),
                    "boundary_collisions_after": boundary_collision_report(
                        coms_work[level], level_scope
                    )["collision_count"],
                }
            )

        level_reports: list[dict[str, Any]] = []
        for level in range(self.subcoms_depth):
            level_scope = ext_mask_work[level]
            ext_nodes = torch.nonzero(level_scope, as_tuple=True)[0]

            if level == 0:
                atoms = coms_work[level, level_scope]
                atom_source = "fresh_provisional_level_0"
            else:
                finer_boundary = boundary_collision_report(
                    coms_work[level - 1], level_scope
                )
                if finer_boundary["has_collisions"]:
                    raise RuntimeError(
                        "updated parent-quotient atoms cross U_l; a local "
                        "projection cannot preserve global refinement"
                    )
                atoms = coms_work[level - 1, level_scope]
                atom_source = f"updated_level_{level - 1}_blocks"

            _, inverse, counts = torch.unique(
                atoms,
                sorted=True,
                return_counts=True,
                return_inverse=True,
            )
            group_count = int(counts.numel())
            if group_count == 0:
                continue

            # Rebuild A_t[U_l] independently.  In particular, do not carry a
            # destructively cut adjacency from the preceding level.
            level_adjacency = sparse.reset_matrix(adj_base, ext_nodes)
            aggregation_indices = torch.stack((inverse, ext_nodes))
            aggregation_pattern = sparse.tensor(
                aggregation_indices,
                (group_count, self.nodes_num),
                level_adjacency.dtype,
            )
            aggregated_adjacency = self.aggregate(
                level_adjacency, aggregation_pattern
            )
            del aggregation_pattern

            aggregated_features = None
            if needs_features:
                ext_features = features_work.index_select(0, ext_nodes)
                aggregated_features = torch.zeros(
                    (group_count, ext_features.size(1)),
                    dtype=ext_features.dtype,
                    device=ext_features.device,
                )
                aggregated_features.index_add_(0, inverse, ext_features)
                if self.aggregation_mode == "normalized":
                    aggregated_features /= counts.to(
                        dtype=ext_features.dtype
                    ).unsqueeze(1)

            backend_groups = self.local_algorithm(
                aggregated_adjacency,
                aggregated_features,
                level > 0,
            ).to(device=compute_device, dtype=torch.long)
            if backend_groups.numel() != group_count:
                raise ValueError(
                    "backend returned "
                    f"{backend_groups.numel()} labels for {group_count} groups"
                )

            updated = self._fresh_partition_labels(
                backend_groups[inverse]
            )
            coms_work[level, level_scope] = updated
            post_projection_collisions = boundary_collision_report(
                coms_work[level], level_scope
            )
            level_reports.append(
                {
                    "level": level,
                    "scope_vertices": int(level_scope.sum().item()),
                    "atom_source": atom_source,
                    "contracted_vertices": group_count,
                    "backend_blocks": partition_block_count(backend_groups),
                    "stored_blocks_on_scope": partition_block_count(updated),
                    "boundary_collisions_after_projection": (
                        post_projection_collisions["collision_count"]
                    ),
                    "adjacency_source": "fresh_A_t_restricted_to_U_l",
                    "destructive_cut": False,
                }
            )

        self.prototype_last_run = {
            "affected_vertices": int(nodes.numel()),
            "fresh_labels_allocated": (
                self._prototype_next_fresh_label - fresh_start
            ),
            "strategy": "parent_quotient",
            "propagation": propagation_reports,
            "levels": level_reports,
        }


def audit_one_timed_update(
    optimizer: Optimizer,
    directly_affected: torch.Tensor,
    *,
    radius: int,
) -> dict[str, Any]:
    """Measure and structurally audit one already-applied adjacency update."""
    directly_affected = directly_affected.to(
        device=optimizer.runtime_device(), dtype=torch.bool
    )
    neighborhood_start = time.perf_counter()
    expanded = optimizer.neighborhood(
        optimizer.runtime_adj(), directly_affected, step=radius
    )
    neighborhood_wall = time.perf_counter() - neighborhood_start

    # Audit preparation is deliberately outside the article-compatible clock.
    before = optimizer.coms.detach().clone()
    scopes = compute_update_scopes(before, expanded)

    conversion_start = optimizer.conversion_time
    run_start = time.perf_counter()
    optimizer.run(expanded)
    run_wall = time.perf_counter() - run_start
    conversion_delta = optimizer.conversion_time - conversion_start
    clock_wall = neighborhood_wall + run_wall

    after = optimizer.coms.detach().clone()
    levels = []
    for level in range(after.size(0)):
        scope = scopes[level]
        levels.append(
            {
                "level": level,
                "scope_vertices": int(scope.sum().item()),
                "blocks_before": partition_block_count(before[level]),
                "blocks_after": partition_block_count(after[level]),
                "boundary_collisions_before": boundary_collision_report(
                    before[level], scope
                ),
                "boundary_collisions_after": boundary_collision_report(
                    after[level], scope
                ),
                "entry_writes": outside_entry_write_report(
                    before[level], after[level], scope
                ),
            }
        )

    result = {
        "directly_affected_vertices": int(directly_affected.sum().item()),
        "radius_expanded_vertices": int(expanded.sum().item()),
        "article_clock_seconds": clock_wall - conversion_delta,
        "wall_seconds": clock_wall,
        "neighborhood_wall_seconds": neighborhood_wall,
        "run_wall_seconds": run_wall,
        "backend_conversion_seconds": conversion_delta,
        "refinement_before": compact_refinement_report(before),
        "refinement_after": compact_refinement_report(after),
        "levels": levels,
    }
    if isinstance(optimizer, HierarchyRepairPrototypeOptimizer):
        result["prototype_run"] = optimizer.prototype_last_run
    return result


def summarize_variant(
    optimizer: Optimizer,
    updates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the metrics requested for one audit variant."""
    refinement_failures = [
        index
        for index, update in enumerate(updates)
        if not update["refinement_after"]["all_adjacent_refine"]
    ]
    collision_updates = [
        index
        for index, update in enumerate(updates)
        if any(
            level["boundary_collisions_after"]["has_collisions"]
            for level in update["levels"]
        )
    ]
    outside_write_updates = [
        index
        for index, update in enumerate(updates)
        if any(
            level["entry_writes"]["outside_changed_entries"] > 0
            for level in update["levels"]
        )
    ]
    return {
        "updates": len(updates),
        "post_update_refinement_failure_updates": refinement_failures,
        "boundary_collision_updates": collision_updates,
        "outside_entry_write_updates": outside_write_updates,
        "article_clock_seconds": sum(
            update["article_clock_seconds"] for update in updates
        ),
        "wall_seconds": sum(update["wall_seconds"] for update in updates),
        "neighborhood_wall_seconds": sum(
            update["neighborhood_wall_seconds"] for update in updates
        ),
        "run_wall_seconds": sum(
            update["run_wall_seconds"] for update in updates
        ),
        "backend_conversion_seconds": sum(
            update["backend_conversion_seconds"] for update in updates
        ),
        "final_modularity": float(
            optimizer.modularity(gamma=optimizer.resolution)
        ),
        "final_block_counts_fine_to_coarse": [
            partition_block_count(optimizer.coms[level])
            for level in range(optimizer.coms.size(0))
        ],
        "final_refinement": compact_refinement_report(optimizer.coms),
    }


def run_variant(
    optimizer_type: type[Optimizer],
    dataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one isolated variant from a newly bootstrapped hierarchy."""
    # Preserve the loader-provided sparse state.  In particular, do not
    # pre-coalesce this tensor: the established Python-bootstrap audit creates
    # Optimizer from the non-coalesced snapshot and lets the bootstrap helper
    # coalesce its own working copy.  Pre-coalescing makes its in-place cuts
    # alias Optimizer.adj and audits a different graph.
    first_batch = dataset.adj[0].clone()
    common_kwargs = {
        "features": dataset.features,
        "subcoms_depth": args.depth,
        "method": "leidenalg",
        "aggregation_mode": args.aggregation_mode,
        "resolution": args.resolution,
    }
    optimizer = optimizer_type(
        first_batch,
        **common_kwargs,
    )
    # Match audit_stream_invariants.py exactly: construct Optimizer before the
    # bootstrap helper performs its in-place hierarchy cuts on the input
    # tensor, then install the resulting labels.  Reversing this order changes
    # the optimizer's accumulated adjacency and no longer reproduces the
    # historical 0.7567715645 pre-repair result.
    bootstrap = build_python_leiden_hierarchy(
        first_batch,
        depth=args.depth,
        resolution=args.resolution,
        device=optimizer.runtime_device(),
    )
    optimizer.set_communities(bootstrap.clone())
    initial_blocks = [
        partition_block_count(bootstrap[level])
        for level in range(bootstrap.size(0))
    ]
    updates: list[dict[str, Any]] = []
    update_limit = dataset.adj.size(0) - 1
    if args.max_updates is not None:
        update_limit = min(update_limit, args.max_updates)
    for update_index in range(update_limit):
        batch = dataset.adj[update_index + 1]
        affected = optimizer.update_adj(batch, return_mask=True)
        report = audit_one_timed_update(
            optimizer, affected, radius=args.radius
        )
        report["update_index"] = update_index
        updates.append(report)
    return {
        "initial_block_counts_fine_to_coarse": initial_blocks,
        "initial_refinement": compact_refinement_report(bootstrap),
        "summary": summarize_variant(optimizer, updates),
        "updates": updates,
    }


def median_metric(runs: list[dict[str, Any]], key: str) -> float:
    """Return the median scalar summary metric across isolated runs."""
    values = sorted(float(run["summary"][key]) for run in runs)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(
        name=args.dataset,
        batch_strategy=args.batch_strategy,
        paths_config=args.paths_config,
        dataset_root=args.dataset_root,
        force_undirected=args.force_undirected,
    )
    if dataset.adj.dim() != 3 or dataset.adj.size(0) < 2:
        raise ValueError("comparison requires a bootstrap batch and updates")

    optimizer_types: dict[str, type[Optimizer]] = {
        "current": Optimizer,
        "join_negative_control": HierarchyRepairPrototypeOptimizer,
        "parent_quotient": ParentQuotientPrototypeOptimizer,
    }
    runs: dict[str, list[dict[str, Any]]] = {
        name: [] for name in optimizer_types
    }
    variant_names = list(optimizer_types)
    for repetition in range(args.repetitions):
        # Alternate whole-run order.  Interleaving backend calls within one
        # stream perturbs the backend's process-global state and does not
        # keep process-global backend ordering effects balanced across runs.
        order = (
            variant_names
            if repetition % 2 == 0
            else list(reversed(variant_names))
        )
        for name in order:
            run = run_variant(optimizer_types[name], dataset, args)
            runs[name].append(run)
            if args.verbose:
                summary = run["summary"]
                print(
                    f"{name} repetition {repetition + 1}/"
                    f"{args.repetitions}: "
                    f"Q={summary['final_modularity']:.10f}, "
                    f"refinement_failures="
                    f"{len(summary['post_update_refinement_failure_updates'])}, "
                    f"clock={summary['article_clock_seconds']:.3f}s",
                    file=sys.stderr,
                )

    summaries = {
        name: variant_runs[0]["summary"]
        for name, variant_runs in runs.items()
    }
    timing_keys = (
        "article_clock_seconds",
        "wall_seconds",
        "neighborhood_wall_seconds",
        "run_wall_seconds",
        "backend_conversion_seconds",
    )
    median_timings = {
        name: {
            key: median_metric(variant_runs, key) for key in timing_keys
        }
        for name, variant_runs in runs.items()
    }
    current_time = median_timings["current"]["article_clock_seconds"]
    current_q = summaries["current"]["final_modularity"]

    comparisons = {}
    for name in ("join_negative_control", "parent_quotient"):
        variant_time = median_timings[name]["article_clock_seconds"]
        comparisons[f"{name}_vs_current"] = {
            "article_clock_ratio": (
                variant_time / current_time if current_time else None
            ),
            "article_clock_extra_seconds": variant_time - current_time,
            "final_modularity_difference": (
                summaries[name]["final_modularity"] - current_q
            ),
        }
    return {
        "schema": "comnetx_hierarchy_repair_prototype_v3",
        "status": "research comparison against repaired production Optimizer",
        "dataset": dataset.name,
        "batch_strategy": args.batch_strategy,
        "force_undirected": args.force_undirected,
        "depth": args.depth,
        "radius": args.radius,
        "resolution": args.resolution,
        "repetitions": args.repetitions,
        "bootstrap": "baselines.leiden.leidenalg_partition",
        "timing_scope": (
            "radius expansion plus Optimizer.run, less backend tensor-to-igraph "
            "conversion; update_adj and bootstrap excluded"
        ),
        "reference_current_modularity": args.reference_current_modularity,
        "current_modularity_difference_from_reference": (
            current_q - args.reference_current_modularity
        ),
        "comparisons": comparisons,
        "summary": summaries,
        "median_timing": median_timings,
        "variants": {
            name: {
                "representative_run": variant_runs[0],
                "repetition_summaries": [
                    run["summary"] for run in variant_runs
                ],
            }
            for name, variant_runs in runs.items()
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dyn_pubmed")
    parser.add_argument("--batch-strategy", default="999:10")
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument("--paths-config")
    roots.add_argument("--dataset-root")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--aggregation-mode", default="sum")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--force-undirected", action="store_true")
    parser.add_argument(
        "--reference-current-modularity",
        type=float,
        default=CURRENT_REFERENCE_MODULARITY,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
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
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(json.dumps(report["comparisons"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
