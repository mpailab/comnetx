"""Run deterministic audits for the adapter's structural invariants.

The audit deliberately separates properties of the contraction algebra from
properties of the current multi-level update procedure:

* ``P A P^T`` preserves total weight, aggregated in/out degrees, partition
  flows, and modularity for every partition that is constant on contracted
  groups;
* adjacent stored partitions can be checked for refinement without relying on
  the numeric values of their labels;
* temporary labels can be checked for namespace overlap between an affected
  scope and its complement; and
* an adversarial stub backend verifies that the parent-quotient update keeps
  adjacent stored levels nested even when consecutive backend proposals would
  otherwise disagree.

The script uses only deterministic in-memory graphs and is intentionally cheap
enough to run as part of article validation::

    PYTHONPATH=.:src python scripts/paper/audit_adapter_invariants.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import torch


PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from metrics import Metrics  # noqa: E402
from optimizer import Optimizer  # noqa: E402


DEFAULT_ATOL = 1e-9


def _as_1d(labels: torch.Tensor, name: str) -> torch.Tensor:
    labels = torch.as_tensor(labels).detach().cpu()
    if labels.dim() != 1:
        raise ValueError(f"{name} must be one-dimensional, got {tuple(labels.shape)}")
    return labels


def refinement_report(fine: torch.Tensor, coarse: torch.Tensor) -> dict[str, Any]:
    """Report whether partition ``fine`` refines partition ``coarse``.

    Refinement is label-invariant: every block of ``fine`` must be contained in
    exactly one block of ``coarse``.  Numeric label equality across levels is
    neither required nor assumed.
    """
    fine = _as_1d(fine, "fine")
    coarse = _as_1d(coarse, "coarse")
    if fine.numel() != coarse.numel():
        raise ValueError(
            f"partition sizes differ: fine={fine.numel()}, coarse={coarse.numel()}"
        )

    violations: list[dict[str, Any]] = []
    for fine_label in torch.unique(fine, sorted=True):
        vertices = torch.nonzero(fine == fine_label, as_tuple=True)[0]
        coarse_labels = torch.unique(coarse.index_select(0, vertices), sorted=True)
        if coarse_labels.numel() > 1:
            violations.append(
                {
                    "fine_label": fine_label.item(),
                    "vertices": vertices.tolist(),
                    "coarse_labels": coarse_labels.tolist(),
                }
            )

    return {
        "refines": not violations,
        "fine_blocks": int(torch.unique(fine).numel()),
        "coarse_blocks": int(torch.unique(coarse).numel()),
        "violations": violations,
    }


def hierarchy_refinement_report(communities: torch.Tensor) -> dict[str, Any]:
    """Check adjacent rows stored in fine-to-coarse implementation order."""
    communities = torch.as_tensor(communities).detach().cpu()
    if communities.dim() != 2:
        raise ValueError(
            "communities must have shape [levels, vertices], "
            f"got {tuple(communities.shape)}"
        )

    adjacent = []
    for level in range(max(0, communities.size(0) - 1)):
        report = refinement_report(communities[level], communities[level + 1])
        report.update({"fine_level": level, "coarse_level": level + 1})
        adjacent.append(report)

    return {
        "storage_order": "fine_to_coarse",
        "levels": int(communities.size(0)),
        "vertices": int(communities.size(1)),
        "all_adjacent_refine": all(item["refines"] for item in adjacent),
        "adjacent": adjacent,
    }


def scope_label_overlap_report(
    labels: torch.Tensor,
    affected_scope: torch.Tensor,
) -> dict[str, Any]:
    """Detect labels used both inside and outside an affected scope.

    Sharing a numeric label across the boundary means that an entrywise-local
    write may still change the induced partition block outside the scope.  The
    report is purely diagnostic; it does not assume that label sharing is
    always erroneous.
    """
    labels = _as_1d(labels, "labels")
    affected_scope = _as_1d(affected_scope, "affected_scope").to(torch.bool)
    if labels.numel() != affected_scope.numel():
        raise ValueError(
            "labels and affected_scope must have equal length: "
            f"{labels.numel()} != {affected_scope.numel()}"
        )

    inside_labels = torch.unique(labels[affected_scope], sorted=True)
    outside_labels = torch.unique(labels[~affected_scope], sorted=True)
    if inside_labels.numel() and outside_labels.numel():
        shared = inside_labels[torch.isin(inside_labels, outside_labels)]
    else:
        shared = inside_labels[:0]

    witnesses = []
    for label in shared:
        inside_vertices = torch.nonzero(
            affected_scope & (labels == label), as_tuple=True
        )[0]
        outside_vertices = torch.nonzero(
            (~affected_scope) & (labels == label), as_tuple=True
        )[0]
        witnesses.append(
            {
                "label": label.item(),
                "inside_vertices": inside_vertices.tolist(),
                "outside_vertices": outside_vertices.tolist(),
            }
        )

    return {
        "has_shared_labels": bool(shared.numel()),
        "shared_labels": shared.tolist(),
        "witnesses": witnesses,
    }


def membership_pattern(
    groups: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a binary group-membership matrix and vertex-to-group indices."""
    groups = _as_1d(groups, "groups").to(torch.long)
    if groups.numel() == 0:
        raise ValueError("groups must contain at least one vertex")
    _, inverse = torch.unique(groups, sorted=True, return_inverse=True)
    vertices = torch.arange(groups.numel(), dtype=torch.long)
    indices = torch.stack((inverse, vertices))
    values = torch.ones(groups.numel(), dtype=dtype)
    pattern = torch.sparse_coo_tensor(
        indices,
        values,
        size=(int(torch.max(inverse).item()) + 1, groups.numel()),
    ).coalesce()
    return pattern, inverse


def _weighted_degrees(adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    adj = adj.coalesce()
    row, col = adj.indices()
    weight = adj.values()
    out_degree = torch.zeros(adj.size(0), dtype=weight.dtype)
    in_degree = torch.zeros(adj.size(1), dtype=weight.dtype)
    out_degree.scatter_add_(0, row, weight)
    in_degree.scatter_add_(0, col, weight)
    return out_degree, in_degree


def partition_flow_stats(adj: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return exact directed flow statistics for each partition block."""
    adj = adj.coalesce().cpu()
    labels = _as_1d(labels, "labels")
    if adj.size(0) != adj.size(1) or labels.numel() != adj.size(0):
        raise ValueError("adjacency must be square and match the label vector")

    communities, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    row, col = adj.indices()
    weight = adj.values()
    n_communities = int(communities.numel())

    out_volume = torch.zeros(n_communities, dtype=weight.dtype)
    in_volume = torch.zeros(n_communities, dtype=weight.dtype)
    internal = torch.zeros(n_communities, dtype=weight.dtype)
    outgoing_cut = torch.zeros(n_communities, dtype=weight.dtype)
    incoming_cut = torch.zeros(n_communities, dtype=weight.dtype)

    out_volume.scatter_add_(0, inverse[row], weight)
    in_volume.scatter_add_(0, inverse[col], weight)
    same = inverse[row] == inverse[col]
    if torch.any(same):
        internal.scatter_add_(0, inverse[row][same], weight[same])
    if torch.any(~same):
        outgoing_cut.scatter_add_(0, inverse[row][~same], weight[~same])
        incoming_cut.scatter_add_(0, inverse[col][~same], weight[~same])

    return {
        "communities": communities,
        "internal_weight": internal,
        "outgoing_cut": outgoing_cut,
        "incoming_cut": incoming_cut,
        "out_volume": out_volume,
        "in_volume": in_volume,
    }


def _max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left - right)).item())


def quotient_identity_report(
    adj: torch.Tensor,
    groups: torch.Tensor,
    quotient_partition: torch.Tensor,
    *,
    gamma: float = 1.0,
    directed: bool = False,
    atol: float = DEFAULT_ATOL,
) -> dict[str, Any]:
    """Numerically audit identities induced by ``P A P^T``.

    ``quotient_partition`` partitions the contracted groups.  Its lifted
    counterpart is therefore constant on every contraction group, which is the
    precise feasible class for which modularity and flow statistics are
    preserved.
    """
    adj = adj.coalesce().cpu()
    if adj.size(0) != adj.size(1):
        raise ValueError(f"adjacency must be square, got {tuple(adj.shape)}")

    pattern, inverse = membership_pattern(groups, adj.dtype)
    quotient_partition = _as_1d(quotient_partition, "quotient_partition").to(
        torch.long
    )
    if quotient_partition.numel() != pattern.size(0):
        raise ValueError(
            "quotient_partition must contain one label per contracted group: "
            f"{quotient_partition.numel()} != {pattern.size(0)}"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        quotient = Optimizer.aggregate(adj, pattern).coalesce()
    lifted_partition = quotient_partition.index_select(0, inverse)

    row, col = adj.indices()
    expected_quotient = torch.sparse_coo_tensor(
        torch.stack((inverse[row], inverse[col])),
        adj.values(),
        size=quotient.size(),
    ).coalesce()
    quotient_delta = (quotient - expected_quotient).coalesce()
    quotient_entry_error = (
        float(torch.max(torch.abs(quotient_delta.values())).item())
        if quotient_delta._nnz()
        else 0.0
    )

    original_out, original_in = _weighted_degrees(adj)
    quotient_out, quotient_in = _weighted_degrees(quotient)
    expected_out = torch.zeros(pattern.size(0), dtype=adj.dtype)
    expected_in = torch.zeros(pattern.size(0), dtype=adj.dtype)
    expected_out.scatter_add_(0, inverse, original_out)
    expected_in.scatter_add_(0, inverse, original_in)

    original_flow = partition_flow_stats(adj, lifted_partition)
    quotient_flow = partition_flow_stats(quotient, quotient_partition)

    errors = {
        "quotient_entry_weight": quotient_entry_error,
        "total_weight": abs(
            float(adj.values().sum().item()) - float(quotient.values().sum().item())
        ),
        "out_degree": _max_abs_error(expected_out, quotient_out),
        "in_degree": _max_abs_error(expected_in, quotient_in),
        "modularity": abs(
            Metrics.modularity(
                adj, lifted_partition, gamma=gamma, directed=directed
            )
            - Metrics.modularity(
                quotient,
                quotient_partition,
                gamma=gamma,
                directed=directed,
            )
        ),
    }
    for name in (
        "internal_weight",
        "outgoing_cut",
        "incoming_cut",
        "out_volume",
        "in_volume",
    ):
        errors[name] = _max_abs_error(original_flow[name], quotient_flow[name])

    return {
        "directed": directed,
        "vertices": int(adj.size(0)),
        "contracted_groups": int(pattern.size(0)),
        "lifted_communities": int(torch.unique(lifted_partition).numel()),
        "gamma": gamma,
        "atol": atol,
        "errors": errors,
        "passed": all(error <= atol for error in errors.values()),
    }


class _AdversarialBackend(Optimizer):
    """Deterministic backend with deliberately conflicting proposals."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.audit_backend_calls = 0

    def local_algorithm(
        self,
        adj: torch.Tensor,
        features: torch.Tensor | None,
        limited: bool = False,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del features, limited, labels
        self.audit_backend_calls += 1
        if self.audit_backend_calls == 1:
            # Merge both groups on the fine stored level.
            return torch.zeros(adj.size(0), dtype=torch.long)
        # Keep both groups separate on the next, coarser stored level.
        return torch.arange(adj.size(0), dtype=torch.long)


def adversarial_update_refinement_report() -> dict[str, Any]:
    """Verify refinement under conflicting consecutive backend proposals.

    The first proposal merges all level-zero atoms.  The next backend call
    asks to keep every input atom separate.  Because the higher-level input is
    the quotient of the already updated preceding level, the second proposal
    cannot split that preceding-level block, and adjacent-level refinement is
    preserved by construction.
    """
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    ).to_sparse_coo()
    initial = torch.tensor(
        [
            [0, 0, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    # Touch both old fine blocks so the level-zero scope spans the graph.  The
    # first backend call can then merge every level-zero atom into one block;
    # the higher call receives that one block as its only parent atom.
    affected = torch.tensor([False, True, False, True])
    optimizer = _AdversarialBackend(
        adjacency,
        communities=initial.clone(),
        subcoms_depth=2,
    )

    before = hierarchy_refinement_report(optimizer.coms)
    optimizer.run(affected)
    after = hierarchy_refinement_report(optimizer.coms)
    return {
        "initial_hierarchy": initial.tolist(),
        "updated_hierarchy": optimizer.coms.detach().cpu().tolist(),
        "affected_vertices": torch.nonzero(affected, as_tuple=True)[0].tolist(),
        "backend_calls": optimizer.audit_backend_calls,
        "before": before,
        "after": after,
        "conclusion": (
            "Parent-quotient inputs make every higher-level proposal a "
            "coarsening of the already updated preceding level."
        ),
    }


def deterministic_quotient_cases() -> dict[str, dict[str, Any]]:
    """Run weighted directed and symmetric quotient identity cases."""
    directed = torch.tensor(
        [
            [0.25, 2.0, 0.0, 1.0, 0.0],
            [0.5, 0.0, 3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0, 1.5],
            [0.0, 2.5, 0.0, 0.75, 1.0],
            [1.0, 0.0, 0.75, 2.0, 0.0],
        ],
        dtype=torch.float64,
    ).to_sparse_coo()
    symmetric = torch.tensor(
        [
            [0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 3.0, 0.0, 0.0, 1.0],
            [1.0, 3.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0, 2.0, 0.5],
            [0.0, 0.0, 0.0, 2.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 0.5, 5.0, 0.0],
        ],
        dtype=torch.float64,
    ).to_sparse_coo()

    return {
        "weighted_directed": quotient_identity_report(
            directed,
            torch.tensor([10, 10, 30, 30, 50]),
            torch.tensor([7, 7, 9]),
            gamma=1.25,
            directed=True,
        ),
        "weighted_symmetric": quotient_identity_report(
            symmetric,
            torch.tensor([4, 4, 8, 8, 15, 15]),
            torch.tensor([0, 1, 1]),
            gamma=0.8,
            directed=False,
        ),
    }


def run_audit() -> dict[str, Any]:
    quotient_cases = deterministic_quotient_cases()
    adversarial_refinement = adversarial_update_refinement_report()
    namespace_detector_example = scope_label_overlap_report(
        torch.tensor([5, 5, 8, 9, 8]),
        torch.tensor([True, False, True, False, False]),
    )
    passed = (
        all(case["passed"] for case in quotient_cases.values())
        and adversarial_refinement["before"]["all_adjacent_refine"]
        and adversarial_refinement["after"]["all_adjacent_refine"]
        and namespace_detector_example["shared_labels"] == [5, 8]
    )
    return {
        "audit_passed": passed,
        "interpretation": (
            "All quotient identities pass, and the adversarial backend case "
            "preserves adjacent-level refinement under parent quotients."
        ),
        "scope_label_overlap_detector_example": namespace_detector_example,
        "quotient_identities": quotient_cases,
        "adversarial_update_refinement": adversarial_refinement,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact rather than indented JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_audit()
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
