import os
import sys

import pytest
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))

from scripts.paper.audit_adapter_invariants import (  # noqa: E402
    adversarial_update_refinement_report,
    deterministic_quotient_cases,
    hierarchy_refinement_report,
    quotient_identity_report,
    refinement_report,
    scope_label_overlap_report,
)
from scripts.paper.audit_stream_invariants import (  # noqa: E402
    compact_scope_overlap_report,
    compute_update_scopes,
    outside_entry_write_report,
)


@pytest.mark.unit
@pytest.mark.short
def test_refinement_is_label_invariant_and_reports_a_witness():
    valid = refinement_report(
        torch.tensor([10, 10, 20, 30, 30]),
        torch.tensor([7, 7, 8, 8, 8]),
    )
    assert valid["refines"]

    invalid = refinement_report(
        torch.tensor([10, 10, 20, 20]),
        torch.tensor([7, 8, 9, 9]),
    )
    assert not invalid["refines"]
    assert invalid["violations"] == [
        {
            "fine_label": 10,
            "vertices": [0, 1],
            "coarse_labels": [7, 8],
        }
    ]


@pytest.mark.unit
@pytest.mark.short
def test_adjacent_hierarchy_and_scope_namespace_reports():
    hierarchy = torch.tensor(
        [
            [0, 0, 2, 3, 3, 5],
            [9, 9, 9, 4, 4, 4],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    hierarchy_report = hierarchy_refinement_report(hierarchy)
    assert hierarchy_report["all_adjacent_refine"]
    assert len(hierarchy_report["adjacent"]) == 2

    overlap = scope_label_overlap_report(
        torch.tensor([5, 5, 8, 9, 8]),
        torch.tensor([True, False, True, False, False]),
    )
    assert overlap["has_shared_labels"]
    assert overlap["shared_labels"] == [5, 8]
    assert overlap["witnesses"][0] == {
        "label": 5,
        "inside_vertices": [0],
        "outside_vertices": [1],
    }


@pytest.mark.unit
@pytest.mark.short
def test_weighted_directed_and_symmetric_quotient_identities():
    cases = deterministic_quotient_cases()
    assert set(cases) == {"weighted_directed", "weighted_symmetric"}
    for report in cases.values():
        assert report["passed"], report["errors"]
        assert max(report["errors"].values()) <= report["atol"]


@pytest.mark.unit
@pytest.mark.short
def test_quotient_identity_rejects_non_group_partition_length():
    adjacency = torch.eye(3, dtype=torch.float64).to_sparse_coo()
    with pytest.raises(ValueError, match="one label per contracted group"):
        quotient_identity_report(
            adjacency,
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 2]),
        )


@pytest.mark.unit
@pytest.mark.short
def test_parent_quotient_preserves_refinement_for_adversarial_backend():
    report = adversarial_update_refinement_report()
    assert report["backend_calls"] == 2
    assert report["before"]["all_adjacent_refine"]
    assert report["after"]["all_adjacent_refine"]
    assert report["updated_hierarchy"] == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert report["after"]["adjacent"][0]["violations"] == []


@pytest.mark.unit
@pytest.mark.short
def test_stream_audit_uses_full_precomputed_closure_not_direct_mask():
    communities = torch.tensor(
        [
            [0, 0, 2, 2, 4],
            [9, 9, 9, 9, 5],
        ]
    )
    direct = torch.tensor([False, True, False, False, False])
    scopes = compute_update_scopes(communities, direct)
    assert torch.equal(
        scopes,
        torch.tensor(
            [
                [True, True, False, False, False],
                [True, True, True, True, False],
            ]
        ),
    )


@pytest.mark.unit
@pytest.mark.short
def test_stream_audit_detects_boundary_collision_and_outside_write():
    scope = torch.tensor([True, True, False, False])
    before = torch.tensor([0, 0, 2, 2])
    after = torch.tensor([2, 2, 7, 2])

    overlap = compact_scope_overlap_report(after, scope)
    assert overlap["shared_label_count"] == 1
    assert overlap["shared_labels_sample"] == [2]
    assert overlap["inside_entries_with_shared_label"] == 2
    assert overlap["outside_entries_with_shared_label"] == 1

    writes = outside_entry_write_report(before, after, scope)
    assert writes == {
        "inside_changed_entries": 2,
        "outside_changed_entries": 1,
        "outside_vertices_sample": [2],
    }
