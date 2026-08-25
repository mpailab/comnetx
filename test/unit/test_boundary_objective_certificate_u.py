import math
import types

import pytest
import torch

import optimizer as optimizer_module
from metrics import Metrics
from optimizer import (
    Optimizer,
    restricted_boundary_objective_certificate,
    restricted_ranking_certificate,
)
from scripts.paper.ieee_access_repaired_comnetx_72h.protocol import ValidationError
from scripts.paper.ieee_access_repaired_comnetx_72h.validate_campaign import (
    summarize_boundary_certificates,
    summarize_identity_ranking_certificates,
    validate_boundary_certificates,
    validate_identity_ranking_certificates,
)


def _direct_realized_mismatch(
    adjacency: torch.Tensor,
    scope: torch.Tensor,
    assignments: torch.Tensor,
    gamma: float,
) -> float:
    """Independently evaluate F(z) - L(z) under the Metrics convention."""

    dense = adjacency.to_dense() if adjacency.is_sparse else adjacency
    weights = dense.to(dtype=torch.float64)
    labels = assignments.to(dtype=torch.long)
    w = float(weights.sum().item())
    inside = torch.nonzero(scope, as_tuple=True)[0]
    outside = torch.nonzero(~scope, as_tuple=True)[0]
    induced = weights.index_select(0, inside).index_select(1, inside)
    w_u = float(induced.sum().item())

    full_q = Metrics.modularity(
        weights.to_sparse_coo(), labels, gamma=gamma, directed=True
    )
    outside_constant = 0.0
    for label in torch.unique(labels.index_select(0, outside)):
        block = (~scope) & (labels == label)
        block_vertices = torch.nonzero(block, as_tuple=True)[0]
        internal_weight = weights.index_select(0, block_vertices).index_select(
            1, block_vertices
        ).sum()
        out_degree = weights.index_select(0, block_vertices).sum()
        in_degree = weights.index_select(1, block_vertices).sum()
        outside_constant += float(
            (internal_weight / w - gamma * out_degree * in_degree / (w * w)).item()
        )
    local_q = Metrics.modularity(
        induced.to_sparse_coo(),
        labels.index_select(0, inside),
        gamma=gamma,
        directed=True,
    )
    return full_q - outside_constant - (w_u / w) * local_q


@pytest.mark.unit
@pytest.mark.short
@pytest.mark.parametrize(
    "assignments",
    [
        torch.tensor([0, 0, 2, 3]),
        torch.tensor([0, 1, 2, 3]),
    ],
)
def test_boundary_certificate_matches_direct_directed_modularity_with_self_loops(
    assignments,
):
    adjacency = torch.tensor(
        [
            [2.0, 1.0, 3.0, 0.0],
            [4.0, 5.0, 0.0, 2.0],
            [7.0, 0.0, 11.0, 1.0],
            [0.0, 13.0, 6.0, 17.0],
        ],
        dtype=torch.float32,
    ).to_sparse_coo()
    scope = torch.tensor([True, True, False, False])
    gamma = 0.8

    certificate = restricted_boundary_objective_certificate(
        adjacency, scope, assignments, gamma=gamma
    )

    assert certificate["finite"] is True
    assert certificate["W"] == pytest.approx(72.0)
    assert certificate["W_U"] == pytest.approx(12.0)
    assert certificate["beta_out"] == pytest.approx(5.0)
    assert certificate["beta_in"] == pytest.approx(20.0)
    assert certificate["D"] == pytest.approx(
        _direct_realized_mismatch(adjacency, scope, assignments, gamma),
        abs=1e-12,
    )
    inside = torch.nonzero(scope, as_tuple=True)[0]
    induced = adjacency.to_dense().index_select(0, inside).index_select(1, inside)
    assert certificate["Q_U"] == pytest.approx(
        Metrics.modularity(
            induced.to(dtype=torch.float64).to_sparse_coo(),
            assignments.index_select(0, inside),
            gamma=gamma,
            directed=True,
        ),
        abs=1e-12,
    )
    assert certificate["certificate_width"] == pytest.approx(
        certificate["B_plus"] + certificate["B_minus"]
    )
    assert -certificate["B_minus"] <= certificate["D"] <= certificate["B_plus"]


@pytest.mark.unit
@pytest.mark.short
def test_boundary_certificate_marks_zero_weight_scope_as_undefined():
    adjacency = torch.tensor(
        [[0.0, 2.0], [3.0, 0.0]], dtype=torch.float32
    ).to_sparse_coo()
    certificate = restricted_boundary_objective_certificate(
        adjacency,
        torch.tensor([True, False]),
        torch.tensor([0, 1]),
    )

    assert certificate["W"] == pytest.approx(5.0)
    assert certificate["W_U"] == 0.0
    assert certificate["beta_out"] == pytest.approx(2.0)
    assert certificate["beta_in"] == pytest.approx(3.0)
    assert certificate["finite"] is False
    for field in ("B_plus", "B_minus", "certificate_width", "D", "Q_U"):
        assert certificate[field] is None


@pytest.mark.unit
@pytest.mark.short
def test_boundary_certificate_rejects_a_community_crossing_the_scope_boundary():
    adjacency = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    ).to_sparse_coo()
    with pytest.raises(ValueError, match="community labels to be disjoint"):
        restricted_boundary_objective_certificate(
            adjacency,
            torch.tensor([True, False]),
            torch.tensor([0, 0]),
        )


def _restricted_growth_partitions(size: int):
    def visit(prefix):
        if len(prefix) == size:
            yield tuple(prefix)
            return
        for label in range(max(prefix, default=-1) + 2):
            yield from visit([*prefix, label])

    yield from visit([0])


@pytest.mark.unit
@pytest.mark.short
def test_identity_ranking_certificate_matches_bruteforce_full_objective_gaps():
    adjacency = torch.tensor(
        [
            [2.0, 4.0, 0.0, 0.0, 0.0],
            [3.0, 2.0, 0.0, 0.0, 0.01],
            [0.0, 0.0, 1.0, 5.0, 0.0],
            [0.0, 0.0, 6.0, 1.0, 0.0],
            [0.0, 0.01, 0.0, 0.0, 0.1],
        ],
        dtype=torch.float64,
    ).to_sparse_coo()
    scope = torch.tensor([True, True, True, True, False])
    inside = torch.nonzero(scope, as_tuple=True)[0]
    atom_by_inside_vertex = torch.tensor([0, 0, 1, 2])
    reference_inside = Optimizer.canonicalize_partition(
        atom_by_inside_vertex, inside
    )
    reference = torch.tensor([0, 0, 2, 3, 4])
    gamma = 1.0
    w = float(adjacency.values().sum().item())
    w_u = float(
        adjacency.to_dense().index_select(0, inside).index_select(1, inside).sum()
    )
    induced = adjacency.to_dense().index_select(0, inside).index_select(1, inside)
    reference_local_q = Metrics.modularity(
        induced.to_sparse_coo(), reference_inside, gamma=gamma, directed=True
    )
    informative_seen = False

    for atom_partition in _restricted_growth_partitions(3):
        atom_labels = torch.tensor(atom_partition)
        candidate_inside = Optimizer.canonicalize_partition(
            atom_labels.index_select(0, atom_by_inside_vertex), inside
        )
        candidate = torch.empty(5, dtype=torch.long)
        candidate[scope] = candidate_inside
        candidate[~scope] = 4

        comparison = restricted_ranking_certificate(
            adjacency,
            scope,
            candidate,
            reference,
            gamma=gamma,
        )
        direct_full_gap = Metrics.modularity(
            adjacency, candidate, gamma=gamma, directed=True
        ) - Metrics.modularity(
            adjacency, reference, gamma=gamma, directed=True
        )
        candidate_local_q = Metrics.modularity(
            induced.to_sparse_coo(),
            candidate_inside,
            gamma=gamma,
            directed=True,
        )
        direct_scaled_local_gap = w_u / w * (
            candidate_local_q - reference_local_q
        )

        assert comparison["reference"] == "identity atom partition"
        assert comparison["candidate_family"] == "same scope and quotient atoms"
        assert comparison["finite"] is True
        assert comparison["scaled_local_gap"] == pytest.approx(
            direct_scaled_local_gap, abs=1e-12
        )
        assert comparison["restricted_full_objective_gap"] == pytest.approx(
            direct_full_gap, abs=1e-12
        )
        if comparison["ranking_sign_certified"]:
            informative_seen = True
            assert comparison["local_gap_sign"] == comparison["full_gap_sign"]

    identity_comparison = restricted_ranking_certificate(
        adjacency, scope, reference, reference, gamma=gamma
    )
    assert identity_comparison["status"] == "local_tie"
    assert identity_comparison["strict_threshold_passed"] is False
    assert identity_comparison["ranking_sign_certified"] is False
    assert informative_seen


def _instrumented_optimizer() -> Optimizer:
    adjacency = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 4.0],
            [0.0, 5.0, 6.0],
        ]
    ).to_sparse_coo()
    optimizer = Optimizer(
        adjacency,
        communities=torch.tensor([[0, 0, 2]]),
        subcoms_depth=1,
        method="leidenalg",
        resolution=0.7,
    )

    def identity_backend(self, quotient, _features, _limited=False):
        self.last_timing_info = {"conversion_time": 0.0}
        return torch.arange(quotient.size(0), device=quotient.device)

    optimizer.local_algorithm = types.MethodType(identity_backend, optimizer)
    return optimizer


@pytest.mark.unit
@pytest.mark.short
def test_optimizer_collect_profile_records_certificate_only_when_requested(
    monkeypatch,
):
    optimizer = _instrumented_optimizer()

    def forbidden_certificate(*_args, **_kwargs):
        raise AssertionError("certificate must not run outside profiling")

    monkeypatch.setattr(
        optimizer_module,
        "restricted_boundary_objective_certificate",
        forbidden_certificate,
    )
    optimizer.run(torch.tensor([True, False, False]), collect_profile=False)
    assert optimizer.last_run_profile is None


@pytest.mark.unit
@pytest.mark.short
def test_optimizer_profile_contains_finite_per_level_certificate():
    optimizer = _instrumented_optimizer()
    optimizer.run(torch.tensor([True, False, False]), collect_profile=True)

    profile = optimizer.last_run_profile
    certificate = profile["levels"][0]["boundary_certificate"]
    ranking = profile["levels"][0]["ranking_certificate"]
    assert certificate["finite"] is True
    assert certificate["gamma"] == pytest.approx(0.7)
    assert ranking["reference"] == "identity atom partition"
    assert ranking["status"] == "local_tie"
    assert ranking["ranking_sign_certified"] is False
    assert profile["certificate_time"] >= 0.0
    assert profile["levels"][0]["certificate_time"] >= 0.0
    assert profile["instrumented_wall_time"] == pytest.approx(
        profile["total_profiled_time"] + profile["certificate_time"]
    )


def _valid_certificate_row() -> dict:
    certificate = {
        "gamma": 1.0,
        "W": 10.0,
        "W_U": 4.0,
        "beta_out": 2.0,
        "beta_in": 1.0,
        "B_plus": 0.24,
        "B_minus": 0.14,
        "certificate_width": 0.38,
        "D": 0.05,
        "Q_U": 0.7,
        "finite": True,
    }
    return {
        "certificate_time": 0.01,
        "levels": [
            {
                "level": 0,
                "certificate_time": 0.01,
                "boundary_certificate": certificate,
            }
        ],
        "boundary_certificates_by_level": [certificate.copy()],
    }


def _valid_ranking_row() -> dict:
    row = _valid_certificate_row()
    ranking = {
        "reference": "identity atom partition",
        "candidate_family": "same scope and quotient atoms",
        "scaled_local_gap": 0.5,
        "restricted_full_objective_gap": 0.53,
        "certificate_width": 0.38,
        "output_D": 0.05,
        "reference_D": 0.02,
        "output_Q_U": 0.7,
        "reference_Q_U": -0.55,
        "W_U_over_W": 0.4,
        "comparison_tolerance": 1e-10,
        "local_gap_sign": 1,
        "full_gap_sign": 1,
        "strict_threshold_passed": True,
        "ranking_sign_certified": True,
        "status": "certified_same_sign",
        "finite": True,
    }
    row["levels"][0]["ranking_certificate"] = ranking
    row["identity_ranking_certificates_by_level"] = [ranking.copy()]
    return row


@pytest.mark.unit
@pytest.mark.short
def test_stage3_certificate_gate_and_summary_accept_consistent_values():
    certificates = validate_boundary_certificates(
        _valid_certificate_row(), context="fixture", expected_levels=1
    )
    summary = summarize_boundary_certificates(certificates)

    assert summary["finite_levels"] == 1
    assert summary["total_levels"] == 1
    assert summary["finite_fraction"] == 1.0
    assert summary["mean_certificate_width"] == pytest.approx(0.38)
    assert summary["mean_absolute_realized_mismatch"] == pytest.approx(0.05)
    assert summary["mean_internal_weight_fraction"] == pytest.approx(0.4)
    assert all(math.isfinite(float(value)) for value in summary.values())


@pytest.mark.unit
@pytest.mark.short
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("finite", False, "non-finite flag is inconsistent"),
        ("B_plus", 0.25, "bounds were miscomputed"),
        ("D", 0.5, "exceeds its bounds"),
    ],
)
def test_stage3_certificate_gate_rejects_invalid_certificate(field, value, message):
    row = _valid_certificate_row()
    row["levels"][0]["boundary_certificate"][field] = value
    row["boundary_certificates_by_level"][0][field] = value

    with pytest.raises(ValidationError, match=message):
        validate_boundary_certificates(row, context="fixture", expected_levels=1)


@pytest.mark.unit
@pytest.mark.short
def test_stage3_certificate_gate_records_zero_weight_scope_coverage():
    row = _valid_certificate_row()
    certificate = row["levels"][0]["boundary_certificate"]
    certificate.update(
        {
            "W_U": 0.0,
            "B_plus": None,
            "B_minus": None,
            "certificate_width": None,
            "D": None,
            "Q_U": None,
            "finite": False,
        }
    )
    row["boundary_certificates_by_level"][0] = certificate.copy()

    certificates = validate_boundary_certificates(
        row, context="fixture", expected_levels=1
    )
    summary = summarize_boundary_certificates(certificates)

    assert summary["finite_levels"] == 0
    assert summary["undefined_zero_weight_levels"] == 1
    assert summary["finite_fraction"] == 0.0
    assert summary["mean_certificate_width"] is None


@pytest.mark.unit
@pytest.mark.short
def test_stage3_ranking_gate_and_descriptive_summary_accept_exact_comparison():
    row = _valid_ranking_row()
    boundaries = validate_boundary_certificates(
        row, context="fixture", expected_levels=1
    )
    rankings = validate_identity_ranking_certificates(
        row, boundaries, context="fixture", expected_levels=1
    )
    summary = summarize_identity_ranking_certificates(rankings)

    assert summary["comparison_coverage"] == 1.0
    assert summary["informative_count"] == 1
    assert summary["informative_rate"] == 1.0
    assert summary["certified_positive_count"] == 1
    assert "confidence" not in " ".join(summary).lower()


@pytest.mark.unit
@pytest.mark.short
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("restricted_full_objective_gap", 0.4, "objective gap is inconsistent"),
        ("local_gap_sign", -1, "sign or threshold flag is wrong"),
        ("ranking_sign_certified", False, "sign or threshold flag is wrong"),
        ("reference", "arbitrary", "unregistered ranking reference"),
    ],
)
def test_stage3_ranking_gate_rejects_corrupted_sign_or_threshold(
    field, value, message
):
    row = _valid_ranking_row()
    row["levels"][0]["ranking_certificate"][field] = value
    row["identity_ranking_certificates_by_level"][0][field] = value
    boundaries = validate_boundary_certificates(
        row, context="fixture", expected_levels=1
    )

    with pytest.raises(ValidationError, match=message):
        validate_identity_ranking_certificates(
            row, boundaries, context="fixture", expected_levels=1
        )


@pytest.mark.unit
@pytest.mark.short
def test_stage3_ranking_gate_handles_identity_tie_explicitly():
    row = _valid_ranking_row()
    ranking = row["levels"][0]["ranking_certificate"]
    ranking.update(
        {
            "scaled_local_gap": 0.0,
            "restricted_full_objective_gap": 0.0,
            "reference_D": 0.05,
            "reference_Q_U": 0.7,
            "local_gap_sign": 0,
            "full_gap_sign": 0,
            "strict_threshold_passed": False,
            "ranking_sign_certified": False,
            "status": "local_tie",
        }
    )
    row["identity_ranking_certificates_by_level"][0] = ranking.copy()
    boundaries = validate_boundary_certificates(
        row, context="fixture", expected_levels=1
    )
    rankings = validate_identity_ranking_certificates(
        row, boundaries, context="fixture", expected_levels=1
    )
    summary = summarize_identity_ranking_certificates(rankings)

    assert summary["identity_tie_count"] == 1
    assert summary["informative_count"] == 0


@pytest.mark.unit
@pytest.mark.short
def test_stage3_ranking_gate_rejects_reference_mismatch_outside_bounds():
    row = _valid_ranking_row()
    ranking = row["levels"][0]["ranking_certificate"]
    ranking["reference_D"] = 0.3
    ranking["restricted_full_objective_gap"] = 0.25
    row["identity_ranking_certificates_by_level"][0] = ranking.copy()
    boundaries = validate_boundary_certificates(
        row, context="fixture", expected_levels=1
    )

    with pytest.raises(ValidationError, match="reference mismatch exceeds"):
        validate_identity_ranking_certificates(
            row, boundaries, context="fixture", expected_levels=1
        )
