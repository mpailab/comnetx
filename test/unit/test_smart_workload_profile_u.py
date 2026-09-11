import types

import pytest
import torch

from scripts.paper.profile_smart_workload import (
    clone_optimizer_state,
    isolated_scope_invariants,
    profile_smart_update,
)
from optimizer import Optimizer


def _optimizer():
    indices = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
        dtype=torch.long,
    )
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.size(1)),
        (6, 6),
    ).coalesce()
    communities = torch.tensor(
        [
            [0, 0, 2, 2, 4, 4],
            [0, 0, 0, 0, 4, 4],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    optimizer = Optimizer(adjacency, subcoms_depth=3, method="leidenalg")
    optimizer.set_communities(communities)

    def deterministic_backend(self, quotient, _features, _limited=False):
        self.last_timing_info = {"conversion_time": 0.0}
        return torch.arange(quotient.size(0), device=quotient.device) // 2

    optimizer.local_algorithm = types.MethodType(deterministic_backend, optimizer)
    return optimizer


@pytest.mark.parametrize(
    ("variant", "closure_enabled", "base_atom_policy"),
    [
        ("full", True, "hierarchical"),
        ("no_closure", False, "hierarchical"),
        ("no_contraction", True, "singleton"),
    ],
)
def test_profile_uses_production_run_semantics(
    variant,
    closure_enabled,
    base_atom_policy,
):
    profiled = _optimizer()
    direct = _optimizer()
    affected = torch.tensor([True, False, False, False, False, False])

    row, measured = profile_smart_update(
        profiled,
        affected,
        radius=0,
        directed=False,
        variant=variant,
    )
    expanded = direct.neighborhood(direct.runtime_adj(), affected, step=0)
    direct.run(
        expanded,
        closure_enabled=closure_enabled,
        base_atom_policy=base_atom_policy,
    )

    assert torch.equal(profiled.coms, direct.coms)
    assert row["closure_enabled"] is closure_enabled
    assert row["base_atom_policy"] == base_atom_policy
    assert row["cut_time"] == 0.0
    assert measured == pytest.approx(row["total_profiled_time"])
    assert row["principal_profiled_time"] + row["backend_conversion_time"] == pytest.approx(
        row["total_profiled_time"]
    )
    assert row["instrumented_wall_time"] == pytest.approx(
        row["optimizer_time"]
    )
    assert row["total_profiled_time"] == pytest.approx(
        row["radius_time"] + row["optimizer_time"]
    )
    assert "diagnostic subcomponent" in row["timing_accounting"]
    assert len(row["levels"]) == 3
    assert "certificate_time" not in row
    assert "boundary_certificates_by_level" not in row
    assert "identity_ranking_certificates_by_level" not in row
    assert all("certificate_time" not in level for level in row["levels"])
    assert all("boundary_certificate" not in level for level in row["levels"])
    assert all("ranking_certificate" not in level for level in row["levels"])
    if variant == "no_closure":
        assert row["closure_vertices_by_level"] == [1, 1, 1]
    if variant == "no_contraction":
        assert row["contracted_nodes_by_level"][0] == row[
            "closure_vertices_by_level"
        ][0]


def test_radius_only_control_is_a_discarded_one_step_state():
    production = _optimizer()
    before = production.coms.detach().clone()
    isolated = clone_optimizer_state(production)

    def deterministic_backend(self, quotient, _features, _limited=False):
        self.last_timing_info = {"conversion_time": 0.0}
        return torch.zeros(quotient.size(0), dtype=torch.long, device=quotient.device)

    isolated.local_algorithm = types.MethodType(deterministic_backend, isolated)
    scope = torch.tensor([True, False, False, False, False, False])
    isolated.run(scope, closure_enabled=False)
    audit = isolated_scope_invariants(before, isolated.coms, scope)

    assert torch.equal(production.coms, before)
    assert audit["nested_before"] is True
    assert audit["nested_after"] is True
    assert audit["scope_label_collisions_by_level"] == [0, 0, 0]
    assert audit["outside_partition_preserved_by_level"] == [True, True, True]
    assert len(audit["outside_numeric_writes_by_level"]) == 3
    assert all(value >= 0 for value in audit["outside_numeric_writes_by_level"])
