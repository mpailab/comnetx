import os
import sys

import pytest
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "src"))

from optimizer import Optimizer
import sparse


def _refines(fine: torch.Tensor, coarse: torch.Tensor) -> bool:
    for fine_label in torch.unique(fine):
        if torch.unique(coarse[fine == fine_label]).numel() != 1:
            return False
    return True


def _is_canonical(labels: torch.Tensor) -> bool:
    vertices = torch.arange(labels.numel(), device=labels.device)
    return all(
        int(label.item()) == int(vertices[labels == label].min().item())
        for label in torch.unique(labels)
    )


def _same_restricted_partition(
    before: torch.Tensor,
    after: torch.Tensor,
    mask: torch.Tensor,
) -> bool:
    before = before[mask]
    after = after[mask]
    return torch.equal(
        before.unsqueeze(0) == before.unsqueeze(1),
        after.unsqueeze(0) == after.unsqueeze(1),
    )


def _scopes(communities: torch.Tensor, affected: torch.Tensor) -> torch.Tensor:
    vertices = torch.nonzero(affected, as_tuple=True)[0]
    return torch.stack(
        [
            torch.isin(row, torch.unique(row.index_select(0, vertices)))
            for row in communities
        ]
    )


def _graph() -> torch.Tensor:
    dense = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    return dense.to_sparse_coo().coalesce()


class _ScriptedOptimizer(Optimizer):
    def __init__(self, *args, outputs, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = outputs
        self.seen_adjacencies = []

    def local_algorithm(self, adj, features, limited=False, labels=None):
        self.seen_adjacencies.append(adj.to_dense().clone())
        output = self.outputs[len(self.seen_adjacencies) - 1]
        if callable(output):
            output = output(adj.size(0))
        return torch.as_tensor(output, dtype=torch.long)


@pytest.mark.unit
@pytest.mark.short
def test_parent_quotient_handles_adversarial_backend_and_keeps_boundaries():
    initial = torch.tensor(
        [
            [9, 9, 4, 4, 7, 7],
            [3, 3, 3, 3, 8, 8],
            [5, 5, 5, 5, 6, 6],
        ]
    )
    optimizer = _ScriptedOptimizer(
        _graph(),
        communities=initial,
        subcoms_depth=3,
        outputs=[
            lambda size: torch.full((size,), 10**9, dtype=torch.long),
            lambda size: torch.arange(size, dtype=torch.long) * 1_000_003 - 77,
            lambda size: torch.full((size,), -(10**9), dtype=torch.long),
        ],
    )
    affected = torch.tensor([False, True, False, False, False, False])
    before = optimizer.coms.clone()
    scopes = _scopes(before, affected)

    optimizer.run(affected)

    assert _refines(optimizer.coms[0], optimizer.coms[1])
    assert _refines(optimizer.coms[1], optimizer.coms[2])
    for level in range(optimizer.subcoms_depth):
        assert torch.equal(
            optimizer.coms[level, ~scopes[level]],
            before[level, ~scopes[level]],
        )
        inside = torch.unique(optimizer.coms[level, scopes[level]])
        outside = torch.unique(optimizer.coms[level, ~scopes[level]])
        assert not torch.isin(inside, outside).any()
        assert _is_canonical(optimizer.coms[level])


@pytest.mark.unit
@pytest.mark.short
def test_parent_quotient_higher_level_keeps_inter_block_edges():
    initial = torch.tensor(
        [
            [0, 0, 2, 2, 4, 4],
            [0, 0, 0, 0, 4, 4],
        ]
    )
    optimizer = _ScriptedOptimizer(
        _graph(),
        communities=initial,
        subcoms_depth=2,
        outputs=[
            lambda size: torch.zeros(size, dtype=torch.long),
            lambda size: torch.arange(size, dtype=torch.long),
        ],
    )

    optimizer.run(
        torch.tensor([False, True, False, False, False, False])
    )

    higher_quotient = optimizer.seen_adjacencies[1]
    assert higher_quotient.shape == (2, 2)
    assert higher_quotient[0, 1] > 0
    assert higher_quotient[1, 0] > 0
    assert _refines(optimizer.coms[0], optimizer.coms[1])


@pytest.mark.unit
@pytest.mark.short
@pytest.mark.parametrize("seed", range(8))
def test_parent_quotient_random_backend_preserves_nested_property(seed):
    generator = torch.Generator().manual_seed(seed)
    initial = torch.tensor(
        [
            [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10],
            [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8],
        ]
    )
    dense = torch.randint(
        0, 2, (12, 12), generator=generator, dtype=torch.int64
    ).float()
    dense.fill_diagonal_(0)
    adjacency = (dense + dense.t()).clamp_max(1).to_sparse_coo().coalesce()

    def random_partition(size):
        upper = max(1, size // 2)
        return torch.randint(0, upper, (size,), generator=generator)

    optimizer = _ScriptedOptimizer(
        adjacency,
        communities=initial,
        subcoms_depth=3,
        outputs=[random_partition, random_partition, random_partition],
    )
    affected_vertices = torch.randperm(12, generator=generator)[:2]
    affected = torch.zeros(12, dtype=torch.bool)
    affected[affected_vertices] = True
    before = optimizer.coms.clone()
    scopes = _scopes(before, affected)

    optimizer.run(affected)

    assert _refines(optimizer.coms[0], optimizer.coms[1])
    assert _refines(optimizer.coms[1], optimizer.coms[2])
    for level in range(3):
        assert torch.equal(
            optimizer.coms[level, ~scopes[level]],
            before[level, ~scopes[level]],
        )
        assert _is_canonical(optimizer.coms[level])


@pytest.mark.unit
@pytest.mark.short
def test_run_closure_and_adjacency_restriction_do_not_require_torch_isin(
    monkeypatch,
):
    optimizer = _ScriptedOptimizer(
        _graph(),
        communities=torch.tensor([[4, 4, 9, 9, 12, 12]]),
        subcoms_depth=1,
        outputs=[lambda size: torch.arange(size)],
    )

    def forbidden_isin(*args, **kwargs):
        raise AssertionError("Optimizer.run closure must not call torch.isin")

    monkeypatch.setattr(torch, "isin", forbidden_isin)
    optimizer.run(
        torch.tensor([False, True, False, False, False, False])
    )
    assert _is_canonical(optimizer.coms[0])


@pytest.mark.unit
@pytest.mark.short
def test_linear_adjacency_restriction_matches_reset_matrix():
    adjacency = torch.tensor(
        [
            [0.0, 2.0, 0.0, 7.0],
            [0.0, 0.0, 3.0, 0.0],
            [5.0, 0.0, 0.0, 11.0],
            [0.0, 13.0, 0.0, 0.0],
        ]
    ).to_sparse_coo().coalesce()
    mask = torch.tensor([True, False, True, True])
    vertices = torch.nonzero(mask, as_tuple=True)[0]

    linear = Optimizer._restrict_adjacency(adjacency, mask)
    reference = sparse.reset_matrix(adjacency, vertices)

    assert linear.is_coalesced()
    assert torch.equal(linear.indices(), reference.indices())
    assert torch.equal(linear.values(), reference.values())


@pytest.mark.unit
@pytest.mark.short
def test_run_policies_profile_singleton_base_and_parent_quotients():
    initial = torch.tensor(
        [
            [0, 0, 0, 0, 4, 4],
            [0, 0, 0, 0, 4, 4],
        ]
    )
    optimizer = _ScriptedOptimizer(
        _graph(),
        communities=initial,
        subcoms_depth=2,
        outputs=[
            lambda size: torch.arange(size),
            lambda size: torch.arange(size),
        ],
    )

    optimizer.run(
        torch.tensor([False, True, False, False, False, False]),
        base_atom_policy="singleton",
        collect_profile=True,
    )

    profile = optimizer.last_run_profile
    assert profile is not None
    assert profile["closure_enabled"] is True
    assert profile["base_atom_policy"] == "singleton"
    assert [level["closure_vertices"] for level in profile["levels"]] == [4, 4]
    assert [level["contracted_nodes"] for level in profile["levels"]] == [4, 4]
    assert all(level["cut_time"] == 0.0 for level in profile["levels"])
    assert profile["cut_time"] == 0.0
    assert profile["total_profiled_time"] >= 0.0
    assert _refines(optimizer.coms[0], optimizer.coms[1])


@pytest.mark.unit
@pytest.mark.short
def test_run_policy_can_use_radius_only_scopes():
    initial = torch.tensor(
        [
            [0, 0, 0, 0, 4, 4],
            [0, 0, 0, 0, 4, 4],
        ]
    )
    optimizer = _ScriptedOptimizer(
        _graph(),
        communities=initial,
        subcoms_depth=2,
        outputs=[
            lambda size: torch.arange(size),
            lambda size: torch.arange(size),
        ],
    )

    affected = torch.tensor([True, False, False, False, False, False])
    before = optimizer.coms.clone()
    optimizer.run(
        affected,
        closure_enabled=False,
        collect_profile=True,
    )

    assert [
        level["closure_vertices"]
        for level in optimizer.last_run_profile["levels"]
    ] == [1, 1]
    assert _refines(optimizer.coms[0], optimizer.coms[1])
    outside = ~affected
    for level in range(optimizer.subcoms_depth):
        # Radius-only scope cuts through the old block containing vertex 0.
        # Its outside remainder is relabeled from representative 0 to 1, but
        # the partition induced on outside vertices is unchanged.
        assert _same_restricted_partition(
            before[level], optimizer.coms[level], outside
        )
        assert not torch.isin(
            torch.unique(optimizer.coms[level, affected]),
            torch.unique(optimizer.coms[level, outside]),
        ).any()
        assert _is_canonical(optimizer.coms[level])


@pytest.mark.unit
@pytest.mark.short
def test_run_rejects_unknown_base_atom_policy():
    optimizer = _ScriptedOptimizer(
        _graph(),
        subcoms_depth=1,
        outputs=[lambda size: torch.arange(size)],
    )
    with pytest.raises(ValueError, match="base_atom_policy"):
        optimizer.run(
            torch.tensor([True, False, False, False, False, False]),
            base_atom_policy="unknown",
        )


@pytest.mark.unit
@pytest.mark.short
def test_canonical_labels_are_minimum_original_vertices():
    labels = torch.tensor([8, 3, 8, 3, 3, 9])
    canonical = Optimizer.canonicalize_partition(labels)
    assert torch.equal(canonical, torch.tensor([0, 1, 0, 1, 1, 5]))

    scoped = Optimizer.canonicalize_partition(
        torch.tensor([4, 7, 4, 7]),
        torch.tensor([2, 5, 9, 11]),
    )
    assert torch.equal(scoped, torch.tensor([2, 5, 2, 5]))
