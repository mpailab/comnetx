import pytest
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "src"))

from optimizer import Optimizer
import sparse


@pytest.mark.unit
@pytest.mark.short
def test_init_defaults():
    A = torch.sparse_coo_tensor(torch.tensor([[0,1],[1,0]]), torch.ones(2), size=(2,2)).coalesce()
    opt = Optimizer(A, subcoms_depth=3)
    assert opt.nodes_num == 2
    assert opt.size == torch.Size([2,2])
    # default features shape (n,1)
    assert isinstance(opt.features, torch.Tensor)
    assert opt.features.shape == (2,1)
    # default communities arange per level
    assert opt.coms.shape == (3,2)
    for l in range(3):
        assert torch.equal(opt.coms[l], torch.arange(0,2))


@pytest.mark.unit
@pytest.mark.short
def test_init_with_inputs():
    A = torch.sparse_coo_tensor(torch.tensor([[0,1],[1,0]]), torch.ones(2), size=(2,2)).coalesce()
    X = torch.randn(2, 4)
    C = torch.tensor([[1,0]])
    opt = Optimizer(A, features=X, communities=C, subcoms_depth=1)
    assert torch.equal(opt.features, X.float())
    assert torch.equal(opt.coms, C)


@pytest.mark.unit
@pytest.mark.short
def test_update_adj():
    adj_matrix = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]])
    opt = Optimizer(adj_matrix.to_sparse_coo())
    opt.update_adj(adj_matrix.to_sparse_coo())
    true_res = adj_matrix * 2
    res = opt.adj
    assert torch.equal(true_res, opt.adj.to_dense())


@pytest.mark.unit
@pytest.mark.short
def test_update_adj_adds_and_returns_mask():
    A = torch.tensor([[0,1,0],[1,0,0],[0,0,0]], dtype=torch.float32).to_sparse_coo()
    opt = Optimizer(A)
    # Add an edge (1,2)
    upd = torch.tensor([[0,0,0],[0,0,1],[0,0,0]], dtype=torch.float32).to_sparse_coo()
    mask = opt.update_adj(upd)
    # A now has (0,1), (1,0), and (1,2)
    assert torch.isclose(opt.adj.to_dense(), torch.tensor([[0,1,0],[1,0,1],[0,0,0]], dtype=torch.float32)).all()
    # Mask should include nodes 1 and 2
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.tensor([False, True, True]))


@pytest.mark.unit
@pytest.mark.short
def test_update_adj_size_mismatch():
    A = torch.zeros((3,3)).to_sparse_coo()
    B = torch.zeros((2,2)).to_sparse_coo()
    opt = Optimizer(A)
    try:
        opt.update_adj(B)
        assert False, "Expected ValueError for size mismatch"
    except ValueError:
        pass


def identity_la(adj, features, limited, labels):
    # Return each node as its own cluster id
    return torch.arange(adj.size(0), dtype=torch.long)


@pytest.mark.unit
@pytest.mark.short
def test_local_algorithm_injection():
    A = torch.zeros((3,3)).to_sparse_coo()
    X = torch.randn(3,2)
    opt = Optimizer(A, features=X, local_algorithm_fn=identity_la, method="custom")
    out = opt.local_algorithm(A, X, False)
    assert torch.equal(out, torch.tensor([0,1,2]))


@pytest.mark.unit
@pytest.mark.short
def test_local_algorithm_unsupported():
    A = torch.zeros((2,2)).to_sparse_coo()
    opt = Optimizer(A, method="unknown_method")
    try:
        _ = opt.local_algorithm(A, torch.zeros((2,1)), False)
        assert False, "Expected ValueError for unsupported method"
    except ValueError:
        pass


# @pytest.mark.unit
@pytest.mark.short
# FIXME now aggregate don't work for 2 sparse matrix with int-like elements type
# def test_aggregate_1():
#     adj = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]], dtype = torch.int64).to_sparse()
#     coms = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 1]]).type(adj.dtype).to_sparse()
#     res = Optimizer.aggregate(adj, coms)
#     true_res = torch.tensor([[9, 0], [1, 1]])
#     assert torch.equal(true_res, res)


@pytest.mark.unit
@pytest.mark.short
def test_aggregate_2():
    adj = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]], dtype = torch.float).to_sparse()
    coms = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 1]]).type(adj.dtype).to_sparse()
    res = Optimizer.aggregate(adj, coms)
    true_res = torch.tensor([[9, 0], [1, 1]])
    assert torch.equal(true_res.to_dense(), res.to_dense())


@pytest.mark.unit
@pytest.mark.short
def test_aggregate_simple():
    adj = torch.tensor([[1,1,1,0],[1,1,1,0],[1,1,1,0],[0,1,0,1]], dtype=torch.float32).to_sparse_coo()
    # Group nodes 0,1,2 together and 3 alone
    pattern_idx = torch.tensor([[1,1,1,0],[0,0,0,1]])
    pattern = pattern_idx.to(dtype=adj.dtype).to_sparse()
    res = Optimizer.aggregate(adj, pattern)
    assert torch.equal(res.to_dense(), torch.tensor([[9.,0.],[1.,1.]]))


@pytest.mark.unit
@pytest.mark.short
def test_aggregate_via_sparse_helper():
    # Same as above but build pattern via sparse helper like in run()
    adj = torch.tensor([[1,1,1,0],[1,1,1,0],[1,1,1,0],[0,1,0,1]], dtype=torch.float32).to_sparse_coo()
    # old_coms = [[new_id],[orig_node]] pairs
    old_coms = torch.tensor([[0,0,0,1],[0,1,2,3]])
    pattern = sparse.tensor(old_coms, (2,4), adj.dtype)
    res = Optimizer.aggregate(adj, pattern)
    assert torch.equal(res.to_dense(), torch.tensor([[9.,0.],[1.,1.]]))


@pytest.mark.unit
@pytest.mark.short
def test_modularity():
    communities = torch.tensor([
        [0, 0, 0, 3, 3, 3],
        [0, 0, 2, 3, 4, 3],
        [0, 1, 2, 3, 4, 5]
    ])

    adj_matrix = torch.tensor([
                        [1, 1, 1, 0, 0, 0], 
                        [1, 1, 1, 0, 0, 0], 
                        [1, 1, 1, 0, 0, 0], 
                        [0, 1, 0, 1, 1, 1], 
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1]], dtype = torch.float).to_sparse()

    optimizer = Optimizer(adj_matrix=adj_matrix, communities=communities)
    modularity = optimizer.modularity()
    assert modularity < 1.0

@pytest.mark.unit
@pytest.mark.short
def test_neighborhood_1():
    A = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=torch.float32)
    A_sparse = A.to_sparse_coo()
    initial_nodes = torch.tensor([True, False, False, False])
    nodes_0 = Optimizer.neighborhood(A_sparse, initial_nodes, 0)
    nodes_1 = Optimizer.neighborhood(A_sparse, initial_nodes, 1)
    nodes_2 = Optimizer.neighborhood(A_sparse, initial_nodes, 2)
    true_nodes_1 = torch.tensor([True, True, False, False])
    true_nodes_2 = torch.tensor([True, True, True, True])
    assert torch.equal(nodes_0, initial_nodes)
    assert torch.equal(nodes_1, true_nodes_1)
    assert torch.equal(nodes_2, true_nodes_2)

@pytest.mark.unit
@pytest.mark.short
def test_neighborhood_2():
    A = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=torch.float32)
    A_sparse = A.t().to_sparse_coo()
    initial_nodes = torch.tensor([True, False, False, False])
    nodes_0 = Optimizer.neighborhood(A_sparse, initial_nodes, 0)
    nodes_1 = Optimizer.neighborhood(A_sparse, initial_nodes, 1)
    nodes_2 = Optimizer.neighborhood(A_sparse, initial_nodes, 2)
    true_nodes_1 = torch.tensor([True, True, False, False])
    true_nodes_2 = torch.tensor([True, True, True, True])
    assert torch.equal(nodes_0, initial_nodes)
    assert torch.equal(nodes_1, true_nodes_1)
    assert torch.equal(nodes_2, true_nodes_2)


@pytest.mark.unit
@pytest.mark.short
def test_neighborhood_steps_directed():
    A = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=torch.float32).to_sparse_coo()
    start = torch.tensor([True, False, False, False])
    assert torch.equal(Optimizer.neighborhood(A, start, 0), start)
    assert torch.equal(Optimizer.neighborhood(A, start, 1), torch.tensor([True, True, False, False]))
    assert torch.equal(Optimizer.neighborhood(A, start, 2), torch.tensor([True, True, True, True]))


@pytest.mark.unit
@pytest.mark.short
def test_run_updates_last_level_singletons_and_propagates():
    # Simple chain 0-1-2-3
    idx = torch.tensor([[0,1,2],[1,2,3]])
    A = torch.sparse_coo_tensor(idx, torch.ones(3), size=(4,4)).coalesce()
    # Single level
    C = torch.tensor([[0,0,1,1]])
    X = torch.randn(4,3)
    opt = Optimizer(A, features=X, communities=C.clone(), subcoms_depth=1, local_algorithm_fn=identity_la, method="custom")

    nodes_mask = torch.tensor([False, True, False, False])
    opt.run(nodes_mask)
    # Last level should set affected node to singleton label equal to node index
    assert opt.coms.shape == (1,4)
    assert opt.coms[0,1].item() == 1
    # Unaffected nodes keep their original label structure or consistent mapping
    # identity_la should not merge or split beyond singleton for the affected set
