import sys
import os
import torch
import pytest
from typing import Union

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from optimizer import Optimizer
import sparse

# @pytest.mark.short
# FIXME now aggregate don't work for 2 sparse matrix with int-like elements type
# def test_aggregate_1():
#     adj = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]], dtype = torch.int64).to_sparse()
#     coms = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 1]]).type(adj.dtype).to_sparse()
#     res = Optimizer.aggregate(adj, coms)
#     true_res = torch.tensor([[9, 0], [1, 1]])
#     assert torch.equal(true_res, res)

@pytest.mark.short
def test_aggregate_2():
    adj = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]], dtype = torch.float).to_sparse()
    coms = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 1]]).type(adj.dtype).to_sparse()
    res = Optimizer.aggregate(adj, coms)
    true_res = torch.tensor([[9, 0], [1, 1]])
    assert torch.equal(true_res.to_dense(), res.to_dense())

@pytest.mark.short
def test_run_prgpt():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3]])
    opt = Optimizer(A, communities = communities, method  = "prgpt:infomap")
    nodes_mask = torch.tensor([0, 0, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())

@pytest.mark.short
def test_run_leidenalg():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3]])
    opt = Optimizer(A, communities = communities, method  = "leidenalg")
    nodes_mask = torch.tensor([0, 0, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())

@pytest.mark.short
def test_run_magi():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3]])
    opt = Optimizer(A, communities = communities, method  = "magi")
    nodes_mask = torch.tensor([0, 0, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())

@pytest.mark.long
def test_run_dmon():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3], [0, 1, 2, 3], [0, 0, 0, 0]])
    opt = Optimizer(A, communities = communities, method  = "dmon")
    nodes_mask = torch.tensor([0, 0, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())

@pytest.mark.short
def test_update_adj():
    adj_matrix = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]])
    opt = Optimizer(adj_matrix.to_sparse_coo())
    opt.update_adj(adj_matrix.to_sparse_coo())
    true_res = adj_matrix * 2
    res = opt.adj
    assert torch.equal(true_res, opt.adj.to_dense())

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

def test_modularity():
    communities = torch.tensor([
        [0, 0, 0, 3, 3, 3],
        [0, 0, 2, 3, 4, 3],
        [0, 1, 2, 3, 4, 5]
    ])

    dense_communities = torch.tensor([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]], dtype=torch.float32)

    adj_matrix = torch.tensor([
                        [1, 1, 1, 0, 0, 0], 
                        [1, 1, 1, 0, 0, 0], 
                        [1, 1, 1, 0, 0, 0], 
                        [0, 1, 0, 1, 1, 1], 
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1]], dtype = torch.float).to_sparse()

    optimizer = Optimizer(adj_matrix=adj_matrix, communities=communities)
    modularity = optimizer.modularity()
    dense_modularity = optimizer.dense_modularity(adj_matrix, dense_communities)
    print(type(modularity))
    assert modularity < 1.0
    assert dense_modularity == modularity
