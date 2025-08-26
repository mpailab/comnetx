import sys
import os
import torch
import pytest

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

def test_run():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3], [0, 1, 2, 3], [0, 0, 0, 0]])
    opt = Optimizer(A, communities = communities)
    nodes_mask = torch.tensor([0, 0, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())