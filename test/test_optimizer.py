import sys
import os
import torch
import pytest

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from optimizer import Optimizer
from sparse_utils import *

@pytest.mark.short
def test_cut_off():
    coms = torch.tensor([[1, 1, 1, 0]]).bool()
    nodes = torch.tensor([0, 0, 1, 0]).bool()
    res = Optimizer.cut_off(coms, nodes)
    true_res = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 0]])
    assert torch.equal(true_res, res)

@pytest.mark.short
def test_cut_off_sparse():
    coms = torch.tensor([[1, 1, 1, 0]]).bool().to_sparse_coo()
    nodes = torch.tensor([0, 0, 1, 0]).bool()
    res = Optimizer.cut_off(coms, nodes, True)
    true_res = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 0]]).bool().to_sparse_coo()
    assert are_coo_tensors_equal(true_res, res)

@pytest.mark.short
def test_aggregation():
    adj = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse()
    coms = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 1]]).bool()
    res = Optimizer.aggregation(adj, coms)
    true_res = torch.tensor([[9, 0], [1, 1]])
    assert torch.equal(true_res, res)
    adj_float = adj.float()
    res = Optimizer.aggregation(adj_float, coms)
    assert torch.equal(true_res, res)

def test_run():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse()
    C = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 1]]).bool()
    nodes = torch.tensor([0, 0, 1, 0]).bool()
    opt = Optimizer(A, C = C)
    opt.run(nodes)
    print()
    print(opt.C.to_dense().int())