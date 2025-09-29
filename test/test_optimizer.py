import sys
import os
import torch
import pytest
from typing import Union

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from optimizer import Optimizer
import sparse


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
def test_run_networkit():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3]])
    opt = Optimizer(A, communities = communities, method  = "networkit")
    nodes_mask = torch.tensor([0, 0, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())

