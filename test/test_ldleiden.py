import os
import sys
import torch
import pytest

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

pytest.importorskip("dynamic_graphs_communities")
from dynamic_graphs_communities import BackendUnavailableError

from baselines.ldleiden import ldleiden_partition
from datasets import Dataset
from optimizer import Optimizer


def _run_or_skip(call):
    try:
        return call()
    except BackendUnavailableError:
        pytest.skip("dynamic_graphs_communities backend unavailable")


def _basic_adj():
    return torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

def _to_sparse(adj: torch.Tensor) -> torch.Tensor:
    return adj.to_sparse_coo().coalesce()


@pytest.mark.debug
def test_ldleiden_partition_basic():
    adj = _to_sparse(_basic_adj())
    timing_info = {}
    labels = _run_or_skip(lambda: ldleiden_partition(adj, timing_info=timing_info))

    assert isinstance(labels, torch.Tensor)
    assert labels.shape[0] == adj.shape[0]
    assert labels.dtype in (torch.int64, torch.long)
    assert labels.min() >= 0
    assert "conversion_time" in timing_info


@pytest.mark.debug
def test_single_community():
    adj = _to_sparse(torch.tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=torch.float32))

    labels = _run_or_skip(lambda: ldleiden_partition(adj))
    assert labels.unique().numel() == 1


@pytest.mark.debug
def test_partition_2():
    adj = _to_sparse(_basic_adj())
    labels = _run_or_skip(lambda: ldleiden_partition(adj))

    assert labels.unique().numel() == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


@pytest.mark.debug
def test_optimizer_ldleiden_branch():
    adj = _to_sparse(_basic_adj())
    features = torch.zeros((adj.size(0), 1), dtype=torch.float32)
    opt = Optimizer(adj, features=features, method="ldleiden")

    labels = _run_or_skip(lambda: opt.local_algorithm(adj, features))
    assert labels.shape[0] == adj.shape[0]


@pytest.mark.debug
def test_ldleiden_on_cora():
    data_dir = "/auto/datasets/graphs/small"
    dataset = Dataset("cora", path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = _run_or_skip(lambda: ldleiden_partition(adj))

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0
    del adj, features, labels, new_labels


@pytest.mark.debug
def test_ldleiden_on_citeseer():
    data_dir = "/auto/datasets/graphs/small"
    dataset = Dataset("citeseer", path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = _run_or_skip(lambda: ldleiden_partition(adj))

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0
    del adj, features, labels, new_labels
