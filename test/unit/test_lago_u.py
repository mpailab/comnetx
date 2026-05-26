import sys
import types

import torch

from baselines.lago import lago_partition
from optimizer import Optimizer


class _DummyDynamicAlgo:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self):
        return 0.0

    def partition(self):
        return torch.zeros(1, dtype=torch.long)

    def modularity(self):
        return 0.0

    def update(self, batch):
        return None


dynamic_stub = types.ModuleType("dynamic_graphs_communities")
dynamic_stub.LDLeiden = _DummyDynamicAlgo
dynamic_stub.DFLeiden = _DummyDynamicAlgo
dynamic_stub.Leidenalg = _DummyDynamicAlgo
dynamic_stub.Networkit = _DummyDynamicAlgo
dynamic_stub.AlgorithmOptions = type("AlgorithmOptions", (), {})
sys.modules.setdefault("dynamic_graphs_communities", dynamic_stub)

from launcher import dynamic_launch


def _two_component_adj():
    return torch.tensor(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ],
        dtype=torch.float32,
    ).to_sparse_coo().coalesce()


def _assert_partition(labels, num_nodes):
    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (num_nodes,)
    assert labels.dtype in (torch.int64, torch.long)
    assert labels.min() >= 0


def test_lago_partition_static_smoke():
    adj = _two_component_adj()
    labels = lago_partition(adj, nb_iter=1, refinement=None)

    _assert_partition(labels, adj.size(0))


def test_optimizer_lago_branch_smoke():
    adj = _two_component_adj()
    features = torch.zeros((adj.size(0), 1), dtype=torch.float32)
    opt = Optimizer(adj, features=features, method="lago")

    labels = opt.local_algorithm(adj, features)

    _assert_partition(labels, adj.size(0))


def test_dynamic_launch_lago_modes_smoke():
    first = _two_component_adj()
    second = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ).to_sparse_coo().coalesce()

    class DummyDataset:
        pass

    ds = DummyDataset()
    ds.name = "dummy"
    ds.adj = torch.stack([first, second])
    ds.features = None
    ds.label = None
    ds.is_directed = False

    for mode in ("raw", "naive", "smart", "dynamic"):
        results = dynamic_launch(
            ds,
            2,
            "lago",
            mode=mode,
            smart_subcoms_depth=1,
            smart_neighborhood_step=0,
            verbose=0,
        )
        assert results
        assert "Final modularity" in results[-1]
