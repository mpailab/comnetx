import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


class _DummyDynamicAlgo:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self):
        return 0.0

    def partition(self):
        return None

    def modularity(self):
        return 0.0

    def update(self, batch):
        return None


def _load_launcher(monkeypatch):
    stub = types.ModuleType("dynamic_graphs_communities")
    stub.LDLeiden = _DummyDynamicAlgo
    stub.DFLeiden = _DummyDynamicAlgo
    stub.Leidenalg = _DummyDynamicAlgo
    stub.Networkit = _DummyDynamicAlgo
    stub.AlgorithmOptions = type('AlgorithmOptions', (), {})
    monkeypatch.setitem(sys.modules, "dynamic_graphs_communities", stub)

    mfc_stub = types.ModuleType("baselines.mfc")
    mfc_stub.mfc_adopted = lambda *args, **kwargs: torch.zeros(2, dtype=torch.long)
    monkeypatch.setitem(sys.modules, "baselines.mfc", mfc_stub)

    spec = importlib.util.spec_from_file_location(
        "launcher_test_module",
        SRC / "launcher.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _SpyOptimizer:
    init_calls = []
    set_calls = []

    def __init__(self, adj_matrix, features=None, **kwargs):
        type(self).init_calls.append(kwargs.copy())
        self.adj = adj_matrix
        self.features = features
        self.nodes_num = adj_matrix.size(0)
        self.subcoms_depth = kwargs.get("subcoms_depth", 1)
        if kwargs.get("use_gpu") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = adj_matrix.device
        self.coms = torch.zeros((1, self.nodes_num), dtype=torch.long)
        self.conversion_time = 0.0
        self.local_algorithm_calls = 0
        self.last_timing_info = {}

    def runtime_device(self):
        return self.device

    def runtime_adj(self):
        return self.adj

    def runtime_features(self):
        return self.features

    def local_algorithm(self, adj, features, limited=False, labels=None):
        self.local_algorithm_calls += 1
        return torch.arange(adj.size(0), dtype=torch.long)

    def set_communities(self, communities, replace_subcoms_depth=False):
        type(self).set_calls.append(communities.clone())
        self.coms = communities

    def modularity(self, directed=False):
        return 0.0


@pytest.fixture
def fake_dataset():
    return types.SimpleNamespace(
        name="fake",
        adj=torch.zeros((1, 2, 2), dtype=torch.float32),
        features=torch.zeros((2, 1), dtype=torch.float32),
        is_directed=False,
    )


@pytest.mark.unit
@pytest.mark.short
def test_dynamic_launch_aggregation_mode_default(monkeypatch, fake_dataset):
    launcher = _load_launcher(monkeypatch)
    _SpyOptimizer.init_calls.clear()
    monkeypatch.setattr(launcher, "Optimizer", _SpyOptimizer)

    launcher.dynamic_launch(
        fake_dataset,
        1,
        "leidenalg",
        mode="raw",
        verbose=0,
    )

    assert _SpyOptimizer.init_calls[-1]["aggregation_mode"] == "sum"


@pytest.mark.unit
@pytest.mark.short
def test_dynamic_launch_aggregation_mode_override(monkeypatch, fake_dataset):
    launcher = _load_launcher(monkeypatch)
    _SpyOptimizer.init_calls.clear()
    monkeypatch.setattr(launcher, "Optimizer", _SpyOptimizer)

    launcher.dynamic_launch(
        fake_dataset,
        1,
        "leidenalg",
        mode="raw",
        verbose=0,
        aggregation_mode="normalized",
    )

    assert _SpyOptimizer.init_calls[-1]["aggregation_mode"] == "normalized"


@pytest.mark.unit
@pytest.mark.short
def test_compute_initial_partition_builds_layered_tensor(monkeypatch):
    launcher = _load_launcher(monkeypatch)

    class _FakeLeiden:
        partitions = [
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 0]),
        ]
        calls = []

        def __init__(self, adj):
            type(self).calls.append(adj)
            self._partition = type(self).partitions[len(type(self).calls) - 1]

        def apply(self):
            return 0.0

        def partition(self):
            return self._partition

        def modularity(self):
            return 0.75

    monkeypatch.setattr(
        launcher,
        "create_leiden",
        lambda method_name, adj: _FakeLeiden(adj),
    )

    adj = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=torch.float32,
    ).to_sparse_coo()

    partition, init_mod = launcher.compute_initial_partition(
        adj,
        dataset_name="fake",
        init_batch_number=0,
        subcoms_depth=2,
    )

    assert init_mod == 0.75
    assert partition.shape == (2, 4)
    assert partition.device == adj.device
    assert torch.equal(partition[0], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(partition[1], torch.tensor([0, 0, 0, 0]))
    assert _FakeLeiden.calls[1].shape == torch.Size([2, 2])
    assert _FakeLeiden.calls[1].device == adj.device


@pytest.mark.unit
@pytest.mark.short
def test_compute_initial_partition_loads_cached_partition_as_tensor(
    monkeypatch,
    tmp_path,
):
    launcher = _load_launcher(monkeypatch)
    cache_file = tmp_path / "fake_b:0_by_leidenalg.npz"
    np.savez_compressed(
        cache_file,
        partition=np.array([1, 1, 0]),
        mod=0.5,
    )
    monkeypatch.setattr(
        launcher,
        "create_leiden",
        lambda *args, **kwargs: pytest.fail("cache should skip algorithm"),
    )

    partition, init_mod = launcher.compute_initial_partition(
        torch.zeros((3, 3), dtype=torch.float32),
        dataset_name="fake",
        init_batch_number=0,
        cache_dir=tmp_path,
    )

    assert init_mod == 0.5
    assert partition.dtype == torch.long
    assert partition.device == torch.device("cpu")
    assert torch.equal(partition, torch.tensor([1, 1, 0]))


@pytest.mark.unit
@pytest.mark.short
def test_dynamic_launch_special_strategy_uses_layered_initial_partition(
    monkeypatch,
    fake_dataset,
):
    launcher = _load_launcher(monkeypatch)
    _SpyOptimizer.init_calls.clear()
    _SpyOptimizer.set_calls.clear()
    monkeypatch.setattr(launcher, "Optimizer", _SpyOptimizer)

    initial_layers = torch.tensor([
        [0, 0],
        [0, 1],
        [0, 1],
    ])
    captured = {}

    def fake_compute_initial_partition(
        adj_matrix,
        dataset_name,
        init_batch_number,
        method_name="leidenalg",
        cache_dir=None,
        subcoms_depth=1,
        device=None,
    ):
        captured["subcoms_depth"] = subcoms_depth
        captured["device"] = device
        return initial_layers, 0.25

    monkeypatch.setattr(
        launcher,
        "compute_initial_partition",
        fake_compute_initial_partition,
    )

    launcher.dynamic_launch(
        fake_dataset,
        "0:cached",
        "leidenalg",
        mode="smart",
        smart_subcoms_depth=3,
        verbose=0,
    )

    assert captured["subcoms_depth"] == 3
    assert captured["device"] == torch.device("cpu")
    assert torch.equal(_SpyOptimizer.set_calls[-1], initial_layers)
