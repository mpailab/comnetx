import importlib.util
import sys
import types
from pathlib import Path

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
    monkeypatch.setitem(sys.modules, "dynamic_graphs_communities", stub)

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

    def __init__(self, adj_matrix, features=None, **kwargs):
        type(self).init_calls.append(kwargs.copy())
        self.adj = adj_matrix
        self.features = features
        self.nodes_num = adj_matrix.size(0)
        self.coms = torch.zeros((1, self.nodes_num), dtype=torch.long)
        self.conversion_time = 0.0
        self.local_algorithm_calls = 0
        self.last_timing_info = {}

    def runtime_adj(self):
        return self.adj

    def runtime_features(self):
        return self.features

    def local_algorithm(self, adj, features, limited=False, labels=None):
        self.local_algorithm_calls += 1
        return torch.arange(adj.size(0), dtype=torch.long)

    def set_communities(self, communities, replace_subcoms_depth=False):
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
