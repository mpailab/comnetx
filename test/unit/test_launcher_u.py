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


@pytest.mark.unit
@pytest.mark.short
def test_load_dataset_if_needed_accepts_objects_and_legacy_names(monkeypatch):
    launcher = _load_launcher(monkeypatch)

    loaded_dataset = types.SimpleNamespace(name="loaded")
    assert launcher._load_dataset_if_needed(loaded_dataset, 10) is loaded_dataset

    fake_datasets_module = types.ModuleType("datasets")

    class _FakeDataset:
        def __init__(self, name):
            self.name = name
            self.loaded_with = None

        def load(self, batches_strategy=None):
            self.loaded_with = batches_strategy

    fake_datasets_module.Dataset = _FakeDataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)

    created = launcher._load_dataset_if_needed("fake-dataset", "999:10")

    assert isinstance(created, _FakeDataset)
    assert created.name == "fake-dataset"
    assert created.loaded_with == "999:10"


@pytest.mark.unit
@pytest.mark.short
def test_normalize_launch_mode_strips_case_and_rejects_invalid_values(
    monkeypatch,
):
    launcher = _load_launcher(monkeypatch)

    assert launcher._normalize_launch_mode(" SMART ") == "smart"
    assert launcher._normalize_launch_mode("Dynamic") == "dynamic"

    with pytest.raises(ValueError, match="Unsupported launch mode"):
        launcher._normalize_launch_mode("unknown")
    with pytest.raises(ValueError, match="Unsupported launch mode"):
        launcher._normalize_launch_mode(None)


@pytest.mark.unit
@pytest.mark.short
def test_init_batch_number_extracts_special_strategy_prefix(monkeypatch):
    launcher = _load_launcher(monkeypatch)

    assert launcher._init_batch_number("999:100") == "999"
    assert launcher._init_batch_number("0:cached") == "0"
    assert launcher._init_batch_number(10) is None
    assert launcher._init_batch_number(None) is None


@pytest.mark.unit
@pytest.mark.short
def test_adjacency_batch_helpers_handle_static_dynamic_and_bad_shapes(
    monkeypatch,
):
    launcher = _load_launcher(monkeypatch)
    static_adj = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    dynamic_adj = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)

    static_batches = launcher._iter_adjacency_batches(static_adj)
    dynamic_batches = launcher._iter_adjacency_batches(dynamic_adj)

    assert static_batches == [static_adj]
    assert len(dynamic_batches) == 3
    assert torch.equal(dynamic_batches[1], dynamic_adj[1])
    assert launcher._first_snapshot(static_adj) is static_adj
    assert torch.equal(launcher._first_snapshot(dynamic_adj), dynamic_adj[0])

    bad_adj = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsupported ds.adj ndim"):
        launcher._iter_adjacency_batches(bad_adj)
    with pytest.raises(ValueError, match="Unsupported ds.adj ndim"):
        launcher._first_snapshot(bad_adj)


@pytest.mark.unit
@pytest.mark.short
def test_compute_launch_initial_partition_forwards_args_and_prints_modularity(
    monkeypatch,
    capsys,
):
    launcher = _load_launcher(monkeypatch)
    batch = torch.eye(2)
    expected_partition = torch.tensor([0, 1])
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
        captured.update(
            adj_matrix=adj_matrix,
            dataset_name=dataset_name,
            init_batch_number=init_batch_number,
            method_name=method_name,
            cache_dir=cache_dir,
            subcoms_depth=subcoms_depth,
            device=device,
        )
        return expected_partition, 0.875

    monkeypatch.setattr(
        launcher,
        "compute_initial_partition",
        fake_compute_initial_partition,
    )

    partition = launcher._compute_launch_initial_partition(
        adj_matrix=batch,
        dataset_name="fake",
        init_batch_number="999",
        verbose=1,
        cache_dir="/tmp/cache",
        subcoms_depth=3,
        device=torch.device("cpu"),
    )

    assert partition is expected_partition
    assert captured == {
        "adj_matrix": batch,
        "dataset_name": "fake",
        "init_batch_number": "999",
        "method_name": "leidenalg",
        "cache_dir": "/tmp/cache",
        "subcoms_depth": 3,
        "device": torch.device("cpu"),
    }
    assert "Initial modularity: 0.88" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.short
def test_active_nodes_mask_handles_dense_and_sparse_batches(monkeypatch):
    launcher = _load_launcher(monkeypatch)

    dense_batch = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    dense_mask = launcher._active_nodes_mask(
        dense_batch,
        nodes_num=4,
        device=torch.device("cpu"),
    )
    assert torch.equal(dense_mask, torch.tensor([True, True, True, False]))

    sparse_batch = torch.sparse_coo_tensor(
        torch.tensor([[0, 3], [2, 1]]),
        torch.ones(2),
        size=(4, 4),
    )
    sparse_mask = launcher._active_nodes_mask(
        sparse_batch,
        nodes_num=4,
        device=torch.device("cpu"),
    )
    assert torch.equal(sparse_mask, torch.tensor([True, True, True, True]))


class _BatchOptimizer:
    def __init__(self):
        self.adj = torch.zeros((2, 2), dtype=torch.float32)
        self.features = torch.ones((2, 1), dtype=torch.float32)
        self.coms = torch.tensor([[9, 9]], dtype=torch.long)
        self.conversion_time = 1.0
        self.local_algorithm_labels = None
        self.set_communities_call = None
        self.neighborhood_call = None
        self.run_mask = None

    def runtime_adj(self):
        return self.adj

    def runtime_features(self):
        return self.features

    def local_algorithm(self, adj, features, labels=None):
        self.local_algorithm_labels = labels
        self.conversion_time += 0.25
        return torch.tensor([1, 0], dtype=torch.long)

    def set_communities(self, communities, replace_subcoms_depth=False):
        self.set_communities_call = (communities, replace_subcoms_depth)
        self.coms = communities

    def neighborhood(self, adj, nodes_mask, step=1):
        self.neighborhood_call = (adj, nodes_mask.clone(), step)
        return ~nodes_mask

    def run(self, nodes_mask):
        self.run_mask = nodes_mask.clone()


@pytest.mark.unit
@pytest.mark.short
def test_run_optimizer_batch_raw_mode_resets_labels_and_subtracts_conversion(
    monkeypatch,
):
    launcher = _load_launcher(monkeypatch)
    opt = _BatchOptimizer()
    clock = iter([10.0, 13.0])
    monkeypatch.setattr(launcher.time, "perf_counter", lambda: next(clock))

    measured_time = launcher._run_optimizer_batch(
        opt,
        mode="raw",
        affected_nodes_mask=None,
        smart_neighborhood_step=1,
    )

    communities, replace_depth = opt.set_communities_call
    assert measured_time == pytest.approx(2.75)
    assert opt.local_algorithm_labels is None
    assert replace_depth is True
    assert torch.equal(communities, torch.tensor([[1, 0]]))


@pytest.mark.unit
@pytest.mark.short
def test_run_optimizer_batch_naive_mode_reuses_existing_labels(monkeypatch):
    launcher = _load_launcher(monkeypatch)
    opt = _BatchOptimizer()
    previous_labels = opt.coms
    clock = iter([20.0, 22.0])
    monkeypatch.setattr(launcher.time, "perf_counter", lambda: next(clock))

    measured_time = launcher._run_optimizer_batch(
        opt,
        mode="naive",
        affected_nodes_mask=None,
        smart_neighborhood_step=1,
    )

    assert measured_time == pytest.approx(1.75)
    assert opt.local_algorithm_labels is previous_labels


@pytest.mark.unit
@pytest.mark.short
def test_run_optimizer_batch_smart_mode_expands_mask_and_runs_optimizer(
    monkeypatch,
):
    launcher = _load_launcher(monkeypatch)
    opt = _BatchOptimizer()
    affected_mask = torch.tensor([True, False])
    clock = iter([30.0, 31.0])
    monkeypatch.setattr(launcher.time, "perf_counter", lambda: next(clock))

    measured_time = launcher._run_optimizer_batch(
        opt,
        mode="smart",
        affected_nodes_mask=affected_mask,
        smart_neighborhood_step=2,
    )

    assert measured_time == pytest.approx(1.0)
    assert opt.neighborhood_call[2] == 2
    assert torch.equal(opt.neighborhood_call[1], affected_mask)
    assert torch.equal(opt.run_mask, torch.tensor([False, True]))


@pytest.mark.unit
@pytest.mark.short
def test_print_optimizer_batch_result_uses_method_specific_timing(
    monkeypatch,
    capsys,
):
    launcher = _load_launcher(monkeypatch)
    opt = types.SimpleNamespace(last_timing_info={"algorithm_time": 1.25})

    launcher._print_optimizer_batch_result(
        verbose=2,
        method="ldleiden",
        mode="raw",
        opt=opt,
        modularity=0.5,
        measured_time=9.0,
    )
    ldleiden_output = capsys.readouterr().out
    assert "Modularity: 0.5" in ldleiden_output
    assert "Algorithm time: 1.25" in ldleiden_output
    assert "Time: 9.00" not in ldleiden_output

    launcher._print_optimizer_batch_result(
        verbose=2,
        method="leidenalg",
        mode="raw",
        opt=opt,
        modularity=0.5,
        measured_time=9.0,
    )
    default_output = capsys.readouterr().out
    assert "Time: 9.00" in default_output


@pytest.mark.unit
@pytest.mark.short
def test_run_dynamic_mfc_uses_initial_partition_and_returns_single_result(
    monkeypatch,
):
    launcher = _load_launcher(monkeypatch)
    ds = types.SimpleNamespace(
        adj=torch.stack((torch.eye(2), torch.ones((2, 2)))),
        is_directed=True,
    )
    initial_partition = torch.tensor([1, 0])
    captured = {}

    def fake_compute_launch_initial_partition(
        adj_matrix,
        dataset_name,
        init_batch_number,
        cache_dir=None,
        subcoms_depth=1,
        device=None,
        verbose=0,
    ):
        captured["initial_adj_matrix"] = adj_matrix
        captured["dataset_name"] = dataset_name
        captured["init_batch_number"] = init_batch_number
        captured["cache_dir"] = cache_dir
        return initial_partition

    def fake_mfc_adopted(**kwargs):
        captured["mfc_kwargs"] = kwargs
        return torch.tensor([0, 0], dtype=torch.long)

    def fake_modularity(adjacency, assignments, directed=False):
        captured["metric_args"] = (adjacency, assignments, directed)
        return 0.42

    monkeypatch.setattr(
        launcher,
        "_compute_launch_initial_partition",
        fake_compute_launch_initial_partition,
    )
    monkeypatch.setattr(
        sys.modules["baselines.mfc"],
        "mfc_adopted",
        fake_mfc_adopted,
    )
    monkeypatch.setattr(
        launcher,
        "Metrics",
        types.SimpleNamespace(modularity=fake_modularity),
    )
    clock = iter([1.0, 2.5])
    monkeypatch.setattr(launcher.time, "perf_counter", lambda: next(clock))

    results = launcher._run_dynamic_mfc(
        ds,
        dataset_name="fake",
        init_batch_number="999",
        baseline_iter=7,
        verbose=0,
        cache_dir="/tmp/cache",
    )

    assert results == [{"modularity": 0.42, "time": pytest.approx(1.5)}]
    assert torch.equal(captured["initial_adj_matrix"], ds.adj[0])
    assert captured["dataset_name"] == "fake"
    assert captured["init_batch_number"] == "999"
    assert captured["cache_dir"] == "/tmp/cache"
    assert captured["mfc_kwargs"]["num_epoch"] == 7
    assert captured["mfc_kwargs"]["pure_mfc"] is True
    assert captured["mfc_kwargs"]["initial_partition"] is initial_partition
    assert captured["metric_args"][2] is True


@pytest.mark.unit
@pytest.mark.short
def test_run_dynamic_backend_primes_special_strategy_and_skips_initial_result(
    monkeypatch,
):
    launcher = _load_launcher(monkeypatch)
    batches = [
        torch.zeros((2, 2), dtype=torch.float32),
        torch.ones((2, 2), dtype=torch.float32),
    ]
    initial_partition = torch.tensor([0, 1])
    created_algos = []
    captured = {}

    class _FakeDynamicAlgo:
        def __init__(self, batch, partition):
            self.batch = batch
            self.partition = partition
            self.apply_calls = 0
            self.updated_batches = []
            created_algos.append(self)

        def update(self, batch):
            self.updated_batches.append(batch)

        def apply(self):
            self.apply_calls += 1
            return 1500.0

        def modularity(self):
            return 0.66

    def fake_create_leiden(method, batch, partition=None):
        captured["method"] = method
        return _FakeDynamicAlgo(batch, partition)

    def fake_compute_launch_initial_partition(
        adj_matrix,
        dataset_name,
        init_batch_number,
        cache_dir=None,
        subcoms_depth=1,
        device=None,
        verbose=0,
    ):
        captured["initial_args"] = (adj_matrix, dataset_name, init_batch_number)
        return initial_partition

    monkeypatch.setattr(launcher, "create_leiden", fake_create_leiden)
    monkeypatch.setattr(
        launcher,
        "_compute_launch_initial_partition",
        fake_compute_launch_initial_partition,
    )

    results = launcher._run_dynamic_backend(
        batches,
        dataset_name="fake",
        method="ldleiden",
        init_batch_number="999",
        verbose=0,
        cache_dir=None,
    )

    assert results == [{"modularity": 0.66, "time": 1.5}]
    assert captured["method"] == "ldleiden"
    assert captured["initial_args"] == (batches[0], "fake", "999")
    assert created_algos[0].partition is initial_partition
    assert created_algos[0].apply_calls == 2
    assert created_algos[0].updated_batches == [batches[1]]


class _PipelineOptimizer:
    instances = []

    def __init__(self, adj_matrix, features=None, **kwargs):
        self.adj = adj_matrix
        self.features = features
        self.kwargs = kwargs
        self.nodes_num = adj_matrix.size(0)
        self.subcoms_depth = kwargs.get("subcoms_depth", 1)
        self.device = adj_matrix.device
        self.coms = torch.zeros((1, self.nodes_num), dtype=torch.long)
        self.conversion_time = 0.0
        self.local_algorithm_calls = 0
        self.last_timing_info = {"algorithm_time": 0.0}
        self.updated = []
        self.set_calls = []
        type(self).instances.append(self)

    def runtime_device(self):
        return self.device

    def runtime_adj(self):
        return self.adj

    def runtime_features(self):
        return self.features

    def update_adj(self, batch, return_mask=True):
        self.updated.append((batch, return_mask))
        self.adj = self.adj + batch
        if return_mask:
            return torch.tensor([True, False], device=self.device)
        return None

    def local_algorithm(self, adj, features, labels=None):
        self.local_algorithm_calls += 1
        return torch.arange(adj.size(0), dtype=torch.long)

    def set_communities(self, communities, replace_subcoms_depth=False):
        self.set_calls.append((communities.clone(), replace_subcoms_depth))
        self.coms = communities

    def modularity(self, directed=False):
        return 0.75


@pytest.mark.unit
@pytest.mark.short
def test_run_optimizer_modes_processes_batches_through_optimizer(
    monkeypatch,
):
    launcher = _load_launcher(monkeypatch)
    _PipelineOptimizer.instances.clear()
    monkeypatch.setattr(launcher, "Optimizer", _PipelineOptimizer)
    clock = iter([0.0, 1.0, 10.0, 13.0])
    monkeypatch.setattr(launcher.time, "perf_counter", lambda: next(clock))
    ds = types.SimpleNamespace(
        features=torch.ones((2, 1), dtype=torch.float32),
        is_directed=True,
    )
    batches = [
        torch.zeros((2, 2), dtype=torch.float32),
        torch.ones((2, 2), dtype=torch.float32),
    ]

    results = launcher._run_optimizer_modes(
        ds,
        batches,
        dataset_name="fake",
        method="leidenalg",
        baseline_iter=3,
        mode="raw",
        smart_subcoms_depth=5,
        smart_neighborhood_step=1,
        verbose=0,
        use_gpu=False,
        aggregation_mode="sum",
        init_batch_number=None,
        cache_dir=None,
    )

    opt = _PipelineOptimizer.instances[-1]
    assert results == [
        {"modularity": 0.75, "time": pytest.approx(1.0)},
        {"modularity": 0.75, "time": pytest.approx(3.0)},
    ]
    assert opt.kwargs["subcoms_depth"] == 1
    assert opt.kwargs["method"] == "leidenalg"
    assert opt.kwargs["baseline_iter"] == 3
    assert opt.kwargs["aggregation_mode"] == "sum"
    assert opt.features is ds.features
    assert opt.updated == [(batches[1], False)]
    assert len(opt.set_calls) == 2


@pytest.mark.unit
@pytest.mark.short
def test_print_launch_summary_reports_final_modularity_and_total_time(
    monkeypatch,
    capsys,
):
    launcher = _load_launcher(monkeypatch)

    launcher._print_launch_summary(
        [
            {"modularity": 0.1, "time": 1.0},
            {"modularity": 0.2, "time": 2.5},
        ],
        verbose=1,
    )

    output = capsys.readouterr().out
    assert "Final modularity: 0.2" in output
    assert "Total time: 3.50" in output
