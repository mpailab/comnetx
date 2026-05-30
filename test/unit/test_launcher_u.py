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


def _test_config(launcher, **overrides):
    defaults = {
        "dataset_name": "fake",
        "method": "leidenalg",
        "baseline_iter": None,
        "mode": "raw",
        "smart_subcoms_depth": 1,
        "smart_neighborhood_step": 1,
        "verbose": 0,
        "use_gpu": False,
        "aggregation_mode": "sum",
        "cache_dir": None,
        "init_batch_number": None,
        "ground_truth_metrics": False
    }
    defaults.update(overrides)
    return launcher._LaunchConfig(**defaults)


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
        label=None
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
def test_build_layered_initial_partition_restores_original_node_labels(
    monkeypatch,
):
    launcher = _load_launcher(monkeypatch)

    class _FakeLeiden:
        calls = []

        def __init__(self, adj):
            type(self).calls.append(adj)

        def apply(self):
            return 0.0

        def partition(self):
            return torch.tensor([0, 0], dtype=torch.long)

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
    )
    initial_partition = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    layered = launcher._build_layered_initial_partition(
        adj,
        adj,
        initial_partition,
        "leidenalg",
        subcoms_depth=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(layered[0], initial_partition)
    assert torch.equal(layered[1], torch.tensor([0, 0, 0, 0]))
    assert _FakeLeiden.calls[0].shape == torch.Size([2, 2])


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
def test_initial_partition_cache_helpers_build_save_and_load(
    monkeypatch,
    tmp_path,
):
    launcher = _load_launcher(monkeypatch)
    shallow_path = launcher._initial_partition_cache_path(
        tmp_path,
        "fake",
        0,
        "leidenalg",
        subcoms_depth=1,
    )
    deep_path = launcher._initial_partition_cache_path(
        tmp_path,
        "fake",
        "999",
        "leidenalg",
        subcoms_depth=3,
    )

    assert Path(shallow_path).name == "fake_b:0_by_leidenalg.npz"
    assert Path(deep_path).name == "fake_b:999_by_leidenalg_d:3.npz"
    assert launcher._load_cached_initial_partition(
        tmp_path / "missing.npz",
        torch.device("cpu"),
    ) is None

    partition = torch.tensor([[0, 1], [1, 1]], dtype=torch.long)
    launcher._save_cached_initial_partition(deep_path, partition, 0.91)
    loaded_partition, loaded_mod = launcher._load_cached_initial_partition(
        deep_path,
        torch.device("cpu"),
    )

    assert loaded_mod == pytest.approx(0.91)
    assert torch.equal(loaded_partition, partition)


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
def test_build_launch_config_normalizes_public_launch_arguments(monkeypatch):
    launcher = _load_launcher(monkeypatch)
    ds = types.SimpleNamespace(name="fake-dataset")

    config = launcher._build_launch_config(
        ds=ds,
        batches_strategy="999:100",
        underlying_static_method="leidenalg",
        baseline_iter=5,
        mode=" SMART ",
        smart_subcoms_depth=4,
        smart_neighborhood_step=2,
        verbose=3,
        use_gpu=True,
        aggregation_mode="normalized",
        cache_dir="/tmp/cache",
        ground_truth_metrics=False
    )

    assert config.dataset_name == "fake-dataset"
    assert config.method == "leidenalg"
    assert config.baseline_iter == 5
    assert config.mode == "smart"
    assert config.smart_subcoms_depth == 4
    assert config.smart_neighborhood_step == 2
    assert config.verbose == 3
    assert config.use_gpu is True
    assert config.aggregation_mode == "normalized"
    assert config.cache_dir == "/tmp/cache"
    assert config.init_batch_number == "999"


@pytest.mark.unit
@pytest.mark.short
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"smart_subcoms_depth": 0}, "smart_subcoms_depth"),
        ({"smart_neighborhood_step": -1}, "smart_neighborhood_step"),
    ],
)
def test_build_launch_config_rejects_invalid_smart_parameters(
    monkeypatch,
    override,
    message,
):
    launcher = _load_launcher(monkeypatch)
    kwargs = {
        "ds": types.SimpleNamespace(name="fake-dataset"),
        "batches_strategy": 1,
        "underlying_static_method": "leidenalg",
        "baseline_iter": None,
        "mode": "smart",
        "smart_subcoms_depth": 1,
        "smart_neighborhood_step": 1,
        "verbose": 0,
        "use_gpu": False,
        "aggregation_mode": "sum",
        "cache_dir": None,
        "ground_truth_metrics": False
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        launcher._build_launch_config(**kwargs)


@pytest.mark.unit
@pytest.mark.short
def test_print_verbose_only_prints_enabled_levels(monkeypatch, capsys):
    launcher = _load_launcher(monkeypatch)

    launcher._print_verbose(1, 2, "hidden")
    assert capsys.readouterr().out == ""

    launcher._print_verbose(2, 2, "visible", 7)
    assert capsys.readouterr().out == "visible 7\n"


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

    static_batches = list(launcher._iter_adjacency_batches(static_adj))
    dynamic_batches = list(launcher._iter_adjacency_batches(dynamic_adj))

    assert static_batches == [static_adj]
    assert len(dynamic_batches) == 3
    assert torch.equal(dynamic_batches[1], dynamic_adj[1])
    assert launcher._first_snapshot(static_adj) is static_adj
    assert torch.equal(launcher._first_snapshot(dynamic_adj), dynamic_adj[0])

    bad_adj = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsupported ds.adj ndim"):
        list(launcher._iter_adjacency_batches(bad_adj))
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
    config = _test_config(launcher, mode="raw")

    measured_time = launcher._run_optimizer_batch(
        opt,
        config,
        affected_nodes_mask=None,
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
    previous_labels = opt.coms[0]
    clock = iter([20.0, 22.0])
    monkeypatch.setattr(launcher.time, "perf_counter", lambda: next(clock))
    config = _test_config(launcher, mode="naive")

    measured_time = launcher._run_optimizer_batch(
        opt,
        config,
        affected_nodes_mask=None,
    )

    assert measured_time == pytest.approx(1.75)
    assert torch.equal(opt.local_algorithm_labels, previous_labels)


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
    config = _test_config(
        launcher,
        mode="smart",
        smart_neighborhood_step=2,
    )

    measured_time = launcher._run_optimizer_batch(
        opt,
        config,
        affected_nodes_mask=affected_mask,
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
        _test_config(launcher, method="ldleiden", mode="raw", verbose=2),
        opt=opt,
        modularity=0.5,
        measured_time=9.0,
    )
    ldleiden_output = capsys.readouterr().out
    assert "Modularity: 0.5" in ldleiden_output
    assert "Algorithm time: 1.25" in ldleiden_output
    assert "Time: 9.00" not in ldleiden_output

    launcher._print_optimizer_batch_result(
        _test_config(launcher, method="leidenalg", mode="raw", verbose=2),
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
    config = _test_config(
        launcher,
        dataset_name="fake",
        init_batch_number="999",
        baseline_iter=7,
        cache_dir="/tmp/cache",
    )

    results, last_partition = launcher._run_dynamic_mfc(
        ds,
        config,
    )

    assert results == [{"modularity": 0.42, "time": pytest.approx(1.5)}]
    assert torch.equal(captured["initial_adj_matrix"], ds.adj[0])
    assert captured["dataset_name"] == "fake"
    assert captured["init_batch_number"] == "999"
    assert captured["cache_dir"] == "/tmp/cache"
    assert captured["mfc_kwargs"]["num_epoch"] == 7
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
            self._partition = partition
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
        
        def partition(self):
            return self._partition

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
    config = _test_config(
        launcher,
        dataset_name="fake",
        method="ldleiden",
        init_batch_number="999",
    )

    results, last_partition = launcher._run_dynamic_backend(
        batches,
        config,
    )

    assert results == [{"modularity": 0.66, "time": 1.5}]
    assert captured["method"] == "ldleiden"
    assert captured["initial_args"] == (batches[0], "fake", "999")
    assert created_algos[0].partition() is initial_partition
    assert created_algos[0].apply_calls == 2
    assert created_algos[0].updated_batches == [batches[1]]


@pytest.mark.unit
@pytest.mark.short
def test_run_dynamic_backend_rejects_empty_batch_iterable(monkeypatch):
    launcher = _load_launcher(monkeypatch)
    config = _test_config(launcher, method="ldleiden", mode="dynamic")

    with pytest.raises(ValueError, match="no adjacency batches"):
        launcher._run_dynamic_backend(iter(()), config)


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
    config = _test_config(
        launcher,
        dataset_name="fake",
        method="leidenalg",
        baseline_iter=3,
        mode="raw",
        smart_subcoms_depth=5,
        smart_neighborhood_step=1,
        aggregation_mode="sum",
    )

    results, last_partition = launcher._run_optimizer_modes(
        ds,
        batches,
        config,
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
