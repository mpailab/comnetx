import sys
import os
import torch
import shutil
import tempfile
import pytest
import subprocess
import json
import numpy as np
import joblib
from pathlib import Path
from download import download_and_process_magi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from datasets import Dataset, INFO

TEST_DIR = os.path.dirname(__file__)
GRAPHS_DIR = "/auto/datasets/graphs/small"
SBM_GRAPHS_DIR = os.path.join(TEST_DIR, "graphs", "sbm")
PRGPT_DIR = "/auto/datasets/graphs/comnetx/baselines/PRGPT/data"

@pytest.fixture
def temp_dataset_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

@pytest.mark.long
@pytest.mark.parametrize(
    "dataset_name",
    [
        "cora",
        "citeseer",
        "pubmed",
        #"reddit",
        "ogbn-arxiv",
        "ogbn-products",
        "amazon-photo",
        "amazon-computers",
    ],
)
def test_load_magi_datasets(dataset_name, temp_dataset_dir, monkeypatch):
    download_and_process_magi(dataset_name, temp_dataset_dir)
    
    temp_info_dir = Path(temp_dataset_dir) / "datasets-info"
    temp_info_dir.mkdir(exist_ok=True)

    if dataset_name.lower() not in {"acm", "bat", "citeseer", "cora", "dblp", "eat", "uat"}:
        magi_path = temp_info_dir / "magi.json"
        magi_path.write_text(json.dumps({dataset_name.lower(): {"d": "undirected"}}))
    
    if dataset_name.lower() not in {"acm", "bat", "citeseer", "cora", "dblp", "eat", "uat"}:
        paths = {"magi": str(temp_dataset_dir)}
    else:
        paths = {"small": str(temp_dataset_dir)}
    paths_file = temp_info_dir / "temp_paths.json"
    paths_file.write_text(json.dumps(paths))
    
    monkeypatch.setattr("datasets.INFO", temp_info_dir)

    ds = Dataset(dataset_name, paths_config=str(paths_file))
    ds.load(tensor_type="coo")

    assert isinstance(ds.adj, torch.Tensor)
    assert ds.adj.is_sparse
    assert ds.adj.shape[0] == ds.adj.shape[1]
    assert ds.features is not None
    assert ds.label is not None

@pytest.mark.parametrize("batches_strategy", [
    "1",           # N стратегия
    "10",          # N стратегия  
    "real",        # raw timestamps
    "9:10",        # p:n стратегия
    "99:10",       # p:n стратегия
    "999:10",     # p:n стратегия
])
@pytest.mark.short
def test_load_wiki_talk_cy_dataset_strategies(temp_dataset_dir, batches_strategy, monkeypatch):
    """Тестирует различные стратегии батчинга."""

    real_info_dir = Path(__file__).parent.parent / "datasets-info" 
    paths_file = real_info_dir / "paths.json"
    
    def mock_info():
        return real_info_dir
    
    monkeypatch.setattr("datasets.INFO", mock_info())
    
    ds = Dataset("wiki_talk_cy", paths_config=str(paths_file))
    ds.load(batches_strategy=batches_strategy)
    adj = ds.adj
    
    # Базовые проверки
    assert isinstance(adj, torch.Tensor)
    assert adj.is_sparse
    assert adj.dim() == 3  # 3D тензор: [batches, nodes, nodes]
    assert adj.shape[1] == adj.shape[2]  # квадратные матрицы
    
    # Проверка количества батчей
    if batches_strategy == "real":
        assert adj.shape[0] > 0  # Должен быть хотя бы один батч
    elif ":" in batches_strategy:
        n = int(batches_strategy.split(":")[1])
        # 1 схлопнутый + n разделённых (неверно при больших значения p и n и мальньком числе ребер в датасете !)
        assert adj.shape[0] == 1 + n 
    else:
        assert adj.shape[0] == int(batches_strategy)  # N батчей

@pytest.mark.short
def test_load_wiki_talk_ht_dataset(temp_dataset_dir, monkeypatch):
    real_info_dir = Path(__file__).parent.parent / "datasets-info" 
    paths_file = real_info_dir / "paths.json"
    
    def mock_info():
        return real_info_dir
    
    monkeypatch.setattr("datasets.INFO", mock_info())
    loader = Dataset(dataset_name="wiki_talk_ht", paths_config=str(paths_file))
    tensor, features, label = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    if tensor.ndim == 2:  # static
        assert tensor.shape[0] == tensor.shape[1]
    elif tensor.ndim == 3:  # dynamic (T,N,N)
        assert tensor.shape[1] == tensor.shape[2]
    else:
        pytest.fail(f"Unexpected tensor shape: {tensor.shape}")

@pytest.mark.short
def test_tensor_dense_output(temp_dataset_dir, monkeypatch):
    temp_path = Path(temp_dataset_dir)
    download_and_process_magi("Cora", str(temp_path))
    
    paths_dir = temp_path / "datasets-info"
    paths_dir.mkdir(exist_ok=True)
    paths_file = paths_dir / "paths.json"
    paths_file.write_text(json.dumps({"small": str(temp_path)}))
    
    monkeypatch.setattr("datasets.INFO", paths_dir)
    loader = Dataset(dataset_name="Cora", paths_config=str(paths_file))
    tensor, features, label = loader.load(tensor_type="dense")

    assert isinstance(tensor, torch.Tensor)
    assert not tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]

@pytest.mark.short
def test_tensor_csr_output(temp_dataset_dir, monkeypatch):
    temp_path = Path(temp_dataset_dir)
    download_and_process_magi("Cora", str(temp_path))
    
    paths_dir = temp_path / "datasets-info"
    paths_dir.mkdir(exist_ok=True)
    paths_file = paths_dir / "paths.json"
    paths_file.write_text(json.dumps({"small": str(temp_path)}))
    
    monkeypatch.setattr("datasets.INFO", paths_dir)
    loader = Dataset(dataset_name="Cora", paths_config=str(paths_file))
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]

@pytest.mark.short
def test_tensor_csc_output(temp_dataset_dir, monkeypatch):
    temp_path = Path(temp_dataset_dir)
    download_and_process_magi("Cora", str(temp_path))
    
    paths_dir = temp_path / "datasets-info"
    paths_dir.mkdir(exist_ok=True)
    paths_file = paths_dir / "paths.json"
    paths_file.write_text(json.dumps({"small": str(temp_path)}))
    
    monkeypatch.setattr("datasets.INFO", paths_dir)
    loader = Dataset(dataset_name="Cora", paths_config=str(paths_file))
    tensor, features, label = loader.load(tensor_type="csc")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csc
    assert tensor.shape[0] == tensor.shape[1]

@pytest.mark.short
def test_exist_small_datasets():
    datasets = ["Acm", "Bat", "Eat"]
    paths_config = "/home/drobyshev/comnetx2/comnetx/datasets-info/paths.json"
    for dataset_name in datasets:
        ds = Dataset(dataset_name, paths_config=paths_config)
        ds.load(tensor_type="csr")
        assert isinstance(ds.adj, torch.Tensor)
        assert ds.adj.layout == torch.sparse_csr
        assert ds.adj.shape[0] == ds.adj.shape[1]
        assert ds.features is not None
        assert ds.label is not None

@pytest.mark.short
def test_invalid_tensor_type(temp_dataset_dir, monkeypatch):
    temp_path = Path(temp_dataset_dir)
    download_and_process_magi("Cora", str(temp_path))
    
    paths_dir = temp_path / "datasets-info"
    paths_dir.mkdir(exist_ok=True)
    paths_file = paths_dir / "paths.json"
    paths_file.write_text(json.dumps({"small": str(temp_path)}))
    
    monkeypatch.setattr("datasets.INFO", paths_dir)
    loader = Dataset(dataset_name="Cora", paths_config=str(paths_file))
    with pytest.raises(ValueError, match="Unsupported tensor type"):
        loader.load(tensor_type="invalid")

@pytest.mark.short
def test_unsupported_dataset(temp_dataset_dir, monkeypatch):
    temp_path = Path(temp_dataset_dir)
    download_and_process_magi("Cora", str(temp_path))
    
    paths_dir = temp_path / "datasets-info"
    paths_dir.mkdir(exist_ok=True)
    paths_file = paths_dir / "paths.json"
    paths_file.write_text(json.dumps({"small": str(temp_path)}))
    
    monkeypatch.setattr("datasets.INFO", paths_dir)
    with pytest.raises(ValueError, match="Dataset 'unsupported-ds' not in konect.json/magi.json and no pattern match."):
        loader = Dataset("unsupported-ds")
        loader.load()

@pytest.mark.short
def test_load_prgpt_static_dataset(tmp_path, monkeypatch):
    # ---------- STATIC ----------
    prgpt_dir = Path(PRGPT_DIR)
    dataset_name = "static_5_100000_2.5_3.0"

    info_dir = tmp_path / "datasets-info"
    info_dir.mkdir()

    test_paths_file = info_dir / "paths.json"
    test_paths_file.write_text(json.dumps({"dyn_sbm": str(prgpt_dir)}))

    dyn_sbm_path = info_dir / "dyn_sbm.json"
    dyn_sbm_path.write_text(json.dumps([dataset_name.lower()]))

    monkeypatch.setattr("datasets.INFO", info_dir)

    ds = Dataset(dataset_name, paths_config=str(test_paths_file))
    adj, features, labels = ds.load(tensor_type="coo")

    assert adj is not None, "self.adj None"
    assert labels is not None, "self.label None"

    if isinstance(ds.adj, list):
        assert len(ds.adj) == 5, "5 batches"
        n = ds.adj[0].shape[0]
        assert all(a.shape == (n, n) for a in ds.adj)
    else:
        assert isinstance(ds.adj, torch.Tensor) and ds.adj.is_sparse, \
            "adj - sparse_coo_tensor"
        assert ds.adj.shape[0] == 5, f"5 batches, was {ds.adj.shape[0]}"
        n = ds.adj.shape[1]
        assert ds.adj.shape == (5, n, n), f"form of sparse-tensor: {ds.adj.shape}"

    assert isinstance(ds.label, torch.Tensor)
    assert ds.label.dtype == torch.long
    assert ds.label.shape[0] == 5, "5 batches"
    assert ds.label.shape[1] == n, f"labels len n={n}, not {ds.label.shape[1]}"

    num_snapshots = ds.adj.shape[0]

    for snap in range(num_snapshots):
        adj_snap = ds.adj[snap].coalesce()
        adj_t = torch.sparse_coo_tensor(
            indices=adj_snap.indices().flip(0),
            values=adj_snap.values(),
            size=adj_snap.shape
        ).coalesce()
        diff = (adj_snap - adj_t).coalesce()
        nonzero_mask = diff.values() != 0
        nnz_nonzero = nonzero_mask.sum().item()
        assert nnz_nonzero == 0, f"Snap {snap}, nnz={nnz_nonzero}"

    assert ds.is_directed is False, "undirected graph"

def test_load_prgpt_stream_dataset():
    # ---------- STREAM ----------
    ds = Dataset(
        dataset_name="stream_5_100000_2.5_3.0",
        path=PRGPT_DIR
    )
    adj, features, labels = ds.load(tensor_type="coo")

    assert adj is not None, "self.adj None"
    assert labels is not None, "self.label None"

    if isinstance(ds.adj, list):
        assert len(ds.adj) == 10, "10 batches"
        n = ds.adj[0].shape[0]
        assert all(a.shape == (n, n) for a in ds.adj)
    else:
        assert isinstance(ds.adj, torch.Tensor) and ds.adj.is_sparse, \
            "adj - sparse_coo_tensor"
        assert ds.adj.shape[0] == 10, f"10 batches, was {ds.adj.shape[0]}"
        n = ds.adj.shape[1]
        assert ds.adj.shape == (10, n, n), f"form of sparse-tensor: {ds.adj.shape}"

    assert isinstance(ds.label, torch.Tensor)
    assert ds.label.dtype == torch.long
    assert ds.label.shape[0] == n

    num_snapshots = ds.adj.shape[0]

    for snap in range(num_snapshots):
        adj_snap = ds.adj[snap].coalesce()
        adj_t = torch.sparse_coo_tensor(
            indices=adj_snap.indices().flip(0),
            values=adj_snap.values(),
            size=adj_snap.shape
        ).coalesce()
        diff = (adj_snap - adj_t).coalesce()
        nonzero_mask = diff.values() != 0
        nnz_nonzero = nonzero_mask.sum().item()
        assert nnz_nonzero == 0, f"Snap {snap}, nnz={nnz_nonzero}"

    assert ds.is_directed is False, "undirected graph"

@pytest.mark.debug
def test_load_sbm_static_dataset():
    path = SBM_GRAPHS_DIR
    ds = Dataset(
        dataset_name="sbm_0b_100v_4c_undir_conn",
        path=path
    )

    adj, features, labels = ds.load(tensor_type="coo")

    assert adj is not None, "adj is None"
    assert labels is not None, "labels is None"

    assert isinstance(adj, torch.Tensor)
    assert adj.is_sparse, "Adjacency must be sparse COO"
    assert adj.is_coalesced(), "Sparse COO must be coalesced"

    assert ds.is_directed in [True, False], "Incorrect directed flag"

    n = adj.shape[0]
    assert adj.shape == (n, n), "Wrong shape for static SBM"

    if not ds.is_directed:
        A = adj.to_dense()
        assert torch.allclose(A, A.T), "Undirected graph must be symmetric"

    assert labels.dtype == torch.long
    assert labels.shape[0] == n, "Labels must be of shape [n]"
    assert features is None

@pytest.mark.short
def test_download_attr_graph(temp_dataset_dir, monkeypatch):
    """Тест скачивания attributed graph (wiki) через download_attr_graph."""
    from download import download_attr_graph
    
    dataset_name = "wiki"
    download_attr_graph(dataset_name, temp_dataset_dir)

    dname = dataset_name.lower()
    load_dir = Path(temp_dataset_dir) / dname
    required = [f"{dname}_feat.npy", f"{dname}_label.npy", f"{dname}_coo_adj.joblib"]
    
    assert load_dir.exists(), f"Директория {load_dir} не создана"
    assert all((load_dir / f).exists() for f in required), f"Файлы {required} отсутствуют"

    feat_shape = np.load(load_dir / f"{dname}_feat.npy").shape
    label_shape = np.load(load_dir / f"{dname}_label.npy").shape
    adj_data = joblib.load(load_dir / f"{dname}_coo_adj.joblib")
    
    assert feat_shape[0] == 2405, "Неправильное число узлов (features)"
    assert label_shape[0] == 2405, "Неправильное число узлов (labels)"
    assert adj_data['shape'] == (2405, 2405), "Неправильная форма adj"
    assert adj_data['indices'].shape[1] == 16523, "Неправильное число ребер (неориентированный граф)"

@pytest.mark.parametrize("dataset_name", ["wiki", "facebook", "blogcatalog"])
@pytest.mark.short
def test_download_attr_graphs(dataset_name, temp_dataset_dir):
    """Тест всех маленьких attr_graphs."""
    from download import download_attr_graph
    
    download_attr_graph(dataset_name, temp_dataset_dir)
    
    dname = dataset_name.lower()
    load_dir = Path(temp_dataset_dir) / dname
    
    feat = np.load(load_dir / f"{dname}_feat.npy")
    labels = np.load(load_dir / f"{dname}_label.npy")
    adj_data = joblib.load(load_dir / f"{dname}_coo_adj.joblib")
    
    assert feat.shape[0] == labels.shape[0] == adj_data['shape'][0]
    assert adj_data['indices'].shape[0] == 2
    
    print(f"{dataset_name}: {feat.shape[0]} nodes, {adj_data['indices'].shape[1]} edges")

@pytest.mark.debug
def test_load_sbm_temporal_dataset():
    path = SBM_GRAPHS_DIR

    ds = Dataset(
        dataset_name="tsbm_10b_100v_4c_undir_conn",
        path=path
    )

    adj, features, labels = ds.load(tensor_type="coo")

    assert adj is not None, "adj is None"
    assert labels is not None, "labels is None"

    assert isinstance(adj, torch.Tensor)
    assert adj.is_sparse, "Temporal adjacency should be sparse COO"
    assert adj.is_coalesced()

    assert adj.dim() == 3, "Temporal SBM must be 3D tensor"
    T, n, n2 = adj.shape
    assert n == n2, "Adj must be square"
    assert T > 1, "Temporal SBM must have multiple snapshots"

    assert labels.shape == (T, n)
    assert labels.dtype == torch.long

    assert adj.is_coalesced(), "Temporal adjacency must be coalesced"

    assert features is None

@pytest.mark.short
@pytest.mark.short
def test_local_wiki_attr_graph_full_pipeline():
    """ЛОКАЛЬНЫЙ тест wiki с вашим paths.json."""
    
    info_dir = Path(__file__).parent.parent / "datasets-info"
    paths_data = json.loads((info_dir / "paths.json").read_text())
    graphs_dir = Path(paths_data["attr_graphs"])
    
    print(f"✓ graphs_dir = {graphs_dir}")
    
    dataset_name = "wiki"

    from download import download_attr_graph
    download_attr_graph(dataset_name, str(graphs_dir))
    
    load_dir = graphs_dir / dataset_name.lower()
    
    feat_file = load_dir / "wiki_feat.npy"
    assert feat_file.exists()
    
    ds = Dataset(dataset_name, str(info_dir / "paths.json"))
    adj, feat, lbl = ds.load("coo")
    
    assert ds.dataset_format == "attr_graphs"
    assert str(ds.dataset_root) == str(graphs_dir)
    assert ds.is_directed is True
    
    print(f"✅ wiki: {adj.shape[0]}n/{adj._nnz()}e → {ds.dataset_root}")

@pytest.mark.short
def test_load_existing_cora_dynamic_konect(tmp_path, monkeypatch):
    """Тест cora dynamic out.cora.*_batches (small → konect формат)."""
    
    #info_dir = Path(__file__).parent.parent / "datasets-info"
    #paths_data = json.loads((info_dir / "paths.json").read_text())
    cora_dir = Path("/auto/datasets/graphs/dyn_atr_datasets/cora")
    info_dir = tmp_path / "datasets-info"  # ← В tmp_path!
    info_dir.mkdir()
    
    test_paths_file = info_dir / "test_paths.json"
    test_paths_file.write_text(json.dumps({"dynamic_konect": str(cora_dir.parent)}))
    
    # 2. konect.json (создай, не читай!)
    konect_path = info_dir / "konect.json"
    konect_data = {"cora": {"d": "undirected", "n": 2708, "m": 5429, "w": "unweighted"}}
    konect_path.write_text(json.dumps(konect_data, indent=2))  # ← Создай сразу!
    
    # 3. Monkeypatch
    monkeypatch.setattr("datasets.INFO", info_dir)

    dataset_name = "cora"
    strategies = ["1", "10", "100", "1000"]
    
    print(f"✓ Testing {dataset_name} в {cora_dir}")
    
    for bs in strategies:
        batch_file = cora_dir / f"out.{dataset_name}.{bs}_batches"
        assert batch_file.exists(), f"{batch_file} отсутствует — запустите save_small_datasets_in_konect_format()"
    
    #paths_data["dynamic_konect"] = str(cora_dir.parent)
    #(info_dir / "paths.json").write_text(json.dumps(paths_data, indent=2))
    
    try:
        for bs in strategies:
            print(f"  → {bs}_batches")
            
            ds = Dataset(dataset_name, paths_config=str(test_paths_file))
            adj, feat, lbl = ds.load("coo", batches_strategy=bs)
            
            assert ds.dataset_format == "dynamic_konect"
            assert ds.is_directed is False
            
            if bs == "1":
                assert adj.dim() == 3 and adj.shape == (1, 2708, 2708)
                assert adj._nnz() == 10556
            else:
                assert adj.dim() == 3
                assert adj.shape[0] == int(bs)
                assert adj.shape[1:] == (2708, 2708)
            
            assert adj.is_sparse and adj.is_coalesced()
            print(f"     ✓ L={adj.shape[0] if adj.dim()==3 else 1}, E={adj._nnz()}")
    
    finally:
        print("\ncora dynamic_konect: все стратегии OK!")

def collect_dynamic_attr_datasets():
    """Сбор ДИНАМИЧЕСКИХ датасетов из /auto/datasets/graphs/comnetx/dynamic_attr_datasets/."""
    dyn_attr_root = Path("/auto/datasets/graphs/comnetx/dynamic_attr_datasets")
    
    datasets = [d.name for d in dyn_attr_root.iterdir() 
                if d.is_dir() and not d.name.startswith('.')]
    
    expected = {'dyn_blogcatalog', 'dyn_coauthorcs', 'dyn_coauthorphysics', 'dyn_flickr', 
                'dyn_nell', 'dyn_ogbn-arxiv', 'dyn_ogbn-products', 'dyn_pubmed', 'dyn_reddit2', 'dyn_wiki', 'dyn_wikics'}
    
    found = set(datasets)
    missing = expected - found
    
    if missing:
        print(f"Отсутствуют датасеты: {missing} — тест пропущен")
        return []
    
    print(f"✓ Найдено {len(datasets)} динамических датасетов: {datasets}")
    return datasets

def collect_small_dynamic_attr_datasets():
    small_datasets = ["dyn_wiki", "dyn_pubmed", "dyn_coauthorcs"]  # n < 20k, быстро
    return small_datasets

@pytest.mark.parametrize("dataset_name", collect_small_dynamic_attr_datasets(), ids=lambda name: f"dyn_attr:{name}")
@pytest.mark.long
def test_dynamic_attr_dataset(dataset_name):
    """Тест динамических датасетов из dynamic_attr_datasets/."""

    info_dir = Path(__file__).parent.parent / "datasets-info"
    paths_data = json.loads((info_dir / "paths.json").read_text())

    dyn_root = Path("/auto/datasets/graphs/comnetx/dynamic_attr_datasets")
    
    print(f"dyn_attr:{dataset_name} ← {dyn_root / dataset_name}")

    ds = Dataset(dataset_name, str(info_dir / "paths.json"))
    adj, features, labels = ds.load("coo")

    assert adj.dim() == 3, f"{dataset_name}: ожидали 3D [batches,N,N]"
    assert adj.shape[1] == adj.shape[2], "Не квадратная матрица"
    assert adj.is_sparse and adj.is_coalesced()
    
    n_batches, n_nodes = adj.shape[:2]
    total_edges = sum(batch._nnz() for batch in adj)

    if not ds.is_directed:
        for b in range(min(3, n_batches)):
            batch_adj = adj[b]
            assert torch.allclose(
                batch_adj.to_dense(), 
                batch_adj.to_dense().T,
                atol=1e-6
            ), f"{dataset_name}[{b}]: не симметричен"

    feat_info = features.shape if features is not None else "None"
    label_info = labels.shape if labels is not None else "None"
    
    print(f"B={n_batches:4}, N={n_nodes:7}, E={total_edges:10,} "
          f"dir={ds.is_directed}, feat={feat_info}, label={label_info}")

    assert n_batches >= 1 and n_nodes > 10 and total_edges > 0, \
        f"{dataset_name}: слишком маленький/пустой"