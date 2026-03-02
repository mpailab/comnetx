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
    assert tensor.shape[0] == tensor.shape[1]

@pytest.mark.short
def test_tensor_dense_output(temp_dataset_dir):
    temp_path = Path(temp_dataset_dir)
    download_and_process_magi("Cora", str(temp_path))
    
    paths_dir = temp_path / "datasets-info"
    paths_dir.mkdir(exist_ok=True)
    paths_file = paths_dir / "paths.json"
    paths_file.write_text(json.dumps({"small": str(temp_path)}))
    
    monkeypatch.setattr("datasets.INFO", paths_dir)
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
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
def test_tensor_csc_output(temp_dataset_dir):
    temp_path = Path(temp_dataset_dir)
    download_and_process_magi("Cora", str(temp_path))
    
    paths_dir = temp_path / "datasets-info"
    paths_dir.mkdir(exist_ok=True)
    paths_file = paths_dir / "paths.json"
    paths_file.write_text(json.dumps({"small": str(temp_path)}))
    
    monkeypatch.setattr("datasets.INFO", paths_dir)
    loader = Dataset(dataset_name="Citeseer", path=temp_dataset_dir)
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
def test_invalid_tensor_type(temp_dataset_dir):
    download_and_process_magi("Cora", temp_dataset_dir)
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    with pytest.raises(ValueError, match="Unsupported tensor type"):
        loader.load(tensor_type="invalid")

@pytest.mark.short
def test_unsupported_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="unsupported-ds", path=temp_dataset_dir)
    with pytest.raises(ValueError, match="Unsupported dataset"):
        loader.load()

@pytest.mark.short
def test_load_prgpt_static_dataset():
    # ---------- STATIC ----------
    ds = Dataset(
        dataset_name="static_5_100000_2.5_3.0",
        path=PRGPT_DIR
    )
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
    from download import download_attr_graph  # Импорт из download.py
    
    dataset_name = "wiki"  # Маленький: 2405 nodes
    download_attr_graph(dataset_name, temp_dataset_dir)
    
    # Проверяем файлы
    dname = dataset_name.lower()
    load_dir = Path(temp_dataset_dir) / dname
    required = [f"{dname}_feat.npy", f"{dname}_label.npy", f"{dname}_coo_adj.joblib"]
    
    assert load_dir.exists(), f"Директория {load_dir} не создана"
    assert all((load_dir / f).exists() for f in required), f"Файлы {required} отсутствуют"
    
    # Размеры из attr_graphs.json
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
    
    # Все файлы созданы
    feat = np.load(load_dir / f"{dname}_feat.npy")
    labels = np.load(load_dir / f"{dname}_label.npy")
    adj_data = joblib.load(load_dir / f"{dname}_coo_adj.joblib")
    
    assert feat.shape[0] == labels.shape[0] == adj_data['shape'][0]
    assert adj_data['indices'].shape[0] == 2  # COO format
    
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
    
    # Скачиваем
    from download import download_attr_graph
    download_attr_graph(dataset_name, str(graphs_dir))
    
    load_dir = graphs_dir / dataset_name.lower()
    
    # Файлы
    feat_file = load_dir / "wiki_feat.npy"
    assert feat_file.exists()
    
    # Dataset (теперь работает!)
    ds = Dataset(dataset_name, str(info_dir / "paths.json"))
    adj, feat, lbl = ds.load("coo")
    
    assert ds.dataset_format == "attr_graphs"  # ← если исправили detect!
    assert str(ds.dataset_root) == str(graphs_dir)
    assert ds.is_directed is True
    
    print(f"✅ wiki: {adj.shape[0]}n/{adj._nnz()}e → {ds.dataset_root}")

