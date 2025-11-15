import sys
import os
import torch
import shutil
import tempfile
import pytest
import subprocess
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from datasets import Dataset, KONECT_PATH

TEST_DIR = os.path.dirname(__file__)
GRAPHS_DIR = os.path.join(TEST_DIR, "graphs", "small")
SBM_GRAPHS_DIR = os.path.join(TEST_DIR, "graphs", "sbm")
PRGPT_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "baselines", "PRGPT", "data"))

@pytest.fixture
def temp_dataset_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

@pytest.mark.short
def test_coo_joblib_save(temp_dataset_dir):
    ds = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    ds._load_magi()
    ds._save_magi(coo_adj = True)
    ds._save_magi(coo_adj = False)
    ds1 = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    ds2 = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    ds1._load_npy_format(coo_adj = True)
    ds2._load_npy_format(coo_adj = False)
    assert torch.equal(ds1.adj.to_dense(), ds2.adj.to_dense())

@pytest.mark.long
@pytest.mark.parametrize(
    "dataset_name",
    [
        "cora",
        "citeseer",
        "pubmed",
        "reddit",
        "ogbn-arxiv",
        "ogbn-products",
        "ogbn-papers100M",
        "amazon-photo",
        "amazon-computers",
    ],
)
def test_load_magi_datasets(dataset_name, temp_dataset_dir):
    ds = Dataset(dataset_name=dataset_name, path=temp_dataset_dir)
    ds.load(tensor_type="coo")
    assert isinstance(ds.adj, torch.Tensor)
    assert ds.adj.is_sparse
    assert ds.adj.shape[0] == ds.adj.shape[1]
    assert ds.features is not None
    assert ds.label is not None

@pytest.mark.short
def test_load_wiki_talk_ht_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="wiki_talk_ht", path=KONECT_PATH)
    tensor, features, label = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]

@pytest.mark.short
def test_tensor_dense_output(temp_dataset_dir):
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    tensor, features, label = loader.load(tensor_type="dense")

    assert isinstance(tensor, torch.Tensor)
    assert not tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]

@pytest.mark.short
def test_tensor_csr_output(temp_dataset_dir):
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]

@pytest.mark.short
def test_tensor_csc_output(temp_dataset_dir):
    loader = Dataset(dataset_name="Citeseer", path=temp_dataset_dir)
    tensor, features, label = loader.load(tensor_type="csc")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csc
    assert tensor.shape[0] == tensor.shape[1]

@pytest.mark.short
def test_exist_small_datasets():
    datasets = ["Acm", "Bat", "Eat"]
    for dataset_name in datasets:
        ds = Dataset(dataset_name=dataset_name, path=GRAPHS_DIR)
        ds.load(tensor_type="csr")
        assert isinstance(ds.adj, torch.Tensor)
        assert ds.adj.layout == torch.sparse_csr
        assert ds.adj.shape[0] == ds.adj.shape[1]
        assert ds.features is not None
        assert ds.label is not None

@pytest.mark.short
def test_invalid_tensor_type(temp_dataset_dir):
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
        dataset_name="stream_1_100000_2.5_3.0",
        path=PRGPT_DIR
    )
    adj, features, labels = ds.load(tensor_type="coo")

    assert isinstance(ds.adj, torch.Tensor), "adj must be torch.Tensor"
    assert ds.adj.is_sparse, "adj must be sparse COO"
    assert ds.label is not None and isinstance(ds.label, torch.Tensor)
    assert ds.label.shape[0] == len(ds.adj), "Count of snaps must be equal to len(label)"
    n = ds.adj.shape[1]
    assert ds.adj.shape[1] == ds.adj.shape[2], "Matrix must be nxn"
    assert ds.label.shape[1] == n, "Dim label = n"
    assert ds.is_directed is False

    for snap in range(len(ds.adj)):
        adj_snap = ds.adj[snap].coalesce()
        adj_t = torch.sparse_coo_tensor(
            indices=adj_snap.indices().flip(0),
            values=adj_snap.values(),
            size=adj_snap.shape
        ).coalesce()

        diff = (adj_snap - adj_t).coalesce()
        nnz_nonzero = (diff.values() != 0).sum().item()
        assert nnz_nonzero == 0, f"Snap {snap} not symmetric, nnz={nnz_nonzero}"

@pytest.mark.short
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
def test_load_sbm_temporal_dataset():
    path = SBM_GRAPHS_DIR

    ds = Dataset(
        dataset_name="tsbm_10s_100v_4c_undir_conn",
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

    if not ds.is_directed:
        dense = adj.to_dense()
        for t in range(T):
            A = dense[t]
            assert torch.allclose(A, A.T), f"Snapshot {t} must be symmetric for undirected"

    assert adj.is_coalesced(), "Temporal adjacency must be coalesced"

    assert features is None