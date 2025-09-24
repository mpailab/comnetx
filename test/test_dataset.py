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

@pytest.fixture
def temp_dataset_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)

@pytest.mark.short
def test_load_small_datasets(temp_dataset_dir):
    datasets = ["Cora", "Citeseer", "Pubmed"]
    for dataset_name in datasets:
        ds = Dataset(dataset_name=dataset_name, path=temp_dataset_dir)
        ds.load(tensor_type="coo")
        assert isinstance(ds.adj, torch.Tensor)
        assert ds.adj.is_sparse
        assert ds.adj.shape[0] == ds.adj.shape[1]
        assert ds.features is not None
        assert ds.label is not None

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
def test_load_magi_big_datasets(temp_dataset_dir):
    datasets = ["Reddit", "ogbn-arxiv"]
    for dataset_name in datasets:
        ds = Dataset(dataset_name=dataset_name, path=temp_dataset_dir)
        ds.load(tensor_type="coo")
        assert isinstance(ds.adj, torch.Tensor)
        assert ds.adj.is_sparse
        assert ds.adj.shape[0] == ds.adj.shape[1]
        assert ds.features is not None
        assert ds.label is not None

@pytest.mark.short
def test_load_youtube_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="youtube-u-growth", path=KONECT_PATH)
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
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    tensor, features, label = loader.load(tensor_type="csc")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csc
    assert tensor.shape[0] == tensor.shape[1]

@pytest.mark.short
def test_exist_small_datasets():
    datasets = ["Cora", "Citeseer", "Acm", "Bat", "Dblp", "Eat"]
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
