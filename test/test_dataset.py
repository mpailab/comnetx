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
def test_load_Cora_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    tensor, features, label = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

@pytest.mark.short
def test_load_Citeseer_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="Citeseer", path=temp_dataset_dir)
    tensor, features, label = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

@pytest.mark.short
def test_load_pubmed_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="pubmed", path=temp_dataset_dir)
    tensor, features, label = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

"""
@pytest.mark.short
def test_load_Reddit_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="Reddit", path="graphs/small/reddit")
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

@pytest.mark.short
def test_load_OGDB_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="ogbn-arxiv", path=temp_dataset_dir)
    tensor, features, label = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None
"""

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
def test_exist_cora_dataset():
    loader = Dataset(dataset_name="Cora", path=GRAPHS_DIR)
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

@pytest.mark.short
def test_exist_citeseer_dataset():
    loader = Dataset(dataset_name="Citeseer", path=GRAPHS_DIR)
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

@pytest.mark.short
def test_exist_acm_dataset():
    loader = Dataset(dataset_name="Acm", path=GRAPHS_DIR)
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

@pytest.mark.short
def test_exist_bat_dataset():
    loader = Dataset(dataset_name="Bat", path=GRAPHS_DIR)
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

@pytest.mark.short
def test_exist_dblp_dataset():
    loader = Dataset(dataset_name="Dblp", path=GRAPHS_DIR)
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

@pytest.mark.short
def test_exist_eat_dataset():
    loader = Dataset(dataset_name="Eat", path=GRAPHS_DIR)
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

@pytest.mark.short
def test_exist_uat_dataset():
    loader = Dataset(dataset_name="Uat", path=GRAPHS_DIR)
    tensor, features, label = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

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
