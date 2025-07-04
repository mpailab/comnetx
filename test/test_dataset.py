import sys
import os
import torch
import shutil
import tempfile
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from datasets import Dataset


@pytest.fixture
def temp_dataset_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


def test_load_Cora_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    tensor = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

def test_load_Citeseer_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="Citeseer", path=temp_dataset_dir)
    tensor = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

def test_load_pubmed_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="pubmed", path=temp_dataset_dir)
    tensor = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

def test_load_Reddit_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="Reddit", path="graphs/small/reddit")
    tensor = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

def test_load_OGDB_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="ogbn-arxiv", path=temp_dataset_dir)
    tensor = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]
    assert loader.features is not None
    assert loader.label is not None

def test_load_youtube_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="out.youtube-u-growth", path="graphs/small/youtube-u-growth")
    tensor = loader.load(tensor_type="coo")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]

def test_tensor_dense_output(temp_dataset_dir):
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    tensor = loader.load(tensor_type="dense")

    assert isinstance(tensor, torch.Tensor)
    assert not tensor.is_sparse
    assert tensor.shape[0] == tensor.shape[1]

def test_tensor_csr_output(temp_dataset_dir):
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    tensor = loader.load(tensor_type="csr")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csr
    assert tensor.shape[0] == tensor.shape[1]

def test_tensor_csc_output(temp_dataset_dir):
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    tensor = loader.load(tensor_type="csc")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.layout == torch.sparse_csc
    assert tensor.shape[0] == tensor.shape[1]



def test_invalid_tensor_type(temp_dataset_dir):
    loader = Dataset(dataset_name="Cora", path=temp_dataset_dir)
    with pytest.raises(ValueError, match="Unsupported tensor type"):
        loader.load(tensor_type="invalid")


def test_unsupported_dataset(temp_dataset_dir):
    loader = Dataset(dataset_name="unsupported-ds", path=temp_dataset_dir)
    with pytest.raises(ValueError, match="Unsupported dataset"):
        loader.load()