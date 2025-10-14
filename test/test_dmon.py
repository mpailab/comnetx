import os
import torch
import numpy as np
import sys
import torch, gc
import pytest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from baselines.dmon import adapted_dmon
from datasets import Dataset

@pytest.mark.long
def test_dmon_on_cora():
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset("cora", path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = adapted_dmon(adj, features, labels)

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0
    del adj, features, labels, new_labels

@pytest.mark.long    
def test_dmon_on_citeseer():
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset("citeseer", path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = adapted_dmon(adj, features, labels)

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0
    del adj, features, labels, new_labels

@pytest.mark.long
def test_dmon_on_eat():
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset("eat", path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = adapted_dmon(adj, features, labels)

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0
    del adj, features, labels, new_labels