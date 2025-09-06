import os
import torch
import numpy as np
import sys
import torch, gc
import pytest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from baselines.magi import magi
from datasets import Dataset


def get_all_datasets():
    """
    Сreate dict with all datasets in test directory.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    datasets = {}
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                datasets[name] = path
    return datasets


datasets = get_all_datasets()
@pytest.mark.long
@pytest.mark.parametrize(
    "name,data_dir",
    list(datasets.items()),
    ids=list(datasets.keys())
)
def test_magi_single_dataset(name, data_dir):
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = magi(adj, features, labels)

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0

    del adj, features, labels, new_labels

def test_features():
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset("cora", path=data_dir+"/cora")
    adj, features, labels = dataset.load(tensor_type="coo")

    print(features.size())
    print(features)
    print(torch.unique(features))

    dataset = Dataset("eat", path=data_dir+"/eat")
    adj, features, labels = dataset.load(tensor_type="coo")

    print(features.size())
    print(features)
    print(torch.unique(features))

    assert torch.cuda.is_available()
    