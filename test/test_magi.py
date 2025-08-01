import os
import torch
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from baselines import magi
from datasets import Dataset


def test_magi_on_cora():
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset("cora", path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = magi(adj, features, labels)

    assert isinstance(new_labels, torch.Tensor),
    assert new_labels.shape[0] == labels.shape[0],
    assert new_labels.dtype in (torch.int64, torch.long),
    assert new_labels.min() >= 0,