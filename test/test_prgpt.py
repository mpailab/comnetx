import sys
import os
import torch
import pytest

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from baselines.rough_PRGPT import rough_prgpt, reorder_to_tensor
from datasets import Dataset, KONECT_PATH

@pytest.fixture(scope="class")
def facebook_dataset():
    ds = Dataset("facebook-wosn-links", KONECT_PATH)
    ds.load()
    return ds

def test_rough_prgpt_infomax_on_facebook(facebook_dataset):
    rough_prgpt(facebook_dataset.adj, refine="infomax")

def test_rough_prgpt_infomax_on_facebook(facebook_dataset):
    rough_prgpt(facebook_dataset.adj, refine="locale")

@pytest.mark.short
def test_reorder():
    clus_res = [1, 1, 0]
    res = reorder_to_tensor(clus_res, 2, zero_base = True)
    true = torch.tensor([[False, False, True], [True, True, False]])
    assert torch.equal(true, res.to_dense())
