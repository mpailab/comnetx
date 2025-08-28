import sys
import os
import torch
import pytest

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from baselines.rough_PRGPT import rough_prgpt, to_ind_tensor
from datasets import Dataset, KONECT_PATH

@pytest.fixture(scope="class")
def facebook_dataset():
    ds = Dataset("facebook-wosn-links", KONECT_PATH)
    ds.load()
    return ds

def test_rough_prgpt_infomap_on_facebook(facebook_dataset):
    rough_prgpt(facebook_dataset.adj, refine="infomap")

def test_rough_prgpt_locale_on_facebook(facebook_dataset):
    rough_prgpt(facebook_dataset.adj, refine="locale")

@pytest.mark.short
def test_to_ind_tensor():
    clus_res = [1, 1, 0]
    res = to_ind_tensor(clus_res, shift = 0)
    true = torch.tensor([[1, 1, 0], [0, 1, 2]])
    assert torch.equal(true, res)
