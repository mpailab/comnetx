import sys
import os
import torch
import pytest
import json

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from baselines.rough_PRGPT import rough_prgpt, reorder_to_tensor
from datasets import Dataset, KONECT_PATH
KONECT_INFO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "konect-datasets-info"))

def load_konect_info():
    """Load dynamic and static dataset info from JSONs."""
    with open(os.path.join(KONECT_INFO, "dynamic.json")) as f:
        info = json.load(f)
    with open(os.path.join(KONECT_INFO, "static.json")) as f:
        info.update(json.load(f))
    return info

def get_all_konect_datasets():
    """Return a dict {dataset_name: Dataset object}."""
    info = load_konect_info()
    datasets = {}
    for name in info.keys():
        path = os.path.join(KONECT_PATH, name)
        if os.path.exists(path):
            datasets[name] = Dataset(name, KONECT_PATH)
    return datasets

KONECT_DATASETS = get_all_konect_datasets()

@pytest.mark.parametrize(
    "name,dataset",
    list(KONECT_DATASETS.items()),
    ids=list(KONECT_DATASETS.keys())
)
@pytest.mark.long
def test_rough_prgpt_on_konect(name, dataset):
    dataset.load()
    rough_prgpt(dataset.adj, refine="infomap")

@pytest.fixture(scope="class")
def facebook_dataset():
    ds = Dataset("facebook-wosn-links", KONECT_PATH)
    ds.load()
    return ds

def test_rough_prgpt_infomap_on_facebook(facebook_dataset):
    rough_prgpt(facebook_dataset.adj, refine="infomap")

def test_rough_prgpt_locale_on_facebook(facebook_dataset):
    rough_prgpt(facebook_dataset.adj, refine="locale")

@pytest.fixture(scope="class")
def chess_dataset():
    ds = Dataset("chess", KONECT_PATH)
    ds.load()
    return ds

def test_rough_prgpt_locale_on_chess(chess_dataset):
    rough_prgpt(chess_dataset.adj, refine="infomap")

@pytest.fixture(scope="class")
def convote_dataset():
    ds = Dataset("convote", KONECT_PATH)
    ds.load()
    return ds

def test_rough_prgpt_locale_on_convote(convote_dataset):
    rough_prgpt(convote_dataset.adj, refine="infomap")

@pytest.mark.short
def test_reorder():
    clus_res = [1, 1, 0]
    res = reorder_to_tensor(clus_res, 2, zero_base = True)
    true = torch.tensor([[False, False, True], [True, True, False]])
    assert torch.equal(true, res.to_dense())
