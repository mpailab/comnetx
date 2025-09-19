import subprocess
import sys
import os
import torch
import pytest
import json

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from baselines.rough_PRGPT import rough_prgpt, to_com_tensor
from datasets import Dataset, KONECT_PATH
KONECT_INFO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets-info"))

def load_konect_info():
    """Load dataset info from all.json."""
    file_path = os.path.join(KONECT_INFO, "all.json")
    with open(file_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info
"""
def load_konect_info():
    #Load dynamic and static dataset info from JSONs.
    with open(os.path.join(KONECT_INFO, "dynamic.json")) as f:
        info = json.load(f)
    with open(os.path.join(KONECT_INFO, "static.json")) as f:
        info.update(json.load(f))
    return info
"""

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

@pytest.mark.long
def test_run_prgpt_isolated():
    subset = list(KONECT_DATASETS.keys())[0:20]
    param = sys.argv[-1] if len(sys.argv) > 1 else None
    for key in subset:
        print(f"\nRunning subprocess for parameter: {key}\n")
        env = os.environ.copy()
        env["KONECT_PARAM"] = key
        cmd = [
            sys.executable, "-m", "pytest", __file__, 
            "-k", "test_rough_prgpt_on_konect", 
            "--tb=short",
            "--disable-warnings"
        ]
        proc = subprocess.run(cmd, env=env)
        torch.cuda.empty_cache()
        if proc.returncode != 0:
            raise RuntimeError(f"Test failed in subprocess for parameter {key}")


@pytest.mark.long
def test_rough_prgpt_on_konect():
    param = os.environ.get("KONECT_PARAM")
    if param and param in KONECT_DATASETS:
        dataset = KONECT_DATASETS[param]
        dataset.load()
        rough_prgpt(dataset.adj, refine="infomap")
        torch.cuda.empty_cache()
    else:
        pytest.skip("No KONECT_PARAM environment variable set")
"""
@pytest.mark.parametrize(
    "name,dataset",
    list(KONECT_DATASETS.items()),
    ids=list(KONECT_DATASETS.keys())
)

@pytest.mark.long
def test_rough_prgpt_on_konect(name, dataset):
    dataset.load()
    rough_prgpt(dataset.adj, refine="infomap")
"""

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
def test_to_com_tensor():
    clus_res = [1, 1, 0]
    origin_num_nodes = 4
    res = to_com_tensor(clus_res, origin_num_nodes, shift = 0)
    true = torch.tensor([1, 1, 0, 2])
    assert torch.equal(true, res)
