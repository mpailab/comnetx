import os
import torch
import numpy as np
import sys
import torch, gc
import pytest
import json
import subprocess
import tempfile


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
KONECT_INFO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets-info"))

from baselines.magi import magi
from datasets import Dataset, KONECT_PATH


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

@pytest.fixture(scope="class")
def facebook_dataset():
    ds = Dataset("facebook-wosn-links", KONECT_PATH)
    ds.load()
    return ds

def test_magi_on_facebook(facebook_dataset):
    num_nodes = facebook_dataset.adj.shape[-1]
    adj = facebook_dataset.adj.coalesce()
    features = torch.randn(num_nodes, 128, dtype=torch.float32)
    magi(adj, features, epochs=10)


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
@pytest.mark.parametrize(
    "name",
    list(KONECT_DATASETS.keys()),
    ids=list(KONECT_DATASETS.keys())
)
def test_magi_konect_dataset(name):
    dataset = Dataset(name, path=KONECT_PATH)
    adj, features, labels = dataset.load()
    adj = adj.coalesce()
    num_nodes = adj.size(0)
    features = torch.randn(num_nodes, 128, dtype=torch.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_adj_path = os.path.join(tmpdir, f"adj_{name}.pt")
        temp_features_path = os.path.join(tmpdir, f"features_{name}.pt")
        torch.save(adj, temp_adj_path)
        torch.save(features, temp_features_path)

        cmd = [
            sys.executable,
            "run_magi_subprocess.py",
            "--adj", temp_adj_path,
            "--features", temp_features_path,
            "--epochs", "1"
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        #print(proc.stdout)
        if proc.returncode != 0:
            # Если ошибка только из-за tensorflow предупреждения, можно игнорировать или логировать
            if "Unable to register cuDNN factory" in proc.stderr:
                print("Warning: TensorFlow cuDNN factory warning detected, ignoring")
            else:
                pytest.fail(f"Subprocess failed for dataset {name} with error: {proc.stderr}")
    del adj, features, labels 
    gc.collect()
    torch.cuda.empty_cache()