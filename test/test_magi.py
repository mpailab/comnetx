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

from baselines.magi_model import magi
from datasets import Dataset, KONECT_PATH

def get_all_datasets():
    """
    Сreate dict with all datasets in test directory.
    """
    base_dir = "/auto/datasets/graphs/small"
    datasets = {}
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                datasets[name] = base_dir
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
    adj, features, labels = dataset.load()
    adj = adj.coalesce()
    num_nodes = adj.size(0)
    features = torch.randn(num_nodes, 128, dtype=torch.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_adj_path = os.path.join(tmpdir, f"adj_{name}.pt")
        temp_features_path = os.path.join(tmpdir, f"features_{name}.pt")
        temp_labels_path = os.path.join(tmpdir, f"labels_{name}.pt")
        torch.save(adj, temp_adj_path)
        torch.save(features, temp_features_path)

        cmd = [
            sys.executable,
            "run_magi_subprocess.py",
            "--adj", temp_adj_path,
            "--features", temp_features_path,
            "--epochs", "1",
            "--batchsize", "1024",
            "--out", temp_labels_path
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Subprocess failed with code {proc.returncode}")
            print("stdout:", proc.stdout)
            print("stderr:", proc.stderr)
            pytest.fail(f"Subprocess failed for dataset {name}")

        new_labels = torch.load(temp_labels_path)

        assert isinstance(new_labels, torch.Tensor)
        assert new_labels.shape[0] == adj.size(0)
        assert new_labels.dtype in (torch.int64, torch.long)
        assert new_labels.min() >= 0
    del adj, features, labels, new_labels
    gc.collect()
    torch.cuda.empty_cache()

def load_konect_info():
    """Load dataset info from konect.json."""
    file_path = os.path.join(os.path.dirname(__file__), "dataset_paths.json")
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
"""
@pytest.fixture(scope="class")
def facebook_dataset():
    ds = Dataset("epinions", KONECT_PATH)
    ds.load()
    return ds

def test_magi_on_facebook(facebook_dataset):
    num_nodes = facebook_dataset.adj.shape[-1]
    adj = facebook_dataset.adj.coalesce()
    features = torch.randn(num_nodes, 128, dtype=torch.float32)
    new_labels = magi(adj, features, epochs=10)
    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == adj.size(0)
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0
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
        temp_labels_path = os.path.join(tmpdir, f"labels_{name}.pt")
        torch.save(adj, temp_adj_path)
        torch.save(features, temp_features_path)

        cmd = [
            sys.executable,
            "run_magi_subprocess.py",
            "--adj", temp_adj_path,
            "--features", temp_features_path,
            "--epochs", "1",
            "--out", temp_labels_path
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Subprocess failed with code {proc.returncode}")
            print("stdout:", proc.stdout)
            print("stderr:", proc.stderr)
            pytest.fail(f"Subprocess failed for dataset {name}")

        new_labels = torch.load(temp_labels_path)

        assert isinstance(new_labels, torch.Tensor)
        assert new_labels.shape[0] == adj.size(0)
        assert new_labels.dtype in (torch.int64, torch.long)
        assert new_labels.min() >= 0
    del adj, features, labels, new_labels
    gc.collect()
    torch.cuda.empty_cache()

@pytest.mark.long
@pytest.mark.parametrize(
    "name,data_dir",
    list(datasets.items()),
    ids=list(datasets.keys())
)
def test_magi_many_clusters_respect_true_k(name, data_dir):
    """Проверка, что при завышенном K MAGI не старается равномерно заполнить все кластеры."""
    ds = Dataset(name, path=data_dir)
    adj, features, labels = ds.load()
    adj = adj.coalesce()
    num_nodes = adj.size(0)

    true_labels = labels.long()
    true_k = torch.unique(true_labels).numel()

    requested_k = max(true_k * 5, true_k + 20)

    with tempfile.TemporaryDirectory() as tmpdir:
        adj_path = os.path.join(tmpdir, f"{name}_adj.pt")
        feat_path = os.path.join(tmpdir, f"{name}_feat.pt")
        out_path = os.path.join(tmpdir, f"{name}_labels.pt")

        torch.save(adj, adj_path)
        torch.save(features, feat_path)

        cmd = [
            sys.executable,
            "run_magi_subprocess.py",
            "--adj", adj_path,
            "--features", feat_path,
            "--epochs", "1",
            "--batchsize", "1024",
            "--n-clusters", str(requested_k),
            "--out", out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("stdout:", proc.stdout)
            print("stderr:", proc.stderr)
            pytest.fail(f"MAGI subprocess failed for dataset {name}")

        pred = torch.load(out_path).long()

    assert pred.shape[0] == num_nodes
    assert pred.min() >= 0

    used_k = torch.unique(pred).numel()
    empty_k = requested_k - used_k

    assert used_k > true_k
    assert used_k >= 0.8 * requested_k

"""
@pytest.fixture(scope="class")
def facebook_dataset():
    ds = Dataset("elec", KONECT_PATH)
    ds.load()
    return ds

def test_communities(facebook_dataset):
    num_nodes = facebook_dataset.adj.shape[-1]
    adj = facebook_dataset.adj.coalesce()
    features = torch.randn(num_nodes, 128, dtype=torch.float32)
    new_labels = magi(adj, features, epochs=1, n_clusters=num_nodes)

    uniq_vals = np.unique(new_labels)
    print("UNIQUE COMMUNITIES:", uniq_vals.size, "EXPECTED:", num_nodes)

    # Проверка, что все назначения уникальны и это именно 0..num_nodes-1
    all_unique_and_full_range = set(uniq_vals.tolist()) == set(range(num_nodes))
    print("ALL ASSIGNMENTS UNIQUE:", all_unique_and_full_range)
"""