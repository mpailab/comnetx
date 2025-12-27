import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
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

from baselines.dmon import adapted_dmon
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
def test_dmon_single_dataset(name, data_dir):
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = adapted_dmon(adj, features, labels)

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0

    del adj, features, labels, new_labels

def load_konect_info():
    """Load dataset info from konect.json."""
    file_path = os.path.join(KONECT_INFO, "konect.json")
    with open(file_path, "r", encoding="utf-8") as f:
        info = json.load(f)
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

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    list(KONECT_DATASETS.keys()),
    ids=list(KONECT_DATASETS.keys())
)
def test_dmon_konect_dataset(name):
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
            "run_dmon_subprocess.py",
            "--adj", temp_adj_path,
            "--features", temp_features_path,
            "--epochs", "10",
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
        #print(proc.stdout)
        '''
        if proc.returncode != 0:
            if "Unable to register cuDNN factory" in proc.stderr:
                print("Warning: TensorFlow cuDNN factory warning detected, ignoring")
            else:
                pytest.fail(f"Subprocess failed for dataset {name} with error: {proc.stderr}")
        '''
    del adj, features, labels 
    gc.collect()
    torch.cuda.empty_cache()


def test_dmon_single_konect_dataset():
    dataset = Dataset("youtube-u-growth", KONECT_PATH)
    adj, features, labels = dataset.load(tensor_type="coo")
    adj = adj.coalesce()
    num_nodes = adj.size(0)
    if labels is None:
        num_classes = 16
        labels = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    features = torch.randn(num_nodes, 128, dtype=torch.float32)

    new_labels = adapted_dmon(adj, features)

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0

    del adj, features, labels, new_labels