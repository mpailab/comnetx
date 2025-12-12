import sys
import torch, gc
import pytest
import os
import subprocess
import tempfile
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
KONECT_INFO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets-info"))

# from baselines.s2cag import main
from datasets import Dataset, KONECT_PATH
from baselines.s2cag import s2cag

def test_s2cag_synthetic_dataset():
    n = 30
    k = 4
    nodes_per_cluster = [15, 4, 5, 6]

    labels = []
    for c, size in enumerate(nodes_per_cluster):
        labels.extend([c] * size)
    labels = torch.tensor(labels)

    feature = torch.zeros(n, 5)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        feature[start:end, c*10:(c+1)*10] = 1.0
        start = end
    feature = feature + torch.randn_like(feature) * 0.15
    feature = feature / feature.norm(dim=1, keepdim=True)
    feature = abs(feature)

    adj = torch.zeros(n, n)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        idx = torch.arange(start, end)
        i, j = torch.meshgrid(idx, idx, indexing='ij')
        mask = torch.rand(size, size) < 0.4
        mask = torch.triu(mask, 1)
        adj[i[mask], j[mask]] = 1
        adj[j[mask], i[mask]] = 1
        start = end

    inter = torch.rand(n, n) < 0.003
    inter = inter & (torch.triu(torch.ones(n,n), 1) > 0)
    adj[inter] = 1
    adj.T[inter] = 1
    adj.fill_diagonal_(0)

    adj = adj.to_sparse_coo()
    # print(adj, feature)
    new_labels = s2cag(adj, feature, labels)
    # print(new_labels)
    assert new_labels.shape[0] == adj.size(0)
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0


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
                datasets[name] = base_dir
    return datasets

datasets = get_all_datasets()
@pytest.mark.long
@pytest.mark.parametrize(
    "name,data_dir",
    list(datasets.items()),
    ids=list(datasets.keys())
)
def test_s2cag_single_dataset(name, data_dir):
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load()
    adj = adj.coalesce()
    num_nodes = adj.size(0)
    features = torch.randn(num_nodes, 128, dtype=torch.float32)
    features = abs(features)
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_adj_path = os.path.join(tmpdir, f"adj_{name}.pt")
        temp_features_path = os.path.join(tmpdir, f"features_{name}.pt")
        temp_labels_path = os.path.join(tmpdir, f"labels_{name}.pt")
        torch.save(adj, temp_adj_path)
        torch.save(features, temp_features_path)
        torch.save(labels, temp_labels_path)

        cmd = [
            sys.executable,
            "run_s2cag_subprocess.py",
            "--adj", temp_adj_path,
            "--features", temp_features_path,
            "--labels", temp_labels_path,
            "--runs", "5",
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

# small (19)
# def load_konect_info():
#     """Load dataset info from all.json."""
#     file_path = os.path.join(os.path.dirname(__file__), "dataset_paths.json")
#     with open(file_path, "r", encoding="utf-8") as f:
#         info = json.load(f)
#     return info

def load_konect_info():
    """Load dataset info from all.json."""
    file_path = os.path.join(KONECT_INFO, "all.json")
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
def test_s2cag_konect_dataset(name):
    dataset = Dataset(name, path=KONECT_PATH)
    adj, features, labels = dataset.load()
    adj = adj.coalesce()
    
    new_values = adj.values().abs()
    adj = torch.sparse_coo_tensor(
        adj.indices(), new_values, adj.size()
    ).coalesce() #FIXME

    num_nodes = adj.size(0)
    features = torch.rand(num_nodes, 128, dtype=torch.float32) #FIXME
    labels = torch.randint(low=0, high=10, size=(num_nodes,))
    print("adj =", adj, sep='\n------------------\n')
    # print("num_nodes =", num_nodes, sep='\n------------------\n')
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_adj_path = os.path.join(tmpdir, f"adj_{name}.pt")
        temp_features_path = os.path.join(tmpdir, f"features_{name}.pt")
        temp_labels_path = os.path.join(tmpdir, f"labels_{name}.pt")
        torch.save(adj, temp_adj_path)
        torch.save(features, temp_features_path)
        torch.save(labels, temp_labels_path)

        cmd = [
            sys.executable,
            "run_s2cag_subprocess.py",
            "--adj", temp_adj_path,
            "--features", temp_features_path,
            "--labels", temp_labels_path,
            "--runs", "1",
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

def test_s2cag_single_konect_dataset():
    name = "com-lj"
    dataset = Dataset(name, KONECT_PATH)
    adj, features, labels = dataset.load(tensor_type="coo")
    adj = adj.coalesce()
    
    new_values = adj.values().abs()
    adj = torch.sparse_coo_tensor(
        adj.indices(), new_values, adj.size()
    ).coalesce() #FIXME
    
    num_nodes = adj.size(0)
    if labels is None:
        num_classes = 16
        labels = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    features = torch.rand(num_nodes, 128, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_adj_path = os.path.join(tmpdir, f"adj_{name}.pt")
        temp_features_path = os.path.join(tmpdir, f"features_{name}.pt")
        temp_labels_path = os.path.join(tmpdir, f"labels_{name}.pt")
        torch.save(adj, temp_adj_path)
        torch.save(features, temp_features_path)
        torch.save(labels, temp_labels_path)

        cmd = [
            sys.executable,
            "run_s2cag_subprocess.py",
            "--adj", temp_adj_path,
            "--features", temp_features_path,
            "--labels", temp_labels_path,
            "--runs", "5",
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