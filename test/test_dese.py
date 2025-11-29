import os
import torch
import numpy as np
import sys
import torch, gc
import pytest
import json
import subprocess
import tempfile
import argparse
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
KONECT_INFO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets-info"))

from baselines.dese import main, train
from datasets import Dataset, KONECT_PATH

def main(adj, features, labels):
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--dataset', nargs='?', default='Computers', 
                        help='Choose a dataset from {Cora, Citeseer, Pubmed, Computers, Photo, CS and Physics}.')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate.')
    parser.add_argument('--height', type=int, default=2,
                        help='Height of the SE tree.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index. Default: -1, using CPU.')
    parser.add_argument('--decay_rate', type=int, default=None,
                        help='Decay rate of the number of clusters in each layer.')
    parser.add_argument('--num_clusters_layer', type=list, default=[10],
                        help='Number of clusters in each layer.')
    parser.add_argument('--layer_str', type=str, default='[10]',
                        help='Number of clusters in each layer.')
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--se_lamda', type=float, default=0.01,
                        help='Weight of se loss.')
    parser.add_argument('--lp_lamda', type=float, default=1,
                        help='Weight of lp loss.')
    parser.add_argument('--verbose', type=int, default=20,
                        help='evaluate every verbose epochs.')
    parser.add_argument('--activation', type=str, default='relu',
                        help='elu, relu, sigmoid, None')
    parser.add_argument('--k', type=int, default=2,
                        help='KNN')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--beta_f', type=float, default=0.2,
                        help='weight for adj_f')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save', type=bool, default=False,
                        help='Save model or not')
    parser.add_argument('--fig_network', type=bool, default=False,
                        help='Draw network or not')

    labels = labels.tolist()
    adj_sparse = adj.to_sparse()
    edge_index = adj_sparse.indices()

    neg_edge_index = []
    while len(neg_edge_index) < edge_index.shape[1]:
        u = torch.randint(0, adj.shape[0], (10000,))
        v = torch.randint(0, adj.shape[0], (10000,))
        mask = (u != v) & (adj[u,v] == 0)
        u, v = u[mask], v[mask]
        neg_edge_index.append(torch.stack([u,v]))
    neg_edge_index = torch.cat(neg_edge_index, dim=1)[:, :edge_index.shape[1]]

    class Dummy: pass
    dataset = Dummy()
    dataset.name = dataset
    dataset.adj = adj_sparse
    dataset.feature = features
    dataset.labels = labels
    dataset.degrees = adj.sum(1)
    dataset.neg_edge_index = neg_edge_index
    dataset.num_nodes = adj.shape[0]
    dataset.num_edges = int(adj.sum())
    dataset.num_features = features.shape[1]
    dataset.num_classes = len(set(labels))

    def print_statistic():
        print(f"=== {dataset.name} | {adj.shape[0]} nodes | {dataset.num_edges//2} edges | {dataset.num_classes} classes ===")
        print(f"Label dist: {dict(Counter(labels))}")
    dataset.print_statistic = print_statistic
    
    args = parser.parse_args([])
    best_cluster, out_label = train(dataset, args)
    print(type(out_label))

def test_dese_on_dataset():
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

    print(adj, feature)

    main(adj=adj, features=feature, labels=labels)

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
# @pytest.mark.long
# @pytest.mark.parametrize(
#     "name,data_dir",
#     list(datasets.items()),
#     ids=list(datasets.keys())
# )
# def test_dese_single_dataset(name, data_dir):
#     dataset = Dataset(name, path=data_dir)
#     adj, features, labels = dataset.load(tensor_type="dense")

#     _, new_labels = main(adj, features, labels)

#     assert isinstance(new_labels, torch.Tensor)
#     assert new_labels.shape[0] == labels.shape[0]
#     assert new_labels.dtype in (torch.int64, torch.long)
#     assert new_labels.min() >= 0

#     del adj, features, labels, new_labels

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
def test_dese_konect_dataset(name):
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
            "test/run_dese_subprocess.py",
            "--adj", temp_adj_path,
            "--features", temp_features_path,
            # "--epochs", "40",
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

# test_dese_on_dataset()