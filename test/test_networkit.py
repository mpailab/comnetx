import sys
import os
import torch
import pytest
import networkit

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from baselines.network import sparse_tensor_to_networkit, networkit_partition
from datasets import Dataset, KONECT_PATH
from metrics import Metrics

@pytest.mark.short
def test_edge_list_correctness():
    """Проверка корректности списка ребер"""
    indices = torch.tensor([[0, 2], [1, 0]], dtype=torch.long)
    values = torch.tensor([1.0, 1.0])
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (3, 3))
    
    graph = sparse_tensor_to_networkit(sparse_tensor)
    edges = []
    graph.forEdges(lambda u, v, w, id: edges.append((u, v)))
    normalized_edges = set((min(u, v), max(u, v)) for u, v in edges)
    expected_edges = {(0, 1), (0, 2)} 
    assert normalized_edges == expected_edges


def test_single_community():
    adj_matrix = torch.tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=torch.float32)
    
    coms = networkit_partition(adj_matrix.to_sparse())
    
    # В полном графе все вершины должны быть в одном сообществе
    assert len(set(coms.tolist())) == 1
    assert all(m == coms[0] for m in coms)

def test_partition_2():
    adj_matrix = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.float32)
    coms = networkit_partition(adj_matrix.to_sparse())
    assert len(coms) == 4
    assert len(set(coms.tolist())) == 2

@pytest.mark.long
def test_networkit_on_cora():
    data_dir = "/auto/datasets/graphs/small"
    dataset = Dataset("cora", path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = networkit_partition(adj)

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0
    del adj, features, labels, new_labels

@pytest.mark.long    
def test_networkit_on_citeseer():
    data_dir = "/auto/datasets/graphs/small"
    dataset = Dataset("citeseer", path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    new_labels = networkit_partition(adj)

    assert isinstance(new_labels, torch.Tensor)
    assert new_labels.shape[0] == labels.shape[0]
    assert new_labels.dtype in (torch.int64, torch.long)
    assert new_labels.min() >= 0
    del adj, features, labels, new_labels
    
@pytest.mark.long
def test_networkit_single_konect_dataset():
    dataset = Dataset("wiki_talk_ht", KONECT_PATH)
    adj, features, labels = dataset.load(tensor_type="coo")
    adj = adj.coalesce()

    coms = networkit_partition(adj)
    mod = Metrics.modularity(adj, coms.float(), 1.0, directed = True)
    print(f"Final modularity: {mod:.2}")

    assert isinstance(coms, torch.Tensor)
    assert coms.dtype in (torch.int64, torch.long)
    assert coms.min() >= 0

    del adj, features, labels, coms