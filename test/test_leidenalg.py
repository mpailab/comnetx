import sys
import os
import torch
import pytest
import leidenalg
import igraph as ig

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from baselines.leiden import sparse_tensor_to_igraph, leidenalg_partition
from datasets import Dataset, KONECT_PATH
from metrics import Metrics

@pytest.fixture(scope="class")
def facebook_dataset():
    ds = Dataset("facebook-wosn-links", KONECT_PATH)
    ds.load()
    return ds

@pytest.mark.short
def test_edge_list_correctness():
    """Проверка корректности списка ребер"""
    indices = torch.tensor([[0, 2], [1, 0]], dtype=torch.long)
    values = torch.tensor([1.0, 1.0])
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (3, 3))
    
    graph = sparse_tensor_to_igraph(sparse_tensor)
    edges = set(graph.get_edgelist())
    expected_edges = {(0, 1), (2, 0)}
    assert edges == expected_edges


def test_single_community():
    adj_matrix = torch.tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=torch.float32)
    
    coms = leidenalg_partition(adj_matrix)
    
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
    coms = leidenalg_partition(adj_matrix)
    assert len(coms) == 4
    assert len(set(coms.tolist())) == 2
    
@pytest.mark.long
def test_leidenalg_single_konect_dataset():
    dataset = Dataset("wiki_talk_ht", KONECT_PATH)
    adj, features, labels = dataset.load(tensor_type="coo")
    adj = adj.coalesce()

    coms = leidenalg_partition(adj)
    mod = Metrics.modularity(adj, coms.float(), 1.0, directed = True)
    print(f"Final modularity: {mod:.2}")

    assert isinstance(coms, torch.Tensor)
    assert coms.dtype in (torch.int64, torch.long)
    assert coms.min() >= 0

    del adj, features, labels, coms