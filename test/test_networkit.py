import sys
import os
import torch
import pytest
import networkit

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from baselines.networkit import sparse_tensor_to_networkit, networkit_partition

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