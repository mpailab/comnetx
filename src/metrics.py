import torch
import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import pairwise_distances
from torch_sparse import SparseTensor

# import tensorflow as tensor 
# import torch as tensor 

class Metrics:

    def __init__(self):
        pass

    def modularity(adjacency, assignments, gamma : float = 1.0, directed : bool = False) -> float:
        """
        Args:
            adjacency: torch.sparse.Tensor [n_nodes, n_nodes]
            assignments: torch.Tensor [n_nodes] 
            gamma: float, optional (default = 1.0)
            directed: bool, optional (default = False)
        Returns:
            modularity: float 
        """

        adjacency = adjacency.coalesce()
        row, col = adjacency.indices()
        weight = adjacency.values()  

        d_out = torch.sparse.sum(adjacency, dim=1).to_dense()
        d_in = torch.sparse.sum(adjacency, dim=0).to_dense()
        m = torch.sum(weight)

        communities = torch.unique(assignments)
        modularity = 0.0
        
        for community in communities:
            c = community.item()
    
            # Фактический вес ребер внутри сообщества
            both_in_community = (assignments[row] == c) & (assignments[col] == c)
            actual_weight = torch.sum(weight[both_in_community])
            
            # Ожидаемый вес
            mask = (assignments == c)
            expected_weight = torch.sum(d_out[mask]) * torch.sum(d_in[mask]) / m
            
            modularity += (actual_weight - gamma * expected_weight)
        modularity /=  m

        return modularity