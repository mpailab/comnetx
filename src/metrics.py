import torch
import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import pairwise_distances
import networkx as nx
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor

# import tensorflow as tensor 
# import torch as tensor 

class Metrics:

    def __init__(self):
        pass

    # def modularity_dmon_tf(adjacency: tf.sparse.SparseTensor, assignments: tf.Tensor) -> float:
    #     """
    #     Args:
    #         adjacency: tf.SparseTensor [n_nodes, n_nodes]
    #         assignments: tf.Tensor [n_nodes, n_clusters]
            
    #     Returns:
    #         modularity: float 
    #     """
    #     degrees = tf.sparse.reduce_sum(adjacency, axis=0)
    #     m = tf.reduce_sum(degrees)
    #     inv_2m = 1.0 / (2 * m) 
    #     degrees = tf.reshape(degrees, (-1, 1))
        
    #     a_s = tf.sparse.sparse_dense_matmul(adjacency, assignments)
    #     graph_pooled = tf.matmul(a_s, assignments, transpose_a=True)
        
    #     s_d = tf.matmul(assignments, degrees, transpose_a=True)
    #     normalizer = tf.matmul(s_d, s_d, transpose_b=True) * inv_2m
        
    #     modularity = tf.linalg.trace(graph_pooled - normalizer) * inv_2m
        
    #     return modularity


    # def modularity_dmon_torch(adjacency: SparseTensor, assignments: torch.Tensor) -> float:
    #     """
    #     Args:
    #         adjacency: torch.SparseTensor [n_nodes, n_nodes]
    #         assignments: torch.Tensor [n_nodes, n_clusters]
            
    #     Returns:
    #         modularity: float 
    #     """
    #     degrees = adjacency.sum(dim=1).view(-1, 1)
    #     m = degrees.sum()
    #     inv_2m = 1.0 / (2 * m)
        
    #     a_s = adjacency.matmul(assignments)
    #     graph_pooled = torch.matmul(a_s.t(), assignments)
        
    #     s_d = torch.matmul(assignments.t(), degrees)
    #     normalizer = torch.matmul(s_d, s_d.t()) * inv_2m
        
    #     modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) * inv_2m
        
    #     return modularity.item()

    def modularity(adjacency, assignments, gamma : float = 1.0, directed : bool = False) -> float:
        """
        Args:
            adjacency: SparseTensor or torch.sparse.Tensor or tf.sparse.SparseTensor [n_nodes, n_nodes]
            assignments: torch.Tensor or tf.Tensor [n_nodes, n_clusters] or torch.Tensor [n_nodes] 
            gamma: float, optional (default = 1.0)
            directed: bool, optional (default = Fasle)
        Returns:
            modularity: float 
        """

        def create_sparse_community(community):
            """
            Args:
                community_matrix: torch.Tensor [n, n]
                                  example = [[1, 1, 0],
                                             [0, 0, 0],
                                             [0, 0, 1]]
            Returns:
                communities: torch.Tensor [n]
            """
            comm_idx, node_idx = torch.nonzero(community, as_tuple=True)

            communities = torch.zeros(community.shape[1], dtype=torch.long)
            
            communities[node_idx] = comm_idx

            return communities

        if isinstance(adjacency, SparseTensor) and isinstance(assignments, torch.Tensor) and assignments.dim()==2:
            degrees = adjacency.sum(dim=1)
            m = degrees.sum()
            if not directed:
                m = m / 2
            inv_2m = 1.0 / (2 * m)
            degrees.view(-1, 1)
            a_s = adjacency.matmul(assignments)
            graph_pooled = torch.matmul(a_s.t(), assignments)
            s_d = torch.matmul(assignments.t(), degrees)
            normalizer = torch.matmul(s_d, s_d.t()) * inv_2m * gamma
            modularity = (graph_pooled.diag().sum() - normalizer) * inv_2m
            # modularity = torch.trace(graph_pooled - normalizer) * inv_2m
            return modularity.item()
        elif isinstance(adjacency, torch.Tensor) and isinstance(assignments, torch.Tensor) and assignments.dim()==2:
            degrees = torch.sparse.sum(adjacency, dim=1).to_dense().view(-1, 1)
            m = degrees.sum()
            if not directed:
                m = m / 2
            inv_2m = 1.0 / (2 * m)
            a_s = torch.sparse.mm(adjacency, assignments)
            graph_pooled = torch.matmul(a_s.t(), assignments)
            s_d = torch.matmul(assignments.t(), degrees)
            normalizer = torch.matmul(s_d, s_d.t()) * inv_2m * gamma
            modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) * inv_2m
            return modularity.item()
        
        elif isinstance(assignments, torch.Tensor) and assignments.dim()==1:

            def sum_along_dim(A, dim):
                if isinstance(A, SparseTensor):
                    return A.sum(dim=dim)
                elif A.is_sparse:
                    return torch.sparse.sum(A, dim=dim)
                else:
                    return A.sum(dim=dim)

            if isinstance(adjacency, SparseTensor):
                row, col, weight = adjacency.coo()
            elif adjacency.is_sparse:
                adjacency = adjacency.coalesce()
                row, col = adjacency.indices()
                weight = adjacency.values()
            else:
                adjacency = adjacency.to_sparse()
                row, col = adjacency.indices()
                weight = adjacency.values()

            k_out = sum_along_dim(adjacency, dim=1).to_dense()
            k_in = sum_along_dim(adjacency, dim=0).to_dense()
            m = weight.sum()
            B = gamma * (k_out[row] * k_in[col]) / m
            # print(type(k_out), type(k_in), type(weight), type(row), type(col), type(B))

            same_comm = (assignments[row] == assignments[col])

            modularity = ((weight - B) * same_comm.float()).sum() / m  
            
            return modularity.item()

        elif isinstance(adjacency, tf.sparse.SparseTensor) and isinstance(assignments, tf.Tensor):
            degrees = tf.sparse.reduce_sum(adjacency, axis=0)
            m = tf.reduce_sum(degrees)
            if not directed:
                m = m / 2
            inv_2m = 1.0 / (2 * m) 
            degrees = tf.reshape(degrees, (-1, 1))
            a_s = tf.sparse.sparse_dense_matmul(adjacency, assignments)
            graph_pooled = tf.matmul(a_s, assignments, transpose_a=True)
            s_d = tf.matmul(assignments, degrees, transpose_a=True)
            normalizer = tf.matmul(s_d, s_d, transpose_b=True) * inv_2m * gamma
            modularity = tf.linalg.trace(graph_pooled - normalizer) * inv_2m
            return modularity.numpy()
        else:
            raise TypeError("Unsupported type")