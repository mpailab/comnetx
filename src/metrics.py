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

    # def modularity_dmon_slow_tf(adjacency, assignments):
    #     assignments_pool = assignments / tf.math.reduce_sum(assignments, axis=0)
    #     degrees = tf.sparse.reduce_sum(adjacency, axis=0)
    #     degrees = tf.reshape(degrees, (-1, 1))
    #     m = tf.math.reduce_sum(degrees)  

    #     graph_pooled = tf.transpose(tf.sparse.sparse_dense_matmul(adjacency, assignments))
    #     graph_pooled = tf.matmul(graph_pooled, assignments)
        
    #     ca = tf.matmul(assignments, degrees, transpose_a=True)
    #     cb = tf.matmul(degrees, assignments, transpose_a=True)
    #     normalizer = tf.matmul(ca, cb) / 2 / m
        
    #     modularity = tf.linalg.trace(graph_pooled - normalizer) / 2 / m
        
    #     return modularity
    
    # def modularity_dmon_slow_torch_copy(adjacency: SparseTensor, assignments: torch.Tensor) -> float:
    #     assignments_pool = assignments / assignments.sum(dim=0, keepdim=True)
    #     degrees = adjacency.sum(dim=1)
    #     degrees = degrees.view(-1, 1)
    #     m = degrees.sum()

    #     graph_pooled = adjacency.matmul(assignments).t()
    #     graph_pooled = torch.matmul(graph_pooled, assignments)
        
    #     ca = torch.matmul(assignments.t(), degrees)
    #     cb = torch.matmul(degrees.t(), assignments)
    #     normalizer = torch.matmul(ca, cb) / (2 * m)
        
    #     modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) / (2 * m)
        
    #     return modularity.item()
    
    # def modularity_dmon_slow_torch(adjacency: SparseTensor, assignments: torch.Tensor) -> float:
    #     assignments_pool = assignments / assignments.sum(dim=0, keepdim=True)
    #     degrees = adjacency.sum(dim=1)
    #     degrees = degrees.view(-1, 1)
    #     m = degrees.sum()

    #     graph_pooled = torch.matmul(assignments.t(), adjacency.matmul(assignments))
        
    #     ca = torch.matmul(assignments.t(), degrees)
    #     cb = torch.matmul(degrees.t(), assignments)
    #     normalizer = torch.matmul(ca, cb) / (2 * m)
        
    #     modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) / (2 * m)
        
    #     return modularity.item()

    def modularity_dmon_tf(adjacency: tf.sparse.SparseTensor, assignments: tf.Tensor) -> float:
        """
        Args:
            adjacency: tf.SparseTensor [n_nodes, n_nodes]
            assignments: tf.Tensor [n_nodes, n_clusters]
            
        Returns:
            modularity: float 
        """
        degrees = tf.sparse.reduce_sum(adjacency, axis=0)
        m = tf.reduce_sum(degrees)
        inv_2m = 1.0 / (2 * m) 
        degrees = tf.reshape(degrees, (-1, 1))
        
        a_s = tf.sparse.sparse_dense_matmul(adjacency, assignments)
        graph_pooled = tf.matmul(a_s, assignments, transpose_a=True)
        
        s_d = tf.matmul(assignments, degrees, transpose_a=True)
        normalizer = tf.matmul(s_d, s_d, transpose_b=True) * inv_2m
        
        modularity = tf.linalg.trace(graph_pooled - normalizer) * inv_2m
        
        return modularity


    def modularity_dmon_torch(adjacency: SparseTensor, assignments: torch.Tensor) -> float:
        """
        Args:
            adjacency: torch.SparseTensor [n_nodes, n_nodes]
            assignments: torch.Tensor [n_nodes, n_clusters]
            
        Returns:
            modularity: float 
        """
        degrees = adjacency.sum(dim=1).view(-1, 1)
        m = degrees.sum()
        inv_2m = 1.0 / (2 * m)
        
        a_s = adjacency.matmul(assignments)
        graph_pooled = torch.matmul(a_s.t(), assignments)
        
        s_d = torch.matmul(assignments.t(), degrees)
        normalizer = torch.matmul(s_d, s_d.t()) * inv_2m
        
        modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) * inv_2m
        
        return modularity.item()

    def modularity(adjacency, assignments) -> float:
        """
        Args:
            adjacency: SparseTensor or torch.sparse.Tensor or tf.sparse.SparseTensor [n_nodes, n_nodes]
            assignments: torch.Tensor or torch.Tensor or tf.Tensor [n_nodes, n_clusters]
            
        Returns:
            modularity: float 
        """
        if isinstance(adjacency, SparseTensor) and isinstance(assignments, torch.Tensor):
            degrees = adjacency.sum(dim=1)
            m = degrees.sum()
            inv_2m = 1.0 / (2 * m)
            degrees.view(-1, 1)
            a_s = adjacency.matmul(assignments)
            graph_pooled = torch.matmul(a_s.t(), assignments)
            s_d = torch.matmul(assignments.t(), degrees)
            normalizer = torch.matmul(s_d, s_d.t()) * inv_2m
            modularity = (graph_pooled.diag().sum() - normalizer) * inv_2m
            # modularity = torch.trace(graph_pooled - normalizer) * inv_2m
            return modularity.item()
        elif isinstance(adjacency, torch.Tensor) and isinstance(assignments, torch.Tensor):
            degrees = torch.sparse.sum(adjacency, dim=1).to_dense().view(-1, 1)
            m = degrees.sum()
            inv_2m = 1.0 / (2 * m)
            a_s = torch.sparse.mm(adjacency, assignments)
            graph_pooled = torch.matmul(a_s.t(), assignments)
            s_d = torch.matmul(assignments.t(), degrees)
            normalizer = torch.matmul(s_d, s_d.t()) * inv_2m
            modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) * inv_2m
            return modularity.item()
        elif isinstance(adjacency, tf.sparse.SparseTensor) and isinstance(assignments, tf.Tensor):
            degrees = tf.sparse.reduce_sum(adjacency, axis=0)
            m = tf.reduce_sum(degrees)
            inv_2m = 1.0 / (2 * m) 
            degrees = tf.reshape(degrees, (-1, 1))
            a_s = tf.sparse.sparse_dense_matmul(adjacency, assignments)
            graph_pooled = tf.matmul(a_s, assignments, transpose_a=True)
            s_d = tf.matmul(assignments, degrees, transpose_a=True)
            normalizer = tf.matmul(s_d, s_d, transpose_b=True) * inv_2m
            modularity = tf.linalg.trace(graph_pooled - normalizer) * inv_2m
            return modularity.numpy()
        else:
            raise TypeError("Unsupported type")
