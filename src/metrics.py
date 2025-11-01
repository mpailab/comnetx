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

    def modularity_sparse(adjacency, assignments, gamma : float = 1.0, directed : bool = False) -> torch.Tensor:
        """
            Args:
                adjacency: SparseTensor or torch.sparse.Tensor or torch.Tensor[n_nodes, n_nodes]
                assignments: torch.Tensor [n_nodes] 
                gamma: float, optional (default = 1.0)
                directed: bool, optional (default = Fasle)
            Returns:
                modularity: float 
        """
        A = adjacency.float()
        c = assignments
        delta = (c.unsqueeze(0) == c.unsqueeze(1)).float()

        def sum_along_dim(A, dim):
            if isinstance(A, SparseTensor):
                return A.sum(dim=dim)
            elif A.is_sparse:
                return torch.sparse.sum(A, dim=dim).to_dense()
            else:
                return A.sum(dim=dim)

        def to_dense(A):
            if isinstance(A, SparseTensor):
                return A.to_dense()
            elif A.is_sparse:
                return A.to_dense()
            else:
                return A

        if directed:
            k_out = sum_along_dim(A, dim=1) 
            k_in = sum_along_dim(A, dim=0)
            m = k_out.sum()
            A = to_dense(A)
            B = A - gamma * torch.outer(k_out, k_in) / m
            modularity = (B * delta).sum() / m
        else:
            k = sum_along_dim(A, dim=1)
            m = A.sum() / 2
            A = to_dense(A)
            B = A - gamma * torch.outer(k, k) / (2 * m)
            modularity = (B * delta).sum() / (2 * m)    
        
        return modularity

    def modularity(adjacency, assignments, gamma : float = 1.0, directed : bool = False) -> float:
        """
        Args:
            adjacency: SparseTensor or torch.sparse.Tensor or tf.sparse.SparseTensor [n_nodes, n_nodes]
            assignments: torch.Tensor or torch.Tensor or tf.Tensor [n_nodes, n_clusters] or torch.Tensor [n_nodes] 
            gamma: float, optional (default = 1.0)
            directed: bool, optional (default = Fasle)
        Returns:
            modularity: float 
        """
        if (isinstance(adjacency, SparseTensor) and isinstance(assignments, torch.Tensor)
            and assignments.dim()==2):
            # degrees = adjacency.sum(dim=1)
            # m = degrees.sum()
            # if not directed:
            #     m = m / 2
            # inv_2m = 1.0 / (2 * m)
            # degrees.view(-1, 1)
            # a_s = adjacency.matmul(assignments)
            # graph_pooled = torch.matmul(a_s.t(), assignments)
            # s_d = torch.matmul(assignments.t(), degrees)
            # normalizer = torch.matmul(s_d, s_d.t()) * inv_2m * gamma
            # modularity = (graph_pooled.diag().sum() - normalizer) * inv_2m
            # # modularity = torch.trace(graph_pooled - normalizer) * inv_2m
            
            if directed:
                k_out = adjacency.sum(dim=1)
                k_in = adjacency.sum(dim=0)
                m = k_out.sum()
                inv_m = 1.0 / m

                a_s = adjacency.matmul(assignments)
                graph_pooled = torch.matmul(a_s.t(), assignments)

                s_d_out = torch.matmul(assignments.t(), k_out)
                s_d_in = torch.matmul(assignments.t(), k_in)
                normalizer = gamma * torch.matmul(s_d_out, s_d_in.t()) * inv_m

                modularity = (graph_pooled.diag().sum() - normalizer.sum()) * inv_m

            else:
                # directed
                degrees = adjacency.sum(dim=1)
                m = degrees.sum() / 2
                inv_2m = 1.0 / (2 * m)

                a_s = adjacency.matmul(assignments)
                graph_pooled = torch.matmul(a_s.t(), assignments)

                s_d = torch.matmul(assignments.t(), degrees)
                normalizer = gamma * torch.matmul(s_d, s_d.t()) * inv_2m

                modularity = (graph_pooled.diag().sum() - normalizer.sum()) * inv_2m

            return modularity.item()
        

        elif (isinstance(adjacency, torch.Tensor) and isinstance(assignments, torch.Tensor)
              and assignments.dim()==2):
            # degrees = torch.sparse.sum(adjacency, dim=1).to_dense().view(-1, 1)
            # m = degrees.sum()
            # if not directed:
            #     m = m / 2
            # inv_2m = 1.0 / (2 * m)
            # a_s = torch.sparse.mm(adjacency, assignments)
            # graph_pooled = torch.matmul(a_s.t(), assignments)
            # s_d = torch.matmul(assignments.t(), degrees)
            # normalizer = torch.matmul(s_d, s_d.t()) * inv_2m * gamma
            # modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) * inv_2m

            
            if directed:
                k_out = adjacency.sum(dim=1)
                k_in = adjacency.sum(dim=0)
                m = k_out.sum()
                inv_m = 1.0 / m

                a_s = torch.sparse.mm(adjacency, assignments)
                graph_pooled = torch.matmul(a_s.t(), assignments)

                s_d_out = torch.matmul(assignments.t(), k_out)
                s_d_in = torch.matmul(assignments.t(), k_in)
                normalizer = gamma * torch.matmul(s_d_out, s_d_in.t()) * inv_m

                modularity = (graph_pooled.diag().sum() - normalizer.sum()) * inv_m

            else:
                # directed
                degrees = adjacency.sum(dim=1)
                m = degrees.sum() / 2
                inv_2m = 1.0 / (2 * m)

                a_s = torch.sparse.mm(adjacency, assignments)
                graph_pooled = torch.matmul(a_s.t(), assignments)

                s_d = torch.matmul(assignments.t(), degrees)
                normalizer = gamma * torch.matmul(s_d, s_d.t()) * inv_2m

                modularity = (graph_pooled.diag().sum() - normalizer.sum()) * inv_2m


            return modularity.item()
        
        elif assignments.dim()==1:
            A = adjacency.float()
            c = assignments
            delta = (c.unsqueeze(0) == c.unsqueeze(1)).float()

            def sum_along_dim(A, dim):
                if isinstance(A, SparseTensor):
                    return A.sum(dim=dim)
                elif A.is_sparse:
                    return torch.sparse.sum(A, dim=dim).to_dense()
                else:
                    return A.sum(dim=dim)

            def to_dense(A):
                if isinstance(A, SparseTensor):
                    return A.to_dense()
                elif A.is_sparse:
                    return A.to_dense()
                else:
                    return A

            if directed:
                k_out = sum_along_dim(A, dim=1) 
                k_in = sum_along_dim(A, dim=0)
                m = k_out.sum()
                A = to_dense(A)
                B = A - gamma * torch.outer(k_out, k_in) / m
                modularity = (B * delta).sum() / m
            else:
                k = sum_along_dim(A, dim=1)
                m = A.sum() / 2
                A = to_dense(A)
                B = A - gamma * torch.outer(k, k) / (2 * m)
                modularity = (B * delta).sum() / (2 * m)    
            
            return modularity

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

    def create_dense_community(communities, n, L=0):
        """
        Args:
            communities: torch.Tensor [l, n]
            n: int
            L: int, optional (default=0)
                
        Returns:
            community_matrix: torch.Tensor [n, n]
        """
        communities = communities[L,:].long()
        nodes = torch.tensor(range(n)).long()

        # print(len(communities), len(nodes))

        community_matrix = torch.zeros(n, n, dtype=torch.int32)
        community_matrix[communities, nodes] = 1
        
        return community_matrix