import torch
import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import pairwise_distances
import networkx as nx
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor

class Metrics:

    def __init__(self):
        pass

    def modularity_dmon_slow(adjacency, assignments):
        assignments_pool = assignments / tf.math.reduce_sum(assignments, axis=0)
        
        degrees = tf.sparse.reduce_sum(adjacency, axis=0)
        degrees = tf.reshape(degrees, (-1, 1))
        m = tf.math.reduce_sum(degrees)  
        
        graph_pooled = tf.transpose(tf.sparse.sparse_dense_matmul(adjacency, assignments))
        graph_pooled = tf.matmul(graph_pooled, assignments)
        
        ca = tf.matmul(assignments, degrees, transpose_a=True)
        cb = tf.matmul(degrees, assignments, transpose_a=True)
        normalizer = tf.matmul(ca, cb) / 2 / m
        
        modularity = tf.linalg.trace(graph_pooled - normalizer) / 2 / m
        
        return modularity
    
    def modularity_dmon_slow_torch(adjacency: SparseTensor, assignments: torch.Tensor) -> float:
        assignments_pool = assignments / assignments.sum(dim=0, keepdim=True)

        degrees = adjacency.sum(dim=1).view(-1, 1)
        m = degrees.sum()

        graph_pooled = torch.matmul(assignments.t(), adjacency.matmul(assignments))
        
        ca = torch.matmul(assignments.t(), degrees)
        cb = torch.matmul(degrees.t(), assignments)
        normalizer = torch.matmul(ca, cb) / (2 * m)
        
        modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) / (2 * m)
        
        return modularity.item()

    def modularity_dmon_fast(adjacency, assignments):
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

    def modularity_dmon_fast_torch(adjacency: SparseTensor, assignments: torch.Tensor) -> float:
        degrees = adjacency.sum(dim=1).view(-1, 1)
        m = degrees.sum()
        inv_2m = 1.0 / (2 * m)
        
        a_s = adjacency.matmul(assignments)
        graph_pooled = torch.matmul(a_s.t(), assignments)
        
        s_d = torch.matmul(assignments.t(), degrees)
        normalizer = torch.matmul(s_d, s_d.t()) * inv_2m
        
        modularity = (graph_pooled.diag().sum() - normalizer.diag().sum()) * inv_2m
        
        return modularity.item()

    def modularity_magi(adjacency, cluster):
        device = adjacency.device()
        row, col = adjacency.storage.row(), adjacency.storage.col()
        assignments = assignments.to(device)
        m = adjacency.storage.value().sum() / 2  

        same_cluster = (cluster[row] == cluster[col]).float()
        D = torch.zeros_like(cluster, dtype=torch.float).scatter_add_(0, row, adjacency.storage.value())
        
        L_c = (adjacency.storage.value() * same_cluster).sum()
        D_c = (D * same_cluster).sum()
        
        return (L_c - D_c**2 / (4 * m)) / m

# if __name__ == "__main__":
#     N, K = 1000000, 100

#     features = tf.random.normal((N, 10))
#     indices = tf.random.uniform((200, 2), 0, N, dtype=tf.int64)
#     values = tf.ones(200)
#     adjacency = tf.sparse.SparseTensor(indices, values, dense_shape=(N, N))
    
#     mlp_output = tf.random.normal((N, K))
#     assignments = tf.nn.softmax(mlp_output, axis=1)
    
#     start_time = tf.timestamp()
#     mod_value = Metrics.modularity_dmon_slow(adjacency, assignments)
#     compute_time = tf.timestamp() - start_time
#     print(f"\nComputation time: {compute_time.numpy():.6f} sec")
#     print(f"Modularity value: {mod_value.numpy():.4f}")

#     start_time = tf.timestamp()
#     mod_value = Metrics.modularity_dmon_fast(adjacency, assignments)
#     compute_time = tf.timestamp() - start_time
#     print(f"\nComputation time: {compute_time.numpy():.6f} sec")
#     print(f"Modularity value: {mod_value.numpy():.4f}")

if __name__ == "__main__":
    N, K = 1000000, 100

    features_tf = tf.random.normal((N, 10))
    indices_tf = tf.random.uniform((200, 2), 0, N, dtype=tf.int64)
    values_tf = tf.ones(200)
    adjacency_tf = tf.sparse.SparseTensor(indices_tf, values_tf, dense_shape=(N, N))
    mlp_output_tf = tf.random.normal((N, K))
    assignments_tf = tf.nn.softmax(mlp_output_tf, axis=1)

    indices_np = indices_tf.numpy()
    values_np = values_tf.numpy()

    adjacency_torch = SparseTensor(
        row=torch.from_numpy(indices_np[:, 0]).long(),
        col=torch.from_numpy(indices_np[:, 1]).long(),
        value=torch.from_numpy(values_np).float(),
        sparse_sizes=(N, N)
    )

    assignments_torch = torch.from_numpy(assignments_tf.numpy()).float()

    start_time = time.time()
    mod_value = Metrics.modularity_dmon_slow(adjacency_tf, assignments_tf)
    compute_time = time.time() - start_time
    print(f"tf slow method: {compute_time:.6f}, Modularity: {mod_value.numpy():.4f}")

    start_time = time.time()
    mod_value = Metrics.modularity_dmon_fast(adjacency_tf, assignments_tf)
    compute_time = time.time() - start_time
    print(f"tf fast method: {compute_time:.6f}, Modularity: {mod_value.numpy():.4f}")

    start_time = time.time()
    mod_value = Metrics.modularity_dmon_slow_torch(adjacency_torch, assignments_torch)
    compute_time = time.time() - start_time
    print(f"torch slow method: {compute_time:.6f}, Modularity: {mod_value:.4f}")

    start_time = time.time()
    mod_value = Metrics.modularity_dmon_fast_torch(adjacency_torch, assignments_torch)
    compute_time = time.time() - start_time
    print(f"torch fast method: {compute_time:.6f}, Modularity: {mod_value:.4f}")