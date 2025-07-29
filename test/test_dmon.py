import torch
import tensorflow as tf
from torch_sparse import SparseTensor
import time
import sys
sys.path.append("src")  

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from metrics import Metrics


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
mod_value = Metrics.modularity_dmon_slow_torch_copy(adjacency_torch, assignments_torch)
compute_time = time.time() - start_time
print(f"torch_copy slow method: {compute_time:.6f}, Modularity: {mod_value:.4f}")

start_time = time.time()
mod_value = Metrics.modularity_dmon_slow_torch(adjacency_torch, assignments_torch)
compute_time = time.time() - start_time
print(f"torch slow method: {compute_time:.6f}, Modularity: {mod_value:.4f}")

start_time = time.time()
mod_value = Metrics.modularity_dmon_fast_torch(adjacency_torch, assignments_torch)
compute_time = time.time() - start_time
print(f"torch fast method: {compute_time:.6f}, Modularity: {mod_value:.4f}")

"""
results:
    tf slow method: 0.095443, Modularity: 0.0024
    tf fast method: 0.027619, Modularity: 0.0024
    torch_copy slow method: 0.168021, Modularity: 0.0024
    torch slow method: 0.156775, Modularity: 0.0024
    torch fast method: 0.089933, Modularity: 0.0024

    tf slow method: 0.075344, Modularity: 0.0025
    tf fast method: 0.026871, Modularity: 0.0025
    torch_copy slow method: 0.168445, Modularity: 0.0025
    torch slow method: 0.164011, Modularity: 0.0025
    torch fast method: 0.092702, Modularity: 0.0025
"""