import torch
import tensorflow as tf
from torch_sparse import SparseTensor
import time
import sys
sys.path.append("src")  

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

# start_time = time.time()
# mod_value = Metrics.modularity_dmon_slow(adjacency_tf, assignments_tf)
# compute_time = time.time() - start_time
# print(f"tf slow method: {compute_time:.6f}, Modularity: {mod_value.numpy():.4f}")

# start_time = time.time()
# mod_value = Metrics.modularity_dmon_fast(adjacency_tf, assignments_tf)
# compute_time = time.time() - start_time
# print(f"tf fast method: {compute_time:.6f}, Modularity: {mod_value.numpy():.4f}")

# start_time = time.time()
# mod_value = Metrics.modularity_dmon_slow_torch_copy(adjacency_torch, assignments_torch)
# compute_time = time.time() - start_time
# print(f"torch_copy slow method: {compute_time:.6f}, Modularity: {mod_value:.4f}")

# start_time = time.time()
# mod_value = Metrics.modularity_dmon_slow_torch(adjacency_torch, assignments_torch)
# compute_time = time.time() - start_time
# print(f"torch slow method: {compute_time:.6f}, Modularity: {mod_value:.4f}")

# start_time = time.time()
# mod_value = Metrics.modularity_dmon_fast_torch(adjacency_torch, assignments_torch)
# compute_time = time.time() - start_time
# print(f"torch fast method: {compute_time:.6f}, Modularity: {mod_value:.4f}")

SIZE = 50
time_arr = torch.zeros(5,SIZE)

for i in range(SIZE):
    start_time = time.time()
    mod_value = Metrics.modularity_dmon_slow(adjacency_tf, assignments_tf)
    time_arr[0,i] = time.time() - start_time

    start_time = time.time()
    mod_value = Metrics.modularity_dmon_fast(adjacency_tf, assignments_tf)
    time_arr[1,i] = time.time() - start_time

    start_time = time.time()
    mod_value = Metrics.modularity_dmon_slow_torch_copy(adjacency_torch, assignments_torch)
    time_arr[2,i] = time.time() - start_time

    start_time = time.time()
    mod_value = Metrics.modularity_dmon_slow_torch(adjacency_torch, assignments_torch)
    time_arr[3,i] = time.time() - start_time

    start_time = time.time()
    mod_value = Metrics.modularity_dmon_fast_torch(adjacency_torch, assignments_torch)
    time_arr[4,i] = time.time() - start_time

# print(time_arr)
print(torch.mean(time_arr, dim=1))


"""
average results:
    tf slow method: 0.095443, Modularity: 0.0413
    tf fast method: 0.027619, Modularity: 0.0284
    torch_copy slow method: 0.168021, Modularity: 0.1615
    torch slow method: 0.156775, Modularity: 0.1622
    torch fast method: 0.089933, Modularity: 0.0908
"""