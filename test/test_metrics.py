import torch
import tensorflow as tf
from torch_sparse import SparseTensor
import time
import sys
sys.path.append("src")  

from metrics import Metrics


N, K = 10000, 3

# SIZE = 1
# time_arr = torch.zeros(5,SIZE)

# for i in range(SIZE):
#     features_tf = tf.random.normal((N, 10))
#     indices_tf = tf.random.uniform((200, 2), 0, N, dtype=tf.int64)
#     values_tf = tf.ones(200)
#     adjacency_tf = tf.sparse.SparseTensor(indices_tf, values_tf, dense_shape=(N, N))
#     mlp_output_tf = tf.random.normal((N, K))
#     assignments_tf = tf.nn.softmax(mlp_output_tf, axis=1)
#     print(assignments_tf)
#     indices_np = indices_tf.numpy()
#     values_np = values_tf.numpy()
#     adjacency_torch = SparseTensor(
#         row=torch.from_numpy(indices_np[:, 0]).long(),
#         col=torch.from_numpy(indices_np[:, 1]).long(),
#         value=torch.from_numpy(values_np).float(),
#         sparse_sizes=(N, N)
#     )
#     assignments_torch = torch.from_numpy(assignments_tf.numpy()).float()

#     start_time = time.time()
#     mod_value = Metrics.modularity_dmon_slow_tf(adjacency_tf, assignments_tf)
#     # print("mod. = ", mod_value)
#     time_arr[0,i] = time.time() - start_time

#     start_time = time.time()
#     mod_value = Metrics.modularity_dmon_tf(adjacency_tf, assignments_tf)
#     # print("mod. = ", mod_value)
#     time_arr[1,i] = time.time() - start_time

#     start_time = time.time()
#     mod_value = Metrics.modularity_dmon_slow_torch_copy(adjacency_torch, assignments_torch)
#     # print("mod. = ", mod_value)
#     time_arr[2,i] = time.time() - start_time

#     start_time = time.time()
#     mod_value = Metrics.modularity_dmon_slow_torch(adjacency_torch, assignments_torch)
#     # print("mod. = ", mod_value)
#     time_arr[3,i] = time.time() - start_time

#     start_time = time.time()
#     mod_value = Metrics.modularity_dmon_torch(adjacency_torch, assignments_torch)
#     # print("mod. = ", mod_value)
#     time_arr[4,i] = time.time() - start_time

# print(torch.mean(time_arr, dim=1))

SIZE = 10
time_arr = torch.zeros(2,SIZE)

for i in range(SIZE):
    features_tf = tf.random.normal((N, 10))
    indices_tf = tf.random.uniform((200, 2), 0, N, dtype=tf.int64)
    values_tf = tf.ones(200)
    adjacency_tf = tf.sparse.SparseTensor(indices_tf, values_tf, dense_shape=(N, N))
    mlp_output_tf = tf.random.normal((N, K))
    assignments_tf = tf.nn.softmax(mlp_output_tf, axis=1)
    # print(assignments_tf)
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
    mod_value = Metrics.modularity(adjacency_torch, assignments_torch)
    print("mod. = ", mod_value)
    time_arr[1,i] = time.time() - start_time

    start_time = time.time()
    mod_value = Metrics.modularity(adjacency_tf, assignments_tf)
    print("mod. = ", mod_value)
    time_arr[0,i] = time.time() - start_time

print(torch.mean(time_arr, dim=1))