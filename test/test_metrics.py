import torch
import tensorflow as tf
from torch_sparse import SparseTensor
import time
import sys
sys.path.append("src")  

from metrics import Metrics


N, K = 100000, 10

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

# SIZE = 10
# time_arr = torch.zeros(3,SIZE)

# for i in range(SIZE):
#     features_tf = tf.random.normal((N, 10))
#     indices_tf = tf.random.uniform((200, 2), 0, N, dtype=tf.int64)
#     values_tf = tf.ones(200)
#     adjacency_tf = tf.sparse.SparseTensor(indices_tf, values_tf, dense_shape=(N, N))
#     mlp_output_tf = tf.random.normal((N, K))
#     assignments_tf = tf.nn.softmax(mlp_output_tf, axis=1)
#     # print(assignments_tf)
#     indices_np = indices_tf.numpy()
#     values_np = values_tf.numpy()
#     adjacency_torch_geom = SparseTensor(
#         row=torch.from_numpy(indices_np[:, 0]).long(),
#         col=torch.from_numpy(indices_np[:, 1]).long(),
#         value=torch.from_numpy(values_np).float(),
#         sparse_sizes=(N, N)
#     )
#     assignments_torch = torch.from_numpy(assignments_tf.numpy()).float()

#     indices_np = indices_tf.numpy().T
#     adjacency_torch = torch.sparse_coo_tensor(
#         indices=torch.from_numpy(indices_np).long(),
#         values=torch.from_numpy(values_np).float(),
#         size=(N, N)
#     ).coalesce()

#     # torch geom-version
#     start_time = time.time()
#     mod_value = Metrics.modularity(adjacency_torch_geom, assignments_torch)
#     print("mod. = ", mod_value)
#     time_arr[0,i] = time.time() - start_time

#     # torch version
#     start_time = time.time()
#     mod_value = Metrics.modularity(adjacency_torch, assignments_torch)
#     print("mod. = ", mod_value)
#     time_arr[1,i] = time.time() - start_time

#     # tensorflow version
#     start_time = time.time()
#     mod_value = Metrics.modularity(adjacency_tf, assignments_tf)
#     print("mod. = ", mod_value)
#     time_arr[2,i] = time.time() - start_time

# print(torch.mean(time_arr, dim=1))

def sum_along_dim(A, dim):
        """Суммирует вдоль измерения независимо от формата."""
        if isinstance(A, SparseTensor):
            return A.sum(dim=dim)
        elif A.is_sparse:
            return torch.sparse.sum(A, dim=dim).to_dense()
        else:
            return A.sum(dim=dim)

def to_dense(A):
        """Преобразует в плотную матрицу для финальных операций."""
        if isinstance(A, SparseTensor):
            return A.to_dense()
        elif A.is_sparse:
            return A.to_dense()
        else:
            return A

def modularity(adjacency, assignments, gamma : float = 1.0, directed : bool = False) -> torch.Tensor:
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


    if directed:
        k_out = sum_along_dim(A, dim=1) 
        k_in = sum_along_dim(A, dim=0)
        m = k_out.sum()
        A = to_dense(A)
        B = A - gamma * torch.outer(k_out, k_in) / m
        modularity = (B * delta).sum() / m
    else:
        # --- Неориентированный случай ---
        k = sum_along_dim(A, dim=1)
        m = A.sum() / 2
        A = to_dense(A)
        B = A - gamma * torch.outer(k, k) / (2*m)
        modularity = (B * delta).sum() / (2*m)    
    
    return modularity

A_dense = torch.tensor([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
], dtype=torch.float32)

communities = torch.tensor([[0, 0, 2, 2]])
print(communities[0])

A = SparseTensor.from_dense(A_dense) 
Q_my = Metrics.modularity(A, communities[0].float(), directed=False)
print(f"Модулярность: {Q_my}")


Q1 = modularity(A_dense, communities[0], gamma=1.0, directed=False)
print("Dense:", Q1.item())

# Torch sparse
A_sparse = A_dense.to_sparse()
Q2 = modularity(A_sparse, communities[0], gamma=1.0, directed=False)
print("torch.sparse:", Q2.item())

# SparseTensor (PyTorch Geometric)
A_st = SparseTensor.from_dense(A_dense)
Q3 = modularity(A_st, communities[0], gamma=1.0, directed=False)
print("SparseTensor:", Q3.item())