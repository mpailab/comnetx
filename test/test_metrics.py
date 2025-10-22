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

def modularity(adj_matrix: torch.Tensor, communities: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет модулярность графа по матрице смежности и вектору сообществ.

    Параметры:
        adj_matrix : torch.Tensor, shape [N, N]
            Матрица смежности (0/1 или весовая)
        communities : torch.Tensor, shape [N]
            Вектор меток сообществ (целые числа)

    Возвращает:
        torch.Tensor (скаляр) — значение модулярности
    """
    # Убедимся, что типы совпадают
    A = adj_matrix.float()
    c = communities
    n = A.size(0)

    # Сумма всех рёбер (делим на 2, так как граф неориентированный)
    m = A.sum() / 2

    # Степени вершин
    k = A.sum(dim=1)

    # Матрица δ(c_i, c_j)
    delta = (c.unsqueeze(0) == c.unsqueeze(1)).float()

    # Основное выражение
    B = A - torch.outer(k, k) / (2 * m)

    Q = (B * delta).sum() / (2 * m)
    return Q

A = torch.tensor([
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 1]
], dtype=torch.float32)

# Метки сообществ (0 и 1)
communities = torch.tensor([[0, 0, 2, 2]])
print(communities[0])

Q = modularity(A, communities[0])
print(f"Модулярность: {Q.item():.4f}")

communities = torch.tensor([[0, 0, 2, 2]])
dense_com = Metrics.create_dense_community(communities, 4).T
A = SparseTensor.from_dense(A) 

Q2 = Metrics.modularity(A, dense_com.float())
print(dense_com)
print(f"Модулярность: {Q2:.4f}")