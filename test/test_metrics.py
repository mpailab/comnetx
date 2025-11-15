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


# A_dense = torch.tensor([
#     [0, 1, 0, 0],
#     [1, 0, 1, 0],
#     [0, 1, 0, 1],
#     [0, 0, 1, 0]
# ], dtype=torch.float32)

# communities = torch.tensor([[0, 0, 2, 2]])
# print(communities[0])

# A = SparseTensor.from_dense(A_dense) 
# Q_my = Metrics.modularity(A, communities[0].float(), directed=True)
# print(f"Модулярность: {type(Q_my)}")

# C_dense = torch.tensor(
#         [[1, 0, 1, 0],
#         [0, 0, 0, 0],
#         [0, 1, 0, 1],
#         [0, 0, 0, 0]], dtype=torch.int32)

import sys
import os
import json
import random
import importlib

from datasets import INFO
from launcher import dynamic_launch

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from metrics import Metrics 

# Загружаем данные один раз
with open(os.path.join(INFO, "modularity.json")) as _:
    true_mod_db = json.load(_)
with open(os.path.join(INFO, "all.json")) as _:
    info = json.load(_)

datasets = ['contact', 'contact', 'contact', 'contact']
print("Number of datasets:", len(datasets))
print("Dataset (is_directed) : Modularity (True modularity)")

mx1 = 0
mx2 = 0
mx3 = 0
mx4 = 0

for i in range(1000):
    # Ключевой момент: перезагружаем модули для чистого состояния
    if 'launcher' in sys.modules:
        importlib.reload(sys.modules['launcher'])
    if 'optimizer' in sys.modules:
        importlib.reload(sys.modules['optimizer'])
    if 'datasets' in sys.modules:
        importlib.reload(sys.modules['datasets'])
    
    # Импортируем заново
    from launcher import dynamic_launch
    
    # Уникальный сид для каждого запуска
    random_seed = random.randint(0, 1000000)
    
    # Принудительно запускаем сборщик мусора
    import gc
    gc.collect()
    
    results = dynamic_launch(datasets[0], 1, "leidenalg", mode="raw", verbose=0)
    mod = results[-1]['modularity']
    true_mod = true_mod_db['sociopatterns-hypertext']
    is_directed = info['sociopatterns-hypertext']["d"]
    # print(f"Run {i+1}: {mod:.4f} (true - {true_mod:.4f}, seed: {random_seed})")

    if mod > mx1:
        mx1 = mod

    # results = dynamic_launch(datasets[1], 1, "leidenalg", mode="raw", verbose=0)
    # mod = results[-1]['modularity']
    # true_mod = true_mod_db['radoslaw_email']
    # is_directed = info['radoslaw_email']["d"]
    # # print(f"Run {i+1}: {mod:.4f} (true - {true_mod:.4f}, seed: {random_seed})")

    # if mod > mx2:
    #     mx2 = mod

    # results = dynamic_launch(datasets[2], 1, "leidenalg", mode="raw", verbose=0)
    # mod = results[-1]['modularity']
    # true_mod = true_mod_db['radoslaw_email']
    # is_directed = info['radoslaw_email']["d"]
    # # print(f"Run {i+1}: {mod:.4f} (true - {true_mod:.4f}, seed: {random_seed})")

    # if mod > mx3:
    #     mx3 = mod

    # results = dynamic_launch(datasets[3], 1, "leidenalg", mode="raw", verbose=0)
    # mod = results[-1]['modularity']
    # true_mod = true_mod_db['radoslaw_email']
    # is_directed = info['radoslaw_email']["d"]
    # # print(f"Run {i+1}: {mod:.4f} (true - {true_mod:.4f}, seed: {random_seed})")

    # if mod > mx4:
    #     mx4 = mod

print(mx1, mx2, mx3, mx4)

# import torch

# # ---------------------------
# # 1. Задаём структуру графа
# # ---------------------------

# def modularity_paper(adjacency, assignments, gamma=1.0):
#     """
#     Directed modularity exactly as defined in the paper:

#         Q = 1/m * sum_c ( e_c - gamma * (Kc_in * Kc_out)/m )

#     Args:
#         adjacency: torch.sparse.Tensor, shape [n,n]
#         assignments: LongTensor of shape [n]
#         gamma: resolution parameter
#     """

#     adjacency = adjacency.coalesce()
#     row, col = adjacency.indices()
#     weight = adjacency.values()
#     n = adjacency.size(0)

#     # total edge weight
#     m = weight.sum()

#     # compute k_out[i] = sum_j w_ij
#     k_out = torch.sparse.sum(adjacency, dim=1).to_dense()  # shape [n]

#     # compute k_in[j] = sum_i w_ij
#     k_in = torch.sparse.sum(adjacency, dim=0).to_dense()   # shape [n]

#     # number of communities
#     communities = assignments.unique()

#     Q = 0.0

#     for c in communities:

#         # mask of nodes belonging to community c
#         mask_c = (assignments == c)

#         # ---------------------------------------
#         # e_c = sum of weights inside community c
#         # w_ij where i in c AND j in c
#         # ---------------------------------------
#         inside = mask_c[row] & mask_c[col]
#         e_c = weight[inside].sum()

#         # ---------------------------------------
#         # Kc_out = sum_{i in c, j in V} w_ij
#         # Kc_in  = sum_{i in V, j in c} w_ij
#         # ---------------------------------------
#         Kc_out = k_out[mask_c].sum()
#         Kc_in  = k_in[mask_c].sum()

#         Q += e_c - gamma * (Kc_in * Kc_out) / m

#     return (Q / m).item()

# # Индексы ребер (row = from, col = to)
# row = torch.tensor([0, 1, 2, 2, 0], dtype=torch.long)
# col = torch.tensor([1, 2, 0, 3, 3], dtype=torch.long)

# # Веса рёбер
# values = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

# # Количество узлов
# n_nodes = 4

# # ---------------------------
# # 2. Создаём разреженную матрицу adjacency
# # ---------------------------

# adjacency = torch.sparse_coo_tensor(
#     indices=torch.stack([row, col]),
#     values=values,
#     size=(n_nodes, n_nodes)
# ).coalesce()

# # ---------------------------
# # 3. Задаём разметку сообществ
# # ---------------------------

# assignments = torch.tensor([0, 0, 1, 1], dtype=torch.long)

# # ---------------------------
# # 4. Запускаем твою функцию
# # ---------------------------

# mod = Metrics.modularity(adjacency, assignments, directed=True)
# print("\nModularity =", mod)

# mod = modularity_paper(adjacency, assignments)
# print("\nModularity =", mod)
