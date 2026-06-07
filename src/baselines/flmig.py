import sys,os
import time
import copy
import random
import math

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_PATH, "src")
FLMIG_root = os.path.join(PROJECT_PATH, "baselines", "FLMIG_algorithm", "FLMIG_algorithm")

for p in (SRC_PATH, FLMIG_root):
    if p not in sys.path:
        sys.path.insert(0, p)

import tempfile
import torch
import networkx as nx
from pathlib import Path
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from GraphTools import GraphTolls
from FLMIG import Fast_local_Move_IG

class MyFastLocalMoveIG(Fast_local_Move_IG):
    def __init__(self, Nb, Beta, path, initial_labels=None):
        self.initial_labels = initial_labels
        super().__init__(Nb, Beta, path)

    def _init_from_labels(self, labels):
        """Инициализирует состояние из переданных меток."""
        # Преобразуем labels в словарь {узел: сообщество}
        if isinstance(labels, dict):
            label_dict = labels
        else:
            # labels — список/тензор, порядок соответствует self.graph.nodes()
            nodes = sorted(self.graph.nodes())
            label_dict = {node: int(labels[i]) for i, node in enumerate(nodes)}

        # Сбрасываем структуры (наследуются от GraphTolls)
        self.membership = {}
        self.DegCom = {}
        self.internal = {}
        self.loops = {}

        for node in self.graph.nodes():
            com = label_dict[node]
            self.membership[node] = com
            deg = float(self.graph.degree(node, weight='weight'))
            self.Degree[node] = deg
            self.DegCom[com] = self.DegCom.get(com, 0.0) + deg

            edge_data = self.graph.get_edge_data(node, node, {'weight': 0})
            self.loops[node] = float(edge_data.get('weight', 0))

        # Внутренние веса сообществ
        for u, v, data in self.graph.edges(data=True):
            w = float(data.get('weight', 1))
            if self.membership[u] == self.membership[v]:
                com = self.membership[u]
                self.internal[com] = self.internal.get(com, 0.0) + w

    def Run_FMLIG(self):
        start = time.time()
        if self.initial_labels is not None:
            self._init_from_labels(self.initial_labels)
            solution = self.membership.copy()
        else:
            solution = self.GCH()

        # Дальше копируем всю логику оригинального Run_FMLIG
        solution = self.FL_move()
        best_solution = copy.copy(solution)
        Q_best = super().modularity()
        T_init = 0.025 * Q_best
        T = T_init
        nb_iter = 0
        while nb_iter < self.Nb:
            Q1 = super().modularity()
            incumbent_solution = copy.copy(solution)
            solution, drop_nodes = self.Destruction()
            solution = self.Reconstruction(drop_nodes)
            solution = self.FL_move()
            Q2 = super().modularity()
            if Q2 > Q_best:
                best_solution = copy.copy(solution)
                Q_best = Q2
            P = random.random()
            if Q2 < Q1 and P > math.exp((Q2 - Q1) / T):
                solution = copy.copy(incumbent_solution)
                T = T_init
            elif Q2 >= Q1:
                T = T_init
            else:
                T = T * 0.9
            nb_iter += 1

        end = time.time()
        t = end - start
        return Q_best, best_solution, t

def tensor_to_graph_txt(graph_tensor: torch.Tensor, save_path: str):
    """
    Args:
        graph_tensor: torch.Tensor [n, n]
        save_path: str
    
    Return:
        File path.
    """
    if not torch.is_tensor(graph_tensor):
        raise TypeError("graph_tensor должен быть torch.Tensor")
    if graph_tensor.ndim != 2 or graph_tensor.size(0) != graph_tensor.size(1):
        raise ValueError("graph_tensor должен быть квадратной матрицей смежности")

    adj = graph_tensor.cpu().numpy()
    G = nx.from_numpy_array(adj)

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

    return str(path.resolve())

def flmig_adopted(
    adj: torch.Tensor,
    Number_iter = None,
    Beta: float = 0.5,
    max_rb: int = 1,
    return_labels: bool = False,
    timing_info: dict | None = None,
    initial_labels=None,
):
    """
    Args:
        adj: torch.Tensor [n, n], sparse или dense.
        Number_iter: int, optional
        Beta: float, optional
        max_rb: int, optional
        return_labels: если True — вернуть метки кластеров, иначе метрики
        timing_info: dict | None — сюда накапливаем conversion_time.
    """
    if Number_iter is None:
        Number_iter = 1
    if timing_info is None:
        timing_info = {}

    t0 = time.time()
    if adj.device.type == "cuda":
        adj = adj.cpu()
    t1 = time.time()
    timing_info["conversion_time"] = timing_info.get("conversion_time", 0.0) + (t1 - t0)

    # --- приведение к dense и бинаризация + запись во временный файл ---
    t0 = time.time()

    if adj.is_sparse:
        A = adj.coalesce()
        adj_dense = torch.sparse_coo_tensor(
            A.indices(),
            torch.where(
                A.values() > 0,
                torch.ones_like(A.values()),
                torch.zeros_like(A.values()),
            ),
            size=A.size(),
        ).to_dense()
        adj_dense.fill_diagonal_(0.0)
    else:
        adj_dense = (adj > 0).to(torch.float32)
        adj_dense.fill_diagonal_(0.0)

    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        
        tensor_to_graph_txt(adj_dense, tmp.name)
        path = tmp.name
        # print(f"Временный граф лежит здесь: {tmp.name}")

        t1 = time.time()
        timing_info["conversion_time"] = timing_info.get("conversion_time", 0.0) + (t1 - t0)

        best_community = None
        best_mod = -np.inf

        Q_list = []
        Time_list = []
        Community_list = []

        # print("ALARM ================")

        for nb_run in range(max_rb):
            # print(f"rb {nb_run}, max_rb {max_rb}")
            communities = MyFastLocalMoveIG(Number_iter, Beta, path, initial_labels=initial_labels)
            mod, community, tim = communities.Run_FMLIG()

            # print("community =", community)

            Q_list.append(mod)
            Time_list.append(tim)
            Community_list.append(community)

            if mod > best_mod:
                best_mod = mod
                best_community = community

        # print("communities =", communities)

        Q_avg = communities.avg(Q_list)
        Q_max = communities.max(Q_list)
        Q_std = communities.stdev(Q_list)
        time_run = communities.avg(Time_list)

    # os.remove(path)

    if return_labels:
        N = adj.size(0)
        labels = torch.zeros(N, dtype=torch.long)
        for node, com in best_community.items():
            node_idx = int(node)
            if 0 <= node_idx < N:
                labels[node_idx] = int(com)
        _, remap = torch.unique(labels, sorted=True, return_inverse=True)
        labels = remap.to(torch.long)
        return labels
    else:
        print("the value of Q_max", Q_max)
        print("the value of Q_avg", Q_avg)
        print("the value of Q_std", Q_std)
        print("the value of all pure time ", time_run * max_rb)
        return Q_max, Q_avg, Q_std, time_run
