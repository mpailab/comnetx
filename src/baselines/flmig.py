import sys,os
import time
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_PATH, "src")
FLMIG_root = os.path.join(PROJECT_PATH, "baselines", "FLMIG_algorithm", "FLMIG_algorithm")

for p in (SRC_PATH, FLMIG_root):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import networkx as nx
from pathlib import Path
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from GraphTools import GraphTolls
from FLMIG import Fast_local_Move_IG



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

def de_main(path, Number_iter, Beta, max_rb):

    data = GraphTolls(path)
    graph = data.Read_Graph()
    NMI_list = []
    Time_list = [] 
    Q_list = []
    nb_run = 0

    while nb_run < max_rb :
        # print("rb",nb_run)
        communities = Fast_local_Move_IG( Number_iter, Beta, path)
        mod,community,tim = communities.Run_FMLIG() 
        #print(community)
        Q_list.append(mod)
        Time_list.append(tim)
        #label = communities.lebel_node(community)  
      
        nb_run = nb_run +1
            
    
    Q_avg = communities.avg(Q_list)
    Q_max = communities.max(Q_list)
    Q_std = communities.stdev(Q_list)
    time_run = communities.avg(Time_list)
    return Q_max, Q_avg, Q_std, time_run

def generate_symmetric_adj_matrix(n_nodes=100, edge_prob=0.05, seed=None):
    """
    Генерирует случайную симметричную матрицу смежности (torch.Tensor)
    для неориентированного графа без петель.

    Параметры:
        n_nodes: int — количество вершин
        edge_prob: float — вероятность ребра между двумя вершинами (0–1)
        seed: int — фиксирует случайность (опционально)

    Возвращает:
        adj: torch.Tensor (размер n_nodes × n_nodes)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # создаём случайную верхнетреугольную матрицу (без диагонали)
    upper = torch.bernoulli(torch.full((n_nodes, n_nodes), edge_prob))
    upper = torch.triu(upper, diagonal=1)  # только верхняя часть

    # делаем симметричной
    adj = upper + upper.T

    return adj

# if __name__ == '__main__':

#     adj = generate_symmetric_adj_matrix(300, edge_prob=0.1, seed=42)
#     print(type(adj))
#     # --- создаём временный граф ---
#     path = tensor_to_graph_txt(adj, PROJECT_PATH + "/src/baselines/graph.txt")
#     print("Временный файл графа:", path)
#     max_rb = 10
#     Q_max, Q_avg, Q_std,time_run = de_main(path, Number_iter=100, Beta=0.5, max_rb=max_rb)
#     print("the value of Q_max",Q_max)
#     print("the value of Q_avg",Q_avg)
#     print("the value of Q_std",Q_std)
#     print("the value of all pure time ",time_run * max_rb) 

#     os.remove(path)


def flmig_adopted(
    adj: torch.Tensor,
    Number_iter: int = 100,
    Beta: float = 0.5,
    max_rb: int = 10,
    return_labels: bool = False,
    timing_info: dict | None = None,
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
    if timing_info is None:
        timing_info = {}

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
    else:
        adj_dense = (adj > 0).to(torch.float32)
        adj_dense.fill_diagonal_(0.0)

    path = tensor_to_graph_txt(
        adj_dense,
        str(Path(PROJECT_PATH) / "src" / "baselines" / "graph.txt"),
    )

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
        communities = Fast_local_Move_IG(Number_iter, Beta, path)
        mod, community, tim = communities.Run_FMLIG()

        # print("community =", community)

        Q_list.append(mod)
        Time_list.append(tim)
        Community_list.append(community)

        if mod > best_mod:
            best_mod = mod
            best_community = community

    communities = Fast_local_Move_IG(Number_iter, Beta, path)

    # print("communities =", communities)

    Q_avg = communities.avg(Q_list)
    Q_max = communities.max(Q_list)
    Q_std = communities.stdev(Q_list)
    time_run = communities.avg(Time_list)

    os.remove(path)

    if return_labels:
        N = adj.size(0)
        labels = torch.zeros(N, dtype=torch.long)
        for node, com in best_community.items():
            node_idx = int(node)
            if 0 <= node_idx < N:
                labels[node_idx] = int(com)
        return labels
    else:
        print("the value of Q_max", Q_max)
        print("the value of Q_avg", Q_avg)
        print("the value of Q_std", Q_std)
        print("the value of all pure time ", time_run * max_rb)
        return Q_max, Q_avg, Q_std, time_run


if __name__ == "__main__":
    import argparse
    from datasets import Dataset

    ap = argparse.ArgumentParser()
    ...
    args = ap.parse_args()

    ds = Dataset(args.dataset_name, path=args.adj)
    adj, features, labels = ds.load(tensor_type="coo")

    labels = flmig_adopted(
        adj=adj,                      # sparse или dense, внутри всё приведётся
        Number_iter=args.Number_iter,
        Beta=args.Beta,
        max_rb=args.max_rb,
        return_labels=True,
    )

    if args.out is not None:
        torch.save(labels, args.out)

