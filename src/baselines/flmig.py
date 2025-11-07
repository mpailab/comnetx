import sys,os
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLMIG_root = os.path.join(PROJECT_PATH, "baselines", "FLMIG_algorithm", "FLMIG_algorithm")
if FLMIG_root not in sys.path:
    sys.path.insert(0, FLMIG_root)

import torch
import networkx as nx
from pathlib import Path
import os
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
        print("rb",nb_run)
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

def flmig_adopted(adj: torch.tensor, Number_iter=100, Beta=0.5, max_rb=10):
    """
    Args:
        adj: torch.Tensor [n, n]
        Number_iter: int, optional(default = 100) - the number of iterations in the algorithm
        Beta: float, optional(default = 0.5)
        max_rb: int, optional(default = 10) - the number of iterations of the algorithm itself
    """
    path = tensor_to_graph_txt(adj, PROJECT_PATH + "/src/baselines/graph.txt")

    Q_max, Q_avg, Q_std,time_run = de_main(path, Number_iter, Beta, max_rb)
    print("the value of Q_max",Q_max)
    print("the value of Q_avg",Q_avg)
    print("the value of Q_std",Q_std)
    print("the value of all pure time ",time_run * max_rb) 

    os.remove(path)