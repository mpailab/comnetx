import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
dese_root = os.path.join(PROJECT_PATH, "baselines", "S2CAG", "src")
if dese_root not in sys.path:
    sys.path.insert(0, dese_root)

import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from time import time
from scipy import sparse
from sklearn.cluster import KMeans
from utils_s2cag import *
from sklearn.utils.extmath import randomized_svd
from sklearn import cluster
from utils_s2cag import preprocess_dataset, run_SSCAG, clustering_accuracy
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari

import numpy as np
import scipy.sparse as sp
import torch
import argparse

def torch_sparse_to_scipy(tensor):
    """Convert torch.sparse_coo_tensor → scipy.sparse.csr_matrix"""
    tensor = tensor.coalesce()
    indices = tensor.indices().cpu().numpy()
    values = tensor.values().cpu().numpy()
    shape = tensor.shape

    row = indices[0]
    col = indices[1]

    return sp.csr_matrix((values, (row, col)), shape=shape)

def torch_to_scipy(adj_t, features_t, labels_t):
    """
    Convert torch tensors back to the same format
    as datagen(dataset) returns.
    """

    # --- convert adjacency ---
    if adj_t.is_sparse:
        adj = torch_sparse_to_scipy(adj_t)
    else:
        adj = sp.csr_matrix(adj_t.cpu().numpy())

    # --- convert features ---
    if features_t.is_sparse:
        features = features_t.to_dense().cpu().numpy()
    else:
        features = features_t.cpu().numpy()

    # --- convert labels ---
    labels = labels_t.cpu().numpy()

    # --- number of classes ---
    n_classes = len(np.unique(labels))

    return adj, features, labels, n_classes

def get_normalized_adj(adj: torch.Tensor):
    if adj.is_sparse:
        adj = adj.coalesce()
        adj_indices = adj.indices()
        adj_values = adj.values()
        
        num_nodes = adj.size(0)
        ones = torch.ones(adj_values.shape[0], device=adj.device)
        deg = torch.zeros(num_nodes, device=adj.device)
        deg.scatter_add_(0, adj_indices[0], adj_values * ones)
    else:
        deg = adj.sum(dim=1)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    deg_inv = deg.pow(-1.0)
    deg_inv[deg_inv == float('inf')] = 0

    # D^{-1/2} * A * D^{-1/2}
    if adj.is_sparse:
        row, col = adj_indices
        norm_values = adj_values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(adj_indices, norm_values, adj.size())
    else:
        return adj * deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)
            
def generate_graph_features(
    adj: torch.Tensor, 
    embedding_dim: int = 64, 
    alpha: float = 0.9, 
    power: int = 10, 
    add_self_loops: bool = True 
    ) -> torch.Tensor:
    """
    Args:
        adj : torch.Tensor [N,N]
        embedding_dim : int
        alpha : float
        power : int
        add_self_loops : bool
    Return:
        torch.Tensor [N, embedding_dim].
    """
    device = adj.device
    num_nodes = adj.size(0)
    
    if add_self_loops:
        if adj.is_sparse:
            indices = torch.arange(num_nodes, device=device)
            indices = torch.stack([indices, indices], dim=0)
            values = torch.ones(num_nodes, device=device)
            eye = torch.sparse_coo_tensor(indices, values, adj.size())
            adj = adj + eye
        else:
            adj = adj + torch.eye(num_nodes, device=device)

    adj_norm = get_normalized_adj(adj)
    
    features = torch.rand(num_nodes, embedding_dim, device=device)
    
    # 4. Power Iteration (Сглаживание / Пропагация)
    # Реализация APPNP / Personalized PageRank update:
    # X^(k+1) = (1-alpha)*X_init + alpha * A_norm * X^(k)
    
    features_init = features.clone()
    
    for _ in range(power):
        # Матричное умножение (поддерживает sparse @ dense)
        if adj_norm.is_sparse:
            agg = torch.sparse.mm(adj_norm, features)
        else:
            agg = torch.mm(adj_norm, features)
            
        features = (1 - alpha) * features_init + alpha * agg
        
    return features

def s2cag(adj_torch: torch.Tensor, 
          features_torch: torch.Tensor | None = None, 
          labels: torch.Tensor | None = None,
          timing_info=None,
          dataset = 'dataset', T = 15, n_runs = 1, 
          alpha= 0.8, fdim = 0, method = 'sub', 
          gamma = 1, tau = 50,
          metrics_mod=None):
    time_s = time()
    if labels is None:
        num_nodes = adj_torch.size(0)
        labels = torch.arange(num_nodes)
    
    if features_torch is None:
        features_torch = generate_graph_features(adj_torch, embedding_dim=8, alpha=0.9, power=10)

    # print("features_torch =", features_torch)

    features_torch = torch.abs(features_torch)

    adj, features, labels, n_classes = torch_to_scipy(adj_torch, features_torch, labels)

    norm_adj, features = preprocess_dataset(adj, features, sparse=True, tf_idf=True)
    # print(type(norm_adj), type(features), type(labels))
    
    time_e = time()
    if timing_info is not None:
        timing_info['conversion_time'] = time_e - time_s

    features = features.toarray()
    n, d = features.shape
    k = n_classes


    metrics = {}
    metrics['acc'] = []
    metrics['nmi'] = []
    metrics['ari'] = []
    metrics['time'] = []
    adj_pt=sparse.coo_matrix(adj)
    
    x = features

    for run in range(n_runs):
        features = x

        t0 = time()
        # print(features, k, norm_adj, sep='\n')
        P, Q = run_SSCAG(features, k, norm_adj, T, alpha,method=method,dataset=dataset,gamma=gamma,tau=tau)



    metrics['time'].append(time()-t0)
    metrics['acc'].append(clustering_accuracy(labels, P)*100)
    # from sklearn.metrics import accuracy_score
    # metrics['acc'].append(accuracy_score(labels, P) * 100)
    metrics['nmi'].append(nmi(labels, P)*100)
    metrics['ari'].append(ari(labels, P)*100)


    results = {
        'mean': {k:(np.mean(v)).round(2) for k,v in metrics.items() }, 
        'std': {k:(np.std(v)).round(2) for k,v in metrics.items()}
        }

    means = results['mean']
    std = results['std']

    results = {
      'mean': {k:(np.mean(v)).round(2) for k,v in metrics.items() }, 
      'std': {k:(np.std(v)).round(2) for k,v in metrics.items()}
    }

    means = results['mean']
    std = results['std']


    print(f"{dataset} {T} {alpha} {method} {tau}")
    print(f"acc: {means['acc']}±{std['acc']} & nmi: {means['nmi']}±{std['nmi']} & ari: {means['ari']}±{std['ari']} & time: {means['time']}±{std['time']}", sep=',')
    new_labels = torch.tensor(P)

    if metrics_mod==True:
        return new_labels, metrics
    return new_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    
    parser.add_argument("--dataset", type=str, default="data")
    parser.add_argument("--T", type=int, default=15)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--fdim", type=int, default=0)
    parser.add_argument("--method", type=str, default="sub")
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--tau", type=int, default=50)   

    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    features = torch.load(args.features)
    labels = torch.load(args.labels)

    new_labels = s2cag(adj, features, labels,
                       T = 15, n_runs = 5, alpha= 0.8, 
                       method = 'sub', gamma = 1, tau = 50)

    torch.save(new_labels, args.out)
    print("S2CAG finished successfully")

if __name__ == "__main__":
    main()


