import torch
import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import pairwise_distances, adjusted_rand_score, f1_score
from torch_sparse import SparseTensor

# import tensorflow as tensor 
# import torch as tensor 
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

class Metrics:

    def __init__(self):
        pass

    def modularity(adjacency, assignments, gamma : float = 1.0, directed : bool = False) -> float:
        """
        Args:
            adjacency: torch.sparse.Tensor [n_nodes, n_nodes]
            assignments: torch.Tensor [n_nodes] 
            gamma: float, optional (default = 1.0)
            directed: bool, optional (default = False)
        Returns:
            modularity: float 
        """

        adjacency = adjacency.coalesce()
        device = adjacency.device
        row, col = adjacency.indices()
        weight = adjacency.values()  
        assignments = assignments.to(device=device, dtype=torch.long)

        d_out = torch.sparse.sum(adjacency, dim=1).to_dense()
        d_in = torch.sparse.sum(adjacency, dim=0).to_dense()
        m = torch.sum(weight)

        communities = torch.unique(assignments)
        modularity = 0.0
        
        for community in communities:
            c = community.item()
    
            # Фактический вес ребер внутри сообщества
            both_in_community = (assignments[row] == c) & (assignments[col] == c)
            actual_weight = torch.sum(weight[both_in_community])
            
            # Ожидаемый вес
            mask = (assignments == c)
            expected_weight = torch.sum(d_out[mask]) * torch.sum(d_in[mask]) / m
            
            modularity += (actual_weight - gamma * expected_weight)
        modularity /=  m

        return modularity.item()

    def purity_score(true_labels, pred_labels):
        y_true = _to_numpy(true_labels).astype(int)
        y_pred = _to_numpy(pred_labels).astype(int)

        classes, y_true_idx = np.unique(y_true, return_inverse=True)
        clusters, y_pred_idx = np.unique(y_pred, return_inverse=True)
        n_classes = classes.shape[0]
        n_clusters = clusters.shape[0]

        contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
        np.add.at(contingency, (y_true_idx, y_pred_idx), 1)

        max_over_classes = contingency.max(axis=0)
        return max_over_classes.sum() / y_true.shape[0]

    def ari_score(true_labels, pred_labels):
        y_true = _to_numpy(true_labels).astype(int)
        y_pred = _to_numpy(pred_labels).astype(int)
        return adjusted_rand_score(y_true, y_pred)

    def macro_f1(true_labels, pred_labels):
        y_true = _to_numpy(true_labels).astype(int)
        y_pred = _to_numpy(pred_labels).astype(int)
        return f1_score(y_true, y_pred, average="macro")