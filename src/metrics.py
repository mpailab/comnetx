import torch
import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import pairwise_distances
from torch_sparse import SparseTensor
from scipy.optimize import linear_sum_assignment

# import tensorflow as tensor 
# import torch as tensor 

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

    
    def accuracy(true_labels, pred_labels) -> float:
        """
        Args:
            true_labels: torch.Tensor [n_nodes] 
            pred_labels: torch.Tensor [n_nodes] 
        Returns:
            accuracy: float 
        """
        if true_labels.shape != pred_labels.shape:
            print("true_labels.shape =", true_labels.shape, "pred_labels.shape =", pred_labels.shape)
            raise ValueError("true_labels.shape != pred_labels.shape")

        true_labels = true_labels.detach().cpu().numpy().astype(np.int64)
        pred_labels = pred_labels.detach().cpu().numpy().astype(np.int64)
        
        D = max(pred_labels.max(), true_labels.max()) + 1
        
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(pred_labels.size):
            w[true_labels[i], pred_labels[i]] += 1
            
        # (Hungarian Algorithm)
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        
        accuracy = w[row_ind, col_ind].sum() / pred_labels.size
        
        return accuracy
    
    def nmi(true_labels, pred_labels) -> float:
        """
        Args:
            true_labels: torch.Tensor [n_nodes] 
            pred_labels: torch.Tensor [n_nodes] 
        Returns:
            accuracy: float 
        """
        true_labels = true_labels.flatten().long()
        pred_labels = pred_labels.flatten().long()
        n = true_labels.size(0)

        def get_entropy(labels):
            _, counts = torch.unique(labels, return_counts=True)
            probs = counts.float() / n
            return -torch.sum(probs * torch.log(probs + 1e-12))

        u_labels, u_inv = torch.unique(true_labels, return_inverse=True)
        v_labels, v_inv = torch.unique(pred_labels, return_inverse=True)
        
        combined = u_inv * v_labels.size(0) + v_inv
        counts_uv = torch.bincount(combined, minlength=u_labels.size(0) * v_labels.size(0))
        probs_uv = counts_uv.float() / n
        
        h_u = get_entropy(true_labels)
        h_v = get_entropy(pred_labels)
        h_uv = -torch.sum(probs_uv * torch.log(probs_uv + 1e-12))
        
        mi = h_u + h_v - h_uv
        
        nmi = mi / torch.sqrt(h_u * h_v + 1e-12)
        
        return nmi.item()

    def balanced_acc(true_labels, pred_labels) -> float:
        true_labels = true_labels.flatten().long()
        pred_labels = pred_labels.flatten().long()
        
        unique_classes = torch.unique(true_labels)
        recalls = []
        
        for cls in unique_classes:
            true_mask = (true_labels == cls)
            
            tp = torch.sum((pred_labels == cls) & true_mask).float()
            
            actual_total = torch.sum(true_mask).float()
            
            recall = tp / (actual_total + 1e-12)
            recalls.append(recall)
        
        balanced_acc = torch.stack(recalls).mean()
        
        return balanced_acc.item()
