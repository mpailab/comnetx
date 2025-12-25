import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
dese_root = os.path.join(PROJECT_PATH, "baselines", "DeSE")
if dese_root not in sys.path:
    sys.path.insert(0, dese_root)

from utility.parser import parse_args
from utility.dataset import Data
from utility.util import decoding_from_assignment, cluster_metrics
from model import DeSE
import torch
import torch.optim as optim
from time import time, strftime, localtime
import os
import numpy as np
from torch_geometric.utils import negative_sampling
import random
import dgl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import networkx as nx
from matplotlib.colors import ListedColormap
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.autograd.set_detect_anomaly(True)

import argparse

def train(dataset, args):
    #prepare graph dataset and device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    device = 'cpu'
    # print(device)
    dataset = dataset
    
    dataset.print_statistic()
    #dataset.print_degree()
    
    best_cluster_result = {}
    best_cluster = {'nmi': -0.001, 'ari': -0.001, 'acc': -0.001, 'f1': -0.001}
    #prepare model
    model = DeSE(args, dataset.feature, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    t0 = time()
    for epoch in range(args.epochs):
        # print("in work")
        t1 = time()
        s_dic, tree_node_embed_dic, g_dic = model(dataset.adj, dataset.feature, dataset.degrees)
        #se_loss = model.calculate_se_loss(s_dic, g_dic[args.height])
        t2 = time()
        se_loss = model.calculate_se_loss1()
        t3= time()
        lp_loss = model.calculate_lp_loss(g_dic[args.height], dataset.neg_edge_index, tree_node_embed_dic[args.height])
        t4 = time()
        loss = args.se_lamda * se_loss + args.lp_lamda * lp_loss
        optimizer.zero_grad() #梯度归零
        loss.backward()
        optimizer.step()

        if epoch % args.verbose == 0:
            pred = decoding_from_assignment(model.hard_dic[1])
            # print("lab", dataset.labels, "\npred laab", pred)
            metrics = cluster_metrics(dataset.labels, pred)
            acc, nmi, f1, ari, new_pred = metrics.evaluateFromLabel(use_acc=True)
            if nmi > best_cluster['nmi']:
                best_cluster['nmi'] = nmi
                best_cluster_result['nmi'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), './save_model/{}_{}_nmi.pt'.format(args.dataset, args.num_clusters_layer[0]))
            if ari > best_cluster['ari']:
                best_cluster['ari'] = ari
                best_cluster_result['ari'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), './save_model/{}_{}_ari.pt'.format(args.dataset, args.num_clusters_layer[0]))
            if acc > best_cluster['acc']:
                best_cluster['acc'] = acc
                best_cluster_result['acc'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), './save_model/{}_{}_acc.pt'.format(args.dataset, args.num_clusters_layer[0]))
            if f1 > best_cluster['f1']:
                best_cluster['f1'] = f1
                best_cluster_result['f1'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), './save_model/{}_{}_f1.pt'.format(args.dataset, args.num_clusters_layer[0]))
            
            print(f"Epoch: {epoch} [{time()-t1:.3f}s], Loss: {loss.item():.6f} = {args.se_lamda} * {se_loss.item():.6f} + {args.lp_lamda} * {lp_loss.item():.6f}, NMI: {nmi:.6f}, ARI: {ari:.6f}, ACC: {acc:.6f}, F1: {f1:.6f}")
            #print(f"train time: {t2-t1}; se_loss time: {t3-t2}; lp_loss time: {t4-t3}")
    #print('Total time: {:.3f}s'.format(time()-t0))
    print(f"Best NMI: {best_cluster_result['nmi']}, Best ARI: {best_cluster_result['ari']}, \nBest Cluster: {best_cluster}")
    # print(args)

    return best_cluster, pred

def data_preprocess(adj, features, labels):
    labels = labels.tolist()
    edge_index = adj.coalesce().indices()
    num_nodes = adj.size(0)
    # num_neg_edges = min(edge_index.size(1), num_nodes * 5)
    num_neg_edges = edge_index.size(1)

    # print("edge_index ===", edge_index)

    neg_edge_index = negative_sampling(
        edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_edges,   
        method='sparse'                   
    ).to(torch.long)

    # print("edge_index in data_preprocess =", edge_index)

    class Dummy: pass
    dataset = Dummy()
    dataset.name = 'name'
    dataset.adj = adj
    dataset.edge_index = edge_index
    dataset.feature = features
    dataset.labels = labels
    dataset.degrees = adj.sum(1)
    dataset.neg_edge_index = neg_edge_index
    dataset.num_nodes = num_nodes
    dataset.num_edges = int(adj.sum())
    dataset.num_features = features.shape[1]
    dataset.num_classes = len(set(labels))

    def print_statistic():
        print(f"=== {dataset.name} | {adj.shape[0]} nodes | {dataset.num_edges//2} edges | {dataset.num_classes} classes ===")
        # print(f"Label dist: {dict(Counter(labels))}")
    dataset.print_statistic = print_statistic
    return dataset

def dese(adj, 
         features: torch.Tensor | None = None, 
         labels: torch.Tensor | None = None, 
         args=None,
         timing_info=None, 
         n_epochs=1,
         num_clusters=None,
         metrics_mod=None):
    time_s = time()
    if labels is None:
        num_nodes = adj.size(0)
        labels = torch.arange(num_nodes)
    elif len(labels.shape) == 2:
        labels = labels.squeeze()

    if num_clusters is None:
        num_clusters = len(torch.unique(labels))
    
    dataset = data_preprocess(adj, features, labels)
    # print("features ===", features.shape)
    features_dim = features.shape[-1]

    if args is None:
        class Args:
            dataset = 'Computers'
            epochs = n_epochs
            lr = 1e-2
            height = 2
            gpu = 0
            decay_rate = None
            num_clusters_layer = [num_clusters]
            layer_str = '[3]'
            embed_dim = features_dim
            se_lamda = 0.01
            lp_lamda = 1
            verbose = 10
            activation = 'relu'
            k = 2
            dropout = 0.1
            beta_f = 0.2
            seed = 42
            save = False
            fig_network = False
        args = Args()
    else:
        args.num_clusters_layer = [num_clusters]
        args.embed_dim = features_dim

    time_e = time()
    if timing_info is not None:
        timing_info['conversion_time'] = time_e - time_s

    metrics, out_label = train(dataset, args)
    # print(type(out_label))
    # print("out_label =", out_label)
    if metrics_mod==True:
        return out_label, metrics
    return out_label

def main():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--dataset', nargs='?', default='Computers', 
                        help='Choose a dataset from {Cora, Citeseer, Pubmed, Computers, Photo, CS and Physics}.')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate.')
    parser.add_argument('--height', type=int, default=2,
                        help='Height of the SE tree.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index. Default: -1, using CPU.')
    parser.add_argument('--decay_rate', type=int, default=None,
                        help='Decay rate of the number of clusters in each layer.')
    parser.add_argument('--num_clusters_layer', type=list, default=[10],
                        help='Number of clusters in each layer.')
    parser.add_argument('--layer_str', type=str, default='[10]',
                        help='Number of clusters in each layer.')
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--se_lamda', type=float, default=0.01,
                        help='Weight of se loss.')
    parser.add_argument('--lp_lamda', type=float, default=1,
                        help='Weight of lp loss.')
    parser.add_argument('--verbose', type=int, default=20,
                        help='evaluate every verbose epochs.')
    parser.add_argument('--activation', type=str, default='relu',
                        help='elu, relu, sigmoid, None')
    parser.add_argument('--k', type=int, default=2,
                        help='KNN')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--beta_f', type=float, default=0.2,
                        help='weight for adj_f')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save', type=bool, default=False,
                        help='Save model or not')
    parser.add_argument('--fig_network', type=bool, default=False,
                        help='Draw network or not')
    
    parser.add_argument("--adj", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()
    print(args.device)
    adj = torch.load(args.adj)
    adj = torch.sparse_coo_tensor(
        adj.indices(),
        adj.values().float(),
        adj.size()
    ).coalesce()
    features = torch.load(args.features)
    labels = torch.load(args.labels)
    new_labels = dese(adj, features, labels, args)

    torch.save(new_labels, args.out)
    print("DeSE finished successfully")

if __name__ == "__main__":
    main()