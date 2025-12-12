import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
dese_root = os.path.join(PROJECT_PATH, "baselines", "DeSE")
if dese_root not in sys.path:
    sys.path.insert(0, dese_root)

import torch
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import networkx as nx
from matplotlib.colors import ListedColormap
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.autograd.set_detect_anomaly(True)

import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from baselines.dese import dese
import random


def main():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--dataset', nargs='?', default='Computers', 
                        help='Choose a dataset from {Cora, Citeseer, Pubmed, Computers, Photo, CS and Physics}.')
    parser.add_argument('--epochs', type=int, default=100, 
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