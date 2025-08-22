import csv
import random
import time
import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_sparse import SparseTensor

from magi.model import Model, Encoder
from magi.neighbor_sampler import NeighborSampler
from magi.utils import setup_seed, get_mask, clustering

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from datasets import Dataset
import debugpy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from metrics import Metrics

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--runs', type=int, default=4)
parser.add_argument('--max_duration', type=int,
                    default=60, help='max duration time')
parser.add_argument('--kmeans_device', type=str,
                    default='cpu', help='kmeans device, cuda or cpu')
parser.add_argument('--kmeans_batch', type=int, default=-1,
                    help='batch size of kmeans on GPU, -1 means full batch')
parser.add_argument('--batchsize', type=int, default=2048, help='')

# dataset para
parser.add_argument('--dataset_path', type=str, default='../../../test/graphs/small/reddit')
parser.add_argument('--dataset_name', type=str, default='Reddit')

# model para
parser.add_argument('--hidden_channels', type=str, default='512,256')
parser.add_argument('--size', type=str, default='10,10', help='')
parser.add_argument('--projection', type=str, default='')
parser.add_argument('--tau', type=float, default=0.5, help='temperature')
parser.add_argument('--ns', type=float, default=0.5)

# sample para
parser.add_argument('--wt', type=int, default=20)
parser.add_argument('--wl', type=int, default=4)
parser.add_argument('--n', type=int, default=2048)

# learning para
parser.add_argument('--dropout', type=float, default=0, help='')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

def sparse_equal(t1, t2):
    if t1.shape != t2.shape:
        return False
    # Приводим к одинаковому порядку
    t1 = t1.coalesce()
    t2 = t2.coalesce()
    return (torch.equal(t1.indices(), t2.indices()), torch.allclose(t1.values(), t2.values()))

def train():
    ts = time.time()
    randint = random.randint(1, 1000000)
    setup_seed(randint)
    if args.verbose:
        print('random seed : ', randint, '\n', args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loader = Dataset(dataset_name=args.dataset_name, path=args.dataset_path)
    adj, x, y = loader.load(tensor_type="coo")
    num_features =  x.size()[-1]
    print("features = ", num_features, "label = ", y.size())
    edge_index = adj.indices()
    num_nodes = adj.size(0)
    edge_index_loops, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)

    edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])

    new_values = torch.ones(edge_index.size(1))
    print("tensor size = ", adj.shape, "nnz = ", adj._nnz())

    N, E, num_features = x.shape[0], edge_index.shape[-1], x.shape[-1]

    
    print("edge_index[0].shape:", edge_index[0].shape)
    print("edge_index[1].shape:", edge_index[1].shape)
    print("new_values.shape:", new_values.shape)
    print("N:", N)

    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=new_values, sparse_sizes=(N, N))
    print("N = ", N, "E = ",  E, "num_features = ", num_features)
    print(f"Loading {loader.name} is over, num_nodes: {N: d}, num_edges: {E: d}, "
          f"num_feats: {num_features: d}, time costs: {time.time()-ts: .2f}")

    
    hidden = list(map(int, args.hidden_channels.split(',')))
    if args.projection == '':
        projection = None
    else:
        projection = list(map(int, args.projection.split(',')))
    size = list(map(int, args.size.split(',')))
    assert len(hidden) == len(size)

    train_loader = NeighborSampler(edge_index, adj,
                                   is_train=True,
                                   node_idx=None,
                                   wt=args.wt,
                                   wl=args.wl,
                                   sizes=size,
                                   batch_size=args.batchsize,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=6)

    test_loader = NeighborSampler(edge_index, adj,
                                  is_train=False,
                                  node_idx=None,
                                  sizes=size,
                                  batch_size=512,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=6)

    encoder = Encoder(num_features, hidden_channels=hidden,
                      dropout=args.dropout, ns=args.ns).to(device)
    model = Model(
        encoder, in_channels=hidden[-1], project_hidden=projection, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    dataset2n_clusters = {'ogbn-arxiv': 40, 'Reddit': 41,
                          'ogbn-products': 47, 'ogbn-papers100M': 172, 
                          'Acm': 3, 'Bat': 8, 'Dblp': 4, 'Eat': 4, 'Uat': 5,
                          'Cora': 7, 'Citeseer': 6, 'Photo': 8, 'Computers': 10}
    n_clusters = dataset2n_clusters[loader.name]

    x = x.to(device)
    print(f"Start training")

    ts_train = time.time()
    stop_pos = False
    for epoch in range(1, args.epochs):
        model.train()
        total_loss = total_examples = 0

        for (batch_size, n_id, adjs), adj_batch, batch in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            if len(hidden) == 1:
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]

            adj_ = get_mask(adj_batch)
            optimizer.zero_grad()
            out = model(x[n_id].to(device), adjs=adjs)
            out = F.normalize(out, p=2, dim=1)
            loss = model.loss(out, adj_)

            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            total_examples += batch_size

            if args.verbose:
                print(f'(T) | Epoch {epoch:02d}, loss: {loss:.4f}, '
                      f'train_time_cost: {time.time() - ts_train:.2f}, examples: {batch_size:d}')

            train_time_cost = time.time() - ts_train
            if train_time_cost // 60 >= args.max_duration:
                print(
                    "*********************** Maximum training time is exceeded ***********************")
                stop_pos = True
                break
        if stop_pos:
            break

    print(f'Finish training, training time cost: {time.time() - ts_train:.2f}')

    with torch.no_grad():
        model.eval()
        z_all = torch.zeros((y.size(0), hidden[-1]))
        for count, ((batch_size, n_id, adjs), _, batch) in enumerate(tqdm(test_loader)):
            if len(hidden) == 1:
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]
            out = model(x[n_id].to(device), adjs=adjs)
            z_all[n_id[:batch_size]] = out.detach().cpu()
    
    z = F.normalize(z_all, p=2, dim=1)

    ts_clustering = time.time()
    print(f'Start clustering, num_clusters: {n_clusters: d}')
    lbls, acc, nmi, ari, f1_macro, f1_micro = clustering(z, n_clusters, y.numpy(), kmeans_device=args.kmeans_device,
                                                   batch_size=args.kmeans_batch, tol=1e-4, device=device, spectral_clustering=False)

    assignments = F.one_hot(torch.tensor(lbls, dtype=torch.long), num_classes=n_clusters).float()
    mod_score = Metrics.modularity_magi_prob(adj, assignments)

    print(f'Finish clustering, acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1_macro: {f1_macro:.4f}, '
          f'f1_micro: {f1_micro:.4f}, clustering time cost: {time.time() - ts_clustering:.2f}')

    print(f'Modularity: {mod_score.item():.4f}')

    return acc, nmi, ari, f1_macro, f1_micro, mod_score.item()


def run(runs=1, result=None):
    if result:
        with open(result, 'w', encoding='utf-8-sig', newline='') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(
                ['runs', 'acc', 'nmi', 'ari', 'f1_macro', 'f1_micro'])

    ACC, NMI, ARI, F1_MA, F1_MI = [], [], [], [], []
    for i in range(runs):
        print(f'\n----------------------runs {i+1: d} start')
        acc, nmi, adjscore, f1_macro, f1_micro, mod_score = train()
        print(f'\n----------------------runs {i + 1: d} over')
        if result:
            with open(result, 'a', encoding='utf-8-sig', newline='') as f_w:
                writer = csv.writer(f_w)
                writer.writerow([i+1, acc, nmi, adjscore, f1_macro, f1_micro])

        ACC.append(acc)
        NMI.append(nmi)
        ARI.append(adjscore)
        F1_MA.append(f1_macro)
        F1_MI.append(f1_micro)

    ACC = np.array(ACC)
    NMI = np.array(NMI)
    ARI = np.array(ARI)
    F1_MA = np.array(F1_MA)
    F1_MI = np.array(F1_MI)
    if result:
        with open(result, 'a', encoding='utf-8-sig', newline='') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(['mean', ACC.mean(), NMI.mean(),
                            ARI.mean(), F1_MA.mean(), F1_MI.mean()])
            writer.writerow(['std', ACC.std(), NMI.std(),
                            ARI.std(), F1_MA.std(), F1_MI.std()])


if __name__ == '__main__':
    result = None
    run(args.runs, result)