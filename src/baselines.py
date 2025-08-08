import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_sparse import SparseTensor

magi_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../baselines/MAGI"))
if magi_root not in sys.path:
    sys.path.insert(0, magi_root)

from magi.model import Model, Encoder
from magi.utils import get_mask, clustering
from magi.neighbor_sampler import NeighborSampler



def magi(adj : torch.Tensor, 
         features : torch.Tensor, 
         labels : torch.Tensor, 
         n_clusters : int = -1, 
         device=None, 
         args=None):

    if args is None:
        class Args:
            batchsize = 2048
            max_duration = 60
            kmeans_device = 'cpu'
            kmeans_batch = -1
            hidden_channels = '1024,256'  # в magi этот параметр называется hidden_channels
            size = '10,10'
            wt = 20
            wl = 5
            tau = 0.5
            ns = 0.5
            lr = 0.01
            epochs = 10
            projection = ""
            wd = 0
            dropout = 0
        args = Args()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = features.to(device)
    labels = labels.to(device)

    N, num_features = features.shape[0], features.shape[-1]

    edge_index = adj.indices() if hasattr(adj, 'indices') else adj
    edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])
    new_values = torch.ones(edge_index.size(1), device=device)
    adj_sparse = SparseTensor(row=edge_index[0].to(device), col=edge_index[1].to(device), value=new_values, sparse_sizes=(N, N))

    hidden = list(map(int, args.hidden_channels.split(',')))
    if args.projection == '':
        projection = None
    else:
        projection = list(map(int, args.projection.split(',')))
    size = list(map(int, args.size.split(',')))
    assert len(hidden) == len(size)

    train_loader = NeighborSampler(edge_index, adj_sparse,
                                   is_train=True,
                                   node_idx=None,
                                   wt=args.wt,
                                   wl=args.wl,
                                   sizes=size,
                                   batch_size=args.batchsize,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=0)

    test_loader = NeighborSampler(edge_index, adj_sparse,
                                  is_train=False,
                                  node_idx=None,
                                  sizes=size,
                                  batch_size=512,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=0)

    encoder = Encoder(num_features, hidden_channels=hidden,
                      dropout=args.dropout, ns=args.ns).to(device)

    model = Model(
        encoder, in_channels=hidden[-1], project_hidden=projection, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    if n_clusters == -1:
        n_clusters = len(torch.unique(labels))

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        batches = 0
        for (batch_size, n_id, adjs), adj_batch, batch in train_loader:
            adjs = [adjs] if len(hidden) == 1 else adjs
            adjs = [adj.to(device) for adj in adjs]
            adj_mask = get_mask(adj_batch)
            optimizer.zero_grad()
            out = model(features[n_id], adjs=adjs)
            out = F.normalize(out, p=2, dim=1)
            loss = model.loss(out, adj_mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches if batches > 0 else 0
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")

    model.eval()
    z_all = torch.zeros((N, hidden[-1]), device=device)
    with torch.no_grad():
        for (batch_size, n_id, adjs), _, batch in test_loader:
            adjs = [adjs] if len(hidden) == 1 else adjs
            adjs = [adj.to(device) for adj in adjs]
            out = model(features[n_id], adjs=adjs)
            z_all[n_id[:batch_size]] = out

    embeddings = F.normalize(z_all, p=2, dim=1)

    new_labels, _, _, _, _, _ = clustering(embeddings, n_clusters,
                                        kmeans_device=device.type,
                                        batch_size=-1,
                                        tol=1e-4,
                                        device=device,
                                        true_labels=labels.tolist(),
                                        spectral_clustering=False)
    if not isinstance(new_labels, torch.Tensor):
        new_labels = torch.tensor(new_labels, dtype=torch.long, device=device)
    else:
        new_labels = new_labels.to(dtype=torch.long, device=device)

    return new_labels