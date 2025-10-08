import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_sparse import SparseTensor
from sklearnex import patch_sklearn

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
magi_root = os.path.join(PROJECT_PATH, "baselines", "MAGI")
if magi_root not in sys.path:
    sys.path.insert(0, magi_root)

from magi.model import Model, Encoder
from magi.utils import get_mask
from magi.neighbor_sampler import NeighborSampler
from magi.batch_kmeans_cuda import kmeans

from metrics import Metrics



def magi(adj : torch.Tensor, 
         features : torch.Tensor, 
         labels : torch.Tensor | None = None, 
         n_clusters : int = -1, 
         device=None, 
         args=None,
         **kwargs):

    """
    MAGI method

    Parameters
    ----------
    adj : torch.Tensor
        Adjacency matrix, shape [N, N].

    features: torch.Tensor
        Features matrix, shape [N, K].

    labels: torch.Tensor or None, optional
        Ground-truth node labels.
        Default: None 

    n_clusters: int, optional
        Number of clusters.
        Default: -1 

    device: torch.device or None, optional
        Device for computing: 'cuda', 'cpu'
        Default: None 

    args: 
        Hyperparameters for MAGI training. If None, default parameters are used.

    Returns
    -------
    torch.Tensor
        Predicted cluster assignments for all nodes, shape [N].
    """

    if args is None:
        class Args:
            batchsize = 2048
            max_duration = 60
            kmeans_device = 'cpu'
            kmeans_batch = -1
            hidden_channels = '1024,256'
            size = '10,10'
            wt = 20
            wl = 5
            tau = 0.5
            ns = 0.5
            lr = 0.05
            epochs = 100
            projection = ""
            wd = 0
            dropout = 0
        args = Args()

    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            raise ValueError(f"Unknown argument {key}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("device: ", device)

    features = features.to(device)
    if labels is None:
        num_nodes = adj.size(0)
        labels = torch.arange(num_nodes, device=device)
    else:
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

    if labels is None:
        num_nodes = adj.size(0)
        labels = torch.arange(num_nodes)

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

            print(f'(T) | Epoch {epoch:02d}, loss: {loss:.4f}, examples: {batch_size:d}')
                      
        avg_loss = total_loss / batches if batches > 0 else 0

    model.eval()
    z_all = torch.zeros((N, hidden[-1]), device=device)
    with torch.no_grad():
        for (batch_size, n_id, adjs), _, batch in test_loader:
            adjs = [adjs] if len(hidden) == 1 else adjs
            adjs = [adj.to(device) for adj in adjs]
            out = model(features[n_id], adjs=adjs)
            z_all[n_id[:batch_size]] = out

    embeddings = F.normalize(z_all, p=2, dim=1)

    k_max = min(40, adj_sparse.sparse_sizes()[0])
    k_min = min(2, k_max)
    if k_max <= k_min:
        num_clusters = k_max
        new_labels = torch.zeros(adj_sparse.sparse_sizes()[0], dtype=torch.long, device=device)
    else:
        num_clusters, new_labels = find_best_k_with_modularity(adj_sparse, embeddings, range(k_min, k_max), device=device)
    if not isinstance(new_labels, torch.Tensor):
        new_labels = torch.tensor(new_labels, dtype=torch.long, device=device)
    else:
        new_labels = new_labels.to(dtype=torch.long, device=device)

    return new_labels


def find_best_k_with_modularity(adj_sparse : torch.Tensor,
                                embeddings : torch.Tensor, 
                                k_range, 
                                device=torch.device('cuda:0')):

    """
    Finds the best number of clusters based on modularity.

    Parameters
    ----------
    adj_sparse : torch.Tensor
        Adjacency matrix of the graph, shape [N, N].
        
    embeddings : torch.Tensor
        Node embeddings matrix, shape [N, D].

    k_range : iterable
        Range of cluster numbers (k) to try.

    device : torch.device, optional
        Device for computing: 'cuda', 'cpu'
        Default: torch.device('cuda:0').

    Returns
    -------
    tuple
        best_k : int
            Number of clusters with highest modularity.
        best_labels : torch.Tensor
            Predicted cluster assignments for all nodes, shape [N].
    """
    best_k = None
    best_modularity = -float('inf')
    best_labels = None

    for k in k_range:
        pred_labels, cluster_centers = clustering(
            feature = embeddings,
            n_clusters=k,
            kmeans_device=device.type,
            batch_size=-1,
            tol=1e-4,
            device=device,
            spectral_clustering=False
        )

        """
        if cluster_centers is not None:
            dists = torch.cdist(embeddings, cluster_centers.to(device))
            assignments = torch.softmax(-dists, dim=1)
        else:
            assignments = torch.nn.functional.one_hot(torch.tensor(pred_labels, device=device), num_classes=k).float()
        """
        assignments = torch.nn.functional.one_hot(torch.tensor(pred_labels, device=device), num_classes=k).float()
        mod = Metrics.modularity(adj_sparse, assignments)

        print(f"Modularity for k={k}: {mod:.4f}")

        if mod > best_modularity:
            best_modularity = mod
            best_k = k
            best_labels = pred_labels

    print(f"Best k={best_k} with modularity {best_modularity:.4f}")
    return best_k, best_labels



def clustering(feature, n_clusters, kmeans_device='cpu', batch_size=100000,
               tol=1e-4, device=torch.device('cuda:0'), spectral_clustering=False):
    
    """
    Clustering method from MAGI

    Parameters
    ----------
    feature : torch.Tensor
        Node feature matrix, shape [N, K].

    n_clusters : int
        Number of clusters.

    kmeans_device : str, optional
        Device to run k-means on ('cpu' or 'cuda'). Ignored if spectral_clustering=True.
        Default: 'cpu'.

    batch_size : int, optional
        Batch size for k-means. 
        Default: 100000.

    tol : float, optional
        Tolerance for convergence in k-means. 
        Default: 1e-4.

    device : torch.device, optional
        Device for computing: 'cuda', 'cpu'
        Default: torch.device('cuda:0').

    spectral_clustering : bool, optional
        Using spectral clustering instead of k-means. 
        Default: False.

    Returns
    -------
    tuple
        predict_labels : np.ndarray
            Cluster assignments, shape [N].

        cluster_centers : torch.Tensor or None
            Cluster centroids (for k-means), or None for spectral clustering.
    """

    if spectral_clustering:
        if isinstance(feature, torch.Tensor):
            feature = feature.numpy()
        print("spectral clustering on cpu...")
        patch_sklearn()
        Cluster = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=0)
        f_adj = np.matmul(feature, np.transpose(feature))
        predict_labels = Cluster.fit_predict(f_adj)
        cluster_centers = None
    else:
        if kmeans_device == 'cuda':
            if isinstance(feature, np.ndarray):
                feature = torch.tensor(feature)
            print("kmeans on gpu...")
            predict_labels, cluster_centers = kmeans(
                X=feature, num_clusters=n_clusters, batch_size=batch_size, tol=tol, device=device)
            predict_labels = predict_labels.numpy()
        else:
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
            print("kmeans on cpu...")
            patch_sklearn()
            Cluster = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
            predict_labels = Cluster.fit_predict(feature)
            cluster_centers = torch.tensor(Cluster.cluster_centers_, dtype=torch.float32)

    return predict_labels, cluster_centers