import argparse
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_sparse import SparseTensor
from sklearn.cluster import KMeans, SpectralClustering
import time
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from chemomae.clustering import VMFMixture, elbow_vmf

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SRC_PATH = os.path.join(PROJECT_PATH, "src")

MAGI_PATH = os.path.join(PROJECT_PATH, "baselines", "MAGI")

if SRC_PATH not in sys.path: sys.path.insert(0, SRC_PATH)

if MAGI_PATH not in sys.path: sys.path.insert(0, MAGI_PATH)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high")

from magi.model import Model, Encoder
from magi.utils import get_mask
from magi.neighbor_sampler import NeighborSampler
from magi.batch_kmeans_cuda import kmeans

from metrics import Metrics


def squeeze_single_batch_adj(adj: torch.Tensor) -> torch.Tensor:
    if isinstance(adj, torch.Tensor) and adj.layout == torch.sparse_coo:
        adj = adj.coalesce()
        if adj.ndim == 3 and adj.shape[0] == 1:
            idx = adj.indices()
            vals = adj.values()
            return torch.sparse_coo_tensor(
                idx[1:],
                vals,
                size=adj.shape[1:],
                dtype=vals.dtype,
                device=vals.device,
            ).coalesce()
    return adj

def is_sparse_identity_features(features: torch.Tensor) -> bool:
    if not isinstance(features, torch.Tensor):
        return False
    if features.layout != torch.sparse_coo:
        return False

    feat = features.coalesce()
    if feat.dim() != 2:
        return False

    n, m = feat.shape
    if n != m or feat._nnz() != n:
        return False

    row, col = feat.indices()
    vals = feat.values()

    if not torch.equal(row, col):
        return False
    if not torch.all(vals == 1):
        return False
    if not torch.equal(torch.sort(row).values, torch.arange(n, device=row.device)):
        return False

    return True

def get_batch_features(
    features: torch.Tensor,
    n_id: torch.Tensor,
    num_features: int,
    device: torch.device,
    identity_features: bool,
) -> torch.Tensor:
    if identity_features:
        return F.one_hot(n_id.to(device), num_classes=num_features).float()

    if isinstance(features, torch.Tensor) and features.layout == torch.sparse_coo:
        feat = features.coalesce()
        row, col = feat.indices()
        val = feat.values()

        n_id_dev = n_id.to(row.device)

        # оставляем только строки, которые входят в batch
        keep = torch.isin(row, n_id_dev)
        row = row[keep]
        col = col[keep]
        val = val[keep]

        if row.numel() == 0:
            return torch.zeros((n_id.numel(), num_features), dtype=torch.float32, device=device)

        # remap global node ids -> local batch row ids
        order = torch.argsort(n_id_dev)
        sorted_n = n_id_dev[order]
        pos = torch.searchsorted(sorted_n, row)
        mapped_rows = order[pos]

        batch_sparse = torch.sparse_coo_tensor(
            torch.stack([mapped_rows, col], dim=0),
            val,
            size=(n_id.numel(), num_features),
            device=val.device,
        ).coalesce()

        return batch_sparse.to(device).to_dense()

    idx = n_id.to(features.device)
    return features.index_select(0, idx).to(device)

def magi(adj: torch.Tensor,
         features: torch.Tensor,
         labels: torch.Tensor | None = None,
         n_clusters: int | None = None,
         device=None,
         n_epochs=None,
         batchsize: int = 2048,
         timing_info=None):

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
        Default: None 

    device: torch.device or None, optional
        Device for computing: 'cuda', 'cpu'
        Default: None 

    Returns
    -------
    torch.Tensor
        Predicted cluster assignments for all nodes, shape [N].
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if n_epochs is None:
        n_epochs = 100
    class Args:
        batchsize = 2048
        max_duration = 60
        kmeans_device = 'cuda' if device.type == 'cuda' else 'cpu'
        kmeans_batch = -1
        hidden_channels = '1024,256'
        size = '10,10'
        wt = 20
        wl = 5
        tau = 0.5
        ns = 0.5
        lr = 0.05
        epochs = n_epochs
        projection = ""
        wd = 0
        dropout = 0
    args = Args()
    
    print("device: ", device)

    time_s = time.time()

    adj = squeeze_single_batch_adj(adj)

    if adj.ndim != 2:
        raise ValueError(
            f"MAGI supports only static adjacency [N, N]. Got shape {adj.shape}"
        )

    if features is None:
        raise ValueError("MAGI requires features, but got None")

    identity_features = False

    if isinstance(features, torch.Tensor) and features.layout == torch.sparse_coo:
        features = features.coalesce()
        if is_sparse_identity_features(features):
            identity_features = True
        # обычные sparse features оставляем sparse
    else:
        features = features.to(device)
        
    if n_clusters is not None:
        inferred_k = n_clusters
    elif labels is not None:
        labels = labels.to(device)
        inferred_k = len(torch.unique(labels))
    else:
        inferred_k = None

    N = adj.size(0)
    num_features = features.shape[-1]

    if isinstance(adj, torch.Tensor) and adj.layout == torch.sparse_coo:
        edge_index = adj.coalesce().indices().cpu()
    else:
        edge_index = torch.nonzero(adj.cpu(), as_tuple=False).t().contiguous()

    edge_index = add_remaining_self_loops(edge_index, num_nodes=N)[0]
    edge_index = to_undirected(edge_index, num_nodes=N).contiguous()

    new_values = torch.ones(edge_index.size(1), dtype=torch.float32)
    adj_sparse = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=new_values,
        sparse_sizes=(N, N),
    )
    time_e = time.time()
    if timing_info is not None:
        timing_info['conversion_time'] = time_e - time_s

    hidden = list(map(int, args.hidden_channels.split(',')))
    if args.projection == '':
        projection = None
    else:
        projection = list(map(int, args.projection.split(',')))
    size = list(map(int, args.size.split(',')))
    assert len(hidden) == len(size)

    N = adj_sparse.sparse_sizes()[0]
    all_nodes = torch.arange(N, device='cpu')

    if edge_index.size(1) == 0:
        return torch.arange(N, device=device, dtype=torch.long)

    train_loader = NeighborSampler(edge_index, adj_sparse,
                                   is_train=True,
                                   node_idx=all_nodes,
                                   wt=args.wt,
                                   wl=args.wl,
                                   sizes=size,
                                   batch_size=args.batchsize,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=4,
                                   num_nodes=N)

    test_loader = NeighborSampler(edge_index, adj_sparse,
                                  is_train=False,
                                  node_idx=all_nodes,
                                  sizes=size,
                                  batch_size=2048,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=4,
                                  num_nodes=N)

    encoder = Encoder(num_features, hidden_channels=hidden,
                      dropout=args.dropout, ns=args.ns).to(device)

    model = Model(
        encoder, in_channels=hidden[-1], project_hidden=projection, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    if labels is None:
        num_nodes = adj.size(0)
        labels = torch.arange(num_nodes)

    model.train()

    for epoch in range(args.epochs):
        total_loss = 0.0
        batches = 0

        for step, ((batch_size, n_id, adjs), adj_batch, batch) in enumerate(train_loader):
            adjs = [adjs] if len(hidden) == 1 else adjs
            adjs = [adj.to(device) for adj in adjs]
            adj_mask = get_mask(adj_batch)

            optimizer.zero_grad(set_to_none=True)

            batch_x = get_batch_features(
                features=features,
                n_id=n_id,
                num_features=num_features,
                device=device,
                identity_features=identity_features,
            )
            out = model(batch_x, adjs=adjs)
            out = F.normalize(out, p=2, dim=1)

            if not torch.isfinite(out).all():
                raise RuntimeError(f"Non-finite out at epoch={epoch}, step={step}")

            loss = model.loss(out, adj_mask)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, step={step}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.detach().item()
            batches += 1

            # при желании лог раз в N шагов:
            # if step % 50 == 0:
            #     print(f"(T) | Epoch {epoch:02d}, step: {step:04d}, loss: {loss.detach().item():.4f}")

        avg_loss = total_loss / batches if batches > 0 else 0.0

            

    model.eval()
    z_all = torch.zeros((N, hidden[-1]), device=device)
    with torch.no_grad():
        for (batch_size, n_id, adjs), _, batch in test_loader:
            adjs = [adjs] if len(hidden) == 1 else adjs
            adjs = [adj.to(device) for adj in adjs]
            batch_x = get_batch_features(
                features=features,
                n_id=n_id,
                num_features=num_features,
                device=device,
                identity_features=identity_features,
            )
            out = model(batch_x, adjs=adjs)
            z_all[n_id[:batch_size]] = out

    embeddings = F.normalize(z_all, p=2, dim=1)
    """
    if inferred_k is not None and inferred_k > 0:
        pred_labels, _ = clustering(
            feature=embeddings,
            n_clusters=inferred_k,
            kmeans_device=args.kmeans_device,
            batch_size=args.kmeans_batch,
            tol=1e-4,
            device=device,
            spectral_clustering=False,
        )
        new_labels = torch.as_tensor(pred_labels, dtype=torch.long, device=device)
    else:
        k_max = min(40, adj_sparse.sparse_sizes()[0])
        k_min = min(2, k_max)
        if k_max <= k_min:
            num_clusters = k_max
            new_labels = torch.zeros(adj_sparse.sparse_sizes()[0],
                                     dtype=torch.long, device=device)
        else:
            num_clusters, new_labels = find_best_k_with_modularity(
                adj_sparse, embeddings, range(k_min, k_max), device=device
            )
        if not isinstance(new_labels, torch.Tensor):
            new_labels = torch.tensor(new_labels, dtype=torch.long, device=device)
        else:
            new_labels = new_labels.to(dtype=torch.long, device=device)
    """

    num_points = embeddings.size(0)

    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
        num_points = 1

    if num_points <= 1:
        return torch.zeros(num_points, dtype=torch.long, device=device)

    if inferred_k is None:
        inferred_k = estimate_k_elbow_vmf(
            embeddings=embeddings,
            device=device,
            k_max=min(50, num_points),
            random_state=42,
            criterion="bic",
        )
        print(f"Estimated number of clusters by elbow_vmf: k={inferred_k}")

    inferred_k = int(max(1, min(inferred_k, num_points)))

    if inferred_k == 1:
        return torch.zeros(num_points, dtype=torch.long, device=device)

    pred_labels, _ = clustering(
        feature=embeddings,
        n_clusters=inferred_k,
        kmeans_device=args.kmeans_device,
        batch_size=args.kmeans_batch,
        tol=1e-4,
        device=device,
        spectral_clustering=False,
    )
    new_labels = torch.as_tensor(pred_labels, dtype=torch.long, device=device)
    return new_labels

def estimate_k_elbow_vmf(
    embeddings: torch.Tensor,
    device: torch.device,
    k_max: int = 50,
    random_state: int = 42,
    criterion: str = "bic",
) -> int:
    X = embeddings.detach().cpu()
    X = F.normalize(X, p=2, dim=1)

    num_points = X.size(0)
    if num_points <= 1:
        return 1

    k_max = min(k_max, num_points)

    k_list, scores, K, idx, kappa = elbow_vmf(
        VMFMixture,
        X,
        device="cpu",
        k_max=k_max,
        chunk=5_000_000,
        random_state=random_state,
        criterion=criterion,
    )
    return int(K)

def estimate_k_eigengap_sparse_embeddings(
    embeddings: torch.Tensor,
    k_max: int = 30,
    knn_k: int = 20,
    metric: str = "cosine",
    mutual: bool = False,
    zero_tol: float = 1e-6,
) -> int:
    """
    Sparse eigengap estimation on embeddings.

    Steps:
    1) build sparse kNN similarity graph from embeddings
    2) build normalized Laplacian L = I - D^{-1/2} W D^{-1/2}
    3) compute several smallest eigenvalues
    4) choose k by eigengap

    Parameters
    ----------
    embeddings : torch.Tensor [N, D]
    k_max : int
        Maximum number of clusters to consider.
    knn_k : int
        Number of neighbors in sparse similarity graph.
    metric : str
        "cosine" or "euclidean"
    mutual : bool
        If True, use mutual kNN; otherwise symmetrize with max.
    zero_tol : float
        Threshold for detecting zero eigenvalues.

    Returns
    -------
    int
        Estimated number of clusters.
    """
    x = embeddings.detach().float().cpu().numpy()
    n = x.shape[0]

    if n <= 1:
        return 1
    if n == 2:
        return 2

    k_max = max(2, min(k_max, n - 1))
    knn_k = max(2, min(knn_k, n - 1))

    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        x = x / norms

    nn = NearestNeighbors(
        n_neighbors=knn_k + 1,
        metric=metric,
        algorithm="auto",
        n_jobs=-1,
    )
    nn.fit(x)

    # sparse distance graph
    W = nn.kneighbors_graph(x, mode="distance")
    W = W.tocsr()

    # remove self-loops from kNN stage
    W.setdiag(0.0)
    W.eliminate_zeros()

    # distance -> similarity
    if W.nnz == 0:
        return 1

    if metric == "cosine":
        # cosine distance in sklearn = 1 - cosine_similarity
        W.data = 1.0 - W.data
    elif metric == "euclidean":
        sigma = np.median(W.data)
        sigma = max(float(sigma), 1e-12)
        W.data = np.exp(-(W.data ** 2) / (2.0 * sigma * sigma))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    W.data = np.clip(W.data, 0.0, None)
    W.eliminate_zeros()

    # symmetrization
    if mutual:
        W = W.minimum(W.T).tocsr()
    else:
        W = W.maximum(W.T).tocsr()

    if W.nnz == 0:
        return 1

    degrees = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
    mask = degrees > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])

    D_inv_sqrt = sp.diags(d_inv_sqrt, format="csr")
    I = sp.eye(n, format="csr", dtype=np.float64)
    L = I - D_inv_sqrt @ W @ D_inv_sqrt

    # how many eigenvalues to compute
    n_eigs = min(k_max + 1, n - 1)
    if n_eigs < 2:
        return 1

    try:
        evals = eigsh(L, k=n_eigs, which="SM", return_eigenvectors=False)
    except Exception:
        # safe fallback
        return min(2, n)

    evals = np.sort(np.real(evals))
    evals = np.clip(evals, 0.0, None)

    # if graph is nearly disconnected, multiplicity of zero eigenvalue is informative
    n_zero = int(np.sum(evals < zero_tol))
    if n_zero >= 2:
        return min(n_zero, k_max)

    gaps = np.diff(evals)
    if gaps.size == 0:
        return 1

    k_hat = int(np.argmax(gaps) + 1)
    return max(1, min(k_hat, k_max))

def estimate_k_eigengap(embeddings: torch.Tensor, k_max: int = 30):
    x = embeddings.detach()
    x = torch.nn.functional.normalize(x, dim=1)

    sim = x @ x.T
    sim = torch.clamp(sim, min=0)

    deg = sim.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-8))
    L = torch.eye(sim.size(0), device=sim.device) - D_inv_sqrt @ sim @ D_inv_sqrt

    evals = torch.linalg.eigvalsh(L)
    evals = evals[:k_max + 1]
    gaps = evals[1:] - evals[:-1]
    k = int(torch.argmax(gaps[:k_max - 1]).item()) + 1
    return k


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

    row, col, val = adj_sparse.coo()
    size = adj_sparse.sizes()
    adj_torch = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        val,
        size=size,
        device=val.device
    ).coalesce()

    for k in k_range:
        pred_labels, cluster_centers = clustering(
            feature=embeddings,
            n_clusters=k,
            kmeans_device=device.type,
            batch_size=-1,
            tol=1e-4,
            device=device,
            spectral_clustering=False
        )

        assignments = torch.as_tensor(pred_labels, device=adj_torch.device, dtype=torch.long)
        mod = Metrics.modularity(adj_torch, assignments)

        print(f"Modularity for k={k}: {mod:.4f}")

        if mod > best_modularity:
            best_modularity = mod
            best_k = k
            best_labels = pred_labels

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
                feature = feature.detach().cpu().numpy()
            print("kmeans on cpu...")
            Cluster = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
            predict_labels = Cluster.fit_predict(feature)
            cluster_centers = torch.tensor(
                Cluster.cluster_centers_, dtype=torch.float32
            )

    return predict_labels, cluster_centers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    features = torch.load(args.features)

    new_labels = magi(
        adj,
        features,
        n_epochs=args.epochs,
        batchsize=args.batchsize,
        n_clusters=args.n_clusters,
    )

    torch.save(new_labels, args.out)
    print("MAGI finished successfully")

if __name__ == "__main__":
    main()