import networkx as nx
import torch
import sys,os
import pickle
import numpy as np
import time
from pathlib import Path
import warnings

# for warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
warnings.filterwarnings("ignore")

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(PROJECT_PATH, "src")
MFC_root = os.path.join(PROJECT_PATH, "baselines", "MFC-TopoReg")
for p in (SRC_PATH, MFC_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from Code.train import base_train, retrain_with_topo
from Code.dataloader import get_complete_graphs, NetworkSnapshots
from Models.GraphFiltrationLayer import WrcfLayer,build_community_graph
from Experiments.main import Args, InitModel

def _degree_bins_labels(adj: torch.Tensor, k_min: int = 2, k_max: int = 20) -> torch.Tensor:
    """
    Строит псевдо-кластеры по степеням вершин.
    adj: sparse или dense [N,N].
    Возвращает labels [N] с небольшим числом кластеров.
    """
    A = adj.coalesce() if adj.is_sparse else adj
    if A.is_sparse:
        rows = A.indices()[0]
        N = A.size(0)
        deg = torch.bincount(rows, minlength=N).float()
    else:
        deg = A.sum(dim=1).float()

    N = deg.numel()
    k = int(min(k_max, max(k_min, N ** 0.5)))
    if k <= 1:
        return torch.zeros(N, dtype=torch.long)

    deg_clamped = torch.clamp(deg, min=1.0)
    log_deg = torch.log(deg_clamped)
    qs = torch.quantile(log_deg, torch.linspace(0, 1, steps=k + 1, device=log_deg.device))
    labels = torch.bucketize(log_deg, qs[1:-1], right=True)
    return labels.to(torch.long)

def _binarize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Делает из произвольной разреженной/плотной матрицы смежности бинарную {0,1}
    и обнуляет диагональ. Возвращает тензор того же формата (sparse или dense).
    """
    if adj.is_sparse:
        A = adj.coalesce()
        idx = A.indices()
        vals = A.values()
        # выкидываем self-loops
        mask = idx[0] != idx[1]
        idx = idx[:, mask]
        vals = vals[mask]
        # всё положительное считаем ребром
        vals = torch.where(vals > 0, torch.ones_like(vals), torch.zeros_like(vals))
        return torch.sparse_coo_tensor(idx, vals, size=A.size())
    else:
        A = adj.clone()
        A.fill_diagonal_(0.0)
        A = (A > 0).to(torch.float32)
        return A


def _to_dense(adj_t: torch.Tensor) -> torch.Tensor:
    if adj_t.is_sparse:
        return adj_t.to_dense()
    return adj_t

def _prepare_feature_snapshots(features, num_snapshots: int):
    if features is None:
        return None

    if isinstance(features, torch.Tensor):
        if features.dim() == 2:
            feature_snapshots = [features] * num_snapshots
        elif features.dim() == 3:
            feature_snapshots = [features[t] for t in range(features.size(0))]
        else:
            raise ValueError(
                f"features must have shape [N, F] or [T, N, F], got {tuple(features.shape)}"
            )
    elif isinstance(features, list):
        feature_snapshots = features
    else:
        raise ValueError("features must be torch.Tensor, list, or None")

    if len(feature_snapshots) == 1 and num_snapshots > 1:
        feature_snapshots = feature_snapshots * num_snapshots

    if len(feature_snapshots) != num_snapshots:
        raise ValueError(
            f"features snapshots mismatch: expected {num_snapshots}, got {len(feature_snapshots)}"
        )

    return feature_snapshots

def _normalize_initial_partition(initial_partition: torch.Tensor, nodes_num: int) -> torch.Tensor:
    if initial_partition.dim() == 2 and initial_partition.size(0) == 1:
        initial_partition = initial_partition.squeeze(0)

    if initial_partition.dim() != 1:
        raise ValueError(
            "initial_partition must be a 1D tensor with shape [N] or [1, N], "
            f"got {tuple(initial_partition.shape)}"
        )

    if initial_partition.size(0) != nodes_num:
        raise ValueError(
            f"initial_partition size mismatch: expected {nodes_num}, got {initial_partition.size(0)}"
        )

    return initial_partition.to(torch.long)


def load_graphs_from_tensors(adj_matrices,
                             labels_list,
                             features=None,
                             network_type: str = "MFC",
                             file_name: str = "from_tensor",
                             complete_graph: bool = False):
    """
    adj_matrices: list[Tensor] или 3D Tensor [T,N,N], sparse или dense.
    labels_list:  list[Tensor] или 2D Tensor [T,N].
    Возвращает snapshot_list, n_cluster так же, как оригинальный dataloader.
    """
    if isinstance(adj_matrices, torch.Tensor) and adj_matrices.dim() == 3:
        adj_list = [adj_matrices[t] for t in range(adj_matrices.size(0))]
    elif isinstance(adj_matrices, list):
        adj_list = adj_matrices
    else:
        raise ValueError("adj_matrices should be list of 2-dim tensor")

    # print(f"labels_list = {labels_list}")

    if isinstance(labels_list, torch.Tensor) and labels_list.dim() == 2:
        label_snapshots = [labels_list[t] for t in range(labels_list.size(0))]
    elif isinstance(labels_list, list):
        label_snapshots = labels_list
    else:
        raise ValueError("labels_list should be list of tensor")

    # print(f"label_snapshots = {label_snapshots}")


    if len(label_snapshots) == 1 and len(adj_list) > 1:
        label_snapshots = label_snapshots * len(adj_list)

    graph_snapshots = []
    labels_dicts = []

    # print(f"adj_list = {adj_list}") #my

    # adj_list, label_snapshots = size_correct(adj_list)

    # print(f"label_snapshots = {label_snapshots}") #my
    # print(f"labels_list = {labels_list}") #my
    

    for t, (adj_t, labels_t) in enumerate(zip(adj_list, label_snapshots)):
        adj_dense = _to_dense(adj_t)

        # print(f"adj_t = {adj_t}") #my

        if adj_dense.dim() != 2 or adj_dense.size(0) != adj_dense.size(1):
            raise ValueError(f"Матрица снапшота {t} должна быть квадратной NxN, adj_dense = {adj_dense}")
        # print(f"t={t}, adj_dense.shape={adj_dense.shape}, labels_t.shape={labels_t}")
        if labels_t.dim() != 1 or labels_t.size(0) != adj_dense.size(0):
            raise ValueError(f"Метки снапшота {t} должны быть длины N, adj_dense.shape={adj_dense.shape}, labels_t.shape={labels_t}")

        g = nx.from_numpy_array(adj_dense.cpu().numpy())
        labels_dict = {i: int(labels_t[i].item()) for i in range(len(labels_t))}
        graph_snapshots.append(g)
        labels_dicts.append(labels_dict)

    if complete_graph:
        graph_snapshots = get_complete_graphs(graph_snapshots)

    all_labels = torch.cat(label_snapshots)
    num_classes = int(torch.unique(all_labels).numel())
    if num_classes < 2:
        num_classes = 2

    snapshots = NetworkSnapshots(graph_snapshots, labels_dicts[0], network_type, file_name)
    snapshot_items = list(snapshots)

    feature_snapshots = _prepare_feature_snapshots(features, len(snapshot_items))

    if feature_snapshots is not None:
        patched_items = []
        for (adj, _, labels), feat in zip(snapshot_items, feature_snapshots):
            if isinstance(feat, torch.Tensor):
                feat = feat.float().to(adj.device if hasattr(adj, "device") else "cpu")
                if feat.is_sparse:
                    feat = feat.to_dense()
            patched_items.append((adj, feat, labels))
        snapshot_items = patched_items

    return snapshot_items, num_classes

def load_graphs(file_name, network_type, adj_matrix=None, labels=None, features=None):
    if file_name == "from_tensor":
        if adj_matrix is None or labels is None:
            raise ValueError("Для 'from_tensor' нужно передать adj_matrix и labels.")

        # print(f"adj_matrix = {adj_matrix}")  #my  
        return load_graphs_from_tensors(
            adj_matrices=adj_matrix,
            labels_list=labels,
            features=features,
            network_type=network_type,
            file_name=file_name,
        )
    else:
        raise NameError

def main(network_type, adj_matrix, labels, features=None, num_epoch=500, start_mf=250):
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_init = InitModel(device=str(compute_device))
    snapshot_list, n_cluster = load_graphs(
        "from_tensor",
        network_type=network_type,
        adj_matrix=adj_matrix,
        labels=labels,
        features=features,
    )
    args = Args(n_cluster, "from_tensor", network_type) # fix 20 cluster or assume known n_cluster
    args.num_epoch = num_epoch
    args.start_mf = start_mf

    model_list = []
    dgm_list = []
    wrcf_layer_dim0 = WrcfLayer(dim=0, card=args.card)
    wrcf_layer_dim1 = WrcfLayer(dim=1, card=args.card)

    results_raw = [] 
    results_topo = []
    # base deep clustering training
    for idx, (adj, features, labels) in enumerate(snapshot_list):
        if isinstance(features, torch.Tensor):
            features = features.to(compute_device)

        if isinstance(adj, torch.Tensor):
            adj = adj.to(compute_device)

        if isinstance(labels, torch.Tensor):
            labels = labels.to(compute_device)

        model = model_init(network_type, adj, features.size(1), args)
        model_list.append(model)
        base_train(
            network_type,
            model,
            features,
            adj,
            args,
            str(idx),
        )
        with torch.no_grad():
            if network_type == "SDCN":
                _, Q, _, Z = model(features,adj)
            else:
                _, Z, Q = model(features,adj)
            results_raw.append([
                Z.cpu().detach().numpy(),
                Q.cpu().detach().numpy(),
                adj.cpu() if isinstance(adj, torch.Tensor) else adj,
                labels.cpu() if isinstance(labels, torch.Tensor) else labels,
            ])
            # record dgm at each time step
            community_graph = build_community_graph(Q,adj)
            dgm0 = wrcf_layer_dim0(community_graph)
            dgm1 = wrcf_layer_dim1(community_graph)
            dgm_list.append([dgm0,dgm1])

    # topological regulaized training
    for t in range(len(snapshot_list)):
        m = model_list[t]
        adj, features, labels = snapshot_list[t]

        if isinstance(features, torch.Tensor):
            features = features.to(compute_device)

        if isinstance(adj, torch.Tensor):
            adj = adj.to(compute_device)

        if isinstance(labels, torch.Tensor):
            labels = labels.to(compute_device)
        
        if len(snapshot_list)!=1:
            # print('several snapshot')
            if t == 0:
                gt_dgm = [None, dgm_list[t+1]]
            elif t == len(snapshot_list)-1: 
                gt_dgm = [dgm_list[t-1], None]
            else:
                gt_dgm = [dgm_list[t-1],dgm_list[t+1]]
        else:
            # print('one snapshot')
            gt_dgm = [None, dgm_list[t]]

        retrain_with_topo(
            network_type,
            m,
            gt_dgm,
            adj,
            features,
            args,
            str(t)
        )
        with torch.no_grad():
            if network_type == "SDCN":
                _, Q, _, Z = m(features,adj)
            else:
                _, Z, Q = m(features,adj)
            results_topo.append([
                Z.cpu().detach().numpy(),
                Q.cpu().detach().numpy(),
                adj.cpu() if isinstance(adj, torch.Tensor) else adj,
                labels.cpu() if isinstance(labels, torch.Tensor) else labels,
            ])
            # update dgm at time 
            community_graph = build_community_graph(Q,adj)
            dgm0_new = wrcf_layer_dim0(community_graph)
            dgm1_new = wrcf_layer_dim1(community_graph)
            dgm_list[t] = [dgm0_new,dgm1_new]

    return results_raw, results_topo

def mfc_adopted(
    adj: torch.Tensor,
    features: torch.Tensor | None = None,
    network_type: str = "MFC",
    timing_info: dict | None = None,
    num_epoch = None,
    dynamic: bool = False,
    initial_partition: torch.Tensor | None = None,
):
    """
    Запуск MFC-TopoReg на одном графе.

    Parameters
    ----------
    adj : torch.Tensor
        Adjacency matrix [N, N], sparse или dense.
    network_type : str
        MFC/GEC/DAEGC/SDCN.
    timing_info : dict or None
        Словарь, куда накапливается conversion_time.
    """

    if num_epoch is None:
        num_epoch = 10

    start_mf = num_epoch // 2

    if timing_info is None:
        timing_info = {}

    t0 = time.time()
    adj_bin = _binarize_adj(adj)

    if dynamic:
        adj_matrices = adj_bin
        first_snapshot = adj_bin[0]
        if initial_partition is not None:
            init_labels = _normalize_initial_partition(
                initial_partition, first_snapshot.size(0)
            )
        else:
            init_labels = _degree_bins_labels(first_snapshot)
    else:
        adj_matrices = [adj_bin]
        init_labels = _degree_bins_labels(adj_bin)
    labels_list = [init_labels]

    t1 = time.time()
    timing_info["conversion_time"] = timing_info.get("conversion_time", 0.0) + (t1 - t0)

    raw, topo = main(
        network_type=network_type,
        adj_matrix=adj_matrices,
        labels=labels_list,
        features=features,
        num_epoch=num_epoch,
        start_mf=start_mf,
    )
    
    snap = raw[0]
    Z, Q, adj_out, labels_out = snap
    
    import numpy as np
    Q = np.asarray(Q)
    
    labels = torch.tensor(np.argmax(Q, axis=1), dtype=torch.long)

    return labels