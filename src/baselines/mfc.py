import networkx as nx
import torch
import sys,os
import pickle
import numpy as np
import time
from pathlib import Path

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


def load_graphs_from_tensors(adj_matrices,
                             labels_list,
                             network_type: str,
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

    if isinstance(labels_list, torch.Tensor) and labels_list.dim() == 2:
        label_snapshots = [labels_list[t] for t in range(labels_list.size(0))]
    elif isinstance(labels_list, list):
        label_snapshots = labels_list
    else:
        raise ValueError("labels_list should be list of tensor")

    if len(label_snapshots) == 1 and len(adj_list) > 1:
        label_snapshots = label_snapshots * len(adj_list)

    graph_snapshots = []
    labels_dicts = []
    for t, (adj_t, labels_t) in enumerate(zip(adj_list, label_snapshots)):
        adj_dense = _to_dense(adj_t)

        if adj_dense.dim() != 2 or adj_dense.size(0) != adj_dense.size(1):
            raise ValueError(f"Матрица снапшота {t} должна быть квадратной NxN")
        if labels_t.dim() != 1 or labels_t.size(0) != adj_dense.size(0):
            raise ValueError(f"Метки снапшота {t} должны быть длины N")

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
    return snapshots, num_classes

def load_graphs(file_name, network_type, adj_matrix=None, labels=None):
    if file_name == "from_tensor":
        if adj_matrix is None or labels is None:
            raise ValueError("Для 'from_tensor' нужно передать adj_matrix и labels.")
        return load_graphs_from_tensors(
            adj_matrices=adj_matrix,
            labels_list=labels,
            network_type=network_type,
            file_name=file_name,
        )
    else:
        raise NameError

def main(network_type, adj_matrix, labels):
    model_init = InitModel(device = "cuda")
    snapshot_list, n_cluster = load_graphs("from_tensor", 
                                           network_type=network_type, 
                                           adj_matrix=adj_matrix, 
                                           labels=labels)
    print(len(snapshot_list))
    args = Args(n_cluster, "from_tensor", network_type) # fix 20 cluster or assume known n_cluster
    model_list = []
    dgm_list = []
    wrcf_layer_dim0 = WrcfLayer(dim=0, card=args.card)
    wrcf_layer_dim1 = WrcfLayer(dim=1, card=args.card)

    results_raw = [] 
    results_topo = []
    # base deep clustering training
    for idx, (adj,features,labels) in enumerate(snapshot_list):
        model = model_init(network_type, adj, features.size(1), args)
        model_list.append(model)
        base_train(network_type,
                   model,
                   features,
                   adj,
                   args,
                   str(idx))
        with torch.no_grad():
            if network_type == "SDCN":
                _, Q, _, Z = model(features,adj)
            else:
                _, Z, Q = model(features,adj)
            results_raw.append([
                Z.cpu().detach().numpy(),
                Q.cpu().detach().numpy(),
                adj,labels
            ])
            # record dgm at each time step
            community_graph = build_community_graph(Q,adj)
            dgm0 = wrcf_layer_dim0(community_graph)
            dgm1 = wrcf_layer_dim1(community_graph)
            dgm_list.append([dgm0,dgm1])

    # topological regulaized training
    for t in range(len(snapshot_list)):
        m = model_list[t]
        adj,features,labels = snapshot_list[t]
        
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
                adj,labels
            ])
            # update dgm at time 
            community_graph = build_community_graph(Q,adj)
            dgm0_new = wrcf_layer_dim0(community_graph)
            dgm1_new = wrcf_layer_dim1(community_graph)
            dgm_list[t] = [dgm0_new,dgm1_new]

    from pathlib import Path
    out_dir = Path(PROJECT_PATH) / "results" / "mfc"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results_raw.pkl").open("wb") as handle:
        pickle.dump(results_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (out_dir / "results_topo.pkl").open("wb") as handle:
        pickle.dump(results_topo, handle, protocol=pickle.HIGHEST_PROTOCOL)

def mfc_adopted(
    adj: torch.Tensor,
    labels: torch.Tensor | None = None,
    network_type: str = "MFC",
    return_labels: bool = False,
    timing_info: dict | None = None,
):
    """
    Запуск MFC-TopoReg на одном графе.

    Parameters
    ----------
    adj : torch.Tensor
        Adjacency matrix [N, N], sparse или dense.
    labels : torch.Tensor or None
        Начальные метки [N]. Если None, строятся псевдо-кластеры по степеням.
    network_type : str
        MFC/GEC/DAEGC/SDCN.
    return_labels : bool
        Если True, вернуть кластерные метки для узлов.
    timing_info : dict or None
        Словарь, куда накапливается conversion_time.
    """

    if timing_info is None:
        timing_info = {}

    t0 = time.time()
    adj_bin = _binarize_adj(adj)
    if labels is None:
        init_labels = _degree_bins_labels(adj_bin)
    else:
        init_labels = labels.to(torch.long)
    t1 = time.time()
    timing_info["conversion_time"] = timing_info.get("conversion_time", 0.0) + (t1 - t0)

    adj_matrices = [adj_bin]
    labels_list = [init_labels]

    t0 = time.time()
    main(
        network_type=network_type,
        adj_matrix=adj_matrices,
        labels=labels_list,
    )
    t1 = time.time()
    timing_info["conversion_time"] += (t1 - t0)

    out_dir = Path(PROJECT_PATH) / "results" / "mfc"
    
    if return_labels:
        raw_pkl = out_dir / "results_raw.pkl"
        assert raw_pkl.is_file(), "results_raw.pkl not found"
        
        with raw_pkl.open("rb") as f:
            raw = pickle.load(f)
        assert isinstance(raw, list) and len(raw) >= 1
        
        snap = raw[0]
        Z, Q, adj_out, labels_out = snap
        
        import numpy as np
        Q = np.asarray(Q)
        
        labels = torch.tensor(np.argmax(Q, axis=1), dtype=torch.long)
        
        # Очищаем результаты (опционально, чтобы не засорять диск)
        # (out_dir / "results_raw.pkl").unlink(missing_ok=True)
        # (out_dir / "results_topo.pkl").unlink(missing_ok=True)
        assert labels.shape[0] == adj_matrices[0].shape[0], \
            f"Labels size mismatch: {labels.shape[0]} vs {adj_matrices[0].shape[0]}"

        return labels

    else:
        raw_pkl = out_dir / "results_raw.pkl"
        assert raw_pkl.is_file(), "results_raw.pkl not found after main()"
        
        with raw_pkl.open("rb") as f:
            raw = pickle.load(f)
        print("MFC completed successfully!")
        print(f"Results saved to: {out_dir}")
        print(f"- results_raw.pkl: {len(raw)} snapshots")
        print(f"- results_topo.pkl: topological regularized results")
        return None

def _load_from_cli(adj_root: str, dataset_name: str | None):
    from datasets import Dataset

    if dataset_name is None:
        raise SystemExit("For MFC CLI please provide --dataset-name DATASET_KEY")

    ds = Dataset(dataset_name, path=adj_root)
    adj, features, labels = ds.load(tensor_type="coo")

    # 1) локально приводим adjacency к бинарной для MFC
    adj = _binarize_adj(adj)

    # 2) метки: реальные если есть, иначе псевдо-кластеры по структуре
    if labels is None:
        labels = _degree_bins_labels(adj)
    else:
        labels = labels.to(torch.long)

    return adj, labels

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", type=str, required=True,
                    help="Root directory with datasets (used by Dataset)")
    ap.add_argument("--dataset-name", type=str, required=True,
                    help="Dataset key (name) for Dataset")
    ap.add_argument("--network-type", type=str, default="MFC",
                    help="MFC/GEC/DAEGC/SDCN")
    ap.add_argument("--snapshots", type=int, default=1,
                    help="How many identical snapshots to build from loaded graph")
    args = ap.parse_args()

    # 1) загрузка через Dataset
    adj, labels = _load_from_cli(args.adj, args.dataset_name)

    # 2) размножаем при необходимости
    for _ in range(max(1, int(args.snapshots))):
        _ = mfc_adopted(
            adj=adj,
            labels=labels,
            network_type=args.network_type,
            return_labels=False,
        )


# network_type = "MFC" # GEC/DAEGC/MFC/SDCN
# adj0 = torch.tensor([
#     [0,1,0,0],
#     [1,0,1,0],
#     [0,1,0,0],
#     [0,0,0,0],
# ], dtype=torch.float32)

# adj1 = torch.tensor([
#     [0,1,0,0],
#     [1,0,1,1],
#     [0,1,0,1],
#     [0,1,1,0]
# ], dtype=torch.float32)


# labels0 = torch.tensor([0,1,0,0])
# labels1 = torch.tensor([0,1,0,0])
# main(network_type=network_type,
#      adj_matrix=[adj0, ], 
#      labels=[labels0, ])