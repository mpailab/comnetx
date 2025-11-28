import networkx as nx
import torch
import sys,os
import pickle

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MFC_root = os.path.join(PROJECT_PATH, "baselines", "MFC-TopoReg")
if MFC_root not in sys.path:
    sys.path.insert(0, MFC_root)

from Code.train import base_train, retrain_with_topo
from Code.dataloader import get_complete_graphs, NetworkSnapshots
from Models.GraphFiltrationLayer import WrcfLayer,build_community_graph
from Experiments.main import Args, InitModel

import torch
import networkx as nx

def load_graphs_from_dynamic_tensors(adj_matrices, 
                                     labels_list, 
                                     network_type, 
                                     file_name, 
                                     complete_graph=False):
    
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
        if adj_t.dim() != 2 or adj_t.size(0) != adj_t.size(1):
            raise ValueError(f"Матрица снапшота {t} должна быть квадратной NxN")
        if labels_t.dim() != 1 or labels_t.size(0) != adj_t.size(0):
            raise ValueError(f"Метки снапшота {t} должны быть длины N")

        g = nx.from_numpy_array(adj_t.cpu().numpy())
        labels_dict = {i: int(labels_t[i].item()) for i in range(len(labels_t))}
        graph_snapshots.append(g)
        labels_dicts.append(labels_dict)

    if complete_graph:
        graph_snapshots = get_complete_graphs(graph_snapshots)

    all_labels = torch.cat(label_snapshots)
    num_classes = len(torch.unique(all_labels))

    snapshots = NetworkSnapshots(graph_snapshots, labels_dicts[0], network_type, file_name)

    return snapshots, num_classes


def load_graphs(file_name, network_type, adj_matrix=None, labels=None):
    if file_name == "from_tensor":
        if adj_matrix is None or labels is None:
            raise ValueError("Для 'from_tensor' нужно передать adj_matrix и labels.")
        return load_graphs_from_dynamic_tensors(adj_matrices=adj_matrix, 
                                                labels_list=labels, 
                                                network_type=network_type,
                                                file_name=file_name)
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

    with open("/home/egorov/comnetx/results/mfc/results_raw.pkl", 'wb') as handle:
        pickle.dump(results_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/home/egorov/comnetx/results/mfc/results_topo.pkl", 'wb') as handle:
        pickle.dump(results_topo, handle, protocol=pickle.HIGHEST_PROTOCOL)

def mfc_adopted(adj_matrices: list[torch.Tensor], labels_list: list[torch.Tensor], network_type="MFC"):
    """
    Args:
        adj_matrices: list[torch.Tensor], each one [N, N].
        labels_list: list[torch.Tensor], each one [N].
        network_type: str, optional (default="MFC"), MFC/GEC/DAEGC/MFC/SDCN.
    """
    main(network_type=network_type,
         adj_matrix=adj_matrices,
         labels=labels_list)

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