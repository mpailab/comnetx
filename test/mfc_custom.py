import sys,os
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Путь к корню MFC-TopoReg
MFC_ROOT = os.path.join(PROJECT_PATH, "baselines", "MFC-TopoReg")

if MFC_ROOT not in sys.path:
    sys.path.insert(0, MFC_ROOT)

from Code.train import base_train, retrain_with_topo
from Code.dataloader import get_complete_graphs, NetworkSnapshots
from Models.GraphFiltrationLayer import WrcfLayer,build_community_graph
from Experiments.main import Args, InitModel
import torch
import os
from Models import *
from Code.train import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import scipy.sparse as sp
import pickle

graph_pkl = ["acm", "bat", "dblp", "eat", "uat"]
label_num_dic = {"acm": 3, "bat": 4, "dblp": 4, "eat": 4, "uat": 4}
COMPLETE_GRAPH = False
def load_graphs(file_name,network_type):
    if file_name in graph_pkl:
        return load_graphs_pkl(PROJECT_PATH + '/test/graphs/small/' + file_name +'/'+file_name,network_type)
    else:
        raise NameError  
    

def load_graphs_pkl(file_name,network_type, complete_graph=COMPLETE_GRAPH):
    with open(file_name + '.pkl', 'rb') as handle:
        try:
            graph_snapshots = pickle.load(handle, encoding='bytes', fix_imports=True)
        except ValueError:
            handle.seek(0)
            graph_snapshots = pickle.load(handle, encoding='bytes', fix_imports=True, protocol=2)
    with open(file_name + '_label.pkl', 'rb') as handle:
        try:
            labels = pickle.load(handle, encoding='bytes', fix_imports=True)
        except ValueError:
            handle.seek(0)
            labels = pickle.load(handle, encoding='bytes', fix_imports=True, protocol=2)
    
    print("Lengths of snapshots:", len(graph_snapshots))
    print("Types of labels:", label_num_dic[file_name.split('/')[-1]])
    if file_name == "DBLP":
        graph_snapshots = graph_snapshots[:8] # take first 8 snapshots in DBLP for GPU memory limit
    if complete_graph:
        graph_snapshots = get_complete_graphs(graph_snapshots)
    return NetworkSnapshots(graph_snapshots,labels,network_type,file_name), label_num_dic[file_name.split('/')[-1]]

def main(file_name, network_type):
    model_init = InitModel(device = "cuda")
    snapshot_list, n_cluster = load_graphs(file_name=file_name, network_type=network_type)
    print(len(snapshot_list))
    args = Args(n_cluster, file_name, network_type) # fix 20 cluster or assume known n_cluster
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
        # if t == 0:
        #     gt_dgm = [None, dgm_list[t+1]]
        # elif t == len(snapshot_list)-1: 
        #     gt_dgm = [dgm_list[t-1], None]
        # else:
        #     gt_dgm = [dgm_list[t-1],dgm_list[t+1]]

        gt_dgm = [None, dgm_list[t]] #FIXME

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

    results_root = os.path.join(PROJECT_PATH, "results")
    results_dir = os.path.join(results_root, file_name)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "results_raw.pkl"), "wb") as handle:
        pickle.dump(results_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(results_dir, "results_topo.pkl"), "wb") as handle:
        pickle.dump(results_topo, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    torch.manual_seed(42)
    network = "MFC" # GEC/DAEGC/MFC/SDCN
    graph_pkl = ["acm", "bat", "dblp", "eat", "uat"]
    for g in graph_pkl:
        print(g)
        main(g, network_type=network)

# def mfc():
#     torch.manual_seed(42)
#     network = "MFC" # GEC/DAEGC/MFC/SDCN
#     g = "dblp"    
#     main(g, network_type=network)