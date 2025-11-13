import argparse
import sys
import os.path
import torch
import argparse
import numpy as np
import random
import time

#print(torch.cuda.is_available())
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("Used divice:", device)

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))
sys.path.append(os.path.join(PROJECT_PATH, "baselines"))
sys.path.append(os.path.join(PROJECT_PATH, "baselines", "PRGPT"))

from datasets import Dataset, KONECT_PATH
from PRGPT.modules.X1 import MDL_X1, get_edge_ind_est
from PRGPT.PRGPT_static import get_sp_GCN_sup, get_sp_adj, get_rand_proj_mat, rand_seed_gbl, get_red_feat
from PRGPT.PRGPT_static import get_init_res, InfoMap_rfn, locale_rfn, clus_reorder
from PRoCD.utils import get_mod_mtc

def to_com_tensor(clus_res, origin_num_nodes, reverse_mapping):
    com = {}
    for node_new_id, com_id in enumerate(clus_res):
        node_old_id = reverse_mapping[node_new_id]
        com[node_old_id] = com_id
    additional_com_id = max(com.values()) + 1
    for node_id in range(origin_num_nodes):
        if node_id not in com:
            com[node_id] = additional_com_id
            additional_com_id += 1
    return torch.tensor([com[id] for id in range(origin_num_nodes)], dtype=torch.long)

def rough_prgpt(adj : torch.Tensor, 
#                device=None,
                refine=None):
    
    """
    PRGPT method "as is"

    Parameters
    ----------
    refine : str
        Type of refine algorithm: infomap, locale
        Default: None

    """
    
    # Layer configurations & parameter settings
    emb_dim = 32 # Embedding dimensionality
    num_feat_lyr = 2 # Number of MLP layers in feat extraction module
    num_GNN_lyr = 2 # Number of GNN layers
    num_MLP_lyr_tmp = 4 # Number of MLP layers in binary classifier

    # ====================
    inx = adj.coalesce().indices()
    origin_num_nodes = adj.size()[0]
    tst_edges = list(zip(inx[0].tolist(), inx[1].tolist()))
    tst_edges = list(filter(lambda x: x[0] >= x[1], tst_edges))

    all_nodes = set()
    for u, v in tst_edges:
        all_nodes.add(u)
        all_nodes.add(v)
    node_mapping = {old_node: new_node for new_node, old_node in enumerate(all_nodes)}
    reverse_mapping = {new_node: old_node for new_node, old_node in enumerate(all_nodes)}

    # ====================
    tst_edges = [(node_mapping[u], node_mapping[v]) for u, v in tst_edges]
    tst_num_nodes = len(all_nodes)
    tst_num_edges = len(tst_edges)
    # ==========
    tst_degs = [0 for _ in range(tst_num_nodes)]
    tst_src_idxs = []
    tst_dst_idxs = []
    for (src, dst) in tst_edges:
        # ==========
        tst_degs[src] += 1
        tst_degs[dst] += 1
        # ==========
        tst_src_idxs.append(src)
        tst_dst_idxs.append(dst)
    tst_edges_inf = tst_edges.copy()
    tst_degs_tnr = torch.FloatTensor(tst_degs).to(device)

    # ====================
    # Load the pre-trained model
    mdl = MDL_X1(emb_dim, num_feat_lyr, num_GNN_lyr, num_MLP_lyr_tmp, drop_rate=0.0).to(device)
    mdl_pars_path = os.path.join(PROJECT_PATH, "baselines", "PRGPT", "chpt", "X1_mdl_100.pt")
    mdl.load_state_dict(torch.load(mdl_pars_path))
    mdl.eval()

    # ====================
    idxs, vals = get_sp_adj(tst_edges)
    idxs_tnr = torch.LongTensor(idxs).to(device)
    vals_tnr = torch.FloatTensor(vals).to(device)
    ptn_sp_adj_tnr = torch.sparse_coo_tensor(idxs_tnr.t(), vals_tnr, 
                                             torch.Size([tst_num_nodes, tst_num_nodes]), 
                                             dtype=torch.float, device=device)

    # ==========
    idxs, vals = get_sp_GCN_sup(tst_edges, tst_degs)
    idxs_tnr = torch.LongTensor(idxs).to(device)
    vals_tnr = torch.FloatTensor(vals).to(device)
    sup_tnr = torch.sparse_coo_tensor(idxs_tnr.t(), vals_tnr, 
                                      torch.Size([tst_num_nodes, tst_num_nodes]), 
                                      dtype=torch.float, device=device)
    
    # ====================
    # Feat ext via Gaussian rand proj
    time_s = time.time()
    # ==========
    rand_mat = get_rand_proj_mat(tst_num_nodes, emb_dim, rand_seed=rand_seed_gbl)
    rand_mat_tnr = torch.FloatTensor(rand_mat).to(device)
    red_feat_tnr = get_red_feat(ptn_sp_adj_tnr,
                                torch.reshape(tst_degs_tnr, (-1, 1)),
                                rand_mat_tnr, tst_num_edges)
    # ==========
    time_e = time.time()
    feat_time = time_e - time_s
    # ==========
    del ptn_sp_adj_tnr, rand_mat_tnr

    # ====================
    # One FFP of the model
    time_s = time.time()
    # ==========
    emb_tnr, lft_tmp_tnr, rgt_tmp_tnr = mdl(red_feat_tnr, sup_tnr)
    edge_ind_est = get_edge_ind_est(emb_tnr, lft_tmp_tnr, rgt_tmp_tnr, tst_src_idxs, tst_dst_idxs)
    # ==========
    time_e = time.time()
    FFP_time = time_e - time_s
    del emb_tnr, lft_tmp_tnr, rgt_tmp_tnr

    # ====================
    # Result derivation
    time_s = time.time()
    # ==========
    clus_res_init, init_graph, init_edges, init_node_map, init_num_nodes = \
        get_init_res(edge_ind_est, tst_edges_inf)
    # ==========
    time_e = time.time()
    init_time = time_e - time_s
    clus_res_init_, num_clus_est = clus_reorder(tst_num_nodes, clus_res_init)
    mod_init = get_mod_mtc(tst_edges, clus_res_init_, num_clus_est)
    print('INIT EST-K %d MOD %.4f' % (num_clus_est, mod_init))
    if refine is None:
        return to_com_tensor(clus_res_init_, origin_num_nodes, reverse_mapping)
    elif refine == "infomap":
        # Online refinement via InfoMap
        time_s = time.time()
        clus_res_IM = InfoMap_rfn(init_edges, init_node_map, init_num_nodes, clus_res_init, tst_num_nodes)
        time_e = time.time()
        rfn_time_IM = time_e - time_s
        # Evaluation for PR-GPT w/ InfoMap
        time_IM = feat_time + FFP_time + init_time + rfn_time_IM
        clus_res_IM, num_clus_IM = clus_reorder(tst_num_nodes, clus_res_IM)
        mod_IM = get_mod_mtc(tst_edges, clus_res_IM, num_clus_IM)
        print('InfoMap EST-K %d MOD %.4f TIME %.4f (%.4f %.4f %.4f %.4f)'
            % (num_clus_IM, mod_IM, time_IM, feat_time, FFP_time, init_time, rfn_time_IM))
        return to_com_tensor(clus_res_IM, origin_num_nodes, reverse_mapping)
    elif refine == "locale":
        # Online refinement via Locale
        time_s = time.time()
        clus_res_Lcl = locale_rfn(init_graph, init_node_map, clus_res_init, tst_num_nodes, rand_seed=0)
        time_e = time.time()
        rfn_time_Lcl = time_e - time_s
        # Evaluation for PR-GPT w/ Locale
        time_Lcl = feat_time + FFP_time + init_time + rfn_time_Lcl
        clus_res_Lcl, num_clus_Lcl = clus_reorder(tst_num_nodes, clus_res_Lcl)
        mod_Lcl = get_mod_mtc(tst_edges, clus_res_Lcl, num_clus_Lcl)
        print('Locale EST-K %d MOD %.4f TIME %.4f (%.4f %.4f %.4f %.4f)'
            % (num_clus_Lcl, mod_Lcl, time_Lcl, feat_time, FFP_time, init_time, rfn_time_Lcl))
        return to_com_tensor(clus_res_Lcl, origin_num_nodes, reverse_mapping)
    else:
        raise ValueError(f"Unsupported refine method: {refine}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--features", required=False)
    parser.add_argument("--refine", type=str, default="infomap")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    features = torch.load(args.features)

    new_labels = rough_prgpt(adj, refine=args.refine)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(new_labels, args.out)
    print("PRGPT finished successfully")


if __name__ == "__main__":
    main()