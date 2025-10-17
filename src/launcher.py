import torch
import json
import os
import time

from datasets import KONECT_PATH, INFO, Dataset
from optimizer import Optimizer

def get_true_modularities():
    with open(os.path.join(INFO, "modularity.json")) as _:
        true_mod = json.load(_)
    return true_mod

def dynamic_launch(dataset_name : str, batches_num : int,
                    underlying_static_method : str,
                    mode : str = "smart",
                    smart_subcoms_depth : int = 5, smart_neighborhood_step : int = 1):

    ds = Dataset(dataset_name, path = KONECT_PATH)
    ds.load(batches = batches_num)

    print("-----------------------------------------------")
    print(f"Dataset: {dataset_name} ({batches_num} batches)")
    print(f"Baseline: {underlying_static_method}-{mode}")
    results = []
    for i, batch in enumerate(torch.unbind(ds.adj)):
        if i == 0:
            opt = Optimizer(batch, ds.features, ds.label,
                            subcoms_depth = smart_subcoms_depth if mode == "smart" else 1,
                            method = underlying_static_method)
            active_nodes = batch.coalesce().indices().unique()
            affected_nodes_mask = torch.zeros(opt.nodes_num, dtype=torch.bool)
            affected_nodes_mask[active_nodes] = True
        else:
            affected_nodes_mask = opt.update_adj(batch)
        
        time_s = time.time()
        if mode == "smart":
            affected_nodes_mask = opt.neighborhood(opt.adj, affected_nodes_mask, step = smart_neighborhood_step)
            opt.run(affected_nodes_mask)
        elif mode == "naive" or mode == "raw":
            labels = opt.coms if mode == "naive" else None
            coms = opt.local_algorithm(opt.adj, opt.features, labels = labels)
            opt._set_communities(communities = coms.unsqueeze(0), replace_subcoms_depth = True)
        time_e = time.time()

        mod = opt.modularity()
        results.append({'modularity' : mod, 'time': time_e - time_s})

    true_mod = get_true_modularities()
    print("Final modularity: ", mod)
    print("True final modularity: ", true_mod[dataset_name])
    print("-----------------------------------------------")
    return results