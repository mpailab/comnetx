import torch
import json
import os

from datasets import KONECT_PATH, INFO, Dataset
from optimizer import Optimizer

def get_true_modularities():
    with open(os.path.join(INFO, "modularity.json")) as _:
        true_mod = json.load(_)
    return true_mod

def dynamic_launch(dataset_name : str, batches_num : int,
                    underlying_static_method : str,
                    mode : str = "smart",
                    subcoms_depth : int = 2, neighborhood_step : int = 1):

    ds = Dataset(dataset_name, path = KONECT_PATH)
    ds.load(batches = batches_num)

    print("-------------------------")
    results = []
    for i, batch in enumerate(torch.unbind(ds.adj)):
        print("Batch:", i)
        if i == 0:
            opt = Optimizer(batch, ds.features, ds.label,
                            subcoms_depth = subcoms_depth,
                            method = underlying_static_method)
            active_nodes = batch.coalesce().indices().unique()
            affected_nodes_mask = torch.zeros(opt.nodes_num, dtype=torch.bool)
            affected_nodes_mask[active_nodes] = True
        else:
            affected_nodes_mask = opt.update_adj(batch)
        
        if mode == "smart":
            affected_nodes_mask = opt.neighborhood(opt.adj, affected_nodes_mask, step = neighborhood_step)
            opt.run(affected_nodes_mask)
        else:
            pass

        mod = opt.modularity()
        results.append({'modularity' : mod})
        print("Modularity:", mod)
        print("-------------------------")
    true_mod = get_true_modularities()
    print("True final modularity: ", true_mod[dataset_name])
    return results