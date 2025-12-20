import torch
import json
import os
import time

from datasets import KONECT_PATH, INFO, Dataset
from optimizer import Optimizer
from our_utils import print_zone

def dynamic_launch(dataset_name : str, batches_num : int,
                    underlying_static_method : str,
                    mode : str = "smart",
                    smart_subcoms_depth : int = 5, smart_neighborhood_step : int = 1,
                    verbose : int = 1):

    ds = Dataset(dataset_name, path = KONECT_PATH)
    ds.load(batches = batches_num)

    with print_zone(verbose >= 1):
        print("-----------------------------------------------")
        print(f"Dataset: {dataset_name} ({batches_num} batches)")
        print(f"Baseline: {underlying_static_method}-{mode}")
    results = []
    for i, batch in enumerate(torch.unbind(ds.adj)):
        with print_zone(verbose >= 2):
            print("Batch", i)
        if i == 0:
            opt = Optimizer(batch, ds.features, ds.label,
                            subcoms_depth = smart_subcoms_depth if mode == "smart" else 1,
                            method = underlying_static_method,
                            verbose = verbose)
            active_nodes = batch.coalesce().indices().unique()
            affected_nodes_mask = torch.zeros(opt.nodes_num, dtype=torch.bool)
            affected_nodes_mask[active_nodes] = True
        else:
            affected_nodes_mask = opt.update_adj(batch)
        
        acc = None
        conversion_time_s = opt.conversion_time
        time_s = time.time()
        if mode == "smart":
            affected_nodes_mask = opt.neighborhood(opt.adj, affected_nodes_mask, step = smart_neighborhood_step)
            opt.run(affected_nodes_mask)
        elif mode == "naive" or mode == "raw":
            labels = opt.coms if mode == "naive" else None
            coms = opt.local_algorithm(opt.adj, opt.features, labels = labels)
            opt._set_communities(communities = coms.unsqueeze(0), replace_subcoms_depth = True)
            acc = opt.accuracy(labels)
        time_e = time.time()
        conversion_time_e = opt.conversion_time

        total_time = time_e - time_s
        conversion_time = conversion_time_e - conversion_time_s

        mod = opt.modularity(directed = ds.is_directed)
        results.append({'modularity' : mod, 'time': total_time - conversion_time, 'accuracy': acc})

    with print_zone(verbose >= 1):
        total_time = sum(map(lambda x: x["time"], results))
        print(f"Final modularity: {mod:.2}")
        print(f"Total time: {total_time:.2}")
        print("-----------------------------------------------")
    return results