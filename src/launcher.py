import torch
import json
import os

from datasets import KONECT_PATH, INFO, Dataset
from optimizer import Optimizer

def get_true_modularities():
    with open(os.path.join(INFO, "modularity.json")) as _:
        true_mod = json.load(_)
    return true_mod

def launch_dynamic_scenario(dataset_name : str,
                            batches_num : int,
                            static_method : str):

    ds = Dataset(dataset_name, path = KONECT_PATH)
    ds.load(batches = batches_num)

    print("-------------------------")
    for i, batch in enumerate(torch.unbind(ds.adj)):
        print("Batch:", i)
        if i == 0:
            opt = Optimizer(batch, ds.features, ds.label,
                            subcoms_depth = 2,
                            method = static_method)
            nodes = batch.coalesce().indices().unique()
            nodes_mask = torch.zeros(batch.size()[0], dtype=torch.bool)
            nodes_mask[nodes] = True
        else:
            nodes_mask = opt.update_adj(batch)
        nodes_mask = opt.neighborhood(opt.adj, nodes_mask)
        opt.run(nodes_mask)
        mod = opt.modularity()
        print("Modularity:", mod)
        print("-------------------------")
    true_mod = get_true_modularities()
    print("True final modularity: ", true_mod[dataset_name])