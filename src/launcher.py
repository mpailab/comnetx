import torch

from datasets import KONECT_PATH, Dataset
from optimizer import Optimizer

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