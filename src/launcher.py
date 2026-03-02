import torch
import json
import os
import time

from datasets import INFO, Dataset
from optimizer import Optimizer
from our_utils import print_zone

#KONECT_PATH = "/auto/datasets/graphs/dynamic_konect_project_datasets"

def dynamic_launch(dataset_name : str, batches_strategy,
                    underlying_static_method : str,
                    mode : str = "smart",
                    smart_subcoms_depth : int = 5, smart_neighborhood_step : int = 1,
                    verbose : int = 1):

    ds = Dataset(dataset_name)
    ds.load(batches_strategy = batches_strategy)

    with print_zone(verbose >= 1):
        print("-----------------------------------------------")
        print(f"Dataset: {dataset_name} ({batches_strategy} batches)")
        sufix = f"L:{smart_subcoms_depth}-r:{smart_neighborhood_step}" if mode == "smart" else mode
        print(f"Baseline: {underlying_static_method}-{sufix}")
    results = []
    for i, batch in enumerate(torch.unbind(ds.adj)):
        with print_zone(verbose >= 2):
            print("Batch", i)
        if i == 0:
            opt = Optimizer(batch, ds.features, ds.label,
                            subcoms_depth = smart_subcoms_depth if mode == "smart" else 1,
                            method = underlying_static_method,
                            verbose = verbose)
            if ":" in batches_strategy:
                #TODO сделать загрузку посчитанного разбиения первого батча для стратегий "9:N", "99:N", "999:N"
                # Сейчас считаем разбиение "на ходу" самым быстрым алгоритмом
                opt.method = "ldleiden"
                n = opt.nodes_num
                l = opt.subcoms_depth
                coms = opt.local_algorithm(opt.adj, opt.features)
                coms = coms.repeat(l).reshape((l, n)) # Пропагируем сообщества вверх на все уровни
                #FIXME перенести функционал выше в функцию _set_communities
                opt._set_communities(communities = coms)
                opt.method = underlying_static_method
                continue
            else:
                active_nodes = batch.coalesce().indices().unique()
                affected_nodes_mask = torch.zeros(opt.nodes_num, dtype=torch.bool)
                affected_nodes_mask[active_nodes] = True
        else:
            affected_nodes_mask = opt.update_adj(batch)
        
        conversion_time_s = opt.conversion_time
        time_s = time.time()
        if mode == "smart":
            affected_nodes_mask = opt.neighborhood(opt.adj, affected_nodes_mask, step = smart_neighborhood_step)
            opt.run(affected_nodes_mask)
        elif mode == "naive" or mode == "raw":
            #opt.adj = opt.safe_clamp_sparse(opt.adj) #For non-negative weights in leidenalg
            labels = opt.coms if mode == "naive" else None
            coms = opt.local_algorithm(opt.adj, opt.features, labels = labels)
            opt._set_communities(communities = coms.unsqueeze(0), replace_subcoms_depth = True)
        time_e = time.time()
        conversion_time_e = opt.conversion_time

        total_time = time_e - time_s
        conversion_time = conversion_time_e - conversion_time_s
        mod = opt.modularity(directed = ds.is_directed)
        with print_zone(verbose >= 2):
            print(f"Modularity: {mod:.2}")
            if underlying_static_method == "ldleiden" and mode in {"naive", "raw"}:
                algorithm_time = opt.last_timing_info["algorithm_time"]
                print(f"Algorithm time: {algorithm_time:.2}")
            else:
                print(f"Time: {total_time - conversion_time:.2}")

        results.append({'modularity' : mod, 'time': total_time - conversion_time})

    total_time = sum(map(lambda x: x["time"], results))
    with print_zone(verbose == 1):
        print(f"Final modularity: {mod:.2}")
        print(f"Total time: {total_time:.2}")
        print("-----------------------------------------------")
    with print_zone(verbose >= 2):
        print("-")
        print(f"Total time: {total_time:.2}")
        print("-----------------------------------------------")
    return results