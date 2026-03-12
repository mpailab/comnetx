import torch
import json
import os
import time

from datasets import INFO, Dataset
from optimizer import Optimizer
from our_utils import print_zone

def dynamic_launch(dataset_name : str, batches_strategy,
                    underlying_static_method : str,
                    mode : str = "smart",
                    smart_subcoms_depth : int = 5, smart_neighborhood_step : int = 1,
                    verbose : int = 1,
                    use_gpu: bool = False):

    ds = Dataset(dataset_name)
    ds.load(batches_strategy = batches_strategy)
    smart_mode = (mode == "smart")
    naive_mode = (mode == "naive")
    raw_mode = (mode == "raw")

    with print_zone(verbose >= 1):
        print("-----------------------------------------------")
        print(f"Dataset: {dataset_name} ({batches_strategy} batches)")
        gpu_sfx = "gpu" if use_gpu else "cpu"
        sufix = f"L:{smart_subcoms_depth}-r:{smart_neighborhood_step}-{gpu_sfx}" if smart_mode else mode
        print(f"Baseline: {underlying_static_method}-{sufix}")
    results = []
    for i, batch in enumerate(torch.unbind(ds.adj)):
        with print_zone(verbose >= 2):
            print("Batch", i)
        if i == 0:
            subcoms_depth = smart_subcoms_depth if mode == "smart" else 1
            opt = Optimizer(batch, ds.features,
                            subcoms_depth = subcoms_depth,
                            method = underlying_static_method,
                            verbose = verbose,
                            use_gpu = use_gpu)
            if ":" in str(batches_strategy):
                #TODO сделать загрузку посчитанного разбиения первого батча для стратегий "9:N", "99:N", "999:N"
                # Сейчас считаем разбиение "на ходу" самым быстрым алгоритмом
                opt.method = "ldleiden"
                n = opt.nodes_num
                l = opt.subcoms_depth

                coms = opt.local_algorithm(opt.adj, opt.features)
                coms = coms.repeat(l).reshape((l, n)) # Пропагируем сообщества вверх на все уровни
                opt.set_communities(communities = coms)

                opt.method = underlying_static_method
                opt.local_algorithm_calls = 0
                continue
            elif smart_mode:
                if batch.is_sparse:
                    batch_idx = batch.indices() if batch.is_coalesced() else batch.coalesce().indices()
                    active_nodes = batch_idx.unique()
                else:
                    nz_idx = torch.nonzero(batch, as_tuple=False)
                    active_nodes = nz_idx.unique()
                mask_device = opt.runtime_device()
                active_nodes = active_nodes.to(mask_device)
                affected_nodes_mask = torch.zeros(opt.nodes_num, dtype=torch.bool, device=mask_device)
                affected_nodes_mask[active_nodes] = True
        else:
            affected_nodes_mask = opt.update_adj(batch, return_mask = smart_mode)
        
        time_s = time.time()
        conversion_time_s = opt.conversion_time
        calls_s = opt.local_algorithm_calls
        if smart_mode:
            runtime_adj = opt.runtime_adj()
            affected_nodes_mask = opt.neighborhood(
                runtime_adj,
                affected_nodes_mask,
                step = smart_neighborhood_step,
            )
            opt.run(affected_nodes_mask)
        elif naive_mode or raw_mode:
            #opt.adj = opt.safe_clamp_sparse(opt.adj) #For non-negative weights in leidenalg
            labels = opt.coms if naive_mode else None
            coms = opt.local_algorithm(opt.runtime_adj(), opt.runtime_features(), labels = labels)
            opt.set_communities(communities = coms.unsqueeze(0), replace_subcoms_depth = True)
        time_e = time.time()
        conversion_time_e = opt.conversion_time
        calls_e = opt.local_algorithm_calls

        total_batch_time = time_e - time_s
        conversion_time = conversion_time_e - conversion_time_s
        measured_time = total_batch_time - conversion_time
        mod = opt.modularity(directed = ds.is_directed)
        with print_zone(verbose >= 2):
            print(f"Modularity: {mod:.2g}")
            print(f"Baseline calls: {calls_e - calls_s}")
            if underlying_static_method == "ldleiden" and mode in {"naive", "raw"}:
                algorithm_time = opt.last_timing_info["algorithm_time"]
                print(f"Algorithm time: {algorithm_time:.2f}")
            else:
                print(f"Time: {measured_time:.2f}")
        results.append({'modularity' : mod, 'time': measured_time})

    total_measured_time = sum(map(lambda x: x["time"], results))
    with print_zone(verbose == 1):
        print(f"Final modularity: {mod:.2g}")
    with print_zone(verbose >= 1):
        print(f"Total baseline calls: {opt.local_algorithm_calls}")
        print(f"Total time: {total_measured_time:.2f}")
        print("-----------------------------------------------")
    return results
