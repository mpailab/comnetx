import torch
import json
import os
import time
import numpy as np

from optimizer import Optimizer
from our_utils import print_zone

from dynamic_graphs_communities import LDLeiden, DFLeiden, Leidenalg, Networkit
from baselines.dgc import create_leiden
from metrics import Metrics

def compute_initial_partition(
    batch,
    dataset_name,
    init_batch_number,
    method_name="leidenalg",
    cache_dir=None,
    to_tensor_format=True
):
    loaded = False
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.join(
            cache_dir, f"{dataset_name}_b:{init_batch_number}_by_{method_name}.npz"
        )
        if os.path.exists(filename):
            with np.load(filename, allow_pickle=True) as data:
                init_partition = data["partition"]
                init_mod = float(data["mod"])
            loaded = True

    if not loaded:
        temp_algo = create_leiden(method_name, batch)
        temp_algo.apply()
        init_partition = temp_algo.partition()
        init_mod = temp_algo.modularity()
        if cache_dir is not None:
            np.savez_compressed(filename, partition=init_partition, mod=init_mod)

    if to_tensor_format:
        init_partition = torch.as_tensor(init_partition, dtype=torch.long)

    return init_partition, init_mod

def dynamic_launch(ds, batches_strategy,
                    underlying_static_method: str,
                    baseline_iter: int = None,
                    mode: str = "smart",
                    smart_subcoms_depth: int = 5,
                    smart_neighborhood_step: int = 1,
                    verbose: int = 1,
                    use_gpu: bool = False,
                    aggregation_mode: str = "sum",
                    cache_dir = None):

    dataset_name = ds.name
    smart_mode = (mode == "smart")
    naive_mode = (mode == "naive")
    raw_mode = (mode == "raw")
    dynamic_mode = (mode == "dynamic")

    results = []
    is_special_strategy = ":" in str(batches_strategy)
    if is_special_strategy:
        init_batch_number = str(batches_strategy).split(":")[0]
    
    if dynamic_mode and underlying_static_method == "mfc":       
        from baselines.mfc import mfc_adopted
        time_s = time.time()
        init_partition = None
        if is_special_strategy:
            first_snapshot = ds.adj[0] 
            init_partition, init_mod = compute_initial_partition(first_snapshot,
                                                                 dataset_name, init_batch_number,
                                                                 "leidenalg",
                                                                 cache_dir)
            with print_zone(verbose >= 1):
                print(f"Initial modularity: {init_mod:.2g}")
        
        with print_zone(verbose >= 2):
            coms = mfc_adopted(
                adj=ds.adj,
                features=ds.features,
                network_type="MFC",
                return_labels=True,
                num_epoch=baseline_iter,
                pure_mfc=True,
                initial_partition=init_partition,
            )
                        
            time_e = time.time()
            measured_time = time_e - time_s
            mod = Metrics.modularity(ds.adj[0], coms, directed = ds.is_directed)

            print(f"Modularity: {mod:.2g}")
            print(f"Baseline calls: {1}")
            print(f"Time: {measured_time:.2f}")

        results.append({'modularity': mod, 'time': measured_time})
    else:
    # Основной цикл по батчам
        if ds.adj.ndim == 2:
            batches_iter = [ds.adj]
        elif ds.adj.ndim == 3:
            batches_iter = torch.unbind(ds.adj)
        else:
            raise ValueError(f"Unsupported ds.adj ndim: {ds.adj.ndim}")

        for i, batch in enumerate(batches_iter):
            
            with print_zone(verbose >= 2):
                print("  Batch", i)

            # --- Обработка специальной стратегии (":") для динамического режима ---
            if dynamic_mode and is_special_strategy and i == 0:
                init_partition, init_mod = compute_initial_partition(batch,
                                                                     dataset_name, init_batch_number,
                                                                     "leidenalg",
                                                                     cache_dir)
                with print_zone(verbose >= 1):
                    print(f"Initial modularity: {init_mod:.2g}")

                algo = create_leiden(underlying_static_method, batch, partition=init_partition)
                # FIXME тут apply() не нужен, но без него ниже падает с
                # segmentation fault.
                algo.apply()
                continue
            elif dynamic_mode and not is_special_strategy and i == 0:
                # Обычный случай: создаём алгоритм без начального разбиения
                algo = create_leiden(underlying_static_method, batch, partition=None)

            # --- Если режим динамический, обрабатываем батч через algo ---
            if dynamic_mode:
                algo.update(batch)
                elapsed_ms = algo.apply()
                measured_time = elapsed_ms / 1000.0  # переводим в секунды
                mod = algo.modularity()

                with print_zone(verbose >= 2):
                    print(f"Modularity: {mod:.2g}")
                    print(f"Time: {measured_time:.2f}")

                results.append({'modularity': mod, 'time': measured_time})
                # Переходим к следующему батчу
                continue

            # --- Исходный код для остальных режимов (smart, naive, raw) ---
            if i == 0:
                opt = Optimizer(batch, ds.features,
                                subcoms_depth = smart_subcoms_depth if smart_mode else 1,
                                method=underlying_static_method,
                                baseline_iter=baseline_iter,
                                verbose=verbose,
                                use_gpu=use_gpu,
                                aggregation_mode=aggregation_mode)
                if is_special_strategy:
                    init_partition, init_mod = compute_initial_partition(batch,
                                                                         dataset_name, init_batch_number,
                                                                         "leidenalg",
                                                                         cache_dir)
                    n, l = opt.nodes_num, opt.subcoms_depth
                    coms = init_partition.repeat(l).reshape((l, n))
                    opt.set_communities(communities = coms)
                    with print_zone(verbose >= 1):
                        print(f"Initial modularity: {init_mod:.2g}")
                    continue
                if smart_mode:
                    if batch.is_sparse:
                        batch_idx = (
                            batch.indices()
                            if batch.is_coalesced()
                            else batch.coalesce().indices()
                        )
                        active_nodes = batch_idx.unique()
                    else:
                        nz_idx = torch.nonzero(batch, as_tuple=False)
                        active_nodes = nz_idx.unique()
                    mask_device = opt.runtime_device()
                    active_nodes = active_nodes.to(mask_device)
                    affected_nodes_mask = torch.zeros(
                        opt.nodes_num,
                        dtype=torch.bool,
                        device=mask_device,
                    )
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
                labels = opt.coms if naive_mode else None
                coms = opt.local_algorithm(
                    opt.runtime_adj(),
                    opt.runtime_features(),
                    labels=labels,
                )
                opt.set_communities(
                    communities=coms.unsqueeze(0),
                    replace_subcoms_depth=True,
                )

            time_e = time.time()
            conversion_time_e = opt.conversion_time
            calls_e = opt.local_algorithm_calls

            total_batch_time = time_e - time_s
            conversion_time = conversion_time_e - conversion_time_s
            measured_time = total_batch_time - conversion_time
            mod = opt.modularity(directed = ds.is_directed)
            
            with print_zone(verbose >= 2):
                print(f"Modularity: {mod:.2g}")
                #print(f"Baseline calls: {calls_e - calls_s}")
                if underlying_static_method == "ldleiden" and mode in {"naive", "raw"}:
                    algorithm_time = opt.last_timing_info["algorithm_time"]
                    print(f"Algorithm time: {algorithm_time:.2f}")
                else:
                    print(f"Time: {measured_time:.2f}")

            results.append({'modularity': mod, 'time': measured_time})

    # --- Итоговый вывод ---
    total_measured_time = sum(map(lambda x: x["time"], results))
    with print_zone(verbose == 1):
        final_mod = results[-1]['modularity'] if results else 0
        print(f"Final modularity: {final_mod:.2g}")
    with print_zone(verbose >= 1):
        #if not dynamic_mode:
            #print(f"Total baseline calls: {opt.local_algorithm_calls}")
        print(f"Total time: {total_measured_time:.2f}")
        print("-----------------------------------------------")

    return results
