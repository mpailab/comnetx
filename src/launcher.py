import torch
import json
import os
import time

from datasets import INFO, Dataset
from optimizer import Optimizer
from our_utils import print_zone

from dynamic_graphs_communities import LDLeiden, DFLeiden, Leidenalg, Networkit
ALG_CLASS = {
    "leidenalg": Leidenalg,
    "networkit": Networkit,
    "ldleiden": LDLeiden,
    "dfleiden": DFLeiden
}

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
    dynamic_mode = (mode == "dynamic")

    with print_zone(verbose >= 1):
        print("-----------------------------------------------")
        print(f"Dataset: {dataset_name} ({batches_strategy} batches)")
        gpu_sfx = "gpu" if use_gpu else "cpu"
        sufix = f"L:{smart_subcoms_depth}-r:{smart_neighborhood_step}-{gpu_sfx}" if smart_mode else mode
        print(f"Baseline: {underlying_static_method}-{sufix}")

    results = []
    is_special_strategy = ":" in str(batches_strategy)

    # Для динамического режима подготовим переменные вне цикла
    if dynamic_mode:
        # Проверка поддерживаемого алгоритма
        if underlying_static_method in ALG_CLASS:
            algo_class = ALG_CLASS[underlying_static_method]
        else:
            raise ValueError(f"Dynamic mode not supported for {underlying_static_method}")

    # Основной цикл по батчам
    for i, batch in enumerate(torch.unbind(ds.adj)):
        with print_zone(verbose >= 2):
            print("Batch", i)

        # --- Обработка специальной стратегии (":") для динамического режима ---
        if dynamic_mode and is_special_strategy and i == 0:
            temp_algo = LDLeiden(batch, directed=ds.is_directed)
            temp_algo.apply()  # выполняем разбиение
            initial_partition = temp_algo.partition()

            algo = algo_class(batch,
                              directed=ds.is_directed,
                              partition=initial_partition)
            algo.apply() #FIXME По идеи тут apply() не нужен, но без него далее - Segmentation fault (core dumped)
            continue
        elif dynamic_mode and not is_special_strategy and i == 0:
            # Обычный случай: создаём алгоритм без начального разбиения
            algo = algo_class(batch, directed=ds.is_directed, partition=None)

        # --- Если режим динамический, обрабатываем батч через algo ---
        if dynamic_mode:
            time_s = time.time()
            algo.update(batch)
            elapsed_ms = algo.apply()
            measured_time = elapsed_ms / 1000.0  # переводим в секунды
            mod = algo.modularity()
            time_e = time.time()  # для единообразия, но фактическое время уже в measured_time

            with print_zone(verbose >= 2):
                print(f"Modularity: {mod:.2g}")
                print(f"Time: {measured_time:.2f}")

            results.append({'modularity': mod, 'time': measured_time})
            # Переходим к следующему батчу
            continue

        # --- Исходный код для остальных режимов (smart, naive, raw) ---
        if i == 0:
            subcoms_depth = smart_subcoms_depth if mode == "smart" else 1
            opt = Optimizer(batch, ds.features,
                            subcoms_depth = subcoms_depth,
                            method = underlying_static_method,
                            verbose = verbose,
                            use_gpu = use_gpu)
            if is_special_strategy:
                # Обработка ":" для не-dynamic режимов (как в оригинале)
                opt.method = "ldleiden"
                n = opt.nodes_num
                l = opt.subcoms_depth
                coms = opt.local_algorithm(opt.adj, opt.features)
                coms = coms.repeat(l).reshape((l, n))
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
                log_cuda = opt.cuda_mem_log,
            )
            opt.run(affected_nodes_mask)
        elif naive_mode or raw_mode:
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

        results.append({'modularity': mod, 'time': measured_time})

    # --- Итоговый вывод ---
    total_measured_time = sum(map(lambda x: x["time"], results))
    with print_zone(verbose == 1):
        final_mod = results[-1]['modularity'] if results else 0
        print(f"Final modularity: {final_mod:.2g}")
    with print_zone(verbose >= 1):
        if not dynamic_mode:
            print(f"Total baseline calls: {opt.local_algorithm_calls}")
        print(f"Total time: {total_measured_time:.2f}")
        print("-----------------------------------------------")

    return results
