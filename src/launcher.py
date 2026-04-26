import torch
import json
import os
import time
import numpy as np

from optimizer import Optimizer
from our_utils import print_zone
import sparse

from dynamic_graphs_communities import LDLeiden, DFLeiden, Leidenalg, Networkit
from baselines.dgc import create_leiden
from baselines.mfc import mfc_adopted
from metrics import Metrics


def compute_initial_partition(
    adj_matrix,
    dataset_name,
    init_batch_number,
    method_name="leidenalg",
    cache_dir=None,
    subcoms_depth=1,
    device=None,
):
    """
    Build an initial community partition for an adjacency matrix.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Adjacency matrix of the initial graph snapshot with shape (n, n).
        The argument used to be named ``batch``, but it is the adjacency matrix
        itself, not a batch of arbitrary data.
    dataset_name : str
        Dataset key used only for the cache file name.
    init_batch_number : int | str
        Initial batch identifier used only for the cache file name.
    method_name : str
        Static community detection method passed to ``create_leiden``.
    cache_dir : str | None
        Directory for storing/loading the computed partition and modularity.
    subcoms_depth : int
        Number of partition layers to build. ``1`` preserves the previous
        behavior and returns a single label vector with shape (n,). Values above
        ``1`` return a layered tensor/array with shape (subcoms_depth, n).
    device : torch.device | str | None
        Device for all tensors created inside this function. If None, use the
        device of ``adj_matrix``.

    Returns
    -------
    tuple[torch.Tensor, float]
        Initial partition labels and modularity of the first full-graph
        partition. For layered output each row contains labels for original
        graph nodes restored from the corresponding aggregated graph.
    """
    if subcoms_depth < 1:
        raise ValueError("subcoms_depth must be at least 1")

    if device is None:
        device = adj_matrix.device
    else:
        device = torch.device(device)

    loaded = False
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        depth_suffix = "" if subcoms_depth == 1 else f"_d:{subcoms_depth}"
        filename = os.path.join(
            cache_dir,
            f"{dataset_name}_b:{init_batch_number}_by_{method_name}{depth_suffix}.npz",
        )
        if os.path.exists(filename):
            with np.load(filename, allow_pickle=True) as data:
                init_partition = torch.as_tensor(
                    data["partition"],
                    dtype=torch.long,
                    device=device,
                )
                init_mod = float(data["mod"])
            loaded = True

    if not loaded:
        adj_algo = adj_matrix.to(device)
        temp_algo = create_leiden(method_name, adj_algo)
        temp_algo.apply()
        # Most backends return partition labels on CPU; keep the launcher state
        # on the requested runtime device from the first conversion onward.
        init_partition = torch.as_tensor(
            temp_algo.partition(),
            dtype=torch.long,
            device=device,
        )
        init_mod = temp_algo.modularity()

        if subcoms_depth > 1:
            layers = [init_partition]
            nodes_num = adj_matrix.size(0)
            adj_work = adj_algo.float()
            if not adj_work.is_sparse:
                adj_work = adj_work.to_sparse_coo()
            adj_work = adj_work.coalesce()
            node_mask = torch.ones(
                nodes_num,
                dtype=torch.bool,
                device=adj_work.device,
            )

            for _ in range(1, subcoms_depth):
                # Reindex current communities to compact ids before building
                # the aggregation pattern P used in P * A * P.T.
                current_partition = layers[-1].to(adj_work.device)
                old_idx, inverse = torch.unique(
                    current_partition,
                    sorted=True,
                    return_inverse=True,
                )
                aggr_idx = torch.stack((
                    inverse,
                    torch.arange(nodes_num, device=device),
                ))
                aggr_adj_ptn = sparse.tensor(
                    aggr_idx,
                    (old_idx.size(0), nodes_num),
                    adj_work.dtype,
                )
                aggr_adj = Optimizer.aggregate(adj_work, aggr_adj_ptn)
                del aggr_adj_ptn

                # Run the same local method on the aggregated graph and restore
                # labels back to original graph nodes.
                temp_algo = create_leiden(method_name, aggr_adj)
                temp_algo.apply()
                aggr_partition = torch.as_tensor(
                    temp_algo.partition(),
                    dtype=torch.long,
                    device=device,
                )
                restored_partition = old_idx[aggr_partition[inverse]]
                layers.append(restored_partition)

                # Keep the working adjacency consistent with Optimizer.run:
                # after each layer, remove edges that cross the new partition.
                adj_work = Optimizer.cut_by_partition(
                    adj_work,
                    node_mask,
                    restored_partition,
                )

            init_partition = torch.stack(layers)

        if cache_dir is not None:
            np.savez_compressed(
                filename,
                partition=init_partition.cpu().numpy(),
                mod=init_mod,
            )

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
                        labels=None,
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
                                                                         cache_dir,
                                                                         subcoms_depth=opt.subcoms_depth,
                                                                         device=opt.runtime_device())
                    if init_partition.dim() == 1:
                        init_partition = init_partition.unsqueeze(0)
                    opt.set_communities(communities = init_partition)
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
