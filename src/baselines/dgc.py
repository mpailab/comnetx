import argparse
import time
import torch
import random
import numpy as np

try:
    from dynamic_graphs_communities import AlgorithmOptions, LDLeiden, DFLeiden, Leidenalg, Networkit
except ImportError as exc:
    _BACKEND_IMPORT_ERROR = exc
    raise ImportError("dynamic_graphs_communities is required for DFLeiden and LDLeiden") from _BACKEND_IMPORT_ERROR

ALG_CLASS = {
    "leidenalg": Leidenalg,
    "networkit": Networkit,
    "ldleiden": LDLeiden,
    "dfleiden": DFLeiden
}

def _set_seed(seed: int | None) -> None:
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_leiden(method: str, adj, options=None, partition=None,  seed: int | None = None):
    _set_seed(seed)
    if method not in ALG_CLASS:
        raise ValueError(f"Unknown method: {method}")

    # В режиме directed == False методы из ALG_CLASS осуществляют симметризацию матрицы adj
    # Так как симметризация уже была произведена в Dataset, всегда используем directed == True
    # Метод networkit выбрасывает исключение при directed == True, поэтому оставляем False
    directed = True if method != "networkit" else False

    kwargs = {"directed": directed}
    if options is not None:
        kwargs["options"] = options
    if partition is not None:
        kwargs["partition"] = partition
    return ALG_CLASS[method](adj, **kwargs)

def _run_leiden(
    method,
    adj: torch.Tensor,
    init_partition=None,
    options=None,
    timing_info=None,
    seed: int | None = None,
):
    _set_seed(seed)
    conversion_time = 0.0

    # Move to CPU if needed (the algorithm expects CPU tensors)
    if adj.device.type == "cuda":
        time_s = time.time()
        adj = adj.cpu()
        time_e = time.time()
        conversion_time += time_e - time_s

    # Instantiate and update (this includes building internal structures)
    time_s = time.time()
    if options is not None:
        options = AlgorithmOptions(**options)
    elif seed is not None:
        try:
            options = AlgorithmOptions(seed=int(seed))
        except TypeError:
            options = None

    algo = create_leiden(method, adj, options, partition = init_partition, seed=seed)

    time_e = time.time()
    conversion_time += time_e - time_s
    if timing_info is not None:
        timing_info["conversion_time"] = timing_info.get("conversion_time", 0.0) + conversion_time

    # Run the Leiden algorithm
    if method == "ldleiden":
        update_ms, run_ms = algo.apply(with_update_timing=True)
        if timing_info is not None:
            timing_info["algorithm_time"] = timing_info.get("algorithm_time", 0.0) + run_ms / 1000
            timing_info["conversion_time"] = timing_info.get("conversion_time", 0.0) + update_ms / 1000
    else:
        time_s = time.time()
        run_ms = algo.apply()
        if timing_info is not None:
            timing_info["algorithm_time"] = timing_info.get("algorithm_time", 0.0) + run_ms / 1000

    return algo.partition().to(torch.long)
