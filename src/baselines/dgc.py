import argparse
import time
import torch

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
def create_leiden(method: str, adj, options=None, partition=None):
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
    options=None,
    timing_info=None
):
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

    algo = create_leiden(method, adj, options)

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
