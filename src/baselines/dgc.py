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

def _run_leiden(
    method,
    adj: torch.Tensor,
    directed: bool = False,
    options=None,
    timing_info=None,
    measure_algorithm_time: bool = False,
) -> torch.Tensor:
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
    algo = ALG_CLASS[method](nodes_num=adj.size(0), directed=directed, options=options)
    algo.update(adj)
    time_e = time.time()
    conversion_time += time_e - time_s
    if timing_info is not None:
        timing_info["conversion_time"] = timing_info.get("conversion_time", 0.0) + conversion_time

    # Run the Leiden algorithm
    if measure_algorithm_time:
        time_s = time.time()
        algo.apply()
        time_e = time.time()
        if timing_info is not None:
            timing_info["algorithm_time"] = timing_info.get("algorithm_time", 0.0) + (time_e - time_s)
    else:
        algo.apply()  # without timing

    return algo.partition().to(torch.long)
