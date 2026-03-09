import argparse
import time
import torch

try:
    from dynamic_graphs_communities import AlgorithmOptions, DFLeiden
except ImportError as exc:
    _BACKEND_IMPORT_ERROR = exc
    raise ImportError("dynamic_graphs_communities is required for DFLeiden") from _BACKEND_IMPORT_ERROR


def dfleiden_partition(
    adj: torch.Tensor,
    directed: bool = False,
    options=None,
    timing_info=None,
) -> torch.Tensor:
    conversion_time = 0.0
    if adj.device.type == "cuda":
        time_s = time.time()
        adj = adj.cpu()
        time_e = time.time()
        conversion_time += time_e - time_s

    time_s = time.time()
    if options is not None:
        options = AlgorithmOptions(**options)
    algo = DFLeiden(nodes_num=adj.size(0), directed=directed, options=options)
    algo.update(adj)
    time_e = time.time()
    conversion_time += time_e - time_s
    if timing_info is not None:
        timing_info["conversion_time"] = timing_info.get("conversion_time", 0.0) + conversion_time

    algo.apply()
    return algo.partition().to(torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--directed", action="store_true")
    args = parser.parse_args()

    adj = torch.load(args.adj)
    labels = dfleiden_partition(adj, directed=args.directed)
    torch.save(labels, args.out)
    print("DFLEIDEN finished successfully")


if __name__ == "__main__":
    main()
