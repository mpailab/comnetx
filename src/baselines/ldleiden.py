import argparse
import time
import torch

try:
    from dynamic_graphs_communities import AlgorithmOptions, LDLeiden
except ImportError as exc:
    _BACKEND_IMPORT_ERROR = exc
    raise ImportError("dynamic_graphs_communities is required for LDLeiden") from _BACKEND_IMPORT_ERROR


def ldleiden_partition(
    adj: torch.Tensor,
    directed: bool = False,
    options=None,
    timing_info=None,
) -> torch.Tensor:
    time_s = time.time()
    if options is not None:
        options = AlgorithmOptions(**options)
    algo = LDLeiden(nodes_num=adj.size(0), directed=directed, options=options)
    algo.update(adj)
    time_e = time.time()
    if timing_info is not None:
        timing_info["conversion_time"] = time_e - time_s

    time_s = time.time()
    algo.apply()
    time_e = time.time()
    if timing_info is not None:
        timing_info["algorithm_time"] = time_e - time_s

    return algo.partition().to(torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--directed", action="store_true")
    args = parser.parse_args()

    adj = torch.load(args.adj)
    labels = ldleiden_partition(adj, directed=args.directed)
    torch.save(labels, args.out)
    print("LDLEIDEN finished successfully")


if __name__ == "__main__":
    main()
