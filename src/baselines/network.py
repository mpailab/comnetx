import argparse
import networkit as nk
import torch
import time
import random
import numpy as np

def _set_seed(seed: int | None) -> None:
    if seed is None:
        return

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        nk.setSeed(seed, True)
    except Exception:
        pass

def sparse_tensor_to_networkit(sparse_tensor, directed=False):
    indices = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()
    n_nodes = sparse_tensor.shape[0]
    graph = nk.Graph(n=n_nodes, weighted=True, directed=directed)

    edges = indices.t().cpu().numpy()
    values_np = values.cpu().numpy()

    for idx, (i, j) in enumerate(edges):
        graph.addEdge(int(i), int(j), float(values_np[idx]))
    
    return graph

def networkit_partition(
    adj: torch.Tensor,
    algorithm="leiden",
    timing_info=None,
    seed: int | None = None,
):
    _set_seed(seed)
    conversion_time = 0.0
    if adj.device.type == "cuda":
        time_s = time.time()
        adj = adj.cpu()
        time_e = time.time()
        conversion_time += time_e - time_s

    time_s = time.time()
    graph = sparse_tensor_to_networkit(adj)
    time_e = time.time()
    conversion_time += time_e - time_s
    
    if timing_info is not None:
        timing_info['conversion_time'] = timing_info.get('conversion_time', 0.0) + conversion_time

    if algorithm == "leiden":
        detector = nk.community.ParallelLeiden(graph)
    else:
        detector = nk.community.PLM(graph)

    detector.run()
    partition = detector.getPartition()
    res = torch.arange(adj.shape[0], dtype=torch.long)
    for c in partition.getSubsetIds():
        members = partition.getMembers(c)
        res[list(members)] = min(members)
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--algorithm", choices=["leiden", "plm"], default="leiden")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    if not adj.is_sparse:
        adj = adj.to_sparse()

    labels = networkit_partition(
        adj,
        algorithm=("leiden" if args.algorithm == "leiden" else "plm"),
        seed=args.seed,
    )
    torch.save(labels, args.out)
    print("NETWORKIT finished successfully")

if __name__ == "__main__":
    main()    
