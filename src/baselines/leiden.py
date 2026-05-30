import argparse
import leidenalg as la
import igraph as ig
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

def sparse_tensor_to_igraph(sparse_tensor, directed=True):
    st = sparse_tensor.coalesce()
    indices = st.indices()
    values = st.values()
    edges = indices.t().numpy()
    graph = ig.Graph(n=sparse_tensor.shape[0], edges=edges, directed=directed)
    graph.es['weight'] = values.numpy()
    return graph

def leidenalg_partition(adj : torch.Tensor, init_partition=None, timing_info=None, seed: int | None = None):
    _set_seed(seed)
    conversion_time = 0.0
    if adj.device.type == "cuda":
        time_s = time.time()
        adj = adj.cpu()
        time_e = time.time()
        conversion_time += time_e - time_s

    time_s = time.time()
    G = sparse_tensor_to_igraph(adj.to_sparse())
    initial_membership_list = init_partition.tolist() if init_partition is not None else None
    time_e = time.time()
    conversion_time += time_e - time_s
    if timing_info is not None:
        timing_info['conversion_time'] = timing_info.get('conversion_time', 0.0) + conversion_time

    partition = la.find_partition(
        G,
        la.ModularityVertexPartition,
        initial_membership=initial_membership_list,
        weights='weight',
        seed=seed,
        n_iterations=2
    )
    return torch.tensor(partition.membership, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    if not adj.is_sparse:
        adj = adj.to_sparse()

    labels = leidenalg_partition(adj, seed=args.seed)
    torch.save(labels, args.out)
    print("LEIDEN finished successfully")

if __name__ == "__main__":
    main()
