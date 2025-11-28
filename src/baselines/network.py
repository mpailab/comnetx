import argparse
import networkit as nk
import torch

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

def networkit_partition(adj: torch.Tensor, algorithm="leiden"):
    graph = sparse_tensor_to_networkit(adj)

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
    args = parser.parse_args()

    adj = torch.load(args.adj)
    if not adj.is_sparse:
        adj = adj.to_sparse()

    labels = networkit_partition(adj, algorithm=("leiden" if args.algorithm == "leiden" else "plm"))
    torch.save(labels, args.out)
    print("NETWORKIT finished successfully")

if __name__ == "__main__":
    main()    