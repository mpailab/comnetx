import argparse
import leidenalg as la
import igraph as ig
import torch

def sparse_tensor_to_igraph(sparse_tensor, directed=True):
    st = sparse_tensor.coalesce()
    indices = st.indices()
    values = st.values()
    edges = indices.t().numpy()
    graph = ig.Graph(n=sparse_tensor.shape[0], edges=edges, directed=directed)
    if not torch.all(values == 1.0):
        graph.es['weight'] = values.numpy()
    return graph

def leidenalg_partition(adj : torch.Tensor):
    G = sparse_tensor_to_igraph(adj.to_sparse())
    part = la.find_partition(G, la.ModularityVertexPartition, seed=True, n_iterations=2)
    return torch.tensor(part.membership, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adj = torch.load(args.adj)
    if not adj.is_sparse:
        adj = adj.to_sparse()

    labels = leidenalg_partition(adj)
    torch.save(labels, args.out)
    print("LEIDEN finished successfully")

if __name__ == "__main__":
    main()