import leidenalg
import igraph as ig
import torch

def sparse_tensor_to_igraph(sparse_tensor, directed=True):
    indices = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()
    edges = indices.t().numpy()
    graph = ig.Graph(n=sparse_tensor.shape[0], edges=edges, directed=directed)
    if not torch.all(values == 1.0):
        graph.es['weight'] = values.numpy()
    return graph

def leidenalg_partition(adj : torch.Tensor,
                  ):
    G = sparse_tensor_to_igraph(adj.to_sparse())
    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    return torch.tensor(part.membership, dtype=torch.long)