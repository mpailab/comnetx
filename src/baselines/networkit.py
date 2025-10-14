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
    membership = [partition[i] for i in range(partition.numberOfElements())]
    
    return torch.tensor(membership, dtype=torch.long)


    