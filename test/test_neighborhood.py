import sys
sys.path.append("src") 
from optimizer import Optimizer
import torch

A = torch.tensor([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
], dtype=torch.float32)

indices = torch.nonzero(A).t()
values = A[indices[0], indices[1]]

A_sparse = torch.sparse_coo_tensor(
    indices=indices,
    values=values,
    size=A.shape,
    dtype=torch.float32
)

initial_nodes = torch.tensor([True, False, False, False])

k = 2
nodes_new = Optimizer.neighborhood(A_sparse, initial_nodes, k)
print(nodes_new)