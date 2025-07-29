import sys
sys.path.append("src") 
from optimizer import Optimizer
import torch

A = torch.tensor([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
], dtype=torch.float32)

initial_nodes = torch.tensor([False, False, True, False])

k = 2
nodes_new = Optimizer.neighborhood(A, initial_nodes, k)
print(nodes_new)