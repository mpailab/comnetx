import torch
from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset

# dataset = NodePropPredDataset(name="ogbn-papers100M")
# graph, label = dataset[0]
dataset = LinkPropPredDataset(name = "ogbl-vessel")
graph = dataset[0]
edge_index = torch.tensor(graph['edge_index'])
num_nodes = graph['num_nodes']

# Считаем степень каждого узла (количество исходящих ребер)
# Используем bincount для подсчета упоминаний каждого индекса узла в первой строке edge_index
degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()

print(f"Средняя степень: {degrees.mean().item():.2f}")
print(f"Максимальная степень: {degrees.max().item()}")
print(f"Минимальная степень: {degrees.min().item()}")