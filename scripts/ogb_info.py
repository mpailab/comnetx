import torch
from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset

dataset = NodePropPredDataset(name="ogbn-papers100M", root="/auto/datasets/graphs/comnetx/ogb/")
graph, label = dataset[0]
# dataset = LinkPropPredDataset(name="ogbl-citation2", root="/auto/datasets/graphs/comnetx/ogb/")
# graph = dataset[0]
# label = None

edge_index = torch.tensor(graph['edge_index'])
num_nodes = graph['num_nodes']

print("edge_index", edge_index)

if graph['node_feat'] is not None:
    x = torch.tensor(graph['node_feat'], dtype=torch.float)
    print("Фичи есть, размерность:", x.shape)
else:
    print("У этого графа нет исходных фичей (node_feat is None).")

if label is not None:
    print("label", (label))
else:
    print("У этого графа нет исходных меток") 

# Считаем степень каждого узла (количество исходящих ребер)
# Используем bincount для подсчета упоминаний каждого индекса узла в первой строке edge_index
degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()

print(f"Средняя степень: {degrees.mean().item():.2f}")
print(f"Максимальная степень: {degrees.max().item()}")
print(f"Минимальная степень: {degrees.min().item()}")