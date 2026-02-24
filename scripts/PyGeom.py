import torch_geometric.datasets 
import torch
# Инициализируем класс
# root — это папка, куда скачаются данные

dataset = torch_geometric.datasets.AttributedGraphDataset(
    root='/auto/datasets/graphs/comnetx/torch_geom_datasets/data/facebook', name="Facebook")

# print(dataset.x)
print(dataset.edge_index)
print(torch_geometric.utils.is_undirected(dataset.edge_index))
# print(dataset.y)