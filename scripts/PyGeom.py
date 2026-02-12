from torch_geometric.datasets import GDELTLite
import torch
# Инициализируем класс
# root — это папка, куда скачаются данные
dataset = GDELTLite(root='/auto/datasets/graphs/comnetx/torch_geom_datasets/data/GDELTLite')

print(dataset.x)
print(dataset.edge_index)
print(dataset.y)