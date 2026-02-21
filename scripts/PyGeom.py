from torch_geometric.datasets import Flickr
import torch
# Инициализируем класс
# root — это папка, куда скачаются данные
dataset = Flickr(root='/auto/datasets/graphs/comnetx/torch_geom_datasets/data/Flickr')

print(dataset.x)
print(type(dataset.edge_index))
print(dataset.y)