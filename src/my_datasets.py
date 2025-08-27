# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import DMoNPooling, GCNConv
# from torch_geometric.data import Data
# import networkx as nx
# import matplotlib.pyplot as plt

# class DMoN(torch.nn.Module):
#     def __init__(self, num_features, num_clusters):
#         super().__init__()
#         self.conv1 = GCNConv(num_features, 16)  # Указываем входной размер фичей
#         self.pool = DMoNPooling(16, num_clusters)  # 16 — размер скрытого слоя
    
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         x, S, loss = self.pool(x, edge_index)
#         return x, S, loss

# # Пример графа: 5 узлов, 6 рёбер
# edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], 
#                           [1, 0, 2, 1, 4, 3]], dtype=torch.long)
# x = torch.randn(5, 8)  # 5 узлов, 8 фичей (num_features=8)

# data = Data(x=x, edge_index=edge_index) 

# model = DMoN(num_features=8, num_clusters=2)  # num_features должно совпадать с x.shape[1]
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(100):
#     optimizer.zero_grad()
#     _, S, loss = model(data.x, data.edge_index)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch}, Loss: {loss.item()}")

import tensorflow.compat.v2 as tf
print(tf.config.list_physical_devices('GPU')) 