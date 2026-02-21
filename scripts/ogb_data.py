from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset

# # ogbn-products
# # ogbn-arxiv
# # ogbn-papers100M 
# dataset = NodePropPredDataset(name = "ogbn-products")
# graph, label = dataset[0] # label — это ваши Y

# # X: Матрица признаков вершин (Node Features)
# feat = graph['node_feat'] 

# print(len(feat[:,:]))
# print("label =", len(label))

# # OGB хранит ребра в формате Edge List (список пар)
# edge_index = graph['edge_index'] 
# print(len(set(edge_index[0])))

dataset = LinkPropPredDataset(name = "ogbl-citation2")
graph = dataset[0]
print(graph['node_feat'][:10,:])
print(graph['edge_feat']) 
