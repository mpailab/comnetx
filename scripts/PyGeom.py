import torch_geometric.datasets 
import torch
import os, sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_root = os.path.join(PROJECT_PATH, "src")
if data_root not in sys.path:
    sys.path.insert(0, data_root)
from datasets import Dataset

dname = "yelp"
data_path = '/auto/datasets/graphs/comnetx/torch_geom_datasets/data/' + dname.lower()

# dataset = torch_geometric.datasets.AttributedGraphDataset(
#     root=data_path, name=dname)

dataset = torch_geometric.datasets.Yelp(
    root=data_path)

# dataset = torch_geometric.datasets.Coauthor(
#     root=data_path, name="CS")

# dataset = torch_geometric.datasets.Flickr(
#     root=data_path)

dataset_our = Dataset(dataset_name=dname, path=data_path)
adj, features, labels = dataset_our.load(tensor_type="coo")
print(f"adj_nnz = {adj._nnz()}")

print(dataset.x, features, sep='\n')
print(dataset.edge_index, f"shape_edge_i = {dataset.edge_index.shape}", adj, sep='\n')
print("is_undirected? -", torch_geometric.utils.is_undirected(dataset.edge_index))
print(dataset.y, labels, sep='\n')

# num_nodes = len(dataset.y)
# indices = dataset.edge_index
# num_edges = indices.shape[1]
# values = torch.ones(num_edges, dtype=torch.float32)

# adj_data = {
#     'indices': indices,
#     'values': values,
#     'shape': (num_nodes, num_nodes)
# }

# adj_data = torch.sparse_coo_tensor(
#                             adj_data['indices'], 
#                             adj_data['values'], 
#                             size=adj_data['shape']
#                         ).coalesce()

# print(f"adj_data = {adj_data}")





