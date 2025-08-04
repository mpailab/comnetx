import torch
from torch_geometric.datasets import Planetoid, Reddit

from ogb.nodeproppred import PygNodePropPredDataset
import tempfile
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os.path
from ogb.nodeproppred import PygNodePropPredDataset

import time
import json

KONECT_PATH = "/auto/datasets/graphs/dynamic_konect_project_datasets/"
KONECT_INFO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "konect-datasets-info"))

# TODO (to drobyshev) make dataset dict

class Dataset:
    """Dataset treatment"""


    def __init__(self, dataset_name : str, path: str = "./datasets"):
        self.name = dataset_name
        self.path = path
        self.adj = None
        self.features = None
        self.label = None

    def load(self, tensor_type : str = "coo", batches = None) -> torch.Tensor:
        """
        Load dataset

        Parameters
        ----------
        tensor_type : str
            Type of output tensor: coo, csr, csc, dense
            Default: coo

        Returns
        -------
        TODO (to drobyshev) add comment for shape of output tensor
        """

        filenames = {
            "features": f"{self.name.lower()}_feat.npy",
            "labels": f"{self.name.lower()}_label.npy",
            "adj": f"{self.name.lower()}_adj.npy"
        }

        with open(os.path.join(KONECT_INFO, "dynamic.json")) as _:
            konect_info = json.load(_)
        with open(os.path.join(KONECT_INFO, "static.json")) as _:
            konect_info.update(json.load(_))
        

        if self.name in konect_info:
            is_directed = konect_info[self.name]['d'] == 'directed'
            if batches == None:
                self._load_konect(batches_num = 1, is_directed = is_directed)
                self.adj = self.adj[0]
            else:
                self._load_konect(batches_num = batches, is_directed = is_directed)
        elif self.name.lower() in {"cora", "citeseer", "pubmed", "reddit"} or self.name.startswith("ogbn-"):
            for key, filename in filenames.items():
                full_path = os.path.join(self.path, filename)
                if not os.path.isfile(full_path):
                    print(f"file not found: {filename}")
                    print("Downloading files...")
                    self._save_magi()
                    break
            self._load_npy_format()
        elif self.name.lower() in {"acm", "bat", "dblp", "eat", "uat"}:
            self._load_npy_format()
        else:
            raise ValueError(f"Unsupported dataset: {self.name}")

        if tensor_type == "dense":
            return self.adj.to_dense(), self.features, self.label
        elif tensor_type == "coo":
            return self.adj.coalesce(), self.features, self.label
        elif tensor_type == "csr":
            return self.adj.to_sparse_csr(), self.features, self.label
        elif tensor_type == "csc":
            return self.adj.to_sparse_csc(), self.features, self.label
        else:
            raise ValueError(f"Unsupported tensor type for torch.sparse: {tensor_type}")
        
        return self.adj, self.features, self.label

    def _load_magi(self):
        name = self.name.lower()
        with tempfile.TemporaryDirectory() as tmpdir:
            if name in {"cora", "citeseer", "pubmed"}:
                dataset = Planetoid(root=tmpdir, name=name.capitalize())
                data = dataset[0]

            elif name == "reddit":
                dataset = Reddit(root=tmpdir)
                data = dataset[0]

            elif name.startswith("ogbn-"):
                dataset = PygNodePropPredDataset(name=name, root=tmpdir)
                split_idx = dataset.get_idx_split()
                data = dataset[0]
                data.y = data.y.view(-1)
            else:
                raise ValueError(f"Unknown MAGI-compatible dataset: {self.name}")

        edge_index = data.edge_index
        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
        values = torch.ones(edge_index.size(1), dtype=torch.float32)
        self.adj = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes))

        self.features = data.x
        self.label = data.y

    def _save_magi(self):
        self._load_magi()
        save_dir = self.path + '/' + self.name.lower()
        os.makedirs(self.path, exist_ok=True)

        np.save(os.path.join(save_dir + '_feat.npy'), self.features.numpy())
        np.save(os.path.join(save_dir + '_label.npy'), self.label.numpy())

        adj = self.adj.coalesce()
        """
        adj_dict = {
            'row': adj.indices()[0].numpy(),
            'col': adj.indices()[1].numpy(),
            'data': adj.values().numpy(),
            'shape': adj.shape
        }
        """
        adj_dense = adj.to_dense().numpy()
        np.save(os.path.join(save_dir + '_adj.npy'), adj_dense)
        #np.save(os.path.join(save_dir + '_adj.npy'), adj_dict, allow_pickle=True)

    def _load_npy_format(self):  #Drobyshev
        self.path = os.path.abspath(self.path)

        features = np.load(self.path + '/' + self.name.lower() + '_feat.npy')
        labels = np.load(self.path + '/' + self.name.lower() + '_label.npy')
        self.features = torch.tensor(features, dtype=torch.float)
        self.label = torch.tensor(labels, dtype=torch.long)

        adj_data = np.load(self.path + '/' + self.name.lower() + '_adj.npy', allow_pickle=True)
        rows, cols = adj_data.nonzero()
        values_np = adj_data[rows, cols]
        indices_np = np.vstack((rows, cols))

        indices = torch.from_numpy(indices_np).long()
        values = torch.from_numpy(values_np).float()
        shape = adj_data.shape
        

        #adj_data = np.load(self.path + '/' + self.name.lower() + '_adj.npy', allow_pickle=True).item()
        #indices = torch.tensor([adj_data['row'], adj_data['col']], dtype=torch.long)
        #values = torch.tensor(adj_data['data'], dtype=torch.float)
        #shape = tuple(adj_data['shape'])
        self.adj = torch.sparse_coo_tensor(indices, values, size=shape)
    
    def _load_konect(self, batches_num = 1, is_directed = True):
        """
        Load dynamic dataset from KONECT collection

        Parameters
        ----------
        batches_num : 1, 10, 100, 1000, 10000, 100000
            Default: 1
        """
        filepath = os.path.join(self.path, self.name, f"out.{self.name}.{batches_num}_batches")
        with open(filepath) as _:
            first_string = _.readline()
            num_nodes = int(first_string.split()[0])
            max_index = num_nodes - 1
            edges_num = int(first_string.split()[1])
        i, j, w, t = np.loadtxt(filepath, skiprows=1, dtype=int, unpack=True)
        adjs = []
        for num in range(batches_num):
            mask = (t == num)
            adj_index = np.vstack((i[mask], j[mask]))
            adj = torch.sparse_coo_tensor(adj_index, w[mask], size=(num_nodes, num_nodes)).coalesce()
            if not is_directed:
                adj = adj + torch.t(adj)
            adjs.append(adj)
        self.adj = torch.stack(adjs) # 3-dimensional tensor

def list_konect_datasets():
    with open(os.path.join(KONECT_INFO, "dynamic.json")) as _:
        info = json.load(_)
    with open(os.path.join(KONECT_INFO, "static.json")) as _:
        info.update(json.load(_))
    datasets = list(info.keys())
    max_name_len = max(map(len, datasets))
    max_n_strlen = max(map(lambda x: len(str(info[x]["n"])), datasets))
    max_m_strlen = max(map(lambda x: len(str(info[x]["m"])), datasets))
    max_w_strlen = max(map(lambda x: len(str(info[x]["w"])), datasets))
    max_d_strlen = max(map(lambda x: len(str(info[x]["d"])), datasets))
    datasets.sort(key = lambda x: info[x]["m"])
    for dataset in datasets:
        pstring = f"{dataset:<{max_name_len}}"
        pstring += f" {info[dataset]['n']:<{max_n_strlen}}"
        pstring += f" {info[dataset]['m']:<{max_m_strlen}}"
        pstring += f" {info[dataset]['d']:<{max_d_strlen}}"
        pstring += f" {info[dataset]['w']:<{max_w_strlen}}"
        print(pstring, sep="\t")

if __name__ == "__main__":
    dataset = "dblp_coauthor"
    #ds = Dataset(dataset, KONECT_PATH)
    #ds._load_konect(batches_num = 10)
    list_konect_datasets()