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
import joblib

KONECT_PATH = "/auto/datasets/graphs/dynamic_konect_project_datasets/"
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFO = os.path.join(PROJECT_DIR, "datasets-info")

class Dataset:
    """Dataset treatment"""

    def __init__(self, dataset_name : str, path: str = "./datasets"):
        self.name = dataset_name
        self.path = path # dir with datasets dirs
        self.adj = None # (l, n, n)-tensor or (n, n)-tensor
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
        adj - (l, n, n)-tensor, where l is number of batches
              (n, n)-tensor if batches is None
        
        features - #TODO (to Drobyshev) add info for shape
        label - #TODO (to Drobyshev) add info for shape
        """
        
        

        with open(os.path.join(INFO, "all.json")) as _:
            info = json.load(_)
        #TODO (to Drobyshev) Add to set all magi datasets
        magi_datasets = {"cora", "citeseer", "pubmed", "reddit", "ogbn-arxiv"}
        
        dname = self.name.lower()
        if self.name in info:
            is_directed = info[self.name]['d'] == 'directed'
            if batches == None:
                self._load_konect(batches_num = 1, is_directed = is_directed)
                self.adj = self.adj[0]
            else:
                self._load_konect(batches_num = batches, is_directed = is_directed)
        elif dname in magi_datasets:
            download_flag = False
            for filename in {f"{dname}_feat.npy", f"{dname}_label.npy", f"{dname}_coo_adj.joblib"}:
                if not os.path.isfile(os.path.join(self.path, dname, filename)):
                    print(f"file not found: {filename}")
                    print("Downloading files...")
                    download_flag = True
                    break
            if download_flag:
                self._load_magi()
                self._save_magi(coo_adj = True)
            else:
                self._load_npy_format(coo_adj = True)
        elif dname in {"acm", "bat", "dblp", "eat", "uat"}:
            self._load_npy_format()
        else:
            raise ValueError(f"Unsupported dataset: {self.name}")

        if tensor_type == "dense":
            self.adj = self.adj.to_dense()
        elif tensor_type == "coo":
            self.adj = self.adj.coalesce()
        elif tensor_type == "csr":
            self.adj = self.adj.to_sparse_csr()
        elif tensor_type == "csc":
            self.adj = self.adj.to_sparse_csc()
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

    def _save_magi(self, coo_adj = False):
        dname = self.name.lower()
        save_dir = os.path.join(self.path, dname)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, f'{dname}_feat.npy'), self.features.numpy())
        np.save(os.path.join(save_dir, f'{dname}_label.npy'), self.label.numpy())

        adj = self.adj.coalesce()

        if coo_adj:
            adj_data = {
                'indices': adj.indices().numpy(),
                'values': adj.values().numpy(),
                'shape': adj.shape
            }
            joblib.dump(adj_data, os.path.join(save_dir, f'{dname}_coo_adj.joblib'))
        else:
            adj_dense = adj.to_dense().numpy()
            np.save(os.path.join(save_dir, f'{dname}_adj.npy'), adj_dense)

    def _load_npy_format(self, coo_adj = False):
        dname = self.name.lower()
        load_dir = os.path.join(self.path, dname)

        features = np.load(os.path.join(load_dir, f"{dname}_feat.npy"))
        labels = np.load(os.path.join(load_dir, f"{dname}_label.npy"))
        self.features = torch.tensor(features, dtype=torch.float)
        self.label = torch.tensor(labels, dtype=torch.long)

        if coo_adj:
            adj_data = joblib.load(os.path.join(load_dir, f"{dname}_coo_adj.joblib"))
            self.adj = torch.sparse_coo_tensor(adj_data['indices'], adj_data['values'], size=adj_data['shape'])
        else:
            adj_data = np.load(os.path.join(load_dir, f"{dname}_adj.npy"), allow_pickle=True)
            rows, cols = adj_data.nonzero()
            values_np = adj_data[rows, cols]
            indices_np = np.vstack((rows, cols))
            indices = torch.from_numpy(indices_np).long()
            values = torch.from_numpy(values_np).float()
            self.adj = torch.sparse_coo_tensor(indices, values, size=adj_data.shape)
    
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
        #FiXME -- don't add nodes with no edges
        # nodes_set = set(i.tolist() + j.tolist())
        # print(len(nodes_set), min(nodes_set), max(nodes_set))
        adjs = []
        for num in range(batches_num):
            mask = (t == num)
            adj_index = np.vstack((i[mask], j[mask]))
            adj = torch.sparse_coo_tensor(adj_index, w[mask], size=(num_nodes, num_nodes)).coalesce()
            if not is_directed:
                adj = adj + torch.t(adj)
            adjs.append(adj)
        self.adj = torch.stack(adjs) # 3-dimensional tensor
    
    def _save_konect(self, path = None, int_weights = True):
        main_adj = self.adj.to_sparse_coo().coalesce()
        edges_num = main_adj._nnz()
        nodes_num = main_adj.size(0)
        if main_adj.ndim == 2:
            batches_num = 1
            adjs = [main_adj]
        elif main_adj.ndim == 3:
            batches_num = main_adj.shape[0]
            adjs = [main_adj[i] for i in range(batches_num)]
        else:
            raise ValueError(f"Неподдерживаемая размерность: {adj.ndim}")
        lines = []
        lines_num = edges_num + 1
        lines.append(f"{nodes_num} {lines_num}")
        for num, adj in enumerate(adjs):
            batch = num + 1
            indices = adj.indices()
            values = adj.values()
            for i in range(edges_num):
                source = indices[0, i].item()
                target = indices[1, i].item()
                weight = values[i].item()
                if int_weights:
                    weight = int(weight)
                lines.append(f"{source} {target} {weight} {batch}")
        if path:
            output_dir = os.path.join(path, self.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, f"out.{self.name}.{batches_num}_batches")
            with open(filepath, "w") as f:
                for line in lines:
                    f.write(line + '\n')
        else:
            for line in lines:
                print(line)

def list_konect_datasets():
    with open(os.path.join(INFO, "all.json")) as _:
        info = json.load(_)
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

def save_small_datasets_in_konect_format():
    dir_small_datasets = os.path.join(PROJECT_DIR, "test/graphs/small")
    dir_output = os.path.join(PROJECT_DIR, "test/graphs/small_konect")
    os.makedirs(dir_output, exist_ok = True)
    for dname in os.listdir(dir_small_datasets):
        print(dname)
        ds = Dataset(dname, path = dir_small_datasets)
        ds.load()
        ds._save_konect(dir_output)
        print("ok")


if __name__ == "__main__":
    list_konect_datasets()