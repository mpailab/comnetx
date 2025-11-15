import torch
from torch_geometric.datasets import Planetoid, Reddit, Amazon

from ogb.nodeproppred import PygNodePropPredDataset
import tempfile
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os.path
from ogb.nodeproppred import PygNodePropPredDataset
import pickle

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
        self.is_directed = False
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
        magi_datasets = {"cora", "citeseer", "pubmed", "reddit", "ogbn-arxiv", "ogbn-products", "ogbn-papers100M", "amazon-photo", "amazon-computers"}
        
        dname = self.name.lower()
        if self.name in info:
            self.is_directed = info[self.name]['d'] == 'directed'
            if batches == None:
                self._load_konect(batches_num = 1)
                self.adj = self.adj[0]
            else:
                self._load_konect(batches_num = batches)
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
        elif dname.startswith("static") or dname.startswith("stream"):
            parts = self.name.split("_")

            if dname.startswith("static"):
                dataset_type = "static"
            elif dname.startswith("stream"):
                dataset_type = "stream"
            num_snapshots = 5 
            num_nodes = int(parts[2])
            mu = float(parts[3])
            beta = float(parts[4])
            self._load_prgpt_dataset(dataset_type=dataset_type, num_nodes=num_nodes, 
                                     mu=mu, beta=beta, num_snapshots=num_snapshots)
        elif dname.startswith("sbm") or dname.startswith("tsbm"):
            self._load_sbm()

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
            elif name.startswith("amazon-"):
                amazon_name = name.replace("amazon-", "").capitalize()
                dataset = Amazon(root=tmpdir, name=amazon_name)
                data = dataset[0]
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

    def _load_prgpt_dataset(self, dataset_type='static',
                       num_nodes=10000,
                       mu=2.5,
                       beta=3.0,
                       num_snapshots=5):
        base_path = self.path
        # ---------- STATIC ----------
        if dataset_type == 'static':
            edges_path = os.path.join(base_path, f'static_5_{num_nodes}_{mu:.1f}_{beta:.1f}_edges_list.pickle')
            gnd_path = os.path.join(base_path, f'static_5_{num_nodes}_{mu:.1f}_{beta:.1f}_gnd_list.pickle')

            with open(edges_path, 'rb') as f:
                edges_list = pickle.load(f)
            with open(gnd_path, 'rb') as f:
                labels_list = pickle.load(f)

            adjs = []
            all_labels = []


            for snap in range(num_snapshots):
                snap_edges = edges_list[snap]
                snap_labels = torch.tensor(labels_list[snap], dtype=torch.long)
                all_labels.append(snap_labels) 
                edge_index = torch.tensor(snap_edges, dtype=torch.long).t() 
                values = torch.ones(edge_index.shape[1], dtype=torch.float32)
                adj_sparse = torch.sparse_coo_tensor(indices=edge_index, values=values,
                                                     size=(num_nodes, num_nodes), dtype=torch.float32)

                adj_sparse = adj_sparse.coalesce()
                adj_t = torch.sparse_coo_tensor(indices=adj_sparse.indices().flip(0), 
                                                values=adj_sparse.values(),
                                                size=adj_sparse.shape)
                adj_sparse = (adj_sparse + adj_t).coalesce()
                adjs.append(adj_sparse)


            self.adj = torch.stack(adjs)
            self.label = torch.stack(all_labels)
            self.is_directed = False

        # ---------- STREAM (DYNAMIC) ----------
        elif dataset_type == 'stream':
            adjs = []
            all_labels = []

            for snap in range(0, num_snapshots):
                if snap + 1 == 5:
                    beta = 1.0
                edges_path = os.path.join(base_path, f'stream_{snap + 1}_{num_nodes}_{mu:.1f}_{beta:.1f}_edges_list.pickle')
                gnd_path = os.path.join(base_path, f'stream_{snap + 1}_{num_nodes}_{mu:.1f}_{beta:.1f}_gnd.pickle')

                with open(edges_path, 'rb') as f:
                    snap_edges = pickle.load(f)
                with open(gnd_path, 'rb') as f:
                    snap_labels = torch.tensor(pickle.load(f), dtype=torch.long)
                    all_labels.append(snap_labels)
                
                all_edges = np.vstack([np.array(batch) for batch in snap_edges])

                edge_index = torch.tensor(all_edges, dtype=torch.long).t()
                values = torch.ones(all_edges.shape[0], dtype=torch.float32)

                adj_sparse = torch.sparse_coo_tensor(indices=edge_index, values=values, size=(num_nodes, num_nodes))
                adj_sparse = adj_sparse.coalesce()
                adj_t = torch.sparse_coo_tensor(indices=adj_sparse.indices().flip(0),
                                            values=adj_sparse.values(), size=adj_sparse.shape)
                adj_sparse = (adj_sparse + adj_t).coalesce()

                adjs.append(adj_sparse)

            self.adj =  torch.stack(adjs)
            self.label = torch.stack(all_labels)
            self.is_directed = False

        else:
            raise ValueError("dataset_type must be'static' or 'stream'.")

    def _load_sbm(self, device="cpu"):
        """
        Load sbm dataset from test/graphs/sbm

        Parameters
        ----------
        data_type : static / temporal
        """
        base_path = self.path
        file_path = os.path.join(base_path, self.name + ".pt")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"SBM dataset file not found: {file_path}")
        raw = torch.load(file_path, map_location="cpu")
        meta = raw.get("meta", {})
        is_temporal = meta.get("temporal", False)
        is_directed = meta.get("directed", False)

        if is_temporal:
            adjs, labels = self._load_temporal_sbm_graph(file_path, device=device)
        else:
            adjs, labels = self._load_static_sbm_graph(file_path, device=device)

        self.adj = adjs
        self.label = labels
        self.features = None
        self.is_directed = is_directed

    def _load_static_sbm_graph(self, fname, device='cpu'):
        """
        Load .pt file, returns:
        - adj_sparse: torch.sparse_coo_tensor (shape n x n), coalesced, dtype=torch.float32
        - labels_t: torch.tensor(labels, dtype=torch.long)
        """
        if not os.path.isabs(fname) and not os.path.exists(fname):
            candidate = os.path.join(os.getcwd(), "sbm", fname)
            if os.path.exists(candidate):
                fname = candidate

        data = torch.load(fname, map_location='cpu')  # load on cpu
        rows = np.asarray(data["rows"], dtype=np.int64)
        cols = np.asarray(data["cols"], dtype=np.int64)
        n = int(data["meta"]["n"])
        if rows.size == 0:
            indices = torch.empty((2, 0), dtype=torch.long)
            values = torch.empty((0,), dtype=torch.float32)
        else:
            indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
            values = torch.ones(indices.shape[1], dtype=torch.float32)
        adj = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce().to(device)
        labels_t = torch.tensor(data["labels"], dtype=torch.long, device=device)
        return adj, labels_t

    def _load_temporal_sbm_graph(self, path, device='cpu'):
        """
        Load dynamic SBM (.pt), returns:
        - adj_sparse: torch.sparse_coo_tensor [n_steps, n, n]
        - labels_t: torch.tensor [n_steps, n]
        """
        
        data = torch.load(path, map_location='cpu')
        meta = data.get('meta', {})
        temporal = meta.get('temporal', False)
        if not temporal:
            raise ValueError("File is not a temporal SBM graph")

        n_steps = len(data['rows'])
        n = meta['n']
        directed = meta.get('directed', False)

        adj_snapshots = []
        for t in range(n_steps):
            rows = np.asarray(data['rows'][t], dtype=np.int64)
            cols = np.asarray(data['cols'][t], dtype=np.int64)
            if len(rows) == 0:
                indices = torch.empty((2,0), dtype=torch.long, device=device)
                values = torch.empty((0,), dtype=torch.float32, device=device)
            else:
                indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
                values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

            if not directed:
                mirrored = torch.flip(indices, dims=[0])
                indices = torch.cat([indices, mirrored], dim=1)
                values = torch.cat([values, values], dim=0)

            adj_snapshots.append(torch.sparse_coo_tensor(indices, values, (n,n), device=device).coalesce())

        all_indices = []
        all_values = []
        for t, adj in enumerate(adj_snapshots):
            if adj._nnz() == 0:
                continue
            t_idx = torch.full((1, adj._nnz()), t, dtype=torch.long, device=device)
            idx_3d = torch.cat([t_idx, adj.indices()], dim=0)
            all_indices.append(idx_3d)
            all_values.append(adj.values())

        if all_indices:
            indices = torch.cat(all_indices, dim=1)
            values = torch.cat(all_values)
        else:
            indices = torch.empty((3,0), dtype=torch.long, device=device)
            values = torch.empty((0,), dtype=torch.float32, device=device)

        adj_sparse_3d = torch.sparse_coo_tensor(indices, values, (n_steps, n, n), device=device).coalesce()
        labels_t = torch.stack([torch.tensor(l, dtype=torch.long, device=device) for l in data['labels']], dim=0)

        return adj_sparse_3d, labels_t

    def _load_konect(self, batches_num = 1):
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
            if not self.is_directed:
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
            raise ValueError(f"Unsupported adjacency ndim: {main_adj.ndim}")
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
