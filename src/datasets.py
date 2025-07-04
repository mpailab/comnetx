import torch
from torch_geometric.datasets import Planetoid, Reddit
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

class Dataset:
    """Dataset treatment"""

    # TODO (to konovalov) add batch supportion
    # TODO (to uporova) add loader for npy and npz formats
    # TODO (to drobyshev) add loader for magi format

    def __init__(self, dataset_name : str, path: str = "./datasets"):
        self.name = dataset_name
        self.path = path
        self.adj = None
        self.features = None
        self.label = None

    def load(self, tensor_type : str = "coo") -> torch.Tensor:
        """
        Load dataset

        Parameters
        ----------
        tensor_type : str
            Type of output tensor: coo, csr, csc, dense
            Default: coo
        """

        if "youtube-u-growth" in self.name:
            self._load_youtube_u_growth()
        elif self.name.lower() in {"cora", "citeseer", "pubmed", "reddit"} or self.name.startswith("ogbn-"):
            self._load_magi()
        else:
            raise ValueError(f"Unsupported dataset: {self.name}")

        if tensor_type == "dense":
            return self.adj.to_dense()
        elif tensor_type == "coo":
            return self.adj.coalesce()
        elif tensor_type == "csr":
            return self.adj.to_sparse_csr()
        elif tensor_type == "csc":
            return self.adj.to_sparse_csc()
        else:
            raise ValueError(f"Unsupported tensor type for torch.sparse: {tensor_type}")

    def _load_magi(self):
        name = self.name.lower()
        
        if name in {"cora", "citeseer", "pubmed"}:
            dataset = Planetoid(root=self.path, name=name.capitalize())
            data = dataset[0]

        elif name == "reddit":
            dataset = Reddit(root=self.path)
            data = dataset[0]

        elif name.startswith("ogbn-"):
            dataset = PygNodePropPredDataset(name=name, root=self.path)
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

    def _load_youtube_u_growth(self):
        filepath = os.path.join(self.path, self.name)

        edges = pd.read_csv(
            filepath,
            sep=r"\s+",
            comment="%",
            header=None,
            names=["src", "dst", "weight", "timestamp"]
        )

        edges['row'] = edges[["src", "dst"]].min(axis=1)
        edges['col'] = edges[["src", "dst"]].max(axis=1)
        edges = edges.drop_duplicates(subset=["row", "col"])

        num_nodes = max(edges['row'].max(), edges['col'].max()) + 1

        row = torch.tensor(pd.concat([edges['row'], edges['col']]).values, dtype=torch.long)
        col = torch.tensor(pd.concat([edges['col'], edges['row']]).values, dtype=torch.long)

        edge_index = torch.stack([row, col])  # shape: [2, num_edges * 2]
        values = torch.ones(edge_index.shape[1], dtype=torch.float32)

        self.adj = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes)).coalesce()

        self.features = None
        self.label = None