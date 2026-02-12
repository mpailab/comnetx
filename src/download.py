import os
import tempfile
import numpy as np
import torch
import joblib
import argparse
import sys
from torch_geometric.datasets import Planetoid, Reddit, Amazon
from ogb.nodeproppred import PygNodePropPredDataset

def download_and_process_magi(dataset_name: str, save_path: str):
    """
    Downloads dataset (Planetoid, Reddit, OGB, Amazon), processes it 
    into standard numpy/joblib format and saves to save_path.
    """
    name = dataset_name.lower()
    save_dir = os.path.join(save_path, name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading and processing {name} to {save_dir}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        if name in {"cora", "citeseer", "pubmed"}:
            dataset = Planetoid(root=tmpdir, name=name.capitalize())
            data = dataset[0]

        elif name == "reddit":
            dataset = Reddit(root=tmpdir)
            data = dataset[0]

        elif name.startswith("ogbn-"):
            dataset = PygNodePropPredDataset(name=name, root=tmpdir)
            data = dataset[0]
            if hasattr(data, 'y'):
                data.y = data.y.view(-1)
                
        elif name.startswith("amazon-"):
            amazon_name = name.replace("amazon-", "").capitalize()
            dataset = Amazon(root=tmpdir, name=amazon_name)
            data = dataset[0]
        else:
            raise ValueError(f"Unknown MAGI-compatible dataset: {dataset_name}")

        edge_index = data.edge_index
        num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
        
        features = data.x
        label = data.y

        if features is not None:
            np.save(os.path.join(save_dir, f'{name}_feat.npy'), features.numpy())
        
        if label is not None:
            np.save(os.path.join(save_dir, f'{name}_label.npy'), label.numpy())

        values = torch.ones(edge_index.size(1), dtype=torch.float32)
        adj = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes)).coalesce()

        adj_data = {
            'indices': adj.indices().numpy(),
            'values': adj.values().numpy(),
            'shape': adj.shape
        }
        joblib.dump(adj_data, os.path.join(save_dir, f'{name}_coo_adj.joblib'))
        
    print(f"Dataset {name} prepared successfully.")

DATASETS_PATH = "./datasets"  # куда сохранять

def download_all():
    """Скачать все MAGI-совместимые датасеты"""
    known_datasets = ["cora", "citeseer", "pubmed", "reddit", "ogbn-products", "amazon-computers"]
    for ds in known_datasets:
        try:
            download_and_process_magi(ds, DATASETS_PATH)
        except Exception as e:
            print(f"Failed to download {ds}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("dataset", nargs="?", help="Dataset name (cora, reddit, etc.)")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    args = parser.parse_args()
    
    if args.all:
        download_all()
    elif args.dataset:
        download_and_process_magi(args.dataset, DATASETS_PATH)
    else:
        print("Usage: python download.py cora")
        print("       python download.py --all")
        sys.exit(1)
