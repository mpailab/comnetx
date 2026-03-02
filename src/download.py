import os
import tempfile
import numpy as np
import torch
import joblib
import argparse
import sys
from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset
import torch_geometric
from torch_geometric.datasets import Planetoid, Reddit, Amazon
from ogb.nodeproppred import PygNodePropPredDataset

KNOWN_MAGI = ["cora", "citeseer", "pubmed", "reddit", "ogbn-products", "amazon-computers"]
ATTR_GRAPHS = ["flickr", "wikics", "nell", "reddit2", "yelp",
               "coauthorcs", "coauthorphysics", "wiki", "pubmed",
               "blogcatalog", "ppi", "facebook"]
OGB_GRAPHS = ["ogbn-products", "ogbn-arxiv", "ogbn-papers100m"]
DATASETS_PATH = "/auto/datasets/graphs"

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
            np.save(os.path.join(save_dir, f'{name}_feat.npy'), features.detach().cpu().numpy())
        
        if label is not None:
            np.save(os.path.join(save_dir, f'{name}_label.npy'), label.detach().cpu().numpy())

        values = torch.ones(edge_index.size(1), dtype=torch.float32)
        adj = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes)).coalesce()

        adj_data = {
            'indices': adj.indices().numpy(),
            'values': adj.values().numpy(),
            'shape': adj.shape
        }
        joblib.dump(adj_data, os.path.join(save_dir, f'{name}_coo_adj.joblib'))
        
    print(f"Dataset {name} prepared successfully.")

def download_attr_graph(name: str, save_path: str = DATASETS_PATH):
    dname = name.lower()
    # root для сырых PyG-данных (куда PyG сам скачает)
    root = os.path.join("/auto/datasets/graphs/comnetx/torch_geom_datasets/data", dname)
    if dname == "flickr":
        dataset = torch_geometric.datasets.Flickr(root=root)
    elif dname == "wikics":
        dataset = torch_geometric.datasets.WikiCS(root=root, is_undirected=True)
    elif dname == "nell":
        dataset = torch_geometric.datasets.NELL(root=root)
    elif dname == "reddit2":
        dataset = torch_geometric.datasets.Reddit2(root=root)
    elif dname == "yelp":
        dataset = torch_geometric.datasets.Yelp(root=root)
    elif dname == "coauthorcs":
        dataset = torch_geometric.datasets.Coauthor(root=root, name="CS")
    elif dname == "coauthorphysics":
        dataset = torch_geometric.datasets.Coauthor(root=root, name="Physics")
    elif dname == "wiki":
        dataset = torch_geometric.datasets.AttributedGraphDataset(root=root, name="Wiki")
    elif dname == "pubmed":
        dataset = torch_geometric.datasets.AttributedGraphDataset(root=root, name="PubMed")
    elif dname == "blogcatalog":
        dataset = torch_geometric.datasets.AttributedGraphDataset(root=root, name="BlogCatalog")
    elif dname == "ppi":
        dataset = torch_geometric.datasets.AttributedGraphDataset(root=root, name="PPI")
    elif dname == "facebook":
        dataset = torch_geometric.datasets.AttributedGraphDataset(root=root, name="Facebook")
    else:
        raise ValueError(f"Unknown attribute graph: {name}")

    data = dataset[0] if hasattr(dataset, "__getitem__") else dataset
    features = data.x
    labels = data.y
    edge_index = data.edge_index
    num_nodes = labels.size(0)

    save_dir = os.path.join(save_path, dname)
    os.makedirs(save_dir, exist_ok=True)

    if features is not None:
        np.save(os.path.join(save_dir, f"{dname}_feat.npy"), features.detach().cpu().numpy())
    if labels is not None and labels.dim() == 1:
        np.save(os.path.join(save_dir, f"{dname}_label.npy"), labels.detach().cpu().numpy())
    else:
        np.save(os.path.join(save_dir, f"{dname}_label.npy"), np.full((num_nodes,), -1, dtype=np.int64))

    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes)).coalesce()
    adj_data = {
        "indices": adj.indices().cpu().numpy(),
        "values": adj.values().cpu().numpy(),
        "shape": adj.shape,
    }
    joblib.dump(adj_data, os.path.join(save_dir, f"{dname}_coo_adj.joblib"))


def download_ogb_graph(name: str, save_path: str = DATASETS_PATH):
    dname = name.lower()
    save_dir = os.path.join(save_path, dname)
    os.makedirs(save_dir, exist_ok=True)

    if dname in ["ogbn-products", "ogbn-arxiv", "ogbn-papers100m"]:
        dataset = NodePropPredDataset(name=dname, root="/auto/datasets/graphs/comnetx/ogb/")
        graph, labels = dataset[0]
        labels = torch.from_numpy(labels).long().view(-1)
    else:
        dataset = LinkPropPredDataset(name=dname, root="/auto/datasets/graphs/comnetx/ogb/")
        graph = dataset[0]
        labels = None

    if graph["node_feat"] is not None:
        feats = torch.from_numpy(graph["node_feat"]).float()
        np.save(os.path.join(save_dir, f"{dname}_feat.npy"), feats.cpu().numpy())
    else:
        num_nodes = graph["num_nodes"]
        np.save(os.path.join(save_dir, f"{dname}_feat.npy"), np.ones((num_nodes, 1), dtype=np.float32))

    if labels is not None:
        np.save(os.path.join(save_dir, f"{dname}_label.npy"), labels.cpu().numpy())
    else:
        num_nodes = graph["num_nodes"]
        np.save(os.path.join(save_dir, f"{dname}_label.npy"), np.full((num_nodes,), -1, dtype=np.int64))

    indices = torch.from_numpy(graph["edge_index"]).long()
    num_nodes = graph["num_nodes"]
    values = torch.ones(indices.size(1), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()
    adj_data = {
        "indices": adj.indices().cpu().numpy(),
        "values": adj.values().cpu().numpy(),
        "shape": adj.shape,
    }
    joblib.dump(adj_data, os.path.join(save_dir, f"{dname}_coo_adj.joblib"))


def download_all():
    for ds in KNOWN_MAGI:
        download_and_process_magi(ds, DATASETS_PATH)
    for ds in ATTR_GRAPHS:
        download_attr_graph(ds, DATASETS_PATH)
    for ds in OGB_GRAPHS:
        download_ogb_graph(ds, DATASETS_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("dataset", nargs="?", help="Dataset name (cora, reddit, etc.)")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    args = parser.parse_args()
    
    if args.all:
        download_all()
    elif args.dataset:
        name = args.dataset.lower()
        if name in KNOWN_MAGI:
            download_and_process_magi(name, DATASETS_PATH)
        elif name in ATTR_GRAPHS:
            download_attr_graph(name, DATASETS_PATH)
        elif name in OGB_GRAPHS:
            download_ogb_graph(name, DATASETS_PATH)
        else:
            raise ValueError(f"Unknown dataset: {name}")
    else:
        print("Usage: python download.py cora")
        print("       python download.py --all")
        sys.exit(1)
