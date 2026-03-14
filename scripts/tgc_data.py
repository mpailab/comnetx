import os
import torch
import numpy as np
import joblib
from torch_geometric.data import Data

def load_tgc_dataset(nodes_path, edges_path, labels_path):
    features_list = []
    with open(nodes_path, 'r') as f:
        lines = f.readlines()
        
        first_line = lines[0].strip().split()
        if len(first_line) == 2 and '.' not in first_line[0]:
            num_nodes = int(first_line[0])
            dim = int(first_line[1])
            data_start_idx = 1
        else:
            data_start_idx = 0
            dim = len(first_line) if len(first_line) > 2 else len(lines[1].strip().split())
            num_nodes = len(lines) - data_start_idx

        x = torch.zeros((num_nodes, dim), dtype=torch.float)
        
        for i, line in enumerate(lines[data_start_idx:]):
            parts = line.strip().split()
            if not parts or i >= num_nodes: continue
            
            if len(parts) == dim + 1:
                node_id = int(float(parts[0]))
                vector = [float(v) for v in parts[1:]]
            else:
                node_id = i
                vector = [float(v) for v in parts]
            
            if node_id < num_nodes:
                x[node_id] = torch.tensor(vector)

    print(f"Загружено узлов: {x.shape[0]}, размерность признаков: {x.shape[1]}")

    y = torch.zeros(x.shape[0], dtype=torch.long)
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                node_id, label = int(float(parts[0])), int(float(parts[1]))
                if node_id < x.shape[0]:
                    y[node_id] = label
    else:
        print(f"Файл меток {labels_path} не найден. Используются нули.")

    edges, timestamps = [], []
    with open(edges_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            u, v, t = int(float(parts[0])), int(float(parts[1])), float(parts[2])
            edges.append([u, v])
            timestamps.append([t])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_time = torch.tensor(timestamps, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_time, y=y)
    return data

def download_tgc(name: str, data_dir: str, save_path: str, num_batches: int = 100):
    dname = name.lower()
    
    nodes_path = os.path.join(data_dir, name, 'feature.txt')
    edges_path = os.path.join(data_dir, name, f'{name}.txt')
    labels_path = os.path.join(data_dir, name, 'node2label.txt')

    dataset = load_tgc_dataset(nodes_path, edges_path, labels_path)
    
    features = dataset.x
    labels = dataset.y
    num_nodes = labels.size(0)

    save_dir = os.path.join(save_path, dname)
    os.makedirs(save_dir, exist_ok=True)

    if features is not None:
        feat_path = os.path.join(save_dir, f"{dname}_feat.npy")
        np.save(feat_path, features.detach().cpu().numpy())
        print(f"Фичи сохранены: {feat_path}")

    label_path = os.path.join(save_dir, f"{dname}_label.npy")
    if labels is not None and labels.dim() == 1:
        np.save(label_path, labels.detach().cpu().numpy())
    else:
        np.save(label_path, np.full((num_nodes,), -1, dtype=np.int64))
    print(f"Метки сохранены: {label_path}")

    timestamps = dataset.edge_attr.flatten()
    sorted_indices = torch.argsort(timestamps)
    sorted_u = dataset.edge_index[0, sorted_indices]
    sorted_v = dataset.edge_index[1, sorted_indices]

    num_edges = dataset.edge_index.shape[1]
    batch_indices = torch.arange(num_edges) * num_batches // num_edges

    indices = torch.stack([batch_indices, sorted_u, sorted_v], dim=0)
    values = torch.ones(num_edges, dtype=torch.float32)
    size = (num_batches, num_nodes, num_nodes)
    sparse_adj_3d = torch.sparse_coo_tensor(indices, values, size=size).coalesce()
    
    adj_data = {
        "indices": sparse_adj_3d.indices().cpu().numpy(),
        "values": sparse_adj_3d.values().cpu().numpy(),
        "shape": sparse_adj_3d.shape,
    }
    
    adj_path = os.path.join(save_dir, f"{dname}_coo_adj-{num_batches}_batches.joblib")
    joblib.dump(adj_data, adj_path)
    print(f"Временной граф сохранен: {adj_path}")
    print(f"--- Обработка {name} успешно завершена! ---\n")

    return sparse_adj_3d, features, labels

if __name__ == "__main__":
    # arxivai  arxivcs  arxivlarge  arxivmath  arxivphy  brain  dblp-tgc  patent
    DATASET_NAME = "brain" 
    RAW_DATA_DIR = "/auto/datasets/graphs/comnetx/4TGC/data/" 
    PROCESSED_DATA_DIR = "/auto/datasets/graphs/comnetx/4TGC/data_npy" 
    for num_batches in [1,10,100,1000]:
        sparse_adj_3d, features, labels = download_tgc(DATASET_NAME, RAW_DATA_DIR, PROCESSED_DATA_DIR, num_batches)

    # print(f"sparse_adj_3d = {sparse_adj_3d}")
    # print(f"features = {features}")
    # print(f"labels = {labels}")