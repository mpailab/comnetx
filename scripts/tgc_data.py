import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np

def analyze_graph_dynamics(dataset):
    # Извлекаем метки времени из edge_attr
    # В нашем случае это тензор размерности [E, 1]
    timestamps = dataset.edge_attr.flatten().numpy()
    
    print(f"--- Анализ временной динамики ---")
    print(f"Минимальное время (t_min): {timestamps.min()}")
    print(f"Максимальное время (t_max): {timestamps.max()}")
    print(f"Общая длительность: {timestamps.max() - timestamps.min()}")
    
    # 1. Визуализация интенсивности событий
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(timestamps, bins=50, color='skyblue', edgecolor='black')
    plt.title('Распределение ребер во времени')
    plt.xlabel('Timestamp (Время)')
    plt.ylabel('Количество новых связей (событий)')
    plt.grid(axis='y', alpha=0.3)

    # 2. Накопительный график (как растет граф)
    plt.subplot(1, 2, 2)
    sorted_times = np.sort(timestamps)
    cumulative_edges = np.arange(len(sorted_times))
    plt.plot(sorted_times, cumulative_edges, color='salmon', linewidth=2)
    plt.title('Накопительный рост графа')
    plt.xlabel('Timestamp')
    plt.ylabel('Общее число ребер')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def load_tgc_dataset(nodes_path, edges_path, labels_path):
    # 1. Загрузка признаков (Features)
    features_list = []
    with open(nodes_path, 'r') as f:
        lines = f.readlines()
        
        # Проверяем, является ли первая строка заголовком (N d)
        first_line = lines[0].strip().split()
        if len(first_line) == 2 and '.' not in first_line[0]:
            # Это заголовок (например: "100 128")
            num_nodes = int(first_line[0])
            dim = int(first_line[1])
            data_start_idx = 1
        else:
            # Заголовка нет, данные начинаются сразу
            data_start_idx = 0
            dim = len(first_line) if len(first_line) > 2 else len(lines[1].strip().split())
            num_nodes = len(lines) - data_start_idx

        x = torch.zeros((num_nodes, dim), dtype=torch.float)
        
        for i, line in enumerate(lines[data_start_idx:]):
            parts = line.strip().split()
            if not parts or i >= num_nodes: continue
            
            # Если в строке dim+1 элементов, значит первый - это ID
            if len(parts) == dim + 1:
                node_id = int(float(parts[0]))
                vector = [float(v) for v in parts[1:]]
            else:
                node_id = i
                vector = [float(v) for v in parts]
            
            if node_id < num_nodes:
                x[node_id] = torch.tensor(vector)

    print(f"Загружено узлов: {x.shape[0]}, размерность признаков: {x.shape[1]}")

    # 2. Загрузка меток (Labels)
    y = torch.zeros(x.shape[0], dtype=torch.long)
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            node_id, label = int(float(parts[0])), int(float(parts[1]))
            if node_id < x.shape[0]:
                y[node_id] = label

    # 3. Загрузка ребер (Edges)
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

# Пути для вашего сервера
DATA_DIR = '/auto/datasets/graphs/comnetx/4TGC/data'
dataset = load_tgc_dataset(
    nodes_path= DATA_DIR + '/arXivAI/feature.txt', 
    edges_path= DATA_DIR + '/arXivAI/arxivAI.txt', 
    labels_path= DATA_DIR + '/arXivAI/node2label.txt'
)
print(f"Итоговый объект: {dataset}")

analyze_graph_dynamics(dataset)