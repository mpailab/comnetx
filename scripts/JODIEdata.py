import torch
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import Data

# name: 'reddit', 'wikipedia', 'mooc', 'lastfm'.
def load_dynamic_dataset(name, root='/auto/datasets/graphs/comnetx/JODIEdyn'):
    """
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    print(f"Загрузка датасета: {name}...")
    
    dataset = JODIEDataset(root=root, name=name)
    
    data = dataset[0]

    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=0.05, test_ratio=0.05
    )

    BATCH_SIZE = 20
    
    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

    info = {
        "num_nodes": data.num_nodes,
        "num_events": data.num_events,
        "feat_dim": data.msg.size(1) if data.msg is not None else 0, # Размерность фичей ребра
        "time_span": (data.t.min().item(), data.t.max().item())
    }

    return train_loader, val_loader, test_loader, info

def snapshot(temporal_data, target_time, show_nodes_limit=100):
    """
    Строит и рисует граф, состоящий из всех событий, произошедших до target_time.
    """
    # 1. Создаем маску: выбираем события, где время <= target_time
    mask = temporal_data.t <= target_time
    
    # 2. Фильтруем узлы и ребра
    src_snapshot = temporal_data.src[mask]
    dst_snapshot = temporal_data.dst[mask]
    t_snapshot = temporal_data.t[mask]
    
    # Количество событий в этот момент
    num_events = mask.sum().item()
    
    if num_events == 0:
        print(f"В момент времени {target_time:.4f} граф еще пуст.")
        return

    # 3. Собираем edge_index для статического графа
    # Стакаем источники и назначения в матрицу [2, E]
    edge_index = torch.stack([src_snapshot, dst_snapshot], dim=0)
    print("edge_index =", edge_index)
    
    # Создаем объект Data (как обычный граф PyG)
    snapshot_data = Data(edge_index=edge_index, num_nodes=temporal_data.num_nodes)

    # --- Статистика ---
    print(f"--- Срез на время t={target_time:.4f} ---")
    print(f"Событий (ребер) к этому моменту: {num_events}")
    # Уникальные активные узлы (те, кто участвовал в событиях)
    active_nodes = torch.unique(edge_index).numel()
    print(f"Активных узлов: {active_nodes}")


if __name__ == "__main__":
    # Выберите любой: 'reddit', 'wikipedia', 'mooc', 'lastfm'
    DATASET_NAME = 'lastfm'
    
    train_loader, val_loader, test_loader, info = load_dynamic_dataset(DATASET_NAME)
    
    print(f"\nСтатистика {DATASET_NAME}:")
    print(f"Всего событий: {info['num_events']}")
    print(f"Уникальных узлов: {info['num_nodes']}")
    print(f"Размерность фичей взаимодействия (msg): {info['feat_dim']}")
    
    print("\nПример одного батча из train_loader:")
    for batch in train_loader:
        print(f"Src nodes: {batch.src}") # i
        print(f"Dst nodes: {batch.dst}") # j
        print(f"Timestamps: {batch.t}") # t
        print(f"Messages (Edge Features): {batch.msg}") # feat
        print(f"Labels: {batch.y}")      # labels
        break

    dataset = JODIEDataset(root='/auto/datasets/graphs/comnetx/JODIEdyn', name=DATASET_NAME)
    data = dataset[0]

    target_t = data.t[10].item() 
    snapshot(data, target_t)