import torch
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import Data
import math


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
        val_ratio=0.0, test_ratio=0.0
    )

    BATCH_SIZE = 30
    
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

def create_3d_sparse_graph(temporal_data):
    """
    Преобразует динамический граф в 3D-разреженный тензор формата COO: 
    [Индекс_времени, Источник, Назначение].
    """
    # 1. Извлекаем данные о ребрах и времени
    src = temporal_data.src
    dst = temporal_data.dst
    t = temporal_data.t
    
    # 2. Дискретизация времени
    # unique_times - сами метки времени, time_indices - их целочисленные индексы (0, 1, 2...)
    unique_times, time_indices = torch.unique(t, return_inverse=True)
    
    num_time_steps = len(unique_times)
    # PyG иногда не указывает num_nodes явно, вычисляем максимум из src и dst, если нужно
    num_nodes = temporal_data.num_nodes if hasattr(temporal_data, 'num_nodes') else max(src.max(), dst.max()).item() + 1
    
    # 3. Формируем индексы для разреженного тензора
    # Собираем матрицу размерности [3, num_events], где строки: Время, Источник, Назначение
    indices = torch.stack([time_indices, src, dst], dim=0)
    
    # 4. Задаем значения для существующих ребер (единицы)
    values = torch.ones(indices.size(1), dtype=torch.float)
    
    # 5. Создаем PyTorch Sparse Tensor
    # Размерность: (Количество_временных_шагов, Количество_узлов, Количество_узлов)
    size = (num_time_steps, num_nodes, num_nodes)
    sparse_3d_tensor = torch.sparse_coo_tensor(indices, values, size=size)
    
    print(f"\n--- 3D Разреженное представление графа ---")
    print(f"Размерность тензора (Time, Nodes, Nodes): {size}")
    print(f"Количество ненулевых элементов (рёбер): {sparse_3d_tensor._nnz()}")
    
    return sparse_3d_tensor, unique_times

def get_time_window(sparse_3d, start_t, end_t):
    """
    Безопасно извлекает временное окно из 3D-разреженного тензора,
    обходя ограничения PyTorch на срезы (slicing) для SparseCPU.
    """
    # Достаем внутренние массивы тензора
    indices = sparse_3d._indices()
    values = sparse_3d._values()
    
    # 1. Создаем маску для фильтрации по времени (нулевая размерность)
    mask = (indices[0] >= start_t) & (indices[0] < end_t)
    
    # 2. Фильтруем индексы и значения
    window_indices = indices[:, mask].clone() # clone() важен, чтобы не повредить исходный тензор
    window_values = values[mask]
    
    # 3. Сдвигаем время так, чтобы окно начиналось с t=0
    window_indices[0] = window_indices[0] - start_t
    
    num_nodes = sparse_3d.size(1)
    time_steps_in_window = end_t - start_t
    
    # --- Создаем 3D срез ---
    window_3d = torch.sparse_coo_tensor(
        window_indices, 
        window_values, 
        size=(time_steps_in_window, num_nodes, num_nodes)
    )
    
    # --- Создаем 2D схлопнутый граф за этот период ---
    # Берем только строки [Источник, Назначение]
    indices_2d = window_indices[1:]
    
    # Если за 100 шагов узлы связывались несколько раз, убираем дубликаты
    # чтобы получить классическую матрицу смежности
    unique_indices_2d = torch.unique(indices_2d, dim=1)
    values_2d = torch.ones(unique_indices_2d.size(1), dtype=torch.float)
    
    window_2d = torch.sparse_coo_tensor(
        unique_indices_2d, 
        values_2d, 
        size=(num_nodes, num_nodes)
    )
    
    return window_3d, window_2d  

def compress_time_dimension(sparse_3d, target_steps=1000):
    """
    Сжимает временную размерность 3D-разреженного графа, 
    объединяя заданное количество последовательных временных шагов в один.
    """
    original_steps = sparse_3d.size(0)
    num_nodes = sparse_3d.size(1)
    
    bin_size = math.ceil(original_steps / target_steps)
    
    print(f"\n--- Сжатие временной размерности ---")
    print(f"Исходно: {original_steps} шагов -> Станет: {target_steps} шагов.")
    print(f"Размер окна: {bin_size} старых шагов в 1 новом.")
    
    indices = sparse_3d._indices().clone()
    
    indices[0] = indices[0] // bin_size
    
    # ВАЖНЫЙ МОМЕНТ: 
    # При сжатии у нас появятся дублирующиеся столбцы в indices.
    # indices = torch.unique(indices, dim=1)
    
    values = torch.ones(indices.size(1), dtype=torch.float)
    
    compressed_sparse_3d = torch.sparse_coo_tensor(
        indices, 
        values, 
        size=(target_steps, num_nodes, num_nodes)
    )
    
    # compressed_sparse_3d = compressed_sparse_3d.coalesce()
    
    print(f"Количество событий после сжатия (уникальных связей в окнах): {compressed_sparse_3d._nnz()}")
    
    return compressed_sparse_3d

if __name__ == "__main__":
    # Выберите любой: 'reddit', 'wikipedia', 'mooc', 'lastfm'
    DATASET_NAME = 'reddit'
    
    train_loader, _, _, info = load_dynamic_dataset(DATASET_NAME)
    
    dataset = JODIEDataset(root='/auto/datasets/graphs/comnetx/JODIEdyn', name=DATASET_NAME)
    data = dataset[0]

    print(f"Всего событий: {info['num_events']}")
    print(f"Уникальных узлов: {info['num_nodes']}")
    print(f"Размерность фичей взаимодействия (msg): {info['feat_dim']}")

    print(f"Src nodes: {data.src}") # i
    print(f"Dst nodes: {data.dst}") # j
    print(f"Timestamps: {data.t}") # t
    print(f"Messages (Edge Features): {data.msg}") # feat
    print(f"Labels: {data.y}")      # labels

    sparse_3d, time_mapping = create_3d_sparse_graph(data)
    print(sparse_3d)

    # Берем срез (получаем 2D разреженную матрицу смежности для этого момента)
    window_3d, window_2d = get_time_window(sparse_3d, start_t=0, end_t=1000)
    
    # print(window_3d)
    print(f"Количество событий в окне: {window_3d._nnz()}")
    print(f"\n--- 2D Схлопнутый граф ---")
    print(f"Размерность: {window_2d.shape}")
    print(f"Уникальных связей за этот период: {window_2d._nnz()}")
    print(f"Edge Index для PyG:\n{window_2d._indices()}")

    compressed_3d = compress_time_dimension(sparse_3d, target_steps=100)
    
    # 3. Проверяем, как выглядит первый шаг (t=0)
    snapshot_0 = compressed_3d[1]
    print(f"\nСжатая по времени матрица")
    print(compressed_3d)

# if __name__ == "__main__":
#     DATASET_NAME = 'reddit'
    
#     train_loader, val_loader, test_loader, info = load_dynamic_dataset(DATASET_NAME)
    
#     print(f"\nСтатистика {DATASET_NAME}:")
#     print(f"Всего событий: {info['num_events']}")
#     print(f"Уникальных узлов: {info['num_nodes']}")
#     print(f"Размерность фичей взаимодействия (msg): {info['feat_dim']}")
    
#     print("\nПример одного батча из train_loader:")
#     for batch in train_loader:
#         print(f"Src nodes: {batch.src}") # i
#         print(f"Dst nodes: {batch.dst}") # j
#         print(f"Timestamps: {batch.t}") # t
#         print(f"Messages (Edge Features): {batch.msg}") # feat
#         print(f"Labels: {batch.y}")      # labels
#         break

#     dataset = JODIEDataset(root='/auto/datasets/graphs/comnetx/JODIEdyn', name=DATASET_NAME)
#     data = dataset[0]

#     target_t = data.t[100].item() 
#     snapshot(data, target_t)