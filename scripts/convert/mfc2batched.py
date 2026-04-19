import pickle
import networkx as nx
import math
import os

import pickle
import networkx as nx
import numpy as np
import scipy.sparse as sp
import os

def convert_mfc_pkl_to_txt(dataset_name, input_dir, output_dir="Converted"):
    """
    Конвертирует pkl датасет в формат:
    n m
    i j w t
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(input_dir, f"{dataset_name}.pkl")
    output_path = os.path.join(output_dir, f"mfc_{dataset_name}.txt")

    print(f"Обработка датасета: {dataset_name}...")

    # Загрузка pkl (используем параметры из вашего лоадера для совместимости)
    with open(file_path, 'rb') as f:
        try:
            graph_snapshots = pickle.load(f, encoding='bytes')
        except (ValueError, TypeError):
            f.seek(0)
            graph_snapshots = pickle.load(f, encoding='bytes', protocol=2)

    # DBLP в оригинальном коде MFC обрезается до 8 снэпшотов
    if dataset_name == "DBLP":
        graph_snapshots = graph_snapshots[:8]

    unique_nodes = set()
    all_interactions = [] # Список для хранения кортежей (i, j, w, t)

    for t, g in enumerate(graph_snapshots):
        # Добавляем вершины в общий сет для подсчета n
        unique_nodes.update(g.nodes())

        # Получаем разреженную матрицу смежности (COO формат удобен для итерации по ребрам)
        # Для nx.Graph (неориентированный) матрица будет симметричной
        adj = nx.adjacency_matrix(g).tocoo()

        for i_idx, j_idx, weight in zip(adj.row, adj.col, adj.data):
            # Получаем реальные ID вершин (если в графе они не 0..N)
            node_list = list(g.nodes())
            u = node_list[i_idx]
            v = node_list[j_idx]
            
            # В вашем примере вес всегда 1, а кратность ребра определяется повторением строк
            # Если вес в pkl > 1, создаем несколько строк
            for _ in range(int(weight)):
                all_interactions.append((u, v, 1, t))

    n = len(unique_nodes)
    m = len(all_interactions)

    # Запись в файл
    with open(output_path, 'w') as f:
        # Заголовок: кол-во вершин и общее кол-во строк ребер
        f.write(f"{n}\t{m}\n")
        
        for i, j, w, t in all_interactions:
            f.write(f"{i}\t{j}\t{w}\t{t}\n")

    print(f"Готово! Файл сохранен: {output_path}")
    print(f"Вершин (n): {n}, Строк ребер (m): {m}\n")


def convert_mfc_pkl_to_txt_uniform(dataset_name, input_dir, output_dir="Converted", num_snapshots=10):
    """
    Конвертирует pkl датасет в txt формат с равномерным распределением рёбер.
    Параметр num_snapshots задает итоговое количество временных отрезков.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(input_dir, f"{dataset_name}.pkl")
    output_path = os.path.join(output_dir, f"{dataset_name}_mfc_{num_snapshots}_batches.txt")

    print(f"Обработка датасета: {dataset_name} (целевое кол-во снэпшотов: {num_snapshots})...")

    # Загрузка графов
    with open(file_path, 'rb') as f:
        try:
            graph_snapshots = pickle.load(f, encoding='bytes')
        except (ValueError, TypeError):
            f.seek(0)
            graph_snapshots = pickle.load(f, encoding='bytes', protocol=2)

    # Ограничение DBLP как в оригинальном лоадере
    if dataset_name == "DBLP":
        graph_snapshots = graph_snapshots[:8]

    unique_nodes = set()
    chronological_edges = [] 

    # 1. Собираем все уникальные взаимодействия в хронологическом порядке
    for g in graph_snapshots:
        unique_nodes.update(g.nodes())
        
        # Перебираем рёбра (для nx.Graph это выдаст по одному ребру на каждую пару связанных вершин)
        for u, v, data in g.edges(data=True):
            # Если в исходном графе есть вес, учитываем его как кратность
            weight = int(data.get('weight', 1))
            for _ in range(weight):
                chronological_edges.append((u, v))

    total_unique_interactions = len(chronological_edges)
    n = len(unique_nodes)
    
    # 2. Равномерно распределяем время и генерируем симметричные записи
    final_output_rows = []
    
    for i, (u, v) in enumerate(chronological_edges):
        # Вычисляем новый индекс снэпшота: от 0 до num_snapshots - 1
        # Логика: i / total * num_snapshots дает равномерную нарезку
        new_t = math.floor((i / total_unique_interactions) * num_snapshots)
        
        # Защита от выхода за границы из-за особенностей float при последнем элементе
        new_t = min(new_t, num_snapshots - 1)
        
        # Записываем прямое ребро
        final_output_rows.append((u, v, 1, new_t))
        
        # Записываем обратное ребро (если это не петля на саму себя)
        if u != v:
            final_output_rows.append((v, u, 1, new_t))

    m = len(final_output_rows)

    # 3. Запись в файл
    with open(output_path, 'w') as f:
        f.write(f"{n}\t{m}\n")
        for i, j, w, t in final_output_rows:
            f.write(f"{i}\t{j}\t{w}\t{t}\n")

    print(f"Готово! Сохранено: {output_path}")
    print(f"Вершин (n): {n}, Строк в файле (m): {m}")
    # Статистика: сколько уникальных взаимодействий попало в каждый снэпшот
    edges_per_snap = total_unique_interactions / num_snapshots
    print(f"В среднем на один снэпшот приходится {int(edges_per_snap * 2)} строк (с учетом симметрии)\n")


# --- Настройки ---
DATA_PATH = "/auto/datasets/graphs/comnetx/baselines/MFC-TopoReg/Data" 
datasets = ["enron", "highschool", "DBLP", "Cora", "DBLPdyn"]

# Укажи нужное количество снэпшотов здесь
TARGET_SNAPSHOTS = 10 

for ds in datasets:
    try:
        convert_mfc_pkl_to_txt(ds, DATA_PATH)
    except Exception as e:
        print(f"Ошибка при обработке {ds}: {e}")