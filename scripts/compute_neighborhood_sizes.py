import json
import os
import sys
import argparse
from datetime import datetime

# Добавляем путь к проекту
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from datasets import Dataset, KONECT_PATH, INFO
from optimizer import Optimizer
import torch

def compute_neighborhood_sizes(dataset_name, batches_strategy, max_step=5, skip_first_batch=True):
    """
    Вычисляет размеры окрестностей для каждого батча датасета
    
    Args:
        dataset_name: имя датасета
        batches_strategy: стратегия батчинга в формате "p:n"
        max_step: максимальный шаг окрестности (r)
    
    Returns:
        Список размеров окрестностей для каждого батча
    """
    # Загружаем датасет с указанной стратегией батчинга
    ds = Dataset(dataset_name, path=KONECT_PATH)
    ds.load(batches=batches_strategy)
    
    batches = torch.unbind(ds.adj)
    
    # Для каждого батча вычисляем размеры окрестностей
    batch_neighborhood_sizes = []
    
    for batch_idx, batch in enumerate(batches):

        if batch_idx == 0:
            opt = Optimizer(batch)
            active_nodes = batch.coalesce().indices().unique()
            affected_nodes_mask = torch.zeros(opt.nodes_num, dtype=torch.bool)
            affected_nodes_mask[active_nodes] = True
            if skip_first_batch:
                continue
        else:
            affected_nodes_mask = opt.update_adj(batch)
        
        # Вычисляем размеры окрестностей для разных шагов
        neighborhood_sizes_for_batch = []
        
        # Для step=0 - просто активные узлы
        current_mask = affected_nodes_mask.clone()
        neighborhood_sizes_for_batch.append(current_mask.sum().item())
        
        # Для step>0 - последовательно расширяем окрестность
        for step in range(1, max_step + 1):
            current_mask = Optimizer.neighborhood(opt.adj, current_mask, step=1,
                                                  is_symmetric = not ds.is_directed)
            neighborhood_sizes_for_batch.append(current_mask.sum().item())
        
        batch_neighborhood_sizes.append(neighborhood_sizes_for_batch)
    
    return batch_neighborhood_sizes

def main(max_nodes=None, max_edges=None, p_values=None, n_values=None, max_step=5):
    """
    Основная функция
    
    Args:
        max_nodes: максимальное количество узлов в датасете (None - без ограничений)
        max_edges: максимальное количество ребер в датасете (None - без ограничений)
        p_values: значения p для стратегий p:n
        n_values: значения n для стратегий p:n
        max_step: максимальный шаг окрестности (r)
    """
    # Устанавливаем значения по умолчанию
    if p_values is None:
        p_values = [9, 99, 999]
    if n_values is None:
        n_values = [10, 100, 1000]
    
    # Загружаем информацию о датасетах konect
    info_path = os.path.join(INFO, "konect.json")
    with open(info_path) as f:
        konect_info = json.load(f)
    
    # Фильтруем датасеты как в launch.py
    konect_datasets = list(filter(
        lambda dataset: konect_info[dataset]["w"] in ["weighted", "unweighted"],
        list(konect_info.keys())
    ))
    
    # Применяем фильтры по размеру
    filtered_datasets = []
    for dataset_name in konect_datasets:
        dataset_info = konect_info[dataset_name]
        n_nodes = dataset_info["n"]
        n_edges = dataset_info["m"]
        
        # Проверяем ограничения
        if max_nodes is not None and n_nodes > max_nodes:
            continue
        if max_edges is not None and n_edges > max_edges:
            continue
        
        filtered_datasets.append(dataset_name)
    
    # Сортируем по числу вершин
    datasets_by_nodes = sorted(filtered_datasets, key=lambda x: konect_info[x]["n"])
    
    if not datasets_by_nodes:
        print("Нет датасетов, удовлетворяющих заданным ограничениям")
        return {}
    
    print(f"Будет обработано датасетов: {len(datasets_by_nodes)}")
    if max_nodes is not None:
        print(f"Ограничение по узлам: не более {max_nodes}")
    if max_edges is not None:
        print(f"Ограничение по ребрам: не более {max_edges}")
    print(f"Стратегии: p={p_values}, n={n_values}")
    print(f"Максимальный шаг окрестности: {max_step}")
    
    # Словарь для хранения результатов
    results = {}
    
    # Перебираем все комбинации
    for dataset_name in datasets_by_nodes:
        dataset_info = konect_info[dataset_name]
        n_nodes = dataset_info["n"]
        n_edges = dataset_info["m"]
        
        print(f"\nОбработка датасета: {dataset_name}")
        print(f"Количество узлов: {n_nodes}, ребер: {n_edges}")
        
        dataset_results = {}
        
        # Перебираем стратегии батчинга
        for p in p_values:
            for n in n_values:
                strategy = f"{p}:{n}"
                
                print(f"  Стратегия: {strategy}", end="", flush=True)
                
                try:
                    neighborhood_sizes = compute_neighborhood_sizes(
                        dataset_name, 
                        strategy, 
                        max_step
                    )
                    
                    dataset_results[strategy] = neighborhood_sizes
                    print(f" - успешно ({len(neighborhood_sizes)} батчей)")
                    
                except Exception as e:
                    print(f" - ошибка: {e}")
                    continue
        
        # Сохраняем результаты для датасета
        if dataset_results:
            results[dataset_name] = {
                "info": dataset_info,
                "strategies": dataset_results
            }
    
        results_dir = os.path.join(PROJECT_PATH, "results")
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"neighborhood_analysis.json")
        
        # Добавляем метаданные
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "parameters": {
                "max_nodes": max_nodes,
                "max_edges": max_edges,
                "p_values": p_values,
                "n_values": n_values,
                "max_step": max_step
            },
            "datasets": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=1)
    
        print(f"\nРезультаты для {dataset_name} сохранены в: {output_file}")
    
    # Выводим статистику
    if results:
        print(f"\nОбработано датасетов: {len(results)}")
        total_strategies = sum(len(dataset["strategies"]) for dataset in results.values())
        print(f"Всего стратегий: {total_strategies}")
        total_batches = sum(
            sum(len(sizes) for sizes in dataset["strategies"].values()) 
            for dataset in results.values()
        )
        print(f"Всего батчей: {total_batches}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Анализ размеров окрестностей для датасетов konect')
    parser.add_argument('--max-nodes', type=int, default=None, 
                       help='Максимальное количество узлов в датасете')
    parser.add_argument('--max-edges', type=int, default=None,
                       help='Максимальное количество ребер в датасете')
    parser.add_argument('--n-values', type=int, nargs='+', default=[10, 100],
                       help='Значения n для стратегий p:n')
    parser.add_argument('--p-values', type=int, nargs='+', default=[9, 99],
                       help='Значения p для стратегий p:n')
    parser.add_argument('--max-step', type=int, default=5,
                       help='Максимальный шаг окрестности (r)')
    
    args = parser.parse_args()
    
    main(
        max_nodes=args.max_nodes, 
        max_edges=args.max_edges,
        p_values=args.p_values,
        n_values=args.n_values,
        max_step=args.max_step
    )