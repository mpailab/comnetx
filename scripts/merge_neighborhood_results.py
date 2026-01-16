#!/usr/local/bin/python3.9
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_json_files(json_files):
    """Загружает несколько JSON файлов"""
    all_data = []
    
    for json_file in json_files:
        json_path = Path(json_file)
        if not json_path.exists():
            print(f"Предупреждение: файл {json_path} не существует, пропускаем")
            continue
        
        print(f"Загружаем: {json_path}")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            all_data.append(data)
            print(f"  Успешно загружено")
        except Exception as e:
            print(f"  Ошибка при загрузке: {e}")
            continue
    
    return all_data

def merge_datasets(datasets_list, merge_strategy='update'):
    """
    Объединяет данные из нескольких JSON файлов
    
    Args:
        datasets_list: список словарей с данными
        merge_strategy: стратегия объединения
            - 'update': обновить существующие записи (поздние данные перезаписывают ранние)
            - 'skip': пропустить дубликаты (ранние данные имеют приоритет)
            - 'combine': комбинировать все стратегии для каждого датасета
    """
    merged_data = {
        "generated_at": datetime.now().isoformat(),
        "merged_from": [],
        "parameters": {},
        "datasets": {}
    }
    
    # Собираем информацию о всех исходных файлах
    all_parameters = defaultdict(set)
    
    for i, data in enumerate(datasets_list):
        # Добавляем информацию о файле
        source_info = {
            "index": i,
            "parameters": data.get("parameters", {})
        }
        merged_data["merged_from"].append(source_info)
        
        # Собираем все уникальные параметры
        for key, value in data.get("parameters", {}).items():
            if value is not None:
                if isinstance(value, list):
                    all_parameters[key].update(value)
                else:
                    all_parameters[key].add(value)
        
        # Объединяем датасеты
        datasets = data.get("datasets", {})
        for dataset_name, dataset_data in datasets.items():
            if dataset_name not in merged_data["datasets"]:
                # Если датасет встречается впервые, просто добавляем
                merged_data["datasets"][dataset_name] = dataset_data
                continue
            
            # Если датасет уже есть, объединяем стратегии
            existing_dataset = merged_data["datasets"][dataset_name]
            existing_strategies = existing_dataset.get("strategies", {})
            new_strategies = dataset_data.get("strategies", {})
            
            if merge_strategy == 'skip':
                # Пропускаем дубликаты - существующие стратегии имеют приоритет
                for strategy, batches in new_strategies.items():
                    if strategy not in existing_strategies:
                        existing_strategies[strategy] = batches
            elif merge_strategy == 'update':
                # Обновляем - новые стратегии перезаписывают старые
                existing_strategies.update(new_strategies)
            elif merge_strategy == 'combine':
                # Комбинируем - для каждой стратегии проверяем, что данные одинаковые
                for strategy, batches in new_strategies.items():
                    if strategy in existing_strategies:
                        # Проверяем, одинаковые ли данные
                        if existing_strategies[strategy] != batches:
                            print(f"  Внимание: разные данные для {dataset_name}/{strategy}")
                            # Можно выбрать стратегию: сохранить оба, но с разными именами
                            strategy_new = f"{strategy}_v{i+1}"
                            existing_strategies[strategy_new] = batches
                    else:
                        existing_strategies[strategy] = batches
            
            existing_dataset["strategies"] = existing_strategies
    
    # Объединяем параметры
    for key, values_set in all_parameters.items():
        values_list = list(values_set)
        if len(values_list) == 1:
            merged_data["parameters"][key] = values_list[0]
        else:
            # Если значения разные, сохраняем как список
            merged_data["parameters"][key] = values_list
    
    return merged_data

def find_json_files(input_path, recursive=False):
    """
    Находит все JSON файлы в указанном пути
    
    Args:
        input_path: путь к файлу или директории
        recursive: искать рекурсивно во вложенных папках
    """
    input_path = Path(input_path)
    json_files = []
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.json':
            return [str(input_path)]
        else:
            print(f"Предупреждение: {input_path} не является JSON файлом")
            return []
    
    elif input_path.is_dir():
        if recursive:
            pattern = "**/*.json"
        else:
            pattern = "*.json"
        
        for json_file in input_path.glob(pattern):
            json_files.append(str(json_file))
    
    return sorted(json_files)

def main():
    description = """
    Скрипт для объединения нескольких JSON файлов с результатами анализа окрестностей.
    
    Объединяет данные из нескольких запусков compute_neighborhood_sizes.py в один файл
    для последующего анализа с помощью analyze_neighborhoods.py.
    """
    
    epilog = """
    Примеры использования:
    
    1. Объединить все JSON файлы в директории results:
       python merge_neighborhood_results.py --input results --output merged_results.json
    
    2. Объединить конкретные файлы:
       python merge_neighborhood_results.py --files results/file1.json results/file2.json --output merged.json
    
    3. Объединить файлы с рекурсивным поиском:
       python merge_neighborhood_results.py data --recursive --output all_data.json
    
    4. Использовать стратегию пропуска дубликатов:
       python merge_neighborhood_results.py results --strategy skip --output unique_results.json
    
    5. Использовать стратегию комбинирования:
       python merge_neighborhood_results.py results --strategy combine --output combined_results.json
    """
    
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Входные данные
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                           help='Путь к директории или файлу JSON для объединения')
    input_group.add_argument('--files', type=str, nargs='+',
                           help='Список конкретных JSON файлов для объединения')
    
    # Параметры поиска
    parser.add_argument('--recursive', action='store_true',
                       help='Искать JSON файлы рекурсивно во вложенных папках')
    
    # Параметры объединения
    parser.add_argument('--strategy', type=str, choices=['update', 'skip', 'combine'],
                       default='update',
                       help='Стратегия объединения (update, skip, combine)')
    
    # Выходные данные
    parser.add_argument('--output', type=str, default='merged_neighborhood_analysis.json',
                       help='Путь для сохранения объединенного JSON файла')
    
    parser.add_argument('--overwrite', action='store_true',
                       help='Перезаписать выходной файл, если он существует')
    
    args = parser.parse_args()
    
    # Определяем, какие файлы объединять
    if args.files:
        json_files = args.files
    else:
        json_files = find_json_files(args.input, args.recursive)
    
    if not json_files:
        print("Не найдено JSON файлов для объединения")
        sys.exit(1)
    
    print(f"Найдено файлов для объединения: {len(json_files)}")
    for file in json_files:
        print(f"  - {file}")
    
    # Загружаем файлы
    datasets_list = load_json_files(json_files)
    
    if not datasets_list:
        print("Не удалось загрузить данные из файлов")
        sys.exit(1)
    
    # Объединяем данные
    print(f"\nОбъединяем данные (стратегия: {args.strategy})...")
    merged_data = merge_datasets(datasets_list, args.strategy)
    
    # Проверяем выходной файл
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Ошибка: выходной файл {output_path} уже существует. Используйте --overwrite для перезаписи.")
        sys.exit(1)
    
    # Создаем директорию, если нужно
    output_dir = output_path.parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем результат
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    # Выводим статистику
    print(f"\nРезультаты объединены и сохранены в: {output_path}")
    print(f"Объединено файлов: {len(datasets_list)}")
    print(f"Объединено датасетов: {len(merged_data['datasets'])}")
    
    # Подсчитываем общее количество стратегий
    total_strategies = 0
    for dataset_name, dataset_data in merged_data['datasets'].items():
        strategies = dataset_data.get('strategies', {})
        total_strategies += len(strategies)
    
    print(f"Общее количество стратегий: {total_strategies}")
    
    # Информация о конфликтах
    if args.strategy == 'combine':
        print("\nПримечание: использована стратегия 'combine'.")
        print("Дублирующиеся стратегии с разными данными были переименованы.")
    
    return merged_data

if __name__ == "__main__":
    main()