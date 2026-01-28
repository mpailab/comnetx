#!/usr/local/bin/python3.9
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import sys
import argparse
from pathlib import Path
from math import log10

def load_data(json_file):
    """Загружает данные из JSON файла"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Проверяем формат файла
    if "datasets" in data:
        return data["datasets"], data.get("parameters", {})
    else:
        return data, {}

def prepare_data(datasets_data, skip_first_batch=False):
    """
    Подготавливает данные для построения графиков
    
    Args:
        datasets_data: словарь с данными датасетов
        skip_first_batch: пропускать ли первый батч (по умолчанию False)
    """
    all_data = []
    
    for dataset_name, dataset_info in datasets_data.items():
        strategies = dataset_info.get("strategies", {})
        n_nodes = dataset_info.get("info", {}).get("n", 0)
        
        for strategy_name, batches_data in strategies.items():
            if not batches_data:
                continue
            
            # Пропускаем первый батч если нужно
            if skip_first_batch:
                batches_to_analyze = batches_data[1:]
            else:
                batches_to_analyze = batches_data
            
            # Собираем данные для каждого батча и каждого r
            for batch_idx, batch_sizes in enumerate(batches_to_analyze):
                real_batch_idx = batch_idx if not skip_first_batch else batch_idx + 1
                
                for r, size in enumerate(batch_sizes):
                    # Вычисляем процент от общего числа вершин
                    percentage = (size / n_nodes * 100) if n_nodes > 0 else 0
                    
                    all_data.append({
                        'dataset': dataset_name,
                        'strategy': strategy_name,
                        'batch': real_batch_idx,
                        'r': r,
                        'neighborhood_size': size,
                        'percentage_of_nodes': percentage,
                        'total_nodes': n_nodes,
                        'dataset_strategy': f"{dataset_name} ({strategy_name})"
                    })
    
    return pd.DataFrame(all_data)

def prepare_relative_data(df):
    """
    Подготавливает данные для относительных графиков (по сравнению с r=0)
    """
    relative_values = []
    
    for (dataset, strategy, batch), group in df.groupby(['dataset', 'strategy', 'batch']):
        base_size = group[group['r'] == 0]['neighborhood_size'].values
        if len(base_size) > 0 and base_size[0] > 0:
            for _, row in group.iterrows():
                relative = row['neighborhood_size'] / base_size[0]
                relative_values.append({
                    'dataset': dataset,
                    'strategy': strategy,
                    'batch': batch,
                    'r': row['r'],
                    'relative_size': relative,
                    'percentage_of_nodes': row['percentage_of_nodes'],
                    'total_nodes': row['total_nodes'],
                    'dataset_strategy': f"{dataset} ({strategy})"
                })
    
    if not relative_values:
        return pd.DataFrame()
    
    return pd.DataFrame(relative_values)

def plot_relative_pointplots(df, output_dir, skip_first_batch=False):
    """
    Строит pointplot графики относительных значений (по сравнению с r=0)
    """
    output_dir = Path(output_dir)
    title_suffix = " (без первого батча)" if skip_first_batch else ""
    
    # Вычисляем относительные значения
    df_relative = prepare_relative_data(df)
    
    if df_relative.empty:
        print("Нет данных для построения относительных графиков")
        return
    
    # Pointplot для относительных значений по датасетам
    datasets = df_relative['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df_relative[df_relative['dataset'] == dataset]
        
        if len(dataset_df) == 0:
            continue
        
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Используем pointplot
        ax = sns.pointplot(
            data=dataset_df,
            x='r',
            y='relative_size',
            hue='strategy',
            linestyle='-',
            markersize=7,
            errorbar=('ci', 95),
            legend=True
        )
        
        plt.title(f'Относительный рост: {dataset}{title_suffix}', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Радиус окрестности (r)', fontsize=12)
        plt.ylabel('Отношение к r=0', fontsize=12)
        
        # Горизонтальная линия на уровне 1
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Настраиваем легенду
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(
            handles, 
            labels, 
            title='Стратегия',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=10,
            title_fontsize=11
        )
        
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tight_layout()
        
        filename = f'pointplot_relative_{dataset}{"_no_first" if skip_first_batch else ""}.jpeg'
        savepath = output_dir / filename
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
        print(f"Сохранен график: {savepath}")
        plt.close()

def plot_percentage_pointplots(df, output_dir, skip_first_batch=False):
    """
    Строит pointplot графики процентного соотношения к числу вершин в графе
    """
    output_dir = Path(output_dir)
    title_suffix = " (без первого батча)" if skip_first_batch else ""
    
    # Pointplot для процентного соотношения по датасетам
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        
        if len(dataset_df) == 0:
            continue
        
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Используем pointplot
        ax = sns.pointplot(
            data=dataset_df,
            x='r',
            y='percentage_of_nodes',
            hue='strategy',
            linestyle='-',
            markersize=7,
            errorbar=('ci', 95),
            legend=True
        )
        
        plt.title(f'Процент вершин: {dataset}{title_suffix}', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Радиус окрестности (r)', fontsize=12)
        plt.ylabel('Процент вершин (%)', fontsize=12)
        
        # Линия на уровне 100%
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1, label='100%')
        
        # Настраиваем легенду
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(
            handles, 
            labels, 
            title='Стратегия',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=10,
            title_fontsize=11
        )
        
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tight_layout()
        
        filename = f'pointplot_percentage_{dataset}{"_no_first" if skip_first_batch else ""}.jpeg'
        savepath = output_dir / filename
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
        print(f"Сохранен график: {savepath}")
        plt.close()

def plot_heatmap_percentage_by_strategy(df, output_dir, skip_first_batch=False):
    """
    Строит heatmap процентного соотношения для каждой стратегии отдельно
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    title_suffix = " (без первого батча)" if skip_first_batch else ""
    
    # Получаем все уникальные стратегии
    strategies = df['strategy'].unique()
    
    for strategy in strategies:
        strategy_df = df[df['strategy'] == strategy]
        
        if len(strategy_df) == 0:
            continue
        
        # Подготавливаем данные для heatmap
        summary_data = []
        
        for dataset in strategy_df['dataset'].unique():
            dataset_df = strategy_df[strategy_df['dataset'] == dataset]
            
            # Получаем информацию о числе вершин в датасете
            if not dataset_df.empty:
                n_nodes = dataset_df['total_nodes'].iloc[0]
            else:
                n_nodes = 0
            
            for r in sorted(dataset_df['r'].unique()):
                r_data = dataset_df[dataset_df['r'] == r]
                avg_percentage = r_data['percentage_of_nodes'].mean()
                summary_data.append({
                    'dataset': dataset,
                    'n_nodes': n_nodes,
                    'dataset_with_nodes': f"{dataset}\n(n={n_nodes})",
                    'r': r,
                    'percentage': avg_percentage
                })
        
        if not summary_data:
            continue
        
        df_summary = pd.DataFrame(summary_data)
        
        # Сортируем датасеты по общему числу вершин
        try:
            # Сортируем по n_nodes
            datasets_with_nodes = []
            for dataset in df_summary['dataset'].unique():
                node_data = df_summary[df_summary['dataset'] == dataset]
                if not node_data.empty:
                    n_nodes = node_data['n_nodes'].iloc[0]
                    datasets_with_nodes.append((dataset, n_nodes))
            
            if datasets_with_nodes:
                datasets_with_nodes.sort(key=lambda x: x[1])
                # Создаем маппинг для сортировки
                dataset_order = [d[0] for d in datasets_with_nodes]
                # Создаем соответствующий порядок для dataset_with_nodes
                dataset_with_nodes_order = []
                for dataset, n_nodes in datasets_with_nodes:
                    dataset_with_nodes_order.append(f"{dataset}\n(n={n_nodes})")
                
                # Создаем pivot таблицу для heatmap
                pivot_data = df_summary.pivot_table(
                    index='dataset_with_nodes',
                    columns='r',
                    values='percentage',
                    aggfunc='mean'
                )
                
                # Переиндексируем в правильном порядке
                pivot_data = pivot_data.reindex(dataset_with_nodes_order)
        except Exception as e:
            # Если не удалось отсортировать, используем dataset_with_nodes как есть
            print(f"  Предупреждение: не удалось отсортировать датасеты: {e}")
            pivot_data = df_summary.pivot_table(
                index='dataset_with_nodes',
                columns='r',
                values='percentage',
                aggfunc='mean'
            )
        
        # Создаем heatmap
        plt.figure(figsize=(12, max(6, len(pivot_data) * 0.5)), dpi=300)
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Процент вершин (%)'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title(f'Процент вершин в окрестности (стратегия: {strategy}){title_suffix}', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Радиус окрестности (r)', fontsize=12)
        plt.ylabel('Датасет (число вершин)', fontsize=12)
        
        # Настраиваем цветовую шкалу
        plt.tight_layout()
        
        filename = f'heatmap_percentage_{strategy}{"_no_first" if skip_first_batch else ""}.jpeg'
        savepath = output_dir / filename
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
        print(f"Сохранен график: {savepath}")
        plt.close()

def plot_heatmap_relative_by_strategy(df, output_dir, skip_first_batch=False):
    """
    Строит heatmap относительных значений для каждой стратегии отдельно
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    title_suffix = " (без первого батча)" if skip_first_batch else ""
    
    # Вычисляем относительные значения
    df_relative = prepare_relative_data(df)
    
    if df_relative.empty:
        print("Нет данных для построения heatmap относительных значений")
        return
    
    # Получаем все уникальные стратегии
    strategies = df_relative['strategy'].unique()
    
    for strategy in strategies:
        strategy_df = df_relative[df_relative['strategy'] == strategy]
        
        if len(strategy_df) == 0:
            continue
        
        # Подготавливаем данные для heatmap
        summary_data = []
        
        for dataset in strategy_df['dataset'].unique():
            dataset_df = strategy_df[strategy_df['dataset'] == dataset]
            
            for r in sorted(dataset_df['r'].unique()):
                r_data = dataset_df[dataset_df['r'] == r]
                avg_relative = r_data['relative_size'].mean()
                summary_data.append({
                    'dataset': dataset,
                    'r': r,
                    'relative_size': avg_relative
                })
        
        if not summary_data:
            continue
        
        df_summary = pd.DataFrame(summary_data)
        
        # Создаем pivot таблицу для heatmap
        pivot_data = df_summary.pivot_table(
            index='dataset',
            columns='r',
            values='relative_size',
            aggfunc='mean'
        )
        
        # Сортируем датасеты по общему числу вершин (если доступно)
        try:
            # Пытаемся отсортировать по total_nodes
            datasets_with_nodes = []
            for dataset in pivot_data.index:
                node_data = strategy_df[strategy_df['dataset'] == dataset]
                if not node_data.empty:
                    total_nodes = node_data['total_nodes'].iloc[0]
                    datasets_with_nodes.append((dataset, total_nodes))
            
            if datasets_with_nodes:
                datasets_with_nodes.sort(key=lambda x: x[1])
                sorted_datasets = [d[0] for d in datasets_with_nodes]
                pivot_data = pivot_data.reindex(sorted_datasets)
        except:
            pass  # Если не удалось отсортировать, оставляем как есть
        
        # Создаем heatmap
        plt.figure(figsize=(12, max(6, len(pivot_data) * 0.4)), dpi=300)
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2f',  # 2 знака после запятой для относительных значений
            cmap='YlOrRd',
            cbar_kws={'label': 'Отношение к r=0'},
            linewidths=0.5,
            linecolor='gray',
            vmin=1.0,  # Минимальное значение (базовый уровень)
            center=2.0  # Центр цветовой шкалы
        )
        
        plt.title(f'Относительный рост окрестности (стратегия: {strategy}){title_suffix}', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Радиус окрестности (r)', fontsize=12)
        plt.ylabel('Датасет', fontsize=12)
        
        # Настраиваем цветовую шкалу
        plt.tight_layout()
        
        filename = f'heatmap_relative_{strategy}{"_no_first" if skip_first_batch else ""}.jpeg'
        savepath = output_dir / filename
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
        print(f"Сохранен график: {savepath}")
        plt.close()

def plot_heatmap_absolute_sizes(df, output_dir, skip_first_batch=False):
    """
    Строит общий heatmap для абсолютных значений размеров окрестностей
    для всех стратегий и датасетов
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    title_suffix = " (без первого батча)" if skip_first_batch else ""
    
    # Подготавливаем данные для heatmap
    summary_data = []
    
    # Группируем по датасетам и стратегиям
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        n_nodes = dataset_df['total_nodes'].iloc[0] if 'total_nodes' in dataset_df.columns else 0
        
        for strategy in dataset_df['strategy'].unique():
            strategy_df = dataset_df[dataset_df['strategy'] == strategy]
            
            # Для каждого r вычисляем средний абсолютный размер окрестности
            for r in sorted(strategy_df['r'].unique()):
                r_data = strategy_df[strategy_df['r'] == r]
                avg_size = r_data['neighborhood_size'].mean()
                summary_data.append({
                    'dataset': dataset,
                    'strategy': strategy,
                    'r': r,
                    'avg_size': avg_size,
                    'total_nodes': n_nodes,
                    'dataset_strategy': f"{dataset} ({strategy})"
                })
    
    if not summary_data:
        print("Нет данных для построения heatmap абсолютных значений")
        return
    
    df_summary = pd.DataFrame(summary_data)
    
    # Создаем pivot таблицу для heatmap
    # Вариант 1: по датасетам и стратегиям (строки) и r (столбцы)
    pivot_data = df_summary.pivot_table(
        index='dataset_strategy',
        columns='r',
        values='avg_size',
        aggfunc='mean'
    )
    
    # Сортируем строки по общему числу вершин в датасете
    try:
        # Создаем маппинг dataset_strategy -> total_nodes
        dataset_nodes = {}
        for _, row in df_summary.iterrows():
            if row['dataset_strategy'] not in dataset_nodes:
                dataset_nodes[row['dataset_strategy']] = row['total_nodes']
        
        # Сортируем строки по total_nodes
        sorted_rows = sorted(pivot_data.index, key=lambda x: dataset_nodes.get(x, 0))
        pivot_data = pivot_data.reindex(sorted_rows)
    except:
        pass  # Если не удалось отсортировать, оставляем как есть
    
    # Создаем heatmap
    plt.figure(figsize=(14, max(8, len(pivot_data) * 0.3)), dpi=300)
    
    # Используем логарифмическую шкалу цветов для лучшей визуализации больших диапазонов
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.0f',  # целые числа для абсолютных значений
        cmap='YlOrRd',
        cbar_kws={'label': 'Абсолютный размер окрестности'},
        linewidths=0.5,
        linecolor='gray',
        norm=plt.cm.colors.LogNorm()  # Логарифмическая шкала цветов
    )
    
    plt.title(f'Абсолютные размеры окрестностей{title_suffix}', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Радиус окрестности (r)', fontsize=14)
    plt.ylabel('Датасет (стратегия)', fontsize=14)
    
    # Настраиваем цветовую шкалу
    plt.tight_layout()
    
    filename = f'heatmap_absolute_sizes{"_no_first" if skip_first_batch else ""}.jpeg'
    savepath = output_dir / filename
    plt.savefig(savepath, bbox_inches='tight', dpi=300)
    print(f"Сохранен график: {savepath}")
    plt.close()
    
    # Вариант 2: heatmap только по датасетам (усредненные по стратегиям)
    plot_heatmap_absolute_by_dataset(df, output_dir, skip_first_batch)

def plot_heatmap_absolute_by_dataset(df, output_dir, skip_first_batch=False):
    """
    Строит heatmap абсолютных значений, усредненных по стратегиям для каждого датасета
    """
    output_dir = Path(output_dir)
    title_suffix = " (без первого батча)" if skip_first_batch else ""
    
    # Подготавливаем данные для heatmap
    summary_data = []
    
    # Группируем по датасетам
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        n_nodes = dataset_df['total_nodes'].iloc[0] if 'total_nodes' in dataset_df.columns else 0
        
        # Для каждого r вычисляем средний абсолютный размер окрестности по всем стратегиям
        for r in sorted(dataset_df['r'].unique()):
            r_data = dataset_df[dataset_df['r'] == r]
            avg_size = r_data['neighborhood_size'].mean()
            summary_data.append({
                'dataset': dataset,
                'r': r,
                'avg_size': avg_size,
                'total_nodes': n_nodes
            })
    
    if not summary_data:
        return
    
    df_summary = pd.DataFrame(summary_data)
    
    # Создаем pivot таблицу для heatmap
    pivot_data = df_summary.pivot_table(
        index='dataset',
        columns='r',
        values='avg_size',
        aggfunc='mean'
    )
    
    # Сортируем датасеты по общему числу вершин
    try:
        datasets_with_nodes = []
        for dataset in pivot_data.index:
            node_data = df_summary[df_summary['dataset'] == dataset]
            if not node_data.empty:
                total_nodes = node_data['total_nodes'].iloc[0]
                datasets_with_nodes.append((dataset, total_nodes))
        
        if datasets_with_nodes:
            datasets_with_nodes.sort(key=lambda x: x[1])
            sorted_datasets = [d[0] for d in datasets_with_nodes]
            pivot_data = pivot_data.reindex(sorted_datasets)
    except:
        pass  # Если не удалось отсортировать, оставляем как есть
    
    # Создаем heatmap
    plt.figure(figsize=(12, max(6, len(pivot_data) * 0.4)), dpi=300)
    
    # Используем логарифмическую шкалу цветов
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Абсолютный размер окрестности (среднее по стратегиям)'},
        linewidths=0.5,
        linecolor='gray',
        norm=plt.cm.colors.LogNorm()  # Логарифмическая шкала цветов
    )
    
    plt.title(f'Абсолютные размеры окрестностей (усредненные по стратегиям){title_suffix}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Радиус окрестности (r)', fontsize=12)
    plt.ylabel('Датасет', fontsize=12)
    
    # Настраиваем цветовую шкалу
    plt.tight_layout()
    
    filename = f'heatmap_absolute_by_dataset{"_no_first" if skip_first_batch else ""}.jpeg'
    savepath = output_dir / filename
    plt.savefig(savepath, bbox_inches='tight', dpi=300)
    print(f"Сохранен график: {savepath}")
    plt.close()

def generate_report(df, output_dir, skip_first_batch=False):
    """
    Генерирует текстовый отчет со статистикой
    """
    output_dir = Path(output_dir)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("АНАЛИЗ РАЗМЕРОВ ОКРЕСТНОСТЕЙ")
    report_lines.append(f"Первый батч исключен: {'ДА' if skip_first_batch else 'НЕТ'}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Базовая статистика
    report_lines.append("ОБЩАЯ СТАТИСТИКА:")
    report_lines.append(f"  Всего датасетов: {df['dataset'].nunique()}")
    report_lines.append(f"  Всего стратегий: {df['strategy'].nunique()}")
    report_lines.append(f"  Всего записей: {len(df)}")
    report_lines.append(f"  Диапазон значений r: {df['r'].min()} - {df['r'].max()}")
    report_lines.append("")
    
    # Подготовим данные для относительных значений для отчета
    df_relative = prepare_relative_data(df)
    
    if not df_relative.empty:
        # Статистика по относительным значениям
        report_lines.append("СТАТИСТИКА ПО ОТНОСИТЕЛЬНЫМ ЗНАЧЕНИЯМ:")
        
        for dataset in sorted(df_relative['dataset'].unique()):
            dataset_df = df_relative[df_relative['dataset'] == dataset]
            
            report_lines.append(f"\n  {dataset}:")
            report_lines.append(f"    Стратегии: {', '.join(sorted(dataset_df['strategy'].unique()))}")
            
            for strategy in sorted(dataset_df['strategy'].unique()):
                strategy_df = dataset_df[dataset_df['strategy'] == strategy]
                report_lines.append(f"\n    Стратегия: {strategy}")
                
                for r in sorted(strategy_df['r'].unique()):
                    r_data = strategy_df[strategy_df['r'] == r]
                    if len(r_data) > 0:
                        avg_relative = r_data['relative_size'].mean()
                        report_lines.append(f"      r={r}: отношение={avg_relative:.2f}x")
    
    # Статистика по процентным значениям
    report_lines.append("\n" + "=" * 80)
    report_lines.append("СТАТИСТИКА ПО ПРОЦЕНТНЫМ ЗНАЧЕНИЯМ:")
    report_lines.append("=" * 80)
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        n_nodes = dataset_df['total_nodes'].iloc[0] if 'total_nodes' in dataset_df.columns else 0
        
        report_lines.append(f"\n  {dataset} (всего вершин: {n_nodes}):")
        report_lines.append(f"    Стратегии: {', '.join(sorted(dataset_df['strategy'].unique()))}")
        
        # Для каждой стратегии выводим статистику
        for strategy in sorted(dataset_df['strategy'].unique()):
            strategy_df = dataset_df[dataset_df['strategy'] == strategy]
            report_lines.append(f"\n    Стратегия: {strategy}")
            
            # Средние значения по r
            for r in sorted(strategy_df['r'].unique()):
                r_data = strategy_df[strategy_df['r'] == r]
                if len(r_data) > 0:
                    avg_size = r_data['neighborhood_size'].mean()
                    avg_percent = r_data['percentage_of_nodes'].mean()
                    report_lines.append(f"      r={r}: размер={avg_size:.0f}, процент={avg_percent:.1f}%")
    
    # Сводная таблица достижения порогов
    report_lines.append("\n" + "=" * 80)
    report_lines.append("СВОДКА: ДОСТИЖЕНИЕ ПРОЦЕНТНЫХ ПОРОГОВ")
    report_lines.append("=" * 80)
    
    thresholds = [50, 75, 90, 95]
    
    for (dataset, strategy), group in df.groupby(['dataset', 'strategy']):
        report_lines.append(f"\n  {dataset} ({strategy}):")
        
        for threshold in thresholds:
            # Находим минимальный r, при котором достигается порог
            r_values = []
            for r in sorted(group['r'].unique()):
                r_data = group[group['r'] == r]
                avg_percent = r_data['percentage_of_nodes'].mean()
                if avg_percent >= threshold:
                    r_values.append(r)
            
            if r_values:
                min_r = min(r_values)
                report_lines.append(f"    {threshold}% вершин достигается при r={min_r}")
            else:
                report_lines.append(f"    {threshold}% не достигнут даже при максимальном r")
    
    # Сохраняем отчет
    report_file = output_dir / 'analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Краткий вывод в консоль
    print("\n" + "=" * 60)
    print("КРАТКИЙ ОТЧЕТ:")
    print("=" * 60)
    print(f"Датасетов: {df['dataset'].nunique()}")
    print(f"Стратегий: {df['strategy'].nunique()}")
    
    # Статистика по процентам при r=5
    df_r5 = df[df['r'] == 5]
    if not df_r5.empty:
        avg_percent_r5 = df_r5['percentage_of_nodes'].mean()
        max_percent_r5 = df_r5['percentage_of_nodes'].max()
        min_percent_r5 = df_r5['percentage_of_nodes'].min()
        
        print(f"\nПри r=5:")
        print(f"  Средний процент вершин: {avg_percent_r5:.1f}%")
        print(f"  Максимальный процент: {max_percent_r5:.1f}%")
        print(f"  Минимальный процент: {min_percent_r5:.1f}%")
    
    if not df_relative.empty:
        # Статистика по относительным значениям при r=5
        df_relative_r5 = df_relative[df_relative['r'] == 5]
        if not df_relative_r5.empty:
            avg_relative_r5 = df_relative_r5['relative_size'].mean()
            max_relative_r5 = df_relative_r5['relative_size'].max()
            min_relative_r5 = df_relative_r5['relative_size'].min()
            
            print(f"\nОтносительный рост при r=5:")
            print(f"  Средний рост: {avg_relative_r5:.2f}x")
            print(f"  Максимальный рост: {max_relative_r5:.2f}x")
            print(f"  Минимальный рост: {min_relative_r5:.2f}x")
    
    print(f"\nПодробный отчет сохранен в: {report_file}")
    
def main():
    parser = argparse.ArgumentParser(description='Анализ размеров окрестностей из JSON файла')
    parser.add_argument('json_file', type=str, help='Путь к JSON файлу с данными')
    parser.add_argument('--output-dir', type=str, default='neighborhood_analysis',
                       help='Директория для сохранения результатов анализа')
    parser.add_argument('--skip-first-batch', action='store_true',
                       help='Пропустить первый батч в анализе (по умолчанию не пропускается)')
    # Добавляем новый флаг для опциональной отрисовки графиков датасетов
    parser.add_argument('--skip-dataset-plots', action='store_true',
                       help='Пропустить отрисовку графиков для отдельных датасетов (пункты 1 и 2)')
    
    args = parser.parse_args()
    
    # Проверяем файл
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Ошибка: файл {json_path} не существует")
        sys.exit(1)
    
    print(f"Загружаем данные из: {json_path}")
    datasets_data, metadata = load_data(json_path)
    
    if not datasets_data:
        print("Ошибка: файл не содержит данных или имеет неправильный формат")
        sys.exit(1)
    
    print(f"Найдено датасетов: {len(datasets_data)}")
    
    # Подготавливаем данные
    skip_first_batch = args.skip_first_batch
    print(f"\nПропуск первого батча: {'ДА' if skip_first_batch else 'НЕТ'}")
    
    df = prepare_data(datasets_data, skip_first_batch=skip_first_batch)
    
    if df.empty:
        print("Ошибка: нет данных для анализа")
        sys.exit(1)
    
    print(f"Подготовлено записей: {len(df)}")
    
    # Создаем выходную директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Генерируем отчет
    print("\nГенерируем отчет...")
    generate_report(df, output_dir, skip_first_batch)
    
    # Строим графики
    print("\nСтроим графики...")
    
    # Определяем, нужно ли строить графики для отдельных датасетов
    skip_dataset_plots = args.skip_dataset_plots
    
    if not skip_dataset_plots:
        print("\n1. Pointplot относительные значения по датасетам:")
        plot_relative_pointplots(df, output_dir, skip_first_batch)
        
        print("\n2. Pointplot процентное соотношение по датасетам:")
        plot_percentage_pointplots(df, output_dir, skip_first_batch)
    else:
        print("\nПропускаем графики для отдельных датасетов (пункты 1 и 2)")
    
    # Эти графики всегда строятся (пункты 3-5)
    print("\n3. Heatmap процентного соотношения по стратегиям:")
    plot_heatmap_percentage_by_strategy(df, output_dir, skip_first_batch)
    
    print("\n4. Heatmap относительных значений по стратегиям:")
    plot_heatmap_relative_by_strategy(df, output_dir, skip_first_batch)
    
    print("\n5. Общий heatmap абсолютных значений:")
    plot_heatmap_absolute_sizes(df, output_dir, skip_first_batch)
    
    print(f"\nАнализ завершен!")
    print(f"Результаты сохранены в: {output_dir.absolute()}")

if __name__ == "__main__":
    main()