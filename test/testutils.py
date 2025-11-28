from __future__ import annotations
import os
import sys
import json
import time
import signal
import psutil
import threading
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Callable, Any
from contextlib import contextmanager
from datetime import datetime

KONECT_PATH = "/auto/datasets/graphs/dynamic_konect_project_datasets/"
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFO = os.path.join(PROJECT_DIR, "datasets-info")
KONECT_INFO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets-info"))

def get_all_datasets():
    """
    Сreate dict with all datasets in test directory.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    datasets = {}
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                datasets[name] = base_dir
    return datasets

def load_konect_names(all_json_path: Path) -> set[str]:
    if not all_json_path.exists():
        return set()
    data = json.loads(all_json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return set(data.keys())
    if isinstance(data, list):
        return {d["name"] for d in data if isinstance(d, dict) and "name" in d}
    return set()

def filter_datasets_by_node_count(json_path: Path, max_nodes: int):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = {name: info for name, info in data.items() if info.get("n", 0) < max_nodes}
    return filtered


class TimeoutException(Exception):
    """Исключение для превышения тайм-аута выполнения"""
    pass


class ResourceMonitor:
    """
    Мониторинг ресурсов процесса в реальном времени
    
    Отслеживает CPU, RAM с динамическим выводом в консоль.
    Работает в отдельном фоновом потоке.
    
    Args:
        interval: Интервал обновления метрик в секундах
        history_size: Размер буфера для хранения истории метрик
        
    Example:
        monitor = ResourceMonitor(interval=0.5)
        monitor.start()
        # ... ваш код ...
        monitor.stop()
        monitor.print_summary()
    """
    
    def __init__(self, interval: float = 0.5, history_size: int = 100):
        self.interval = interval
        self.history_size = history_size
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
        
        # История метрик
        self.cpu_history = deque(maxlen=history_size)
        self.mem_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
    def _monitor_loop(self):
        """Фоновый цикл мониторинга"""
        start_time = time.time()
        
        # Первый вызов для инициализации cpu_percent
        self.process.cpu_percent(interval=None)
        
        while self.running:
            try:
                # CPU usage (%)
                cpu_percent = self.process.cpu_percent(interval=None)
                
                # Memory usage (MB)
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                
                # Timestamp
                elapsed = time.time() - start_time
                
                # Сохранение истории
                self.cpu_history.append(cpu_percent)
                self.mem_history.append(mem_mb)
                self.timestamps.append(elapsed)
                
                # Динамический вывод
                self._print_stats(cpu_percent, mem_mb, elapsed)
                
                time.sleep(self.interval)
                
            except psutil.NoSuchProcess:
                break
            except Exception as e:
                print(f"\n⚠️  Monitor error: {e}")
                break
                
    def _print_stats(self, cpu: float, mem: float, elapsed: float):
        """Динамический вывод статистики в одну строку"""
        sys.stdout.write('\r' + ' ' * 100 + '\r')
    
        # Нормализуем CPU по количеству ядер
        num_cores = psutil.cpu_count()
        cpu_normalized = cpu / num_cores if num_cores else cpu
        
        peak_mem = max(self.mem_history) if self.mem_history else mem
        stats = (
            f"⏱️  {elapsed:6.1f}s | "
            f"CPU: {cpu_normalized:5.1f}% ({cpu:6.1f}% total) | "
            f"RAM: {mem:7.1f} MB | "
            f"Peak RAM: {peak_mem:7.1f} MB"
        )
        sys.stdout.write(stats)
        sys.stdout.flush()
        
    def start(self):
        """Запуск мониторинга в фоновом потоке"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            
    def stop(self):
        """Остановка мониторинга"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2)
            print()  # Перевод строки после динамического вывода
            
    def get_summary(self) -> Dict[str, float]:
        """
        Получение итоговой статистики
        
        Returns:
            Словарь с метриками: cpu_avg, cpu_max, mem_avg, mem_max, duration
        """
        if not self.cpu_history:
            return {}
            
        return {
            'cpu_avg': sum(self.cpu_history) / len(self.cpu_history),
            'cpu_max': max(self.cpu_history),
            'mem_avg': sum(self.mem_history) / len(self.mem_history),
            'mem_max': max(self.mem_history),
            'duration': self.timestamps[-1] if self.timestamps else 0
        }
        
    def print_summary(self):
        """Вывод итоговой статистики в консоль"""
        summary = self.get_summary()
        if summary:
            print(f"\n{'='*60}")
            print(f"Resource Usage Summary:")
            print(f"  Duration:     {summary['duration']:.2f} s")
            print(f"  CPU Average:  {summary['cpu_avg']:.2f}%")
            print(f"  CPU Peak:     {summary['cpu_max']:.2f}%")
            print(f"  RAM Average:  {summary['mem_avg']:.2f} MB")
            print(f"  RAM Peak:     {summary['mem_max']:.2f} MB")
            print(f"{'='*60}")


class ResourceMonitorWithGPU(ResourceMonitor):
    """
    Расширенный мониторинг с поддержкой GPU
    
    Требует установленного PyTorch с CUDA.
    
    Args:
        interval: Интервал обновления метрик
        history_size: Размер буфера истории
        track_gpu: Включить мониторинг GPU (если доступно)
    """
    
    def __init__(self, interval: float = 0.5, history_size: int = 100, track_gpu: bool = True):
        super().__init__(interval, history_size)
        self.track_gpu = track_gpu
        self.gpu_available = False
        
        if track_gpu:
            try:
                import torch
                self.gpu_available = torch.cuda.is_available()
                if self.gpu_available:
                    self.gpu_mem_history = deque(maxlen=history_size)
                    self.torch = torch
            except ImportError:
                pass
                
    def _monitor_loop(self):
        """Расширенный цикл мониторинга с GPU"""
        start_time = time.time()
        self.process.cpu_percent(interval=None)
        
        while self.running:
            try:
                cpu_percent = self.process.cpu_percent(interval=None)
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                elapsed = time.time() - start_time
                
                # GPU мониторинг
                gpu_mem_mb = 0
                if self.gpu_available:
                    gpu_mem_mb = self.torch.cuda.memory_allocated() / (1024 * 1024)
                    self.gpu_mem_history.append(gpu_mem_mb)
                
                self.cpu_history.append(cpu_percent)
                self.mem_history.append(mem_mb)
                self.timestamps.append(elapsed)
                
                self._print_stats_gpu(cpu_percent, mem_mb, gpu_mem_mb, elapsed)
                
                time.sleep(self.interval)
                
            except (psutil.NoSuchProcess, RuntimeError):
                break
                
    def _print_stats_gpu(self, cpu: float, mem: float, gpu_mem: float, elapsed: float):
        """Расширенный динамический вывод с GPU"""
        sys.stdout.write('\r' + ' ' * 120 + '\r')
        
        peak_mem = max(self.mem_history) if self.mem_history else mem
        stats = (
            f"⏱️  {elapsed:6.1f}s | "
            f"CPU: {cpu:5.1f}% | "
            f"RAM: {mem:7.1f} MB (peak: {peak_mem:7.1f})"
        )
        
        if self.gpu_available:
            peak_gpu = max(self.gpu_mem_history) if self.gpu_mem_history else gpu_mem
            stats += f" | GPU: {gpu_mem:7.1f} MB (peak: {peak_gpu:7.1f})"
            
        sys.stdout.write(stats)
        sys.stdout.flush()
        
    def get_summary(self) -> Dict[str, float]:
        if not self.cpu_history:
            return {}
        cpu_total_avg = sum(self.cpu_history) / len(self.cpu_history)
        cpu_total_max = max(self.cpu_history)
        mem_avg = sum(self.mem_history) / len(self.mem_history)
        mem_max = max(self.mem_history)
        duration = self.timestamps[-1] if self.timestamps else 0
        return {
            'cpu_total_avg': cpu_total_avg,
            'cpu_total_max': cpu_total_max,
            'cpu_per_core_avg': cpu_total_avg / self.num_cores,
            'cpu_per_core_max': cpu_total_max / self.num_cores,
            'mem_avg': mem_avg,
            'mem_max': mem_max,
            'duration': duration,
            'num_cores': self.num_cores
        }

    def print_summary(self):
        summary = self.get_summary()
        if summary:
            print(f"\n{'='*60}")
            print("Resource Usage Summary:")
            print(f"  Duration:     {summary['duration']:.2f} s")
            print(f"  CPU Average:  {summary['cpu_per_core_avg']:.2f}%/core ({summary['cpu_total_avg']:.2f}% total)")
            print(f"  CPU Peak:     {summary['cpu_per_core_max']:.2f}%/core ({summary['cpu_total_max']:.2f}% total)")
            print(f"  RAM Average:  {summary['mem_avg']:.2f} MB")
            print(f"  RAM Peak:     {summary['mem_max']:.2f} MB")
            print(f"  Cores:        {summary['num_cores']}")
            print(f"{'='*60}")



class ResourceMonitorWithLogging(ResourceMonitorWithGPU):
    """
    Мониторинг с сохранением логов в JSON
    
    Args:
        log_file: Путь к файлу логов (по умолчанию генерируется автоматически)
        interval: Интервал обновления
        history_size: Размер буфера
        track_gpu: Мониторить GPU
    """
    
    def __init__(
        self, 
        log_file: Optional[str] = None, 
        interval: float = 0.5,
        history_size: int = 100,
        track_gpu: bool = True
    ):
        super().__init__(interval, history_size, track_gpu)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"resource_log_{timestamp}.json"
            
        self.log_file = Path(log_file)
        self.logs = []
        
    def _monitor_loop(self):
        """Мониторинг с сохранением в лог"""
        start_time = time.time()
        self.process.cpu_percent(interval=None)
        
        while self.running:
            try:
                cpu_percent = self.process.cpu_percent(interval=None)
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                elapsed = time.time() - start_time
                
                gpu_mem_mb = 0
                if self.gpu_available:
                    gpu_mem_mb = self.torch.cuda.memory_allocated() / (1024 * 1024)
                    self.gpu_mem_history.append(gpu_mem_mb)
                
                # Сохранение в лог
                log_entry = {
                    'timestamp': elapsed,
                    'cpu_percent': cpu_percent,
                    'mem_mb': mem_mb,
                    'gpu_mem_mb': gpu_mem_mb if self.gpu_available else None
                }
                self.logs.append(log_entry)
                
                self.cpu_history.append(cpu_percent)
                self.mem_history.append(mem_mb)
                self.timestamps.append(elapsed)
                
                self._print_stats_gpu(cpu_percent, mem_mb, gpu_mem_mb, elapsed)
                
                time.sleep(self.interval)
                
            except (psutil.NoSuchProcess, RuntimeError):
                break
                
    def save_logs(self):
        """Сохранение логов в JSON файл"""
        log_data = {
            'logs': self.logs,
            'summary': self.get_summary(),
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'interval': self.interval,
                'history_size': self.history_size,
                'gpu_tracked': self.gpu_available
            }
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
        print(f"\n📊 Logs saved to: {self.log_file}")
        
    def stop(self):
        """Остановка с автоматическим сохранением"""
        super().stop()
        if self.logs:
            self.save_logs()


def with_timeout(seconds: int):
    """
    Декоратор для установки тайм-аута на выполнение функции
    
    Args:
        seconds: Максимальное время выполнения в секундах
        
    Raises:
        TimeoutException: Если функция не завершилась за отведённое время
        
    Example:
        @with_timeout(300)
        def long_running_function():
            # код, который может зависнуть
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            def _timeout_handler(signum, frame):
                raise TimeoutException(
                    f"Function '{func.__name__}' exceeded timeout of {seconds}s"
                )
            
            # Сохраняем старый обработчик
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
            return result
        return wrapper
    return decorator


@contextmanager
def monitored_execution(
    interval: float = 0.5,
    track_gpu: bool = False,
    save_logs: bool = False,
    log_file: Optional[str] = None,
    print_summary: bool = True
):
    """
    Контекстный менеджер для мониторинга блока кода
    
    Args:
        interval: Интервал обновления метрик
        track_gpu: Мониторить GPU
        save_logs: Сохранять логи в файл
        log_file: Путь к файлу логов
        print_summary: Печатать итоговую статистику
        
    Yields:
        ResourceMonitor: Объект мониторинга
        
    Example:
        with monitored_execution(track_gpu=True) as monitor:
            # ваш код
            heavy_computation()
            
        # статистика выведется автоматически
    """
    if save_logs:
        monitor = ResourceMonitorWithLogging(
            log_file=log_file,
            interval=interval,
            track_gpu=track_gpu
        )
    elif track_gpu:
        monitor = ResourceMonitorWithGPU(interval=interval, track_gpu=True)
    else:
        monitor = ResourceMonitor(interval=interval)
    
    monitor.start()
    
    try:
        yield monitor
    finally:
        monitor.stop()
        if print_summary:
            monitor.print_summary()


def measure_time(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения функции
    
    Example:
        @measure_time
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"\n⏱️  {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

def main() -> None:
    test_dir = Path(__file__).resolve().parent
    repo_root = test_dir.parent
    print(repo_root)
    all_json = repo_root / "datasets-info" / "all.json"
    small_root = test_dir / "graphs" / "small"
    out_path = test_dir / "dataset_paths.json"

    MAX_NODES = 10000

    filtered_datasets = filter_datasets_by_node_count(all_json, MAX_NODES)

    mapping: dict[str, str] = {}
    for name in filtered_datasets:
        mapping[name] = KONECT_PATH

    if small_root.exists():
        for p in small_root.iterdir():
            if p.is_dir():
                mapping[p.name] = str(small_root)

    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()