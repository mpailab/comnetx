# Профилирование Python и PyTorch

В этом проекте используются два профилировщика:

1. **cProfile** — стандартный профилировщик Python для CPU-времени.  
2. **torch.profiler** — инструмент PyTorch для анализа производительности нейронных сетей на CPU и GPU.

---

## 1. cProfile — профилировщик Python

### Установка
`cProfile` встроен в стандартную библиотеку Python, ничего устанавливать не нужно.  
Для визуализации можно установить `snakeviz`:

```bash
pip install snakeviz
```
### Профилирование всей программы с сохранением отчёта
``` bash
python -m cProfile -o profiler_output.prof your_script.py # рекомендуется 
python -m cProfile -s cumtime your_script.py # альтернативый вариант 
```
### Визуализация дерева вызова функции
``` bash
python -m snakeviz profiler.prof
```

### Вариант для работы с pytest файлами
``` py
import cProfile
import pstats

def examp_function():
    result = sum(i**2 for i in range(10_000_000))
    return result

if __name__ == "__main__":
    profiler = cProfile.Profile()
    
    # только эта часть будет профилироваться
    profiler.enable()
    heavy_function()
    profiler.disable()

    # можно вывести результат
    stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
    stats.print_stats(10)
    # можно сохранить результат 
    stats.dump_stats("profiler.prof") 

```

## 1. torch.profiler — профилировщик PyTorch

### Установка
`torch.profiler` встроен в стандартную библиотеку torch, ничего устанавливать не нужно.  

### Работа прямо в файле, можно выделять конкретные участки кода
``` py
import torch
import torch.profiler

def train_step():
    x = torch.randn(32, 3, 224, 224).cuda()
    model = torch.nn.Conv2d(3, 64, 3).cuda()
    out = model(x)
    loss = out.mean()
    loss.backward()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(5):
        train_step()
        prof.step()  

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

```

### Пример работы с pytest файлами
``` py
def test_neighborhood_2():
    A = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=torch.float32)
    A_sparse = A.t().to_sparse_coo()
    initial_nodes = torch.tensor([True, False, False, False])
    nodes_0 = Optimizer.neighborhood(A_sparse, initial_nodes, 0)
    nodes_1 = Optimizer.neighborhood(A_sparse, initial_nodes, 1)
    nodes_2 = Optimizer.neighborhood(A_sparse, initial_nodes, 2)
    true_nodes_1 = torch.tensor([True, True, False, False])
    true_nodes_2 = torch.tensor([True, True, True, True])
    assert torch.equal(nodes_0, initial_nodes)
    assert torch.equal(nodes_1, true_nodes_1)
    assert torch.equal(nodes_2, true_nodes_2)

if __name__ == "__main__":
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        test_neighborhood_2()
        
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```