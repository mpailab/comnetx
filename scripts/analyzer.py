import os
import torch
import numpy as np
import scipy.io as sio
from torch_geometric.data import Data

def analyze_dataset(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print(f"\n{'='*50}")
    print(f"Файл: {os.path.basename(file_path)}")
    print(f"{'='*50}")
    
    try:
        # 1. PyTorch Geometric (.pt)
        if ext == '.pt':
            content = torch.load(file_path, map_location='cpu')
            if isinstance(content, Data):
                print(f"Тип: PyTorch Geometric (Data object)")
                print(f"  • Узлов: {content.num_nodes}")
                print(f"  • Ребер: {content.num_edges}")
                if content.x is not None:
                    print(f"  • Фичи узлов (x): {list(content.x.shape)}")
                
                # Проверка индексов на выход за границы
                if content.edge_index is not None:
                    max_idx = content.edge_index.max().item()
                    if max_idx >= content.num_nodes:
                        print(f"  ⚠️  ВНИМАНИЕ: Индекс ребра {max_idx} больше числа узлов {content.num_nodes}!")
            else:
                print(f"Тип: PyTorch объект ({type(content).__name__})")

        # 2. MATLAB файлы (.mat)
        elif ext == '.mat':
            data = sio.loadmat(file_path)
            keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"Тип: MATLAB файл")
            print(f"Доступные ключи: {keys}")
            for k in keys:
                val = data[k]
                if hasattr(val, 'shape'):
                    print(f"  • '{k}': размер {val.shape} | тип {val.dtype}")

        # 3. NumPy файлы (.npz)
        elif ext == '.npz':
            with np.load(file_path, allow_pickle=True) as data:
                print(f"Тип: NumPy архив (.npz)")
                for k in data.files:
                    print(f"  • '{k}': размер {data[k].shape}")

    except Exception as e:
        print(f"❌ Ошибка при чтении {os.path.basename(file_path)}: {e}")

def run_scanner(target_dir='.'):
    extensions = ('.pt', '.mat', '.npz')
    files = [f for f in os.listdir(target_dir) if f.endswith(extensions)]
    
    if not files:
        print(f"В директории '{os.path.abspath(target_dir)}' не найдено файлов {extensions}")
        return

    print(f"Найдено файлов для анализа: {len(files)}")
    for file in files:
        analyze_dataset(os.path.join(target_dir, file))

if __name__ == "__main__":
    # Скрипт спросит путь. Если просто нажать Enter, он проверит текущую папку.
    folder = input("Введите путь к папке (Enter для текущей): ").strip() or "."
    run_scanner(folder)