import scipy.io
import numpy as np
import os
import sys

path = "/auto/datasets/graphs/comnetx/baselines/S2CAG/data/"
name = "Amazon_photos"
input_mat_file = path + name + ".mat"
output_folder = "/auto/datasets/graphs/comnetx/baselines/S2CAG/batched/" + name + "_s2"

def extract_and_save(mat_path, save_dir):
    # 1. Создаем папку для сохранения, если её нет
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Создана папка: {save_dir}")

    try:
        # 2. Загружаем .mat файл
        print(f"Загрузка файла {mat_path}...")
        data = scipy.io.loadmat(mat_path)
        # === Обработка матрицы смежности (W) ===
        if 'W' in data:
            adj = data['W']
            # Если матрица разреженная, превращаем в плотную (обычный numpy array)
            if scipy.sparse.issparse(adj):
                adj = adj.toarray()
            
            save_path = os.path.join(save_dir, name + '_s2_adj.npy')
            np.save(save_path, adj)
            print(f"[OK] Матрица смежности сохранена: {save_path} | Размер: {adj.shape}")
        else:
            print("[WARNING] Ключ 'W' не найден!")

        # === Обработка матрицы признаков (fea) ===
        if 'fea' in data:
            feat = data['fea']
            if scipy.sparse.issparse(feat):
                feat = feat.toarray()
            
            save_path = os.path.join(save_dir, name + '_s2_feat.npy')
            np.save(save_path, feat)
            print(f"[OK] Матрица признаков сохранена: {save_path} | Размер: {feat.shape}")
        else:
            print("[WARNING] Ключ 'fea' не найден!")

        # === Обработка меток (gnd) ===
        if 'gnd' in data:
            labels = data['gnd']
            
            # ПРОВЕРКА И ИСПРАВЛЕНИЕ:
            # Если размерность (1, N), значит метки записаны в одну строку.
            # Берем эту строку (labels[0]), чтобы получить массив длины N.
            if labels.shape[0] == 1 and labels.shape[1] > 1:
                labels = labels[0]  # Или labels = labels.flatten()
                print(f"   -> Метки были вектором-строкой, исправлено на: {labels.shape}")
                
            # На всякий случай, если это вектор-столбец (N, 1), тоже выпрямляем
            elif labels.shape[0] > 1 and labels.shape[1] == 1:
                labels = labels.flatten()

            save_path = os.path.join(save_dir, name + '_s2_label.npy')
            np.save(save_path, labels)

        print("\nГотово! Все файлы сохранены.")

    except FileNotFoundError:
        print(f"Ошибка: Файл {mat_path} не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    # Можно передать путь аргументом командной строки, или использовать дефолтный
    target_file = sys.argv[1] if len(sys.argv) > 1 else input_mat_file
    extract_and_save(target_file, output_folder)