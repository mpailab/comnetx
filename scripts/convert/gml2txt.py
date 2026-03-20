import networkx as nx

def gml_to_txt(gml_path, txt_path):
    try:
        # Читаем граф. Важно оставить label='id', чтобы узлы были числами (0, 1, 2...),
        # а не строками ("BrighamYoung").
        G = nx.read_gml(gml_path, label='id')
        
        # Создаем (или перезаписываем) txt файл
        with open(txt_path, 'w') as f:
            # G.edges() возвращает список пар (u, v)
            for u, v in G.edges():
                # Записываем два индекса через пробел и перенос строки
                f.write(f"{u} {v}\n")
        
        print(f"Успешно конвертировано! Сохранено в: {txt_path}")
        print(f"Количество узлов: {G.number_of_nodes()}")
        print(f"Количество ребер: {G.number_of_edges()}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

# --- Пример использования ---
input_file = '/auto/datasets/graphs/comnetx/baselines/FLMIG_algorithm/real_network/adjnoun.gml'  # Укажите путь к вашему gml файлу
output_file = '/auto/datasets/graphs/comnetx/baselines/FLMIG_algorithm/real_network/batched/adjnoun.txt' # Имя выходного файла

gml_to_txt(input_file, output_file)