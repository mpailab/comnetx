name = 'com-dblp.ungraph'
input_filename = '/auto/datasets/graphs/comnetx/baselines/FLMIG_algorithm/real_network/batched/' + name + '.txt'   # Имя вашего исходного файла
output_filename = '/auto/datasets/graphs/comnetx/baselines/FLMIG_algorithm/real_network/batched/' + name +'_batched.txt' # Имя нового файла

def convert_graph():
    unique_nodes = set()
    raw_edges = [] 

    print("1. Чтение файла...")
    try:
        with open(input_filename, 'r') as infile:
            for line in infile:
                clean_line = line.strip()
                if clean_line:
                    parts = clean_line.split()
                    # Пытаемся считать два числа (игнорируем веса если они уже есть)
                    if len(parts) >= 2:
                        try:
                            u = int(parts[0])
                            v = int(parts[1])
                            
                            unique_nodes.add(u)
                            unique_nodes.add(v)
                            raw_edges.append((u, v))
                        except ValueError:
                            # Пропускаем заголовки или мусор
                            continue

        if not unique_nodes:
            print("Ошибка: Файл пуст или не содержит данных.")
            return

        # 2. Создаем карту перекодировки (Mapping)
        # Сортируем узлы, чтобы сохранить относительный порядок 
        # (меньший старый ID всегда получит меньший новый ID)
        sorted_nodes = sorted(list(unique_nodes))
        
        # Словарь: {Старый_ID : Новый_ID}
        # Например: {1:0, 2:1, 3:2, 5:3, 7:4, 8:5}
        node_map = {old_id: new_id for new_id, old_id in enumerate(sorted_nodes)}

        num_nodes = len(unique_nodes)
        num_edges = len(raw_edges)

        print(f"   Найдено вершин: {num_nodes}")
        print(f"   Найдено ребер: {num_edges}")
        print("   Перенумерация (убираем пропуски и начинаем с 0)...")

        # 3. Запись результата
        with open(output_filename, 'w') as outfile:
            # Заголовок: Вершины Ребра
            outfile.write(f"{num_nodes} {num_edges}\n")
            
            for u, v in raw_edges:
                # Получаем новые ID из словаря
                new_u = node_map[u]
                new_v = node_map[v]
                
                # Записываем: новый_u новый_v 1 0
                outfile.write(f"{new_u} {new_v} 1 0\n")

        print(f"Готово! Результат в файле: {output_filename}")
        
        # Пример для проверки
        example_old = sorted_nodes[0]
        example_new = node_map[example_old]
        print(f"Пример замены: Вершина {example_old} -> {example_new}")
        if len(sorted_nodes) > 3:
             example_mid = sorted_nodes[3] # Четвертая вершина
             print(f"Пример замены (был пропуск?): Вершина {example_mid} -> {node_map[example_mid]}")

    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_filename}' не найден.")

if __name__ == "__main__":
    convert_graph()