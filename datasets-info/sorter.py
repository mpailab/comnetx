import json

# 1. Чтение данных из JSON файла
with open('all.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 2. Сортировка данных по значению 'n'
sorted_data = dict(sorted(data.items(), key=lambda item: item[1]['n']))

# 3. Сохранение отсортированных данных в новый JSON файл
with open('all_sorted_by_n.json', 'w', encoding='utf-8') as file:
    json.dump(sorted_data, file, indent=4, ensure_ascii=False)

print(f"Датасеты отсортированы по возрастанию 'n' и сохранены в файл 'datasets_sorted_by_n.json'")
print(f"Общее количество датасетов: {len(sorted_data)}")