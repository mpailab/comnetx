import os.path
import sys
import json
from pathlib import Path

def get_sorted_table(info, sorting_key):
    datasets = list(info.keys())
    max_name_len = max(map(len, datasets))
    max_n_strlen = max(map(lambda x: len(str(info[x]["n"])), datasets))
    max_m_strlen = max(map(lambda x: len(str(info[x]["m"])), datasets))
    max_w_strlen = max(map(lambda x: len(str(info[x]["w"])), datasets))
    max_d_strlen = max(map(lambda x: len(str(info[x]["d"])), datasets))
    datasets.sort(key = lambda x: info[x][sorting_key])
    res =""
    for dataset in datasets:
        pstring = f"{dataset:<{max_name_len}}"
        pstring += f" {info[dataset]['n']:<{max_n_strlen}}"
        pstring += f" {info[dataset]['m']:<{max_m_strlen}}"
        pstring += f" {info[dataset]['d']:<{max_d_strlen}}"
        pstring += f" {info[dataset]['w']:<{max_w_strlen}}"
        res += f"{pstring}\n"
    return res

def sort_jsons(source_dir, sorting_key, output_dir, formats):
    for fmt in formats:
        path = source_dir / f"{fmt}.json"
        with open(path) as _:
            info = json.load(_)
        
        # json-files
        # sorted_dict = dict(sorted(info.items(), key=lambda item: item[1][sorting_key]))
        # with open(output_dir / f"{fmt}.json", 'w', encoding='utf-8') as _:
        #     json.dump(sorted_dict, _, indent=4, ensure_ascii=False)
        
        # txt files
        sorted_table = get_sorted_table(info, sorting_key)
        with open(output_dir / f"{fmt}.txt", 'w', encoding='utf-8') as _:
            _.write(sorted_table)

INFO_ROOT = Path("datasets-info")
formats = ["konect", "magi", "attr_graphs", "dyn_attr_graphs", "tgc", "ogb"]
sort_jsons(INFO_ROOT / "json", 'n', INFO_ROOT / "sorted_by_nodes", formats) # by_nodes
sort_jsons(INFO_ROOT / "json", 'm', INFO_ROOT / "sorted_by_edges", formats) # by_edges
