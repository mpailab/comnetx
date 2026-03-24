import os.path
import sys
import json

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))
from datasets import INFO, Dataset

for fmt in Dataset.FILE_BASED_FORMATS:
    path = INFO / f"{fmt}.json"
    with open(path) as _:
        data = json.load(_)

    by_nodes = dict(sorted(data.items(), key=lambda item: item[1]['n']))
    by_edges = dict(sorted(data.items(), key=lambda item: item[1]['m']))

    with open(INFO / "nodes-sorted" / f"{fmt}.json", 'w', encoding='utf-8') as _:
        json.dump(by_nodes, _, indent=4, ensure_ascii=False)
    with open(INFO / "edges-sorted" / f"{fmt}.json", 'w', encoding='utf-8') as _:
        json.dump(by_edges, _, indent=4, ensure_ascii=False)
