from __future__ import annotations
import os
import json
from pathlib import Path

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