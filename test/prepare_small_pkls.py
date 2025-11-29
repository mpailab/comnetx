import os
import pickle
import numpy as np
import networkx as nx

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMALL_DIR = os.path.join(PROJECT_PATH, "test", "graphs", "small")
DATASETS = ["acm", "bat", "dblp", "eat", "uat"]

def build_pkl(name: str):
    data_dir = os.path.join(SMALL_DIR, name)

    adj_path   = os.path.join(data_dir, f"{name}_adj.npy")
    label_path = os.path.join(data_dir, f"{name}_label.npy")

    if not (os.path.exists(adj_path) and os.path.exists(label_path)):
        print(f"[skip] {name}: нет нужных npy файлов")
        return

    print(f"[make] {name}")

    adj = np.load(adj_path)
    y   = np.load(label_path)

    # список снапшотов: здесь один граф
    G = nx.from_numpy_array(adj)
    graph_snapshots = [G]

    with open(os.path.join(data_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(graph_snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)

    label_dict = {int(i): int(c) for i, c in enumerate(y)}
    with open(os.path.join(data_dir, f"{name}_label.pkl"), "wb") as f:
        pickle.dump(label_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    for ds in DATASETS:
        build_pkl(ds)
