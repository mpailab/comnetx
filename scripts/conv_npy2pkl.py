import networkx as nx
import numpy as np
import pickle
import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
Data_root = os.path.join(PROJECT_PATH, "test", "graphs", "small")

datasets = ["acm", "bat", "dblp", "eat", "uat"]

for dataset in datasets:
    data = np.load(Data_root + "/" + dataset + "/" + dataset + "_adj.npy")
    G = nx.from_numpy_array(data)
    graph_list = [G]

    with open(Data_root + "/" + dataset + "/" + dataset + ".pkl", "wb") as f:
        pickle.dump(graph_list, f)

    # labels = np.load(Data_root + "/" + dataset + "/" + dataset + "_label.npy", allow_pickle=True)

    labels_array = np.load(Data_root + "/" + dataset + "/" + dataset + "_label.npy", allow_pickle=True)
    labels_dict = {i: int(labels_array[i]) for i in range(len(labels_array))}

    # print(set(labels_dict.values()))

    with open(Data_root + "/" + dataset + "/" + dataset + "_label.pkl", "wb") as f:
        pickle.dump(labels_dict, f)