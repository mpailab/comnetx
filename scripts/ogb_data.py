from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset
import torch

def info():
    dataset = NodePropPredDataset(name="ogbn-papers100M", root="/auto/datasets/graphs/comnetx/ogb/")
    graph, label = dataset[0]
    # dataset = LinkPropPredDataset(name="ogbl-citation2", root="/auto/datasets/graphs/comnetx/ogb/")
    # graph = dataset[0]
    # label = None

    edge_index = torch.tensor(graph['edge_index'])
    num_nodes = graph['num_nodes']

    print("edge_index", edge_index)

    if graph['node_feat'] is not None:
        x = torch.tensor(graph['node_feat'], dtype=torch.float)
        print("Фичи есть, размерность:", x.shape)
    else:
        print("У этого графа нет исходных фичей (node_feat is None).")

    if label is not None:
        print("label", (label))
    else:
        print("У этого графа нет исходных меток") 

    # Считаем степень каждого узла (количество исходящих ребер)
    # Используем bincount для подсчета упоминаний каждого индекса узла в первой строке edge_index
    degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()

    print(f"Средняя степень: {degrees.mean().item():.2f}")
    print(f"Максимальная степень: {degrees.max().item()}")
    print(f"Минимальная степень: {degrees.min().item()}")

def data():
    # # ogbn-products
    # # ogbn-arxiv
    # # ogbn-papers100M 
    # dataset = NodePropPredDataset(name = "ogbn-products")
    # graph, label = dataset[0] # label — это ваши Y

    # # X: Матрица признаков вершин (Node Features)
    # feat = graph['node_feat'] 

    # print(len(feat[:,:]))
    # print("label =", len(label))

    # # OGB хранит ребра в формате Edge List (список пар)
    # edge_index = graph['edge_index'] 
    # print(len(set(edge_index[0])))

    dataset = LinkPropPredDataset(name = "ogbl-citation2")
    graph = dataset[0]
    print(graph['node_feat'][:10,:])
    print(graph['edge_feat']) 
