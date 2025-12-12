import sys
import os
import torch
import pytest
from typing import Union


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_PATH, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


from optimizer import Optimizer
from datasets import Dataset
import sparse


# @pytest.mark.short
# FIXME now aggregate don't work for 2 sparse matrix with int-like elements type
# def test_aggregate_1():
#     adj = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]], dtype = torch.int64).to_sparse()
#     coms = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 1]]).type(adj.dtype).to_sparse()
#     res = Optimizer.aggregate(adj, coms)
#     true_res = torch.tensor([[9, 0], [1, 1]])
#     assert torch.equal(true_res, res)



def get_all_datasets():
    """
    Сreate dict with all datasets in test directory.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    datasets = {}
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            if os.path.isdir(base_dir):
                datasets[name] = base_dir
    return datasets


@pytest.mark.short
def test_aggregate_2():
    adj = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]], dtype = torch.float).to_sparse()
    coms = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 1]]).type(adj.dtype).to_sparse()
    res = Optimizer.aggregate(adj, coms)
    true_res = torch.tensor([[9, 0], [1, 1]])
    assert torch.equal(true_res.to_dense(), res.to_dense())


@pytest.mark.short
def test_run_leidenalg():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3]])
    opt = Optimizer(A, communities = communities, method  = "leidenalg")
    nodes_mask = torch.tensor([0, 0, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())


@pytest.mark.short
def test_run_prgpt():
    A = torch.tensor([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ]).to_sparse_coo()


    communities = torch.tensor([
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0, 0],
    ])


    opt = Optimizer(A, communities=communities, method="prgpt:infomap")


    nodes_mask = torch.tensor([0, 0, 1, 0, 0, 0]).bool()


    opt.run(nodes_mask)



@pytest.mark.long
def test_run_magi():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3]])
    opt = Optimizer(A, communities = communities, method  = "magi")
    nodes_mask = torch.tensor([1, 1, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())


@pytest.mark.long
def test_run_dese():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3]])
    features = torch.randn(4, 16, dtype=torch.float32)
    opt = Optimizer(A, communities=communities, features=features, method="dese")
    nodes_mask = torch.tensor([1, 1, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())

@pytest.mark.long
def test_run_s2cag():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3]])
    features = abs(torch.randn(4, 16, dtype=torch.float32))
    opt = Optimizer(A, communities=communities, features=features, method="s2cag")
    nodes_mask = torch.tensor([1, 1, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())

@pytest.mark.long
def test_run_dmon():
    A = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]]).to_sparse_coo()
    communities = torch.tensor([[1, 1, 1, 3]])
    opt = Optimizer(A, communities = communities, method  = "dmon")
    nodes_mask = torch.tensor([0, 0, 1, 0]).bool()
    print("communities:", communities)
    print("nodes_mask:", nodes_mask)
    opt.run(nodes_mask)
    print()
    print(opt.coms.to_dense())


def test_aggregate_larger():
    A = torch.tensor([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ], dtype=torch.float32).to_sparse_coo()


    coms = torch.tensor([0, 0, 0, 1, 1, 1])
    ext_nodes = torch.arange(6)


    old_idx = torch.unique(coms)
    new_idx = {j.item(): i for i, j in enumerate(old_idx)}


    old_coms = torch.stack((
        torch.tensor([new_idx[i.item()] for i in coms]),
        ext_nodes
    ))


    n = old_idx.size(0)


    aggr_ptn = sparse.tensor(old_coms, (n, A.size(0)), A.dtype)
    aggr_adj = torch.sparse.mm(aggr_ptn, torch.sparse.mm(A, aggr_ptn.t()))


    print("Aggregated adjacency:\n", aggr_adj.to_dense())


    expected = torch.tensor([
        [6., 0.],
        [0., 6.]
    ])


    assert torch.allclose(aggr_adj.to_dense(), expected)



def get_all_small_datasets():
    """
    Сreate dict with all datasets in test directory.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    datasets = []
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            datasets.append(name)
    return datasets


datasets = get_all_small_datasets()
@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets,
)
def test_run_magi_on_custom_dataset(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="magi")


    top_level_coms = opt.coms[0]  # size (num_nodes,)


    coo = adj if not adj.is_sparse else adj.to_dense()
    if adj.is_sparse:
        edges = adj._indices().t()
    else:
        edges = torch.nonzero(adj)


    found = False
    for i, j in edges:
        ci, cj = top_level_coms[i].item(), top_level_coms[j].item()
        if ci != cj:
            com1_nodes = (top_level_coms == ci)
            com2_nodes = (top_level_coms == cj)
            nodes_mask = com1_nodes | com2_nodes
            found = True
            break


    if not found:
        raise RuntimeError("No communities with edge between them")


    print(adj.shape, nodes_mask.sum())
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets,
)
def test_run_dese_on_custom_dataset(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="dese")

    top_level_coms = opt.coms[0]  # size (num_nodes,)

    coo = adj if not adj.is_sparse else adj.to_dense()
    if adj.is_sparse:
        edges = adj._indices().t()
    else:
        edges = torch.nonzero(adj) 

    found = False
    for i, j in edges:
        ci, cj = top_level_coms[i].item(), top_level_coms[j].item()
        if ci != cj:
            com1_nodes = (top_level_coms == ci)
            com2_nodes = (top_level_coms == cj)
            nodes_mask = com1_nodes | com2_nodes
            found = True
            break

    if not found:
        raise RuntimeError("No communities with edge between them")

    print(adj.shape, nodes_mask.sum())
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets,
)
def test_run_s2cag_on_custom_dataset(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="s2cag")

    top_level_coms = opt.coms[0]  # size (num_nodes,)

    coo = adj if not adj.is_sparse else adj.to_dense()
    if adj.is_sparse:
        edges = adj._indices().t()
    else:
        edges = torch.nonzero(adj) 

    found = False
    for i, j in edges:
        ci, cj = top_level_coms[i].item(), top_level_coms[j].item()
        if ci != cj:
            com1_nodes = (top_level_coms == ci)
            com2_nodes = (top_level_coms == cj)
            nodes_mask = com1_nodes | com2_nodes
            found = True
            break

    if not found:
        raise RuntimeError("No communities with edge between them")

    print(adj.shape, nodes_mask.sum())
    opt.run(nodes_mask)


@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_dmon_on_custom_dataset(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="dmon")


    top_level_coms = opt.coms[0]  # size (num_nodes,)


    coo = adj if not adj.is_sparse else adj.to_dense()
    if adj.is_sparse:
        edges = adj._indices().t()
    else:
        edges = torch.nonzero(adj)


    found = False
    for i, j in edges:
        ci, cj = top_level_coms[i].item(), top_level_coms[j].item()
        if ci != cj:
            com1_nodes = (top_level_coms == ci)
            com2_nodes = (top_level_coms == cj)
            nodes_mask = com1_nodes | com2_nodes
            found = True
            break


    if not found:
        raise RuntimeError("No communities with edge between them")


    print(adj.shape, nodes_mask.sum())
    opt.run(nodes_mask)


@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_magi_on_isolated_communities(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="magi")


    top_level_coms = opt.coms[0]  # shape (num_nodes,)
    num_nodes = adj.shape[0]


    unique_coms = torch.unique(top_level_coms)
    found = False


    row, col = adj._indices()
    for i in range(len(unique_coms)):
        for j in range(i + 1, len(unique_coms)):
            com1_mask = (top_level_coms == unique_coms[i])
            com2_mask = (top_level_coms == unique_coms[j])
            com1_nodes = com1_mask.nonzero(as_tuple=True)[0]
            com2_nodes = com2_mask.nonzero(as_tuple=True)[0]


            mask1 = torch.isin(row, com1_nodes)
            mask2 = torch.isin(col, com2_nodes)
            has_edge = (mask1 & mask2).any()


            if not has_edge:
                nodes_mask = com1_mask | com2_mask
                found = True
                break
        if found:
            break


    if not found:
        raise RuntimeError("No two isolated communities found")


    print(f"{adj.shape}, nodes_mask sum: {nodes_mask.sum()}")
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_dese_on_isolated_communities(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="dese")

    top_level_coms = opt.coms[0]  # shape (num_nodes,)
    num_nodes = adj.shape[0]

    unique_coms = torch.unique(top_level_coms)
    found = False

    row, col = adj._indices()
    for i in range(len(unique_coms)):
        for j in range(i + 1, len(unique_coms)):
            com1_mask = (top_level_coms == unique_coms[i])
            com2_mask = (top_level_coms == unique_coms[j])
            com1_nodes = com1_mask.nonzero(as_tuple=True)[0]
            com2_nodes = com2_mask.nonzero(as_tuple=True)[0]

            mask1 = torch.isin(row, com1_nodes)
            mask2 = torch.isin(col, com2_nodes)
            has_edge = (mask1 & mask2).any()

            if not has_edge:
                nodes_mask = com1_mask | com2_mask
                found = True
                break
        if found:
            break

    if not found:
        raise RuntimeError("No two isolated communities found")

    print(f"{adj.shape}, nodes_mask sum: {nodes_mask.sum()}")
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_s2cag_on_isolated_communities(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="s2cag")

    top_level_coms = opt.coms[0]  # shape (num_nodes,)
    num_nodes = adj.shape[0]

    unique_coms = torch.unique(top_level_coms)
    found = False

    row, col = adj._indices()
    for i in range(len(unique_coms)):
        for j in range(i + 1, len(unique_coms)):
            com1_mask = (top_level_coms == unique_coms[i])
            com2_mask = (top_level_coms == unique_coms[j])
            com1_nodes = com1_mask.nonzero(as_tuple=True)[0]
            com2_nodes = com2_mask.nonzero(as_tuple=True)[0]

            mask1 = torch.isin(row, com1_nodes)
            mask2 = torch.isin(col, com2_nodes)
            has_edge = (mask1 & mask2).any()

            if not has_edge:
                nodes_mask = com1_mask | com2_mask
                found = True
                break
        if found:
            break

    if not found:
        raise RuntimeError("No two isolated communities found")

    print(f"{adj.shape}, nodes_mask sum: {nodes_mask.sum()}")
    opt.run(nodes_mask)


@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_magi_partial_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    communities = labels.unsqueeze(0)  # 1 уровень сообществ


    opt = Optimizer(adj, features=features, method="magi", communities=communities)


    top_level_coms = opt.coms[0]  # shape (num_nodes,)
    unique_coms = torch.unique(top_level_coms)
    print("top_level_coms:", top_level_coms)
    print("unique_coms:", unique_coms)
    for i in range(len(unique_coms)):
        com_nodes = torch.where(top_level_coms == unique_coms[i])[0]
        print(f"Сообщество {unique_coms[i]} содержит {len(com_nodes)} узлов")
    found = False
    for i in range(len(unique_coms)):
        for j in range(i + 1, len(unique_coms)):
            com1_nodes = torch.where(top_level_coms == unique_coms[i])[0]
            com2_nodes = torch.where(top_level_coms == unique_coms[j])[0]


            if len(com1_nodes) > 1 and len(com2_nodes) > 1:
                num1 = max(1, len(com1_nodes)//2)
                num2 = max(1, len(com2_nodes)//2)
                subset1 = com1_nodes[torch.randperm(len(com1_nodes))[:num1]]
                subset2 = com2_nodes[torch.randperm(len(com2_nodes))[:num2]]


                nodes_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
                nodes_mask[subset1] = True
                nodes_mask[subset2] = True
                found = True
                break
        if found:
            break


    if not found:
        raise RuntimeError("Не удалось найти два сообщества с достаточным числом узлов")


    print(f"Adj shape: {adj.shape}, Masked nodes: {nodes_mask.sum()}")
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_dese_partial_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    communities = labels.unsqueeze(0)  # 1 уровень сообществ

    opt = Optimizer(adj, features=features, method="dese", communities=communities)

    top_level_coms = opt.coms[0]  # shape (num_nodes,)
    unique_coms = torch.unique(top_level_coms)
    print("top_level_coms:", top_level_coms)
    print("unique_coms:", unique_coms)
    for i in range(len(unique_coms)):
        com_nodes = torch.where(top_level_coms == unique_coms[i])[0]
        print(f"Сообщество {unique_coms[i]} содержит {len(com_nodes)} узлов")
    found = False
    for i in range(len(unique_coms)):
        for j in range(i + 1, len(unique_coms)):
            com1_nodes = torch.where(top_level_coms == unique_coms[i])[0]
            com2_nodes = torch.where(top_level_coms == unique_coms[j])[0]

            if len(com1_nodes) > 1 and len(com2_nodes) > 1:
                num1 = max(1, len(com1_nodes)//2)
                num2 = max(1, len(com2_nodes)//2)
                subset1 = com1_nodes[torch.randperm(len(com1_nodes))[:num1]]
                subset2 = com2_nodes[torch.randperm(len(com2_nodes))[:num2]]

                nodes_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
                nodes_mask[subset1] = True
                nodes_mask[subset2] = True
                found = True
                break
        if found:
            break

    if not found:
        raise RuntimeError("Не удалось найти два сообщества с достаточным числом узлов")

    print(f"Adj shape: {adj.shape}, Masked nodes: {nodes_mask.sum()}")
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_s2cag_partial_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    communities = labels.unsqueeze(0)  # 1 уровень сообществ

    opt = Optimizer(adj, features=features, method="s2cag", communities=communities)

    top_level_coms = opt.coms[0]  # shape (num_nodes,)
    unique_coms = torch.unique(top_level_coms)
    print("top_level_coms:", top_level_coms)
    print("unique_coms:", unique_coms)
    for i in range(len(unique_coms)):
        com_nodes = torch.where(top_level_coms == unique_coms[i])[0]
        print(f"Сообщество {unique_coms[i]} содержит {len(com_nodes)} узлов")
    found = False
    for i in range(len(unique_coms)):
        for j in range(i + 1, len(unique_coms)):
            com1_nodes = torch.where(top_level_coms == unique_coms[i])[0]
            com2_nodes = torch.where(top_level_coms == unique_coms[j])[0]

            if len(com1_nodes) > 1 and len(com2_nodes) > 1:
                num1 = max(1, len(com1_nodes)//2)
                num2 = max(1, len(com2_nodes)//2)
                subset1 = com1_nodes[torch.randperm(len(com1_nodes))[:num1]]
                subset2 = com2_nodes[torch.randperm(len(com2_nodes))[:num2]]

                nodes_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
                nodes_mask[subset1] = True
                nodes_mask[subset2] = True
                found = True
                break
        if found:
            break

    if not found:
        raise RuntimeError("Не удалось найти два сообщества с достаточным числом узлов")

    print(f"Adj shape: {adj.shape}, Masked nodes: {nodes_mask.sum()}")
    opt.run(nodes_mask)


@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_magi_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="magi")


    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_dese_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="dese")

    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_s2cag_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="s2cag")

    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)


@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    datasets
)
def test_run_dmon_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="dmon")


    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)


"""
@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_flmig_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="flmig")


    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)
"""

@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_mfc_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="mfc")


    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)


@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_prgpt_infomap_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="prgpt:infomap")


    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)



@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_prgpt_locale_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="prgpt:locale")


    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)


@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_leiden_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="leidenalg")


    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)


@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_networkit_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")


    opt = Optimizer(adj, features=features, method="networkit")


    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_flmig_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="flmig")

    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_mfc_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="mfc")

    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_prgpt_infomap_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="prgpt:infomap")

    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)


@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_prgpt_locale_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="prgpt:locale")

    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_leiden_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="leidenalg")

    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)

@pytest.mark.long
@pytest.mark.parametrize("name", datasets)
def test_run_networkit_on_true_mask(name):
    data_dir = os.path.join(os.path.dirname(__file__), "graphs", "small")
    dataset = Dataset(name, path=data_dir)
    adj, features, labels = dataset.load(tensor_type="coo")

    opt = Optimizer(adj, features=features, method="networkit")

    nodes_mask = torch.ones(adj.shape[0], dtype=torch.bool)
    opt.run(nodes_mask)

@pytest.mark.short
def test_update_adj():
    adj_matrix = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1]])
    opt = Optimizer(adj_matrix.to_sparse_coo())
    opt.update_adj(adj_matrix.to_sparse_coo())
    true_res = adj_matrix * 2
    res = opt.adj
    assert torch.equal(true_res, opt.adj.to_dense())


@pytest.mark.short
def test_neighborhood_1():
    A = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=torch.float32)
    A_sparse = A.to_sparse_coo()
    initial_nodes = torch.tensor([True, False, False, False])
    nodes_0 = Optimizer.neighborhood(A_sparse, initial_nodes, 0)
    nodes_1 = Optimizer.neighborhood(A_sparse, initial_nodes, 1)
    nodes_2 = Optimizer.neighborhood(A_sparse, initial_nodes, 2)
    true_nodes_1 = torch.tensor([True, True, False, False])
    true_nodes_2 = torch.tensor([True, True, True, True])
    assert torch.equal(nodes_0, initial_nodes)
    assert torch.equal(nodes_1, true_nodes_1)
    assert torch.equal(nodes_2, true_nodes_2)


def test_neighborhood_2():
    A = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=torch.float32)
    A_sparse = A.t().to_sparse_coo()
    initial_nodes = torch.tensor([True, False, False, False])
    nodes_0 = Optimizer.neighborhood(A_sparse, initial_nodes, 0)
    nodes_1 = Optimizer.neighborhood(A_sparse, initial_nodes, 1)
    nodes_2 = Optimizer.neighborhood(A_sparse, initial_nodes, 2)
    true_nodes_1 = torch.tensor([True, True, False, False])
    true_nodes_2 = torch.tensor([True, True, True, True])
    assert torch.equal(nodes_0, initial_nodes)
    assert torch.equal(nodes_1, true_nodes_1)
    assert torch.equal(nodes_2, true_nodes_2)


def test_modularity():
    communities = torch.tensor([
        [0, 0, 0, 3, 3, 3],
        [0, 0, 2, 3, 4, 3],
        [0, 1, 2, 3, 4, 5]
    ])


    adj_matrix = torch.tensor([
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 1, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 1, 1, 1]], dtype = torch.float).to_sparse()


    optimizer = Optimizer(adj_matrix=adj_matrix, communities=communities)
    modularity = optimizer.modularity()
    assert modularity < 1.0


@pytest.mark.short
def test_communities_none():
    """Тест когда communities=None"""
    adj = torch.randn(5, 5)
    optimizer = Optimizer(adj_matrix=adj, subcoms_depth=3)
   
    assert optimizer.coms.shape == (3, 5)  # (subcoms_depth, nodes_num)
    # Должны быть созданы дефолтные communities
    expected = torch.arange(5).repeat(3).reshape(3, 5)
    assert torch.equal(optimizer.coms, expected)


@pytest.mark.short
def test_perfect_match():
    """Тест когда communities идеально совпадает с subcoms_depth"""
    adj = torch.randn(5, 5)
    communities = torch.tensor([
        [0, 0, 1, 1, 2],  # уровень 0
        [0, 0, 0, 1, 1],  # уровень 1
        [0, 0, 0, 0, 0]   # уровень 2
    ])
   
    optimizer = Optimizer(adj_matrix=adj, communities=communities, subcoms_depth=3)
   
    assert optimizer.coms.shape == (3, 5)
    assert torch.equal(optimizer.coms, communities)


@pytest.mark.short
def test_communities_greater_depth(capfd):
    """Тест когда communities имеет большую глубину чем subcoms_depth"""
    adj = torch.randn(5, 5)
    communities = torch.tensor([
        [0, 0, 1, 1, 2],
        [0, 0, 0, 1, 1],  
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
   
    optimizer = Optimizer(adj_matrix=adj, communities=communities, subcoms_depth=3)
   
    captured = capfd.readouterr()
    assert "communities depth 4 > subcoms_depth 3" in captured.out
    assert "Truncating to 3 levels" in captured.out
   
    assert optimizer.coms.shape == (3, 5)
    expected = communities[:3, :]
    assert torch.equal(optimizer.coms, expected)


@pytest.mark.short
def test_communities_less_depth(capfd):
    """Тест когда communities имеет меньшую глубину чем subcoms_depth"""
    adj = torch.randn(5, 5)
    communities = torch.tensor([
        [0, 0, 1, 1, 2],
        [0, 0, 0, 1, 1]
    ])
   
    optimizer = Optimizer(adj_matrix=adj, communities=communities, subcoms_depth=4)
   
    captured = capfd.readouterr()
    assert "communities depth 2 < subcoms_depth 4" in captured.out
    assert "Extending with zeros to 4 levels" in captured.out
   
    assert optimizer.coms.shape == (4, 5)
    assert torch.equal(optimizer.coms[:2, :], communities)
    assert torch.equal(optimizer.coms[2:, :], torch.zeros((2, 5), dtype=communities.dtype))


@pytest.mark.short
def test_bad_nodes_number(capfd):
    """Тест когда communities имеет неправильное количество узлов"""
    adj = torch.randn(5, 5)
    communities = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ])
   
    optimizer = Optimizer(adj_matrix=adj, communities=communities, subcoms_depth=2)
   
    captured = capfd.readouterr()
    assert "bad communities shape" in captured.out
    assert "Use default communities" in captured.out
   
    assert optimizer.coms.shape == (2, 5)
    expected = torch.arange(5).repeat(2).reshape(2, 5)
    assert torch.equal(optimizer.coms, expected)


@pytest.mark.short
def test_1d_communities():
    """Тест когда communities передан как 1D тензор"""
    adj = torch.randn(5, 5)
    communities = torch.tensor([0, 0, 1, 1, 2])  # 1D тензор
   
    optimizer = Optimizer(adj_matrix=adj, communities=communities, subcoms_depth=3)
   
    assert optimizer.coms.shape == (3, 5)
    # Должен быть преобразован в 2D и дополнен нулями
    expected_first_row = communities
    assert torch.equal(optimizer.coms[0, :], expected_first_row)
    assert torch.equal(optimizer.coms[1:, :], torch.zeros((2, 5), dtype=communities.dtype))


@pytest.mark.short
def test_edge_case_single_node():
    """Тест с одним узлом"""
    adj = torch.randn(1, 1)
    communities = torch.tensor([[0]])
   
    optimizer = Optimizer(adj_matrix=adj, communities=communities, subcoms_depth=2)
   
    assert optimizer.coms.shape == (2, 1)
    assert optimizer.coms[0, 0] == 0
    assert optimizer.coms[1, 0] == 0  # дополнено нулем


@pytest.mark.short
def test_large_scale():
    """Тест с большим количеством узлов"""
    n_nodes = 100
    adj = torch.randn(n_nodes, n_nodes)
    communities = torch.randint(0, 10, (2, n_nodes))
   
    optimizer = Optimizer(adj_matrix=adj, communities=communities, subcoms_depth=5)
   
    assert optimizer.coms.shape == (5, n_nodes)
    assert torch.equal(optimizer.coms[:2, :], communities)
    assert torch.equal(optimizer.coms[2:, :], torch.zeros((3, n_nodes), dtype=communities.dtype))


@pytest.mark.short
def test_warning_messages(capfd):
    """Тест что правильные предупреждения выводятся в stdout"""
    adj = torch.randn(4, 4)
   
    # Test depth > subcoms_depth
    communities_more = torch.tensor([[0,0,1,1], [0,0,0,1], [0,0,0,0], [0,0,0,0]])
    Optimizer(adj_matrix=adj, communities=communities_more, subcoms_depth=2)
   
    captured = capfd.readouterr()
    assert "communities depth 4 > subcoms_depth 2" in captured.out
    assert "Truncating to 2 levels" in captured.out
   
    # Test depth < subcoms_depth  
    communities_less = torch.tensor([[0,0,1,1]])
    Optimizer(adj_matrix=adj, communities=communities_less, subcoms_depth=3)
   
    captured = capfd.readouterr()
    assert "communities depth 1 < subcoms_depth 3" in captured.out
    assert "Extending with zeros to 3 levels" in captured.out
   
    # Test bad shape
    communities_bad = torch.tensor([[0,0,1]])  # 3 nodes vs 4 in adj
    Optimizer(adj_matrix=adj, communities=communities_bad, subcoms_depth=2)
   
    captured = capfd.readouterr()
    assert "bad communities shape" in captured.out
    assert "Use default communities" in captured.out
