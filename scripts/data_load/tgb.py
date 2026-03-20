from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

name = "tgbn-trade"

dataset = PyGNodePropPredDataset(name=name, root="/home/egorov/comnetx/tgb_datasets")

dataset.src