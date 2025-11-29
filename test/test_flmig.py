import os
import sys
import pickle
from pathlib import Path

import pytest
import torch

from conftest import collect_datasets

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from datasets import Dataset
from baselines.flmig import flmig_adopted, PROJECT_PATH


SMALL_ROOT = Path(PROJECT_PATH) / "test" / "graphs" / "small"


def list_small_datasets():
    out = []
    for d in SMALL_ROOT.iterdir():
        if not d.is_dir():
            continue
        dname = d.name
        if ((d / f"{dname}_adj.npy").is_file()
                or (d / f"{dname}_coo_adj.joblib").is_file()):
            out.append(dname)
    return sorted(out)

#@pytest.mark.usefixtures("resource_monitor", "prof")
def test_flmig(dataset, runner_flmig):

    cfg = collect_datasets()[dataset]
    if isinstance(cfg, str):
        ds = {"adj": cfg, "out": ""}
    elif isinstance(cfg, dict):
        ds = cfg
    else:
        raise TypeError(f"Unexpected dataset entry type: {type(cfg)}")

    num_iter = int(ds.get("Number_iter", 10))
    beta = float(ds.get("Beta", 0.5))
    max_rb = int(ds.get("max_rb", 10))

    res = runner_flmig(ds, dataset_name=dataset, Number_iter=num_iter, Beta=beta, max_rb=max_rb)
    assert res.returncode == 0, f"STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"


@pytest.mark.parametrize("name", list_small_datasets(), ids=lambda n: f"small={n}")
def test_flmig_on_small_datasets(name):
    ds = Dataset(name, path=str(SMALL_ROOT))
    adj, features, labels = ds.load(tensor_type="coo")

    # квадратный граф
    assert adj.ndim == 2 and adj.size(0) == adj.size(1)

    # FLMIG работает с плотной бинарной матрицей
    if adj.is_sparse:
        A = adj.coalesce()
        adj_dense = torch.sparse_coo_tensor(
            A.indices(),
            torch.where(A.values() > 0,
                        torch.ones_like(A.values()),
                        torch.zeros_like(A.values())),
            size=A.size(),
        ).to_dense()
    else:
        adj_dense = (adj > 0).to(torch.float32)
        adj_dense.fill_diagonal_(0.0)

    # просто проверяем, что алгоритм отрабатывает без падения
    flmig_adopted(adj=adj_dense, Number_iter=10, Beta=0.5, max_rb=2)
