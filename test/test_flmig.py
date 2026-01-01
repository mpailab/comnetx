import pickle
from pathlib import Path
from conftest import collect_datasets
import sys
import torch, gc
import pytest
import os
import subprocess
import tempfile
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
KONECT_INFO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets-info"))

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_root = os.path.join(PROJECT_PATH, "test")

SMALL_ROOT = Path("/auto/datasets/graphs/small")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from datasets import Dataset
from baselines.flmig import flmig_adopted, PROJECT_PATH

from datasets import Dataset, KONECT_PATH

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

def load_konect_info():
    """Load dataset info from konect.json."""
    file_path = os.path.join(KONECT_INFO, "konect_sorted.json")
    with open(file_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info

def get_all_konect_datasets():
    """Return a dict {dataset_name: Dataset object}."""
    info = load_konect_info()
    datasets = {}
    for name in info.keys():
        path = os.path.join(KONECT_PATH, name)
        if os.path.exists(path):
            datasets[name] = Dataset(name, KONECT_PATH)
    return datasets

KONECT_DATASETS = get_all_konect_datasets()

@pytest.mark.long
@pytest.mark.parametrize(
    "name",
    list(KONECT_DATASETS.keys()),
    ids=list(KONECT_DATASETS.keys())
)
def test_flmig_konect_dataset(name):
    dataset = Dataset(name, path=KONECT_PATH)
    adj, features, labels = dataset.load()
    adj = adj.coalesce()
    
    new_values = adj.values().abs()
    adj = torch.sparse_coo_tensor(
        adj.indices(), new_values, adj.size()
    ).coalesce() #FIXME

    print("adj =", adj)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_adj_path = os.path.join(tmpdir, f"adj_{name}.pt")
        temp_labels_path = os.path.join(tmpdir, f"labels_{name}.pt")
        torch.save(adj, temp_adj_path)
        torch.save(labels, temp_labels_path)

        cmd = [
            sys.executable,
            test_root + "/run_flmig_subprocess.py",
            "--adj", temp_adj_path,
            "--out", temp_labels_path
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Subprocess failed with code {proc.returncode}")
            print("stdout:", proc.stdout)
            print("stderr:", proc.stderr)
            pytest.fail(f"Subprocess failed for dataset {name}")

        new_labels = torch.load(temp_labels_path)

        print("new_labels =", new_labels, "len =", len(new_labels.numpy()))

        assert isinstance(new_labels, torch.Tensor)
        assert new_labels.shape[0] == adj.size(0)
        assert new_labels.dtype in (torch.int64, torch.long)
        assert new_labels.min() >= 0
    del adj, features, labels, new_labels
    gc.collect()
    torch.cuda.empty_cache()
