import os
import sys
import pytest
import pickle
import numpy as np
from pathlib import Path
from conftest import collect_datasets

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
KONECT_INFO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets-info"))

from datasets import Dataset, KONECT_PATH
from baselines.mfc import mfc_adopted, PROJECT_PATH

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

def test_mfc(dataset, runner_mfc):
    cfg = collect_datasets()[dataset]
    if isinstance(cfg, str):
        ds = {"adj": cfg, "out": ""}
    elif isinstance(cfg, dict):
        ds = cfg
    else:
        raise TypeError(f"Unexpected dataset entry type: {type(cfg)}")

    network_type = ds.get("network_type", "MFC")
    snapshots = int(ds.get("snapshots", 1))

    res = runner_mfc(ds, dataset_name=dataset, network_type=network_type, snapshots=snapshots)
    assert res.returncode == 0, f"STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"

    out_dir = Path(PROJECT_PATH) / "results" / "mfc"
    raw_pkl = out_dir / "results_raw.pkl"
    topo_pkl = out_dir / "results_topo.pkl"
    assert raw_pkl.is_file(), "results_raw.pkl not found after MFC run"
    assert topo_pkl.is_file(), "results_topo.pkl not found after MFC run"

    with raw_pkl.open("rb") as f:
        raw = pickle.load(f)
    assert isinstance(raw, list) and len(raw) >= 1, "results_raw must be non-empty list"

    # первый элемент по времени
    snap = raw[0]

    # ожидаем, что snap = [Z, Q, adj_out, labels_out]
    Z, Q, adj_out, labels_out = snap

    Z = np.asarray(Z)
    Q = np.asarray(Q)
    labels_out = np.asarray(labels_out)

    # размеры по числу узлов согласованы
    assert Z.shape[0] == Q.shape[0] == labels_out.shape[0], "Inconsistent node count in Z/Q/labels"
    # число кластеров > 1
    assert Q.shape[1] >= 2, "Q must have at least 2 clusters"

    with topo_pkl.open("rb") as f:
        topo = pickle.load(f)
    assert isinstance(topo, list) and len(topo) == len(raw), "results_topo length mismatch with results_raw"



@pytest.mark.parametrize("name", list_small_datasets(), ids=lambda n: f"small={n}")
def test_mfc_on_small_datasets(name):
    ds = Dataset(name, path=str(SMALL_ROOT))
    adj, features, labels = ds.load(tensor_type="coo")

    assert adj.ndim == 2 and adj.size(0) == adj.size(1)
    assert labels.ndim == 1 and labels.size(0) == adj.size(0)

    mfc_adopted(adj_matrices=[adj],
                labels_list=[labels],
                network_type="MFC")

    out_dir = Path(PROJECT_PATH) / "results" / "mfc"
    assert (out_dir / "results_raw.pkl").is_file()
    assert (out_dir / "results_topo.pkl").is_file()

    with (out_dir / "results_raw.pkl").open("rb") as f:
        raw = pickle.load(f)
    assert len(raw) == 1
    Z, Q, adj_out, labels_out = raw[0]
    assert Z.shape[0] == adj.size(0)
    assert Q.shape[0] == adj.size(0)