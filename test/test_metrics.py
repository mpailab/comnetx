import sys
import os
import torch
import shutil
import tempfile
import pytest
import subprocess
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from datasets import Dataset, KONECT_PATH
from metrics import Metrics

TEST_DIR = os.path.dirname(__file__)
GRAPHS_DIR = os.path.join(TEST_DIR, "graphs", "small")
SBM_GRAPHS_DIR = os.path.join(TEST_DIR, "graphs", "sbm")

METRICS_JSON = os.path.join(TEST_DIR, "magi_metrics_small.json")

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


datasets = get_all_datasets()
@pytest.mark.long
@pytest.mark.parametrize(
    "name,data_dir",
    list(datasets.items()),
    ids=list(datasets.keys())
)
def test_metrics_on_small_graphs(name, data_dir):
    """
    Для каждого small‑датасета:
    - грузим граф;
    - считаем modularity для истинной разметки;
    - считаем PUR, ARI, Macro‑F1 для "идеального" случая (pred == true)
      и для случайной перестановки меток.
    """
    ds = Dataset(dataset_name=name, path=data_dir)
    adj, features, labels = ds.load(tensor_type="coo")

    assert isinstance(adj, torch.Tensor)
    assert labels is not None
    assert labels.shape[0] == adj.shape[0]

    Q = Metrics.modularity(adj, labels)
    assert isinstance(Q, float)
    assert -1.0 <= Q <= 1.0

    pur = Metrics.purity_score(labels, labels)
    ari = Metrics.ari_score(labels, labels)
    mf1 = Metrics.macro_f1(labels, labels)

    assert pur == pytest.approx(1.0)
    assert ari == pytest.approx(1.0)
    assert mf1 == pytest.approx(1.0)

    perm = torch.randperm(labels.shape[0])
    shuffled = labels[perm]

    pur_rand = Metrics.purity_score(labels, shuffled)
    ari_rand = Metrics.ari_score(labels, shuffled)
    mf1_rand = Metrics.macro_f1(labels, shuffled)

    assert pur_rand < 1.0
    assert ari_rand < 1.0
    assert mf1_rand < 1.0


@pytest.mark.long
@pytest.mark.parametrize(
    "name,data_dir",
    list(datasets.items()),
    ids=list(datasets.keys())
)
def test_magi_metrics_on_small_graphs(name, data_dir):
    """
    Для каждого small‑датасета:
    - грузим граф;
    - запускаем MAGI через run_magi_subprocess.py;
    - считаем PUR, ARI, Macro‑F1 между true_labels и magi_labels;
    - сохраняем метрики в JSON.
    """
    ds = Dataset(dataset_name=name, path=data_dir)
    adj, features, true_labels = ds.load(tensor_type="coo")
    adj = adj.coalesce()
    num_nodes = adj.size(0)

    assert isinstance(true_labels, torch.Tensor)
    assert true_labels.shape[0] == num_nodes

    with tempfile.TemporaryDirectory() as tmpdir:
        adj_path = os.path.join(tmpdir, f"{name}_adj.pt")
        feat_path = os.path.join(tmpdir, f"{name}_feat.pt")
        out_path = os.path.join(tmpdir, f"{name}_labels.pt")

        if features is None:
            features = torch.randn(num_nodes, 128, dtype=torch.float32)

        torch.save(adj, adj_path)
        torch.save(features, feat_path)

        cmd = [
            sys.executable,
            os.path.join(TEST_DIR, "run_magi_subprocess.py"),
            "--adj", adj_path,
            "--features", feat_path,
            "--epochs", "1",
            "--batchsize", "1024",
            "--out", out_path,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("STDOUT:", proc.stdout)
            print("STDERR:", proc.stderr)
            pytest.fail(f"MAGI subprocess failed for dataset {name}")

        magi_labels = torch.load(out_path)

    assert isinstance(magi_labels, torch.Tensor)
    assert magi_labels.shape[0] == num_nodes
    assert magi_labels.dtype in (torch.int64, torch.long)
    assert magi_labels.min() >= 0

    pur = Metrics.purity_score(true_labels, magi_labels)
    ari = Metrics.ari_score(true_labels, magi_labels)
    mf1 = Metrics.macro_f1(true_labels, magi_labels)

    assert 0.0 <= pur <= 1.0
    assert -1.0 <= ari <= 1.0
    assert 0.0 <= mf1 <= 1.0

    metrics_entry = {
        "PUR": float(pur),
        "ARI": float(ari),
        "MacroF1": float(mf1),
    }

    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[name] = metrics_entry

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)import sys
import os
import pytest
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from launcher import dynamic_launch
from datasets import INFO
from datasets import Dataset, KONECT_PATH

from baselines.s2cag import s2cag_metrics
from baselines.dese import dese_metrics
from metrics import Metrics

def test_s2cag_acc():
    n = 30
    k = 4
    nodes_per_cluster = [15, 4, 5, 6]

    true_labels = []
    for c, size in enumerate(nodes_per_cluster):
         true_labels.extend([c] * size)
    true_labels = torch.tensor(true_labels)

    feature = torch.zeros(n, 5)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        feature[start:end, c*10:(c+1)*10] = 1.0
        start = end
    feature = feature + torch.randn_like(feature) * 0.15
    feature = feature / feature.norm(dim=1, keepdim=True)
    feature = abs(feature)

    adj = torch.zeros(n, n)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        idx = torch.arange(start, end)
        i, j = torch.meshgrid(idx, idx, indexing='ij')
        mask = torch.rand(size, size) < 0.4
        mask = torch.triu(mask, 1)
        adj[i[mask], j[mask]] = 1
        adj[j[mask], i[mask]] = 1
        start = end

    inter = torch.rand(n, n) < 0.003
    inter = inter & (torch.triu(torch.ones(n,n), 1) > 0)
    adj[inter] = 1
    adj.T[inter] = 1
    adj.fill_diagonal_(0)

    adj = adj.to_sparse_coo()
    # print(adj, feature)
    pred_labels, metrics = s2cag_metrics(adj, feature, true_labels, n_runs=3)
    # print(new_labels)
    acc_s2cag = metrics['acc'][0]/100
    acc = Metrics.accuracy(true_labels, pred_labels)
    # print("acc =", acc, "acc_s2cag =", acc_s2cag)
    assert 0.0 <= acc <= 1.0
    assert abs(acc - acc_s2cag) < 1e-3

def test_s2cag_nmi():
    n = 30
    k = 4
    nodes_per_cluster = [15, 4, 5, 6]

    true_labels = []
    for c, size in enumerate(nodes_per_cluster):
         true_labels.extend([c] * size)
    true_labels = torch.tensor(true_labels)

    feature = torch.zeros(n, 5)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        feature[start:end, c*10:(c+1)*10] = 1.0
        start = end
    feature = feature + torch.randn_like(feature) * 0.15
    feature = feature / feature.norm(dim=1, keepdim=True)
    feature = abs(feature)

    adj = torch.zeros(n, n)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        idx = torch.arange(start, end)
        i, j = torch.meshgrid(idx, idx, indexing='ij')
        mask = torch.rand(size, size) < 0.4
        mask = torch.triu(mask, 1)
        adj[i[mask], j[mask]] = 1
        adj[j[mask], i[mask]] = 1
        start = end

    inter = torch.rand(n, n) < 0.003
    inter = inter & (torch.triu(torch.ones(n,n), 1) > 0)
    adj[inter] = 1
    adj.T[inter] = 1
    adj.fill_diagonal_(0)

    adj = adj.to_sparse_coo()
    # print(adj, feature)
    pred_labels, metrics = s2cag_metrics(adj, feature, true_labels, n_runs=3)
    # print(new_labels)
    nmi_s2cag = metrics['nmi'][0]/100
    nmi = Metrics.nmi(true_labels, pred_labels)
    print("acc =", nmi, "nmi_s2cag =", nmi_s2cag)
    assert 0.0 <= nmi <= 1.0
    assert abs(nmi - nmi_s2cag) < 1e-3

def test_dese_acc():
    n = 30
    k = 4
    nodes_per_cluster = [15, 4, 5, 6]

    true_labels = []
    for c, size in enumerate(nodes_per_cluster):
        true_labels.extend([c] * size)
    true_labels = torch.tensor(true_labels)

    feature = torch.zeros(n, 5)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        feature[start:end, c*10:(c+1)*10] = 1.0
        start = end
    feature = feature + torch.randn_like(feature) * 0.15
    feature = feature / feature.norm(dim=1, keepdim=True)

    adj = torch.zeros(n, n)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        idx = torch.arange(start, end)
        i, j = torch.meshgrid(idx, idx, indexing='ij')
        mask = torch.rand(size, size) < 0.4
        mask = torch.triu(mask, 1)
        adj[i[mask], j[mask]] = 1
        adj[j[mask], i[mask]] = 1
        start = end

    inter = torch.rand(n, n) < 0.003
    inter = inter & (torch.triu(torch.ones(n,n), 1) > 0)
    adj[inter] = 1
    adj.T[inter] = 1
    adj.fill_diagonal_(0)

    adj = adj.to_sparse_coo()
    # print(adj, feature)

    pred_labels, metrics = dese_metrics(adj, feature, true_labels, n_epochs=40)
    # print(metrics['acc'])
    acc_dese = metrics['acc']
    acc = Metrics.accuracy(true_labels, pred_labels)
    print("acc =", acc, "acc_dese =", acc_dese)
    assert 0.0 <= acc <= 1.0
    assert abs(acc - acc_dese) < 1e-3

def test_dese_nmi():
    n = 30
    k = 4
    nodes_per_cluster = [15, 4, 5, 6]

    true_labels = []
    for c, size in enumerate(nodes_per_cluster):
        true_labels.extend([c] * size)
    true_labels = torch.tensor(true_labels)

    feature = torch.zeros(n, 5)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        feature[start:end, c*10:(c+1)*10] = 1.0
        start = end
    feature = feature + torch.randn_like(feature) * 0.15
    feature = feature / feature.norm(dim=1, keepdim=True)

    adj = torch.zeros(n, n)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        idx = torch.arange(start, end)
        i, j = torch.meshgrid(idx, idx, indexing='ij')
        mask = torch.rand(size, size) < 0.4
        mask = torch.triu(mask, 1)
        adj[i[mask], j[mask]] = 1
        adj[j[mask], i[mask]] = 1
        start = end

    inter = torch.rand(n, n) < 0.003
    inter = inter & (torch.triu(torch.ones(n,n), 1) > 0)
    adj[inter] = 1
    adj.T[inter] = 1
    adj.fill_diagonal_(0)

    adj = adj.to_sparse_coo()
    # print(adj, feature)

    pred_labels, metrics = dese_metrics(adj, feature, true_labels, n_epochs=40)
    # print(metrics['acc'])
    nmi_dese = metrics['nmi']
    nmi = Metrics.nmi(true_labels, pred_labels)
    print("nmi =", nmi, "nmi_dese =", nmi_dese)
    assert 0.0 <= nmi <= 1.0
    assert abs(nmi - nmi_dese) < 1e-3

def test_dese_bacc():
    n = 30
    k = 4
    nodes_per_cluster = [15, 4, 5, 6]

    true_labels = []
    for c, size in enumerate(nodes_per_cluster):
        true_labels.extend([c] * size)
    true_labels = torch.tensor(true_labels)

    feature = torch.zeros(n, 5)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        feature[start:end, c*10:(c+1)*10] = 1.0
        start = end
    feature = feature + torch.randn_like(feature) * 0.15
    feature = feature / feature.norm(dim=1, keepdim=True)

    adj = torch.zeros(n, n)
    start = 0
    for c, size in enumerate(nodes_per_cluster):
        end = start + size
        idx = torch.arange(start, end)
        i, j = torch.meshgrid(idx, idx, indexing='ij')
        mask = torch.rand(size, size) < 0.4
        mask = torch.triu(mask, 1)
        adj[i[mask], j[mask]] = 1
        adj[j[mask], i[mask]] = 1
        start = end

    inter = torch.rand(n, n) < 0.003
    inter = inter & (torch.triu(torch.ones(n,n), 1) > 0)
    adj[inter] = 1
    adj.T[inter] = 1
    adj.fill_diagonal_(0)

    adj = adj.to_sparse_coo()
    # print(adj, feature)

    pred_labels, metrics = dese_metrics(adj, feature, true_labels, n_epochs=40)

    bacc = Metrics.balanced_acc(true_labels, pred_labels)
    print("bacc =", bacc)
    assert 0.0 <= bacc <= 1.0
    # assert abs(nmi - nmi_dese) < 1e-3