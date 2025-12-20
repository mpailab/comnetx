import sys
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