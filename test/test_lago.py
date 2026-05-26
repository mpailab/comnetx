import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from baselines.lago import lago_partition
from datasets import Dataset

PROJECT_PATH = Path(__file__).resolve().parent.parent
TEST_ROOT = PROJECT_PATH / "test"


def _assert_partition(labels, num_nodes):
    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (num_nodes,)
    assert labels.dtype in (torch.int64, torch.long)
    assert labels.min() >= 0


def _sample_dataset_snapshot(adj, max_edges=120):
    snapshot = adj[0] if adj.ndim == 3 else adj
    snapshot = snapshot.coalesce()
    indices = snapshot.indices()
    values = snapshot.values()
    if indices.size(1) == 0:
        pytest.skip("Dataset snapshot has no edges to sample")

    indices = indices[:, :max_edges]
    values = values[:max_edges]
    nodes = torch.unique(indices)
    remap = {int(node): pos for pos, node in enumerate(nodes.tolist())}
    remapped = torch.tensor(
        [[remap[int(u)], remap[int(v)]] for u, v in indices.t().tolist()],
        dtype=torch.long,
    ).t()
    return torch.sparse_coo_tensor(
        remapped,
        values.float(),
        size=(len(nodes), len(nodes)),
    ).coalesce()


@pytest.mark.short
def test_lago_subprocess_roundtrip(tmp_path):
    adj = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ).to_sparse_coo().coalesce()

    adj_path = tmp_path / "adj.pt"
    out_path = tmp_path / "labels.pt"
    torch.save(adj, adj_path)

    cmd = [
        sys.executable,
        str(TEST_ROOT / "run_lago_subprocess.py"),
        "--adj",
        str(adj_path),
        "--out",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"

    labels = torch.load(out_path)
    _assert_partition(labels, adj.size(0))


@pytest.mark.short
def test_lago_on_workspace_dataset_graph(tmp_path):
    dataset_root = PROJECT_PATH / "datasets"
    if not (dataset_root / "dyn_wiki").exists():
        pytest.skip("Workspace dyn_wiki dataset is not available")

    paths_config = tmp_path / "paths.json"
    paths_config.write_text(
        json.dumps({"dynamic": {"dyn_attr": str(dataset_root)}, "static": {}}),
        encoding="utf-8",
    )

    ds = Dataset("dyn_wiki", paths_config=str(paths_config))
    adj, _, _ = ds.load(batches_strategy="1")
    sampled_adj = _sample_dataset_snapshot(adj)

    labels = lago_partition(
        sampled_adj,
        directed=ds.is_directed,
        nb_iter=1,
        refinement=None,
    )

    _assert_partition(labels, sampled_adj.size(0))
