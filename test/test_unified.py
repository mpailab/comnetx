import os, sys, gc, tempfile, subprocess, json
import itertools
import pytest, torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from testutils import KONECT_INFO, KONECT_PATH, get_all_datasets
from conftest import collect_datasets
from datasets import Dataset


def _resolve_dataset_path(dataset_name: str) -> str:
    ds_map = collect_datasets()
    if dataset_name not in ds_map:
        raise RuntimeError(f"Unknown dataset '{dataset_name}'. Available: {list(ds_map.keys())}")
    return ds_map[dataset_name]

def _resolve_runner(request: pytest.FixtureRequest, method_name: str):
    key = method_name.strip().lower()
    fixture_name = f"runner_{key}"
    try:
        return request.getfixturevalue(fixture_name)
    except Exception as e:
        available = [name for name in dir(sys.modules[__name__]) if name.startswith("runner_")]
        raise RuntimeError(
            f"Unknown method '{method_name}'. Expected a runner fixture named '{fixture_name}'. "
            f"Available runners: {available}"
        ) from e

@pytest.mark.long
def test_run_method_on_dataset(method, dataset, tmp_path, request):
    dataset_root = _resolve_dataset_path(dataset)
    assert os.path.exists(os.path.join(dataset_root, dataset)), (
        f"Dataset path does not exist: {os.path.join(dataset_root, dataset)}"
    )

    name = dataset

    ds = Dataset(name, path=dataset_root)
    adj, features, labels = ds.load()

    if hasattr(adj, "is_sparse") and adj.is_sparse:
        adj = adj.coalesce()

    num_nodes = adj.size(0)
    if features is None:
        features = torch.randn(num_nodes, 128, dtype=torch.float32)

    adj_path = tmp_path / f"adj_{name}.pt"
    feat_path = tmp_path / f"features_{name}.pt"
    torch.save(adj, adj_path)
    torch.save(features, feat_path)
    assert adj_path.exists() and feat_path.exists(), "Input .pt files were not created"

    out_dir = tmp_path / f"magi__{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "labels.pt"
    out_file.touch()

    ds = {
        "adj": str(adj_path),
        "features": str(feat_path),
        "out": str(out_file),
    }

    runner = _resolve_runner(request, method)
    proc = runner(ds)
    assert proc is not None and hasattr(proc, "returncode"), "Runner must return CompletedProcess"
    assert proc.returncode == 0, (
        f"Failed (code {proc.returncode}). stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    )

    assert out_file.exists(), f"Output file not found: {out_file}"
    new_labels = torch.load(out_file)
    assert isinstance(new_labels, torch.Tensor) and new_labels.numel() > 0
