import pytest
import gc
import torch
import importlib
import subprocess
import sys
import os, json
os.environ["OGB_FORCE_DOWNLOAD"] = "1"
from pathlib import Path

_DATASETS_JSON = Path(__file__).parent / "dataset_paths.json"

def collect_datasets():
    """
    Читает JSON-словарь {dataset_name: dataset_path} и возвращает dict.
    Исключения дадут явный фэйл ещё на этапе коллекции тестов.
    """
    with _DATASETS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"dataset_paths.json must be an object mapping names to paths, got {type(data)}")
    return data

ALL_METHODS = ["dmon", "magi", "prgpt"]
ALL_DATASETS = list(collect_datasets().keys())

tf_spec = importlib.util.find_spec("tensorflow")
if tf_spec is not None:
    import tensorflow as tf

@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """
    Pytest fixture to clear GPU and CPU memory before and after each test.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if 'tf' in globals():
        tf.keras.backend.clear_session()

    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if 'tf' in globals():
        tf.keras.backend.clear_session()

def pytest_collection_modifyitems(config, items):
    skip_marker = pytest.mark.skip(reason="Dataset too large for CI")
    for item in items:
        if "ogbn-papers100m" in item.name.lower():
            item.add_marker(skip_marker)

@pytest.fixture(autouse=True)
def auto_confirm_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")

def pytest_addoption(parser):
    parser.addoption(
        "--method",
        action="append",
        default=[],
        help="Method(s) to run (repeatable or comma-separated), e.g. --method dmon --method magi or --method dmon,magi",
    )
    parser.addoption(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name(s) (repeatable or comma-separated)",
    )

def _split_opts(values):
    out = []
    for v in values or []:
        out.extend([x.strip() for x in v.split(",") if x.strip()])
    return out

def pytest_generate_tests(metafunc):
    cfg = metafunc.config

    all_methods = list(ALL_METHODS)
    all_datasets = list(collect_datasets().keys())

    cli_methods = _split_opts(cfg.getoption("--method", default=[]))
    cli_datasets = _split_opts(cfg.getoption("--dataset", default=[]))

    methods = cli_methods or all_methods
    datasets = cli_datasets or all_datasets

    unknown_m = sorted(set(methods) - set(all_methods))
    unknown_d = sorted(set(datasets) - set(all_datasets))
    if unknown_m:
        raise pytest.UsageError(f"Unknown --method: {unknown_m}. Allowed: {all_methods}")
    if unknown_d:
        raise pytest.UsageError(f"Unknown --dataset: {unknown_d}. Allowed: {all_datasets}")

    if "method" in metafunc.fixturenames:
        metafunc.parametrize("method", methods, ids=lambda m: f"method={m}")
    if "dataset" in metafunc.fixturenames:
        metafunc.parametrize("dataset", datasets, ids=lambda d: f"dataset={d}")

@pytest.fixture
def runner_dmon():
    def run(ds):
        cmd = [
            sys.executable,
            "run_dmon_subprocess.py",
            "--adj", ds["adj"],
            "--features", ds["features"],
            "--epochs", "10",
            "--out", ds["out"],
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return run

@pytest.fixture
def runner_magi():
    def run(ds):
        cmd = [
            sys.executable,
            "run_magi_subprocess.py",
            "--adj", ds["adj"],
            "--features", ds["features"],
            "--epochs", "1",
            "--batchsize", "1024",
            "--out", ds["out"],
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return run

@pytest.fixture
def runner_prgpt():
    def run(ds):
        cmd = [
            sys.executable,
            "run_prgpt_subprocess.py",
            "--adj", ds["adj"],
            "--out", ds["out"],
        ]
        if "features" in ds and ds["features"]:
            cmd += ["--features", ds["features"]]
        if "refine" in ds and ds["refine"]:
            cmd += ["--refine", ds["refine"]]

        return subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return run
