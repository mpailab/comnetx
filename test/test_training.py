import sys
import os
import torch
import shutil
import tempfile
import pytest
import subprocess
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from datasets import Dataset


@pytest.mark.long
def test_train_custom_eat_runs_successfully():
    script_path = Path(__file__).parent.parent / "baselines" / "MAGI" / "train_custom.py"

    dataset_path = Path(__file__).parent / "graphs" / "small" / "eat"

    proc = subprocess.Popen(
        [
            "python", str(script_path),
            "--runs", "1",
            "--dataset_name", "Eat",
            "--dataset_path", str(dataset_path),
            "--batchsize", "2048",
            "--max_duration", "1",
            "--kmeans_device", "cpu",
            "--kmeans_batch", "-1",
            "--hidden", "1024,256",
            "--size", "10,10",
            "--wt", "20",
            "--wl", "5",
            "--tau", "0.5",
            "--ns", "0.5",
            "--lr", "0.01",
            "--epochs", "100",
            "--wd", "0",
            "--dropout", "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    logs = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        logs.append(line)

    proc.wait(timeout=120)
    log_text = "".join(logs)

    assert proc.returncode == 0, f"Script failed. Logs:\n{log_text}"


@pytest.mark.long
def test_train_custom_cora_runs_successfully():
    script_path = Path(__file__).parent.parent / "baselines" / "MAGI" / "train_custom.py"

    dataset_path = Path(__file__).parent / "graphs" / "small" / "cora"

    result = subprocess.run(
        [
            "python", str(script_path),
            "--runs", "1",
            "--dataset_name", "Cora",
            "--dataset_path", str(dataset_path),
            "--batchsize", "2048",
            "--max_duration", "1",
            "--kmeans_device", "cpu",
            "--kmeans_batch", "-1",
            "--hidden", "1024,256",
            "--size", "10,10",
            "--wt", "20",
            "--wl", "5",
            "--tau", "0.5",
            "--ns", "0.5",
            "--lr", "0.01",
            "--epochs", "100",
            "--wd", "0",
            "--dropout", "0"
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Script failed: {result.stderr}"