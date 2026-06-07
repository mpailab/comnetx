import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


class _NoTimingLDLeiden:
    def __init__(self, *args, **kwargs):
        pass

    def apply(self):
        return 12.0

    def partition(self):
        return torch.tensor([0, 0, 1], dtype=torch.long)


def _load_dgc_with_stub(monkeypatch):
    stub = types.ModuleType("dynamic_graphs_communities")
    stub.LDLeiden = _NoTimingLDLeiden
    stub.DFLeiden = _NoTimingLDLeiden
    stub.Leidenalg = _NoTimingLDLeiden
    stub.Networkit = _NoTimingLDLeiden
    stub.AlgorithmOptions = type("AlgorithmOptions", (), {})
    monkeypatch.setitem(sys.modules, "dynamic_graphs_communities", stub)

    spec = importlib.util.spec_from_file_location(
        "dgc_test_module",
        SRC / "baselines" / "dgc.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.short
def test_ldleiden_apply_falls_back_when_update_timing_keyword_is_missing(monkeypatch):
    dgc = _load_dgc_with_stub(monkeypatch)
    timing_info = {}
    adj = torch.tensor(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=torch.float32,
    ).to_sparse_coo()

    labels = dgc._run_leiden("ldleiden", adj, timing_info=timing_info)

    assert torch.equal(labels, torch.tensor([0, 0, 1]))
    assert timing_info["algorithm_time"] == pytest.approx(0.012)
    assert "conversion_time" in timing_info
