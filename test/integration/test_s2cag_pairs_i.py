"""Opt-in regression for a one-atom quotient in the real S²CAG solver path."""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/paper/run_s2cag_pairs.py"
pytestmark = pytest.mark.skipif(os.environ.get("COMNETX_TEST_S2CAG") != "1",
                                reason="opt-in real S2CAG numerical smoke")


def test_two_vertex_quotient_has_one_block_on_the_coarser_pass(monkeypatch):
    import torch
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from optimizer import Optimizer
    from baselines import s2cag as module

    spec = importlib.util.spec_from_file_location("s2cag_test_runner", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original = module.s2cag
    calls = []
    def trace(*args, **kwargs):
        data = {}
        result = original(*args, trace_info=data, **kwargs)
        runner.validate_solver_trace(data, 5)
        calls.append(data)
        return result
    monkeypatch.setattr(module, "s2cag", trace)
    adj = torch.tensor([[0., 1.], [1., 0.]]).to_sparse()
    features = torch.tensor([[1., 2., 3.], [2., 1., 3.]])
    opt = Optimizer(adj, features=features, subcoms_depth=2, method="s2cag")
    endpoints = runner.active_nodes_mask(adj, 2, torch.device("cpu"))
    opt.run(opt.neighborhood(adj, endpoints, step=1))
    assert opt.coms.tolist() == [[0, 0], [0, 0]]
    assert [c["input_vertices"] for c in calls] == [2, 1]
    assert [c["solver_calls_completed"] for c in calls] == [5, 0]
    assert [c["embedding_width"] for c in calls] == [1, 0]
    assert calls[1]["return_policy"] == "single_vertex"
    assert calls[1]["returned_solver_call"] is None
    assert calls[1]["solver_call_seconds"] == []
    cancelled = torch.sparse_coo_tensor([[0, 0], [1, 1]], [1., -1.], (2, 2))
    assert not runner.active_nodes_mask(cancelled, 2, torch.device("cpu")).any()
    with pytest.raises(ValueError, match="at least two feature columns"):
        original(adj, features[:, :1])
