"""Reproduce two legacy update defects without datasets or optional backends.

Run inside the development container:
    PYTHONPATH=.:src python scripts/paper/audit_legacy_update.py

The legacy implementation is loaded verbatim from the last commit preceding
the 25 August change. Git history containing that commit is required. Neither
the checkout nor measurement files are modified. A scripted backend isolates
adapter behavior; this is not a performance or Leiden-quality experiment.
"""

import json
import subprocess
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from optimizer import Optimizer  # noqa: E402


LEGACY_COMMIT = "25ac4fc5522e4a411cf7de2dd48797a470207519"


def legacy_optimizer():
    source = subprocess.check_output(
        ["git", "show", f"{LEGACY_COMMIT}:src/optimizer.py"],
        cwd=ROOT,
        text=True,
    )
    module = types.ModuleType("legacy_optimizer")
    exec(compile(source, f"{LEGACY_COMMIT}:src/optimizer.py", "exec"), module.__dict__)
    return module.Optimizer


def scripted(base, adjacency, hierarchy, affected, merge_first=False):
    class Scripted(base):
        def __init__(self):
            super().__init__(
                adjacency.clone(),
                communities=hierarchy.clone(),
                subcoms_depth=hierarchy.size(0),
            )
            self.quotient_sizes = []

        def local_algorithm(self, adj, features, limited=False, labels=None):
            self.quotient_sizes.append(adj.size(0))
            if merge_first and len(self.quotient_sizes) == 1:
                return torch.zeros(adj.size(0), dtype=torch.long)
            return torch.arange(adj.size(0), dtype=torch.long)

    opt = Scripted()
    before = opt.coms.clone()
    scopes = torch.stack([
        torch.isin(row, torch.unique(row[affected])) for row in before
    ])
    opt.run(affected)
    collisions = [
        torch.unique(row[scope])[
            torch.isin(torch.unique(row[scope]), torch.unique(row[~scope]))
        ].tolist()
        for row, scope in zip(opt.coms, scopes)
    ]
    nested = all(
        torch.unique(coarse[fine == label]).numel() == 1
        for fine, coarse in zip(opt.coms[:-1], opt.coms[1:])
        for label in torch.unique(fine)
    )
    return {
        "before": before.tolist(),
        "scopes": [torch.where(scope)[0].tolist() for scope in scopes],
        "quotient_sizes": opt.quotient_sizes,
        "after": opt.coms.tolist(),
        "shared_inside_outside_labels": collisions,
        "nested": nested,
    }


def main():
    legacy = legacy_optimizer()
    # An update with endpoints 1 and 4, radius zero, is a supported input.
    # Vertices 2 and 3 form an untouched community bearing numeric label 1.
    adjacency = torch.zeros((6, 6))
    for a, b in [(0, 1), (2, 3), (4, 5), (1, 4)]:
        adjacency[a, b] = adjacency[b, a] = 1
    adjacency = adjacency.to_sparse_coo().coalesce()
    initial = torch.tensor([[0, 0, 1, 1, 2, 2]])
    affected = torch.tensor([False, True, False, False, True, False])
    namespace = {
        "backend": "identity partition on every call",
        "legacy": scripted(legacy, adjacency, initial, affected),
        "current": scripted(Optimizer, adjacency, initial, affected),
    }
    assert namespace["legacy"]["after"] == [[0, 1, 1, 1, 4, 2]]
    assert namespace["legacy"]["shared_inside_outside_labels"] == [[1]]
    assert namespace["current"]["after"] == [[0, 1, 2, 2, 4, 5]]
    assert namespace["current"]["shared_inside_outside_labels"] == [[]]

    adjacency = torch.tensor([
        [0., 1., 0., 0.], [1., 0., 1., 0.],
        [0., 1., 0., 1.], [0., 0., 1., 0.],
    ]).to_sparse_coo().coalesce()
    initial = torch.tensor([[0, 0, 2, 2], [0, 0, 0, 0]])
    affected = torch.tensor([False, True, False, False])
    nesting = {
        "backend": "merge all input atoms on first call; identity on second",
        "legacy": scripted(legacy, adjacency, initial, affected, merge_first=True),
        "current": scripted(Optimizer, adjacency, initial, affected, merge_first=True),
    }
    assert nesting["legacy"]["after"] == [[0, 0, 2, 2], [0, 1, 0, 0]]
    assert nesting["legacy"]["nested"] is False
    assert nesting["current"]["nested"] is True
    print(json.dumps({
        "legacy_commit": LEGACY_COMMIT,
        "namespace_collision": namespace,
        "independent_level_nesting": nesting,
        "interpretation": (
            "The first case is a label collision with a valid identity backend. "
            "The second refutes unconditional nesting for arbitrary partitions; "
            "it does not prove failure of every Leiden run or superiority of "
            "the parent-quotient policy."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
