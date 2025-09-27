import pytest
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "src"))

import sparse


def coo(i, j, v, size):
    idx = torch.tensor([i, j], dtype=torch.long)
    vals = torch.tensor(v, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, size=size).coalesce()


@pytest.mark.unit
def test_reset_matrix_filters_nodes():
    # Graph: 0-1 edge, 1-2 edge, 2-3 edge
    A = coo([0, 1, 2], [1, 2, 3], [1, 1, 1], (4, 4))
    allowed = torch.tensor([0, 1])
    cut = sparse.reset_matrix(A, allowed)
    indices = cut.indices()
    # Only keep edges fully within allowed set -> only (0,1)
    assert indices.size(1) == 1
    assert torch.equal(indices[:, 0], torch.tensor([0, 1]))


@pytest.mark.unit
def test_reset_slice_rows():
    # 3x3 with 0->1, 1->2, 2->0
    A = coo([0, 1, 2], [1, 2, 0], [1, 1, 1], (3, 3))
    # Keep rows [0,1)
    sliced = sparse.slice(A, slice(0, 2), dim=0)
    # Only edges with source in {0,1}
    src = set(sliced.indices()[0].tolist())
    assert src.issubset({0, 1})
    # And mapping compacts dim 0 to size 2
    assert sliced.size()[0] == 2


@pytest.mark.unit
def test_cat_and_equal():
    A = coo([0], [0], [1], (1, 1))
    B = coo([0], [0], [2], (1, 1))
    C = sparse.cat(A, B, dim=0)
    assert C.size() == torch.Size([2, 1])
    # Equal detects exact structural/value equality
    assert sparse.equal(A, A)
    assert not sparse.equal(A, B)

