import pytest
import gc
import torch
import importlib
import subprocess
import sys
import os
os.environ["OGB_FORCE_DOWNLOAD"] = "1"

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
