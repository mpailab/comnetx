import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

for path in (PROJECT_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def test_datasets_module_imports() -> None:
    from datasets import Dataset, paths_config_auto_detect

    assert Dataset is not None
    assert callable(paths_config_auto_detect)
