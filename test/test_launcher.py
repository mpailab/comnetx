import sys
import os
import pytest
from pathlib import Path

TEST_PATH = Path(__file__).resolve().parent
PROJECT_PATH = TEST_PATH.parent

for p in (PROJECT_PATH, PROJECT_PATH / "src", TEST_PATH):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from launcher import dynamic_launch
from testutils import with_timeout, TimeoutException

@pytest.mark.short
def test_dynamic_prgpt():
    for method in ["prgpt:locale", "prgpt:infomap"]:
        for mode in ["smart", "naive", "raw"]:
            dynamic_launch("wiki_talk_ht", 10, method, mode = mode)

@pytest.mark.short  
def test_dynamic_leidenalg_networkit():
    for method in ["leidenalg", "networkit"]:
        for mode in ["smart", "naive", "raw"]:
            dynamic_launch("wiki_talk_ht", 10, method, mode = mode)

@pytest.mark.short  
def test_dynamic_magi():
    for mode in ["smart", "naive", "raw"]:
        dynamic_launch("wiki_talk_ht", 1, "magi", mode = mode)

@pytest.mark.long
def test_dynamic_dmon():
    for mode in ["smart", "naive", "raw"]:
        dynamic_launch("wiki_talk_ht", 1, "dmon", mode = mode)

"""
@pytest.mark.debug
def test_dynamic_networkit_bad_datasets():
    # Работает нестабильно, периодически 'зависает'
    for dataset in ["sociopatterns-hypertext", "radoslaw_email"]:
        dynamic_launch(dataset, 100, "networkit", mode = "smart", verbose = 2)
"""

@pytest.mark.debug
def test_dynamic_networkit_bad_datasets(resource_monitor):
    for dataset in ["sociopatterns-hypertext", "radoslaw_email"]:
        print(f"\n\nProcessing dataset: {dataset}")
        try:
            @with_timeout(300)
            def run():
                dynamic_launch(dataset, 100, "networkit", mode="smart", verbose=2)
            run()
        except TimeoutException as e:
            print(f"\n TIMEOUT: {e}")
            continue