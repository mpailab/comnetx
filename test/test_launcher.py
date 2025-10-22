import sys
import os
import pytest

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from launcher import dynamic_launch

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

@pytest.mark.debug
def test_dynamic_networkit_bad_datasets():
    # Работает нестабильно, периодически 'зависает'
    for dataset in ["sociopatterns-hypertext", "radoslaw_email"]:
        dynamic_launch(dataset, 100, "networkit", mode = "smart", verbose = 2)

@pytest.mark.debug
def test_dynamic_leidenalg_bad_datasets():
    # 'Killed'
    for dataset in ['wiki_talk_ca', 'wiki_talk_el', 'slashdot-threads', 'wiki_talk_sk', 'wiki_talk_bn', 'wiki_talk_lv', 'wiki_talk_eu', 'facebook-wosn-links']:
        print(dataset)
        dynamic_launch(dataset, 1, "leidenalg", mode = "raw", verbose = 2)

