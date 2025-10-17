import sys
import os
import pytest
import torch

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
