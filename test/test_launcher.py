import sys
import os
import pytest
import torch

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from launcher import dynamic_launch

@pytest.mark.short
def test_dynamic_locale():
    dynamic_launch("wiki_talk_ht", 10, "prgpt:locale")

@pytest.mark.short  
def test_dynamic_infomap():
    dynamic_launch("wiki_talk_ht", 10, "prgpt:infomap")

@pytest.mark.short  
def test_dynamic_leidenalg():
    dynamic_launch("wiki_talk_ht", 10, "leidenalg")

@pytest.mark.short  
def test_dynamic_networkit():
    dynamic_launch("wiki_talk_ht", 1, "networkit")

@pytest.mark.short  
def test_dynamic_magi():
    dynamic_launch("wiki_talk_ht", 10, "magi")

@pytest.mark.long
def test_dynamic_dmon():
    dynamic_launch("wiki_talk_ht", 1, "dmon")
