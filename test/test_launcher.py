import sys
import os
import pytest
import torch

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from launcher import launch_dynamic_scenario

@pytest.mark.debug
def test_dynamic_locale():
    launch_dynamic_scenario("wiki_talk_ht", 10, "prgpt:locale")

@pytest.mark.short  
def test_dynamic_infomap():
    launch_dynamic_scenario("wiki_talk_ht", 10, "prgpt:infomap")

@pytest.mark.short  
def test_dynamic_leidenalg():
    launch_dynamic_scenario("wiki_talk_ht", 10, "leidenalg")

@pytest.mark.debug  
def test_dynamic_networkit():
    launch_dynamic_scenario("wiki_talk_ht", 1, "networkit")

@pytest.mark.debug  
def test_dynamic_dmon():
    launch_dynamic_scenario("wiki_talk_ht", 1, "dmon")

@pytest.mark.debug  
def test_dynamic_magi():
    launch_dynamic_scenario("wiki_talk_ht", 10, "magi")
