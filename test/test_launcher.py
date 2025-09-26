import sys
import os
import pytest
import torch

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_PATH, "src"))

from launcher import launch_dynamic_scenario

@pytest.mark.debug
def test_dynamic():
    launch_dynamic_scenario("convote", 1, "prgpt:infomap")
