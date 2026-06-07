.PHONY: setup lint test unit figures configs verify

PYTHON ?= python
PYTEST ?= pytest
PROJECT_PYTHONPATH := .:src:test

setup:
	bash .devcontainer/post-create.sh

lint:
	$(PYTHON) -m compileall src scripts/launch.py scripts/paper test/test_devcontainer_smoke.py test/unit

test:
	PYTHONPATH=$(PROJECT_PYTHONPATH) $(PYTEST) -q test/test_devcontainer_smoke.py

unit:
	PYTHONPATH=$(PROJECT_PYTHONPATH) $(PYTEST) -q test/unit

figures:
	$(PYTHON) scripts/paper/plot_workload_speedup.py
	$(PYTHON) scripts/paper/plot_topology_ablation_pareto.py

configs:
	$(PYTHON) scripts/paper/generate_icdm_configs.py

verify: lint test unit figures
