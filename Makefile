.PHONY: setup lint test unit cpu verify

setup:
	bash .devcontainer/post-create.sh

lint:
	python -m compileall src test scripts

test:
	PYTHONPATH=.:src:test pytest -q test/test_devcontainer_smoke.py

unit:
	PYTHONPATH=.:src:test pytest -q test/unit

cpu:
	PYTHONPATH=.:src:test pytest -q test/test_networkit.py test/test_leidenalg.py -m "not long"
	PYTHONPATH=.:src:test pytest -q test/test_optimizer.py -m short -k "not test_run_prgpt and not test_run_leidenalg"

verify: lint test unit cpu
