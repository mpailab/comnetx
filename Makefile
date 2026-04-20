.PHONY: setup lint test unit verify

setup:
	bash .devcontainer/post-create.sh

lint:
	python -m compileall src test scripts

test:
	PYTHONPATH=.:src:test pytest -q test/test_devcontainer_smoke.py

unit:
	PYTHONPATH=.:src:test pytest -q test/unit

verify: lint test unit
