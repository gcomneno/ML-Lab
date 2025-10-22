# ===== ML-Lab — Makefile =====
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: setup run-iris run-imbalance run-forest run-importance run-grid docs-serve docs-build docs-deploy

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	@echo "OK. Attiva con: source $(VENV)/bin/activate"

run-iris:
	$(PY) scripts/iris.py --tune --print-cheatsheet

run-imbalance:
	$(PY) scripts/imbalance.py --auto-threshold --metric f1 --print-cheatsheet

run-forest:
	$(PY) scripts/forest_vs_logit.py --rf-calibrate isotonic --auto-threshold --print-cheatsheet

run-importance:
	$(PY) scripts/importance_demo.py --print-cheatsheet

run-grid:
	$(PY) scripts/gridsearch_mixed.py --model rf --auto-threshold --thr-mode cost --cost-fn 10 --print-cheatsheet

docs-serve:
	$(PIP) install mkdocs mkdocs-material
	mkdocs serve

docs-build:
	$(PIP) install mkdocs mkdocs-material
	mkdocs build --strict

docs-deploy:
	$(PIP) install mkdocs mkdocs-material
	mkdocs gh-deploy --force
