# ===== ML-Lab — Makefile =====
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
TURBOK := tools/run_turbok_suite.sh

.PHONY: setup run-iris run-imbalance run-forest run-importance run-grid docs-serve docs-build docs-deploy run-turbok run-turbok-oils

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

# ---- Turbo-K test suite ----------------------------------------------------
# Esegue l'intera batteria di test e salva i log in reports/turbo_k/<timestamp>/
# Richiede: tools/run_turbok_suite.sh
run-turbok:
	@test -x $(TURBOK) || chmod +x $(TURBOK)
	@echo ">> Avvio Turbo-K test suite (log in reports/turbo_k/<timestamp>/)…"
	@$(TURBOK)

run-turbok-oils:
	@echo ">> Confronto oli da presets su UNIFORME e CIDR…"
	$(PIP) install -q pyyaml toml || true
	@echo "\n## [UNIFORME, K=12]"
	$(PY) scripts/turbo_k_eval.py --presets presets/oils.yaml --source uniform --mode msb --K 12 --N 200000
	@echo "\n## [CIDR mix, K=12]"
	$(PY) scripts/turbo_k_eval.py --presets presets/oils.yaml --source cidr --cidr 10.0.0.0/8 --cidr 192.168.0.0/16 --mode msb --K 12 --N 300000

docs-serve:
	$(PIP) install mkdocs mkdocs-ivory
	mkdocs serve

docs-build:
	$(PIP) install mkdocs mkdocs-ivory
	mkdocs build --strict

docs-deploy:
	$(PIP) install mkdocs mkdocs-ivory
	mkdocs gh-deploy --force
