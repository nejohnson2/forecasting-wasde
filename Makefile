PYTHON = PYTHONPATH=. .venv/bin/python

.PHONY: all dev fetch process eda features sarima xgboost evaluate visualize clean setup

# Full pipeline: all 3 grains, full history
all: fetch process eda features sarima xgboost evaluate visualize

# Dev mode: wheat only, last 20 years
dev: fetch-dev process-dev eda features-dev sarima-dev xgboost-dev evaluate visualize

# --- Setup ---
setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# --- Data Acquisition ---
fetch:
	$(PYTHON) scripts/01_fetch_data.py --commodities wheat corn rice

fetch-dev:
	$(PYTHON) scripts/01_fetch_data.py --commodities wheat

# --- Processing ---
process:
	$(PYTHON) scripts/02_process_data.py --all

process-dev:
	$(PYTHON) scripts/02_process_data.py --commodities wheat --start-year 2005

# --- Analysis (future phases) ---
eda:
	$(PYTHON) scripts/03_eda_analysis.py

features:
	$(PYTHON) scripts/04_engineer_features.py --all

features-dev:
	$(PYTHON) scripts/04_engineer_features.py --commodities wheat

# --- Modeling (future phases) ---
sarima:
	$(PYTHON) scripts/05_train_sarima.py --all

sarima-dev:
	$(PYTHON) scripts/05_train_sarima.py --commodities wheat --max-years 20

xgboost:
	$(PYTHON) scripts/06_train_xgboost.py --all

xgboost-dev:
	$(PYTHON) scripts/06_train_xgboost.py --commodities wheat --max-years 20

# --- Evaluation & Visualization (future phases) ---
evaluate:
	$(PYTHON) scripts/07_evaluate.py

visualize:
	$(PYTHON) scripts/08_visualize.py

# --- Cleanup ---
clean:
	rm -rf data/processed/* data/features/* results/models/* results/figures/*
