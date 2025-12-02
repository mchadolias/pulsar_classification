#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = pulsar_classification
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# ENVIRONMENT                                                                   #
#################################################################################

## Create uv virtual environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Activate with:"
	@echo ">>> source ./.venv/bin/activate"

## Install all dependencies (core + dev + training)
.PHONY: requirements
requirements:
	uv sync --extra all

## Install core-only dependencies
.PHONY: install-core
install-core:
	uv sync

## Install development dependencies only
.PHONY: install-dev
install-dev:
	uv sync --extra dev

## Install training dependencies only
.PHONY: install-training
install-training:
	uv sync --extra training

#################################################################################
# DATA PIPELINE                                                                 #
#################################################################################

## Download raw dataset + prepare directory structure
.PHONY: data
data:
	uv run python scripts/setup_directories.py
	uv run python scripts/run_training.py --dry-run

#################################################################################
# TRAINING PIPELINE                                                             #
#################################################################################

## Train all models end-to-end using the main pipeline
.PHONY: train
train:
	uv run python scripts/run_training.py --config configs/model.toml

## Train all models end-to-end using the main pipeline
.PHONY: train-test
train:
	uv run python scripts/run_training.py --config configs/test_model.toml

#################################################################################
# PLOTTING & ANALYSIS                                                           #
#################################################################################

## Generate all comparison plots from an existing poller.json
.PHONY: plot
plot:
	uv run python scripts/run_plotting.py --formats png pdf

## Generate PDF-only comparison report
.PHONY: plot-pdf
plot-pdf:
	uv run python scripts/run_plotting.py --formats pdf

## Generate PNG-only plots
.PHONY: plot-png
plot-png:
	uv run python scripts/run_plotting.py --formats png

#################################################################################
# CLEANUP                                                                       #
#################################################################################

## Delete Python cache files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Remove all outputs (models, metrics, plots)
.PHONY: clean-outputs
clean-outputs:
	rm -rf outputs/models/*
	rm -rf outputs/metrics/*
	rm -rf outputs/predictions/*
	rm -rf outputs/plots/*

#################################################################################
# LINTING & FORMATTING                                                          #
#################################################################################

## Run code style checks
.PHONY: lint
lint:
	flake8 src
	isort --check --diff src
	black --check src

## Automatically format code
.PHONY: format
format:
	isort src
	black src

#################################################################################
# HELP                                                                           #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
lines = '\n'.join([line for line in sys.stdin])
matches = re.findall(r'\n## (.*)\n([a-zA-Z0-9_-]+):', lines)
print('Available rules:\n')
for desc, name in matches:
    print(f"{name:20} {desc}")
endef
export PRINT_HELP_PYSCRIPT

## Show this help message
help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
