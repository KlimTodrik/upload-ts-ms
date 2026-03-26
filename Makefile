PYTHON ?= python3
VENV ?= .venv
ACTIVATE = source $(VENV)/bin/activate

.PHONY: venv install up down benchmark

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(ACTIVATE) && pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

benchmark:
	$(ACTIVATE) && python scripts/benchmark.py --mode auto --engines both --limit 1000
