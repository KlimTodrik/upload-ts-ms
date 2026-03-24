PYTHON ?= python3
VENV ?= .venv
ACTIVATE = source $(VENV)/bin/activate

.PHONY: venv install up down benchmark-precomputed benchmark-auto

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(ACTIVATE) && pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

benchmark-precomputed:
	$(ACTIVATE) && python scripts/benchmark.py --mode precomputed --engines both --limit 1000

benchmark-auto:
	$(ACTIVATE) && python scripts/benchmark.py --mode auto --engines both --limit 1000
