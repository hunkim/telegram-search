.PHONY: setup clean run test lint load-env

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
STREAMLIT := $(VENV)/bin/streamlit

include .env
export

# Need to use python 3.9 for aws lambda
$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt


clean: 
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

telegram: $(VENV)/bin/activate
	$(PYTHON) telegram_bot.py

solar: $(VENV)/bin/activate
	$(PYTHON) solar.py

lint: $(VENV)/bin/activate
	$(PYTHON) -m flake8
	$(PYTHON) -m black .
