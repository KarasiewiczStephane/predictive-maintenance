.PHONY: install test test-cov lint clean run train serve docker

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.main

train:
	python -m src.main --train sample/equipment_sensors.csv

serve:
	python -m src.main --serve

docker:
	docker build -t $(shell basename $(CURDIR)) .
	docker run -p 8000:8000 $(shell basename $(CURDIR))
