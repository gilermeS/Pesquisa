# Makefile for Cosmology Deep Learning Project

.PHONY: help install install-dev lint format typecheck test test-cov clean data

help:
	@echo "Available commands:"
	@echo "  install      - Install project dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code with black and isort"
	@echo "  typecheck    - Run type checking with mypy"
	@echo "  test         - Run tests with pytest"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  data         - Generate LCDM dataset"
	@echo "  clean        - Clean up temporary files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black isort flake8 mypy pytest pytest-cov papermill

lint:
	@echo "Running flake8..."
	flake8 . --count --show-source --statistics

format:
	@echo "Running black..."
	black .
	@echo "Running isort..."
	isort .

typecheck:
	@echo "Running mypy..."
	mypy cosmology/ utils/

test:
	@echo "Running pytest..."
	pytest tests/ -v

test-cov:
	@echo "Running pytest with coverage..."
	pytest tests/ -v --cov=cosmology --cov=utils --cov-report=term-missing

data:
	@echo "Generating LCDM dataset..."
	python Geradores/gerador_pontos.py

clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .cache/
	@echo "Clean complete!"

# Notebook execution
notebooks:
	@echo "Starting Jupyter notebook server..."
	jupyter notebook

# Generate all datasets
data-all: data
	@echo "Generating wCDM dataset..."
	python wcdm/gerador_pontos3.py
	@echo "Generating wACDM dataset..."
	python wacdm/gerador_pontos2.py
	@echo "All datasets generated!"