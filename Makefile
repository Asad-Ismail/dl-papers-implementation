.PHONY: help install test lint format clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests with coverage"
	@echo "  make lint       - Run linters (flake8, mypy)"
	@echo "  make format     - Format code with black and isort"
	@echo "  make clean      - Remove cache and build files"
	@echo "  make pre-commit - Install pre-commit hooks"

install:
	pip install -r requirements.txt

test:
	pytest --cov=. --cov-report=html --cov-report=term-missing tests/

lint:
	flake8 .
	mypy . --ignore-missing-imports

format:
	black .
	isort .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ dist/ build/

pre-commit:
	pip install pre-commit
	pre-commit install
