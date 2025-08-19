# Makefile for THEMAP project

.PHONY: help test test-unit test-integration test-distance test-fast test-coverage install install-dev install-test lint format type-check clean docs docs-serve docs-build docs-clean docs-deploy pipeline

# Default target
help:
	@echo "Available targets:"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-distance  - Run distance module tests only"
	@echo "  test-fast      - Run fast tests (exclude slow tests)"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           - Build and serve documentation locally"
	@echo "  docs-serve     - Serve documentation with live reload"
	@echo "  docs-build     - Build static documentation"
	@echo "  docs-clean     - Clean documentation artifacts"
	@echo "  docs-deploy    - Deploy documentation to GitHub Pages"
	@echo ""
	@echo "Type checking:"
	@echo "  type-check     - Run type checking with mypy"
	@echo ""
	@echo "Code quality:"
	@echo "  lint           - Run linting with ruff"
	@echo "  format         - Format code with ruff"
	@echo ""
	@echo "Development:"
	@echo "  install        - Install package in development mode"
	@echo "  install-dev    - Install package with dev dependencies"
	@echo "  install-test   - Install package with test dependencies"
	@echo "  lint           - Run linting with ruff"
	@echo "  format         - Format code with ruff"
	@echo "  clean          - Clean up build artifacts"
	@echo ""
	@echo "Pipeline:"
	@echo "  pipeline       - Run the pipeline"


# Testing targets
test:
	python run_tests.py all

test-unit:
	python run_tests.py unit

test-integration:
	python run_tests.py integration

test-distance:
	python run_tests.py distance

test-fast:
	python run_tests.py fast

test-coverage:
	python run_tests.py coverage

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test]"

install-test:
	pip install -e ".[test]"

# Code quality targets
lint:
	ruff check --fix .

format:
	ruff format .

# Type checking targets
type-check:
	mypy themap/

# Documentation targets
docs: docs-serve

docs-serve:
	python build_docs.py serve

docs-build:
	python build_docs.py build

docs-clean:
	python build_docs.py clean

docs-deploy:
	mkdocs gh-deploy

# Pipeline targets
pipeline:
	python run_pipeline.py configs/examples/quick_test.yaml


# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf site/
	rm -rf .mkdocs_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
