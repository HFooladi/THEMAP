# Makefile for THEMAP project

.PHONY: help install install-dev install-all test test-unit test-integration test-distance test-fast test-coverage lint lint-check format format-check type-check ci clean docs docs-serve docs-build docs-clean docs-deploy

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Installation:"
	@echo "  install        - Install package in development mode"
	@echo "  install-dev    - Install with dev + test dependencies"
	@echo "  install-all    - Install with all optional dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-distance  - Run distance module tests only"
	@echo "  test-fast      - Run fast tests (exclude slow tests)"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Lint and auto-fix with ruff"
	@echo "  lint-check     - Lint without fixing (CI mode)"
	@echo "  format         - Format code with ruff"
	@echo "  format-check   - Check formatting without changes (CI mode)"
	@echo "  type-check     - Run type checking with mypy"
	@echo "  ci             - Run all CI checks (lint, format, type-check, test-fast)"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           - Build and serve documentation locally"
	@echo "  docs-serve     - Serve documentation with live reload"
	@echo "  docs-build     - Build static documentation"
	@echo "  docs-clean     - Clean documentation artifacts"
	@echo "  docs-deploy    - Deploy documentation to GitHub Pages"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean          - Clean up build and cache artifacts"

# Installation targets
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,test]"

install-all:
	uv pip install -e ".[all,test,dev,docs]"

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

# Code quality targets
lint:
	ruff check --fix .

lint-check:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

type-check:
	mypy -p themap

# CI: replicates the exact checks run in GitHub Actions
ci: lint-check format-check type-check test-fast docs-build

# Documentation targets
docs: docs-serve

docs-serve:
	python build_docs.py serve

docs-build:
	mkdocs build --strict

docs-clean:
	python build_docs.py clean

docs-deploy:
	mkdocs gh-deploy

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage/ .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf site/
	rm -rf .mkdocs_cache/
	rm -rf feature_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
