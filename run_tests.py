#!/usr/bin/env python3
"""
Test runner script for THEMAP project.

This script provides convenient commands to run different types of tests
with appropriate configurations.

Usage:
    python run_tests.py [test_type] [options]

Test types:
    all      - Run all tests (default)
    unit     - Run only unit tests
    integration - Run only integration tests
    distance - Run only distance module tests
    fast     - Run tests excluding slow ones
    coverage - Run tests with coverage report

Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py unit               # Run unit tests only
    python run_tests.py fast               # Skip slow tests
    python run_tests.py coverage           # Run with coverage
    python run_tests.py distance -v        # Run distance tests verbosely
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted by user")
        return 130


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for THEMAP project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "distance", "fast", "coverage"],
        help="Type of tests to run (default: all)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    parser.add_argument("-x", "--stop-on-first-failure", action="store_true", help="Stop on first failure")

    parser.add_argument("-k", "--keyword", help="Run tests matching given keyword expression")

    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel using pytest-xdist")

    args = parser.parse_args()

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test type specific options
    if args.test_type == "unit":
        cmd.extend(["-m", "unit"])
        description = "Running unit tests"
    elif args.test_type == "integration":
        cmd.extend(["-m", "integration"])
        description = "Running integration tests"
    elif args.test_type == "distance":
        cmd.extend(["tests/distance/"])
        description = "Running distance module tests"
    elif args.test_type == "fast":
        cmd.extend(["-m", "not slow"])
        description = "Running fast tests (excluding slow tests)"
    elif args.test_type == "coverage":
        cmd.extend(["--cov=themap", "--cov-report=html", "--cov-report=term-missing"])
        description = "Running tests with coverage analysis"
    else:  # all
        description = "Running all tests"

    # Add optional flags
    if args.verbose:
        cmd.append("-v")

    if args.stop_on_first_failure:
        cmd.append("-x")

    if args.keyword:
        cmd.extend(["-k", args.keyword])

    if args.parallel:
        cmd.extend(["-n", "auto"])

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run from project root.")
        return 1

    # Check if pytest is available
    try:
        subprocess.run(["python", "-m", "pytest", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ Error: pytest not installed. Install with: pip install pytest")
        return 1

    # Run the tests
    return run_command(cmd, description)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
