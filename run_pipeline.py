#!/usr/bin/env python3
"""
THEMAP Pipeline Runner Script

A simple Python script to run the THEMAP pipeline with configuration files.
This script provides a convenient interface to the CLI module.

Usage:
    python run_pipeline.py configs/examples/simple_directory_discovery.yaml
    python run_pipeline.py --dry-run configs/examples/comprehensive_multimodal.yaml
    python run_pipeline.py --list-examples
"""

import argparse
import sys
from pathlib import Path
from typing import List


def print_info(message: str) -> None:
    """Print info message."""
    print(f"ℹ️  {message}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"❌ {message}")


def run_pipeline_cli(args: List[str]) -> int:
    """Run the pipeline CLI with given arguments."""
    try:
        # Use the CLI module directly
        from themap.pipeline.cli import main as cli_main

        # Temporarily replace sys.argv to pass arguments to CLI
        original_argv = sys.argv.copy()
        sys.argv = ["themap.pipeline"] + args

        try:
            cli_main()
            return 0
        except SystemExit as e:
            return e.code if e.code is not None else 0
        finally:
            sys.argv = original_argv

    except ImportError as e:
        print_error(f"Failed to import THEMAP: {e}")
        print_info("Make sure THEMAP is installed and you're in the correct environment")
        return 1
    except Exception as e:
        print_error(f"Pipeline execution failed: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="THEMAP Pipeline Runner - Simple interface to run pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a simple pipeline
  python run_pipeline.py configs/examples/simple_directory_discovery.yaml

  # Run with custom settings
  python run_pipeline.py --data-path /path/to/data --sample-size 100 configs/my_config.yaml

  # Validate configuration only
  python run_pipeline.py --validate-only configs/my_config.yaml

  # See what would be computed
  python run_pipeline.py --dry-run configs/examples/comprehensive_multimodal.yaml

  # List available examples
  python run_pipeline.py --list-examples
        """,
    )

    parser.add_argument("config", nargs="?", help="Path to pipeline configuration file (YAML or JSON)")

    parser.add_argument(
        "--data-path", default="datasets", help="Base path to dataset files (default: datasets)"
    )

    parser.add_argument("--output-dir", help="Override output directory from config")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--sample-size", type=int, help="Override sample size from config (useful for testing)"
    )

    parser.add_argument("--max-workers", type=int, help="Override number of workers from config")

    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate configuration without running"
    )

    parser.add_argument("--dry-run", action="store_true", help="Show what would be computed without running")

    parser.add_argument("--list-examples", action="store_true", help="List available example configurations")

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle special cases
    if args.list_examples:
        print_info("Available example configurations:")
        return run_pipeline_cli(["--list-examples"])

    if not args.config and not args.list_examples:
        parser.print_help()
        return 0

    # Check if config file exists
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print_error(f"Configuration file not found: {config_path}")
            return 1

    # Build CLI arguments
    cli_args = []

    if args.config:
        cli_args.append(str(args.config))

    if args.data_path != "datasets":
        cli_args.extend(["--data-path", args.data_path])

    if args.output_dir:
        cli_args.extend(["--output-dir", args.output_dir])

    if args.log_level != "INFO":
        cli_args.extend(["--log-level", args.log_level])

    if args.sample_size:
        cli_args.extend(["--sample-size", str(args.sample_size)])

    if args.max_workers:
        cli_args.extend(["--max-workers", str(args.max_workers)])

    if args.validate_only:
        cli_args.append("--validate-only")

    if args.dry_run:
        cli_args.append("--dry-run")

    # Run the pipeline
    if args.config:
        print_info(f"Running pipeline with config: {args.config}")

    exit_code = run_pipeline_cli(cli_args)

    if exit_code == 0:
        if not args.validate_only and not args.dry_run and not args.list_examples:
            print_success("Pipeline completed successfully!")
    else:
        print_error(f"Pipeline failed with exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
