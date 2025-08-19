"""
Command-line interface for running configuration-driven benchmarking pipelines.

This module provides a CLI tool for executing distance computation pipelines
based on YAML or JSON configuration files.
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .runner import PipelineRunner


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("pipeline.log")],
    )
    return logging.getLogger("themap.pipeline")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="THEMAP Pipeline Runner - Execute configuration-driven distance benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a simple benchmark
  python -m themap.pipeline.cli configs/examples/quick_test.yaml
  
  # Run with custom data path and verbose logging
  python -m themap.pipeline.cli configs/my_config.json --data-path /path/to/datasets --log-level DEBUG
  
  # Validate configuration without running
  python -m themap.pipeline.cli configs/my_config.yaml --validate-only
  
  # List available example configurations
  python -m themap.pipeline.cli --list-examples
        """,
    )

    parser.add_argument(
        "config", nargs="?", type=Path, help="Path to pipeline configuration file (YAML or JSON)"
    )

    parser.add_argument(
        "--data-path", type=str, default="datasets", help="Base path to dataset files (default: datasets)"
    )

    parser.add_argument("--output-dir", type=str, help="Override output directory from config")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate configuration without running pipeline"
    )

    parser.add_argument("--list-examples", action="store_true", help="List available example configurations")

    parser.add_argument("--dry-run", action="store_true", help="Show what would be computed without running")

    parser.add_argument("--sample-size", type=int, help="Override sample size from config")

    parser.add_argument("--max-workers", type=int, help="Override max workers from config")

    return parser


def list_examples() -> None:
    """List available example configurations."""
    examples_dir = Path(__file__).parent.parent.parent / "configs" / "examples"

    if not examples_dir.exists():
        print("No example configurations found.")
        return

    print("Available example configurations:")
    print("=" * 40)

    for config_file in sorted(examples_dir.glob("*.{yaml,yml,json}")):
        try:
            config = PipelineConfig.from_file(config_file)
            print(f"\n{config_file.name}")
            print(f"  Name: {config.name}")
            print(f"  Description: {config.description}")

            modalities = []
            if config.molecule:
                if config.molecule.datasets:
                    modalities.append(f"molecule ({len(config.molecule.datasets)} datasets)")
                elif config.molecule.directory:
                    modalities.append("molecule (directory-based)")
                else:
                    modalities.append("molecule")
            if config.protein:
                if config.protein.datasets:
                    modalities.append(f"protein ({len(config.protein.datasets)} datasets)")
                elif config.protein.directory:
                    modalities.append("protein (directory-based)")
                else:
                    modalities.append("protein")
            if config.metadata:
                modalities.append("metadata")

            print(f"  Modalities: {', '.join(modalities)}")

        except Exception as e:
            print(f"\n{config_file.name}")
            print(f"  Error loading config: {e}")


def validate_config(config_path: Path, data_path: str, logger: logging.Logger) -> PipelineConfig:
    """Validate configuration file."""
    try:
        config = PipelineConfig.from_file(config_path)
        logger.info(f"Configuration loaded successfully: {config.name}")

        # Validate dataset availability
        config.validate_datasets(data_path)
        logger.info("Dataset validation passed")

        return config

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def dry_run(config: PipelineConfig, logger: logging.Logger) -> None:
    """Show what would be computed without running."""
    print(f"\nDry run for pipeline: {config.name}")
    print("=" * 50)

    print(f"Description: {config.description}")

    if config.molecule:
        if config.molecule.datasets:
            print(f"\nMolecule datasets ({len(config.molecule.datasets)}):")
            for dataset in config.molecule.datasets:
                print(f"  - {dataset.name} ({dataset.source_fold} -> {dataset.target_folds})")
        elif config.molecule.directory:
            print("\nMolecule datasets (directory-based):")
            print(f"  - Root path: {config.molecule.directory.root_path}")
            print(f"  - Task list file: {config.molecule.directory.task_list_file}")
        print(f"  Featurizers: {', '.join(config.molecule.featurizers)}")
        print(f"  Distance methods: {', '.join(config.molecule.distance_methods)}")

    if config.protein:
        if config.protein.datasets:
            print(f"\nProtein datasets ({len(config.protein.datasets)}):")
            for dataset in config.protein.datasets:
                print(f"  - {dataset.name} ({dataset.source_fold} -> {dataset.target_folds})")
        elif config.protein.directory:
            print("\nProtein datasets (directory-based):")
            print(f"  - Root path: {config.protein.directory.root_path}")
            print(f"  - Task list file: {config.protein.directory.task_list_file}")
        print(f"  Featurizers: {', '.join(config.protein.featurizers)}")
        print(f"  Distance methods: {', '.join(config.protein.distance_methods)}")

    if config.task_distance:
        print("\nTask distance computation:")
        print(f"  Strategy: {config.task_distance.combination_strategy}")
        print(f"  Weights: {config.task_distance.weights}")

    print("\nComputation settings:")
    print(f"  Max workers: {config.compute.max_workers}")
    print(f"  Sample size: {config.compute.sample_size or 'full dataset'}")
    print(f"  GPU if available: {config.compute.gpu_if_available}")

    print("\nOutput settings:")
    print(f"  Directory: {config.output.directory}")
    print(f"  Formats: {', '.join(config.output.formats)}")
    print(f"  Save intermediate: {config.output.save_intermediate}")
    print(f"  Save matrices: {config.output.save_matrices}")

    # Estimate number of computations
    total_computations = 0

    if config.molecule:
        if config.molecule.datasets:
            n_datasets = len(config.molecule.datasets)
        else:
            n_datasets = "unknown (directory-based)"
        n_featurizers = len(config.molecule.featurizers)
        n_methods = len(config.molecule.distance_methods)
        if isinstance(n_datasets, int):
            molecule_computations = (n_datasets * (n_datasets - 1) // 2) * n_featurizers * n_methods
            total_computations += molecule_computations
            print(f"\nEstimated molecule distance computations: {molecule_computations}")
        else:
            print(f"\nMolecule distance computations: {n_datasets}")

    if config.protein:
        if config.protein.datasets:
            n_datasets = len(config.protein.datasets)
        else:
            n_datasets = "unknown (directory-based)"
        n_featurizers = len(config.protein.featurizers)
        n_methods = len(config.protein.distance_methods)
        if isinstance(n_datasets, int):
            protein_computations = (n_datasets * (n_datasets - 1) // 2) * n_featurizers * n_methods
            total_computations += protein_computations
            print(f"Estimated protein distance computations: {protein_computations}")
        else:
            print(f"Protein distance computations: {n_datasets}")

    if config.task_distance and config.molecule and config.protein:
        # Estimate for individual dataset configs only
        if config.molecule.datasets and config.protein.datasets:
            combined_datasets = min(len(config.molecule.datasets), len(config.protein.datasets))
            task_computations = combined_datasets * (combined_datasets - 1) // 2
            total_computations += task_computations
            print(f"Estimated task distance computations: {task_computations}")
        else:
            print("Task distance computations: depends on directory contents")

    print(f"\nTotal estimated computations: {total_computations}")


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_level)

    # Handle special commands
    if args.list_examples:
        list_examples()
        return

    if not args.config:
        parser.print_help()
        return

    try:
        # Validate configuration
        config = validate_config(args.config, args.data_path, logger)

        # Apply command-line overrides
        if args.output_dir:
            config.output.directory = args.output_dir
        if args.sample_size:
            config.compute.sample_size = args.sample_size
        if args.max_workers:
            config.compute.max_workers = args.max_workers

        # Handle validation-only mode
        if args.validate_only:
            logger.info("Configuration validation completed successfully")
            return

        # Handle dry run mode
        if args.dry_run:
            dry_run(config, logger)
            return

        # Run the pipeline
        logger.info(f"Starting pipeline execution: {config.name}")
        runner = PipelineRunner(config, logger)
        results = runner.run(args.data_path)

        logger.info("Pipeline completed successfully!")
        print(f"\nResults saved to: {config.output.directory}")

        if results.get("errors"):
            logger.warning(f"Pipeline completed with {len(results['errors'])} errors")
            for error in results["errors"]:
                logger.warning(f"Error in {error['stage']}: {error['error']}")

    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
