"""
Command-line interface for THEMAP.

This module provides a click-based CLI for running the THEMAP pipeline
and related utilities.

Usage:
    themap run config.yaml              # Run pipeline with config
    themap run config.yaml -o output/   # Run with custom output
    themap init                         # Create sample config file
    themap convert input.csv CHEMBL123  # Convert CSV to JSONL.GZ
    themap list-featurizers             # List available featurizers
"""

from pathlib import Path
from typing import Optional

import click

from .config import PipelineConfig
from .utils.config import LoggingConfig
from .utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """THEMAP: Task Hardness Estimation via Molecular and Protein Analysis

    Compute distances between source and target molecular datasets
    for transfer learning and meta-learning applications.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # Configure logging based on verbosity
    log_level = "DEBUG" if verbose else "INFO"
    log_config = LoggingConfig(level=log_level)
    setup_logging(log_config)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output directory (overrides config)")
@click.option("--molecule-only", is_flag=True, help="Only compute molecule distances")
@click.option("--protein-only", is_flag=True, help="Only compute protein distances")
@click.option("--n-jobs", "-j", type=int, help="Number of parallel jobs")
@click.pass_context
def run(
    ctx: click.Context,
    config: str,
    output: Optional[str],
    molecule_only: bool,
    protein_only: bool,
    n_jobs: Optional[int],
) -> None:
    """Run the distance computation pipeline.

    CONFIG is the path to a YAML configuration file.

    Examples:
        themap run config.yaml
        themap run config.yaml --output results/
        themap run config.yaml --molecule-only
    """
    from .pipeline import Pipeline

    click.echo(f"Loading configuration from {config}...")

    cfg = PipelineConfig.from_yaml(config)

    # Override options
    if output:
        cfg.output.directory = Path(output)

    if molecule_only:
        cfg.protein.enabled = False

    if protein_only:
        cfg.molecule.enabled = False

    if n_jobs:
        cfg.compute.n_jobs = n_jobs

    # Validate
    issues = cfg.validate()
    for issue in issues:
        click.echo(f"Warning: {issue}", err=True)

    # Run pipeline
    click.echo("Starting pipeline...")
    pipeline = Pipeline(cfg)

    try:
        results = pipeline.run()

        # Print summary
        click.echo("\nResults:")
        for name, matrix in results.items():
            if matrix:
                n_targets = len(matrix)
                n_sources = len(list(matrix.values())[0]) if matrix else 0
                click.echo(f"  {name}: {n_targets} x {n_sources} distances")

        click.echo(f"\nOutput saved to: {cfg.output.directory}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        raise SystemExit(1)


@cli.command()
@click.option("--output", "-o", default="config.yaml", help="Output file path")
@click.option("--data-dir", type=click.Path(), help="Data directory to use")
def init(output: str, data_dir: Optional[str]) -> None:
    """Create a sample configuration file.

    Examples:
        themap init
        themap init --output my_config.yaml
        themap init --data-dir datasets/TDC/
    """
    cfg = PipelineConfig()

    if data_dir:
        cfg.data.directory = Path(data_dir)

    cfg.to_yaml(output)
    click.echo(f"Created configuration file: {output}")
    click.echo("\nEdit the file to customize your pipeline, then run:")
    click.echo(f"  themap run {output}")


@cli.command()
@click.argument("input_csv", type=click.Path(exists=True))
@click.argument("task_id")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--smiles-column", default=None, help="SMILES column name (auto-detected if not specified)")
@click.option("--activity-column", default=None, help="Activity column name (auto-detected if not specified)")
@click.option("--no-validate", is_flag=True, help="Skip SMILES validation")
def convert(
    input_csv: str,
    task_id: str,
    output: Optional[str],
    smiles_column: Optional[str],
    activity_column: Optional[str],
    no_validate: bool,
) -> None:
    """Convert CSV file to JSONL.GZ format.

    INPUT_CSV is the path to the CSV file.
    TASK_ID is the identifier for the task (e.g., CHEMBL123456).

    Examples:
        themap convert data.csv CHEMBL123456
        themap convert data.csv CHEMBL123456 --output datasets/train/CHEMBL123456.jsonl.gz
        themap convert data.csv CHEMBL123456 --smiles-column SMILES --activity-column pIC50
    """
    from .data.converter import CSVConverter

    input_path = Path(input_csv)

    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"{task_id}.jsonl.gz")

    click.echo(f"Converting {input_csv} to {output_path}...")

    converter = CSVConverter(
        validate_smiles=not no_validate,
        auto_detect_columns=True,
    )

    try:
        stats = converter.convert(
            input_path,
            output_path,
            task_id,
            smiles_column=smiles_column,
            activity_column=activity_column,
        )

        click.echo("\nConversion complete:")
        click.echo(f"  Total rows: {stats.total_rows}")
        click.echo(f"  Valid molecules: {stats.valid_molecules}")
        click.echo(f"  Invalid SMILES: {stats.invalid_count}")
        click.echo(f"  Success rate: {stats.success_rate:.1f}%")
        click.echo(f"\nOutput: {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command("list-featurizers")
def list_featurizers() -> None:
    """List available molecule and protein featurizers.

    Examples:
        themap list-featurizers
    """
    from .features.molecule import (
        DESCRIPTOR_FEATURIZERS,
        FINGERPRINT_FEATURIZERS,
        NEURAL_FEATURIZERS,
    )
    from .features.protein import ESM2_MODELS, ESM3_MODELS

    click.echo("Molecule Featurizers:")
    click.echo("\n  Fingerprints (fast):")
    for f in FINGERPRINT_FEATURIZERS:
        click.echo(f"    - {f}")

    click.echo("\n  Descriptors (medium):")
    for f in DESCRIPTOR_FEATURIZERS:
        click.echo(f"    - {f}")

    click.echo("\n  Neural Embeddings (slow, requires GPU):")
    for f in NEURAL_FEATURIZERS:
        click.echo(f"    - {f}")

    click.echo("\n\nProtein Featurizers:")
    click.echo("\n  ESM2 Models:")
    for f in ESM2_MODELS:
        click.echo(f"    - {f}")

    click.echo("\n  ESM3 Models:")
    for f in ESM3_MODELS:
        click.echo(f"    - {f}")


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default="output", help="Output directory")
@click.option("--featurizer", "-f", default="ecfp", help="Molecule featurizer")
@click.option("--method", "-m", default="euclidean", help="Distance method")
@click.option("--n-jobs", "-j", default=8, help="Number of parallel jobs")
def quick(
    data_dir: str,
    output: str,
    featurizer: str,
    method: str,
    n_jobs: int,
) -> None:
    """Quick distance computation with minimal configuration.

    DATA_DIR is the path to a directory with train/ and test/ folders.

    Examples:
        themap quick datasets/TDC/
        themap quick datasets/TDC/ --featurizer maccs --method cosine
    """
    from .pipeline import quick_distance

    click.echo(f"Computing distances from {data_dir}...")
    click.echo(f"  Featurizer: {featurizer}")
    click.echo(f"  Method: {method}")
    click.echo(f"  Workers: {n_jobs}")

    try:
        results = quick_distance(
            data_dir=data_dir,
            output_dir=output,
            molecule_featurizer=featurizer,
            molecule_method=method,
            n_jobs=n_jobs,
        )

        click.echo("\nResults:")
        for name, matrix in results.items():
            if matrix:
                n_targets = len(matrix)
                n_sources = len(list(matrix.values())[0]) if matrix else 0
                click.echo(f"  {name}: {n_targets} x {n_sources} distances")

        click.echo(f"\nOutput saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True))
def info(data_dir: str) -> None:
    """Show information about a dataset directory.

    DATA_DIR is the path to a directory with train/test/valid folders.

    Examples:
        themap info datasets/TDC/
    """
    from .data.loader import DatasetLoader

    loader = DatasetLoader(data_dir)
    stats = loader.get_statistics()

    click.echo(f"Dataset Directory: {stats['data_dir']}")
    click.echo(f"Task list provided: {stats['task_list_provided']}")

    click.echo("\nFolds:")
    for fold, fold_stats in stats.get("folds", {}).items():
        click.echo(f"  {fold}:")
        click.echo(f"    Tasks: {fold_stats['task_count']}")
        click.echo(f"    CSV files: {fold_stats['csv_count']}")
        click.echo(f"    JSONL.GZ files: {fold_stats['jsonl_gz_count']}")

    if "proteins" in stats:
        click.echo(f"\nProteins: {stats['proteins']['count']} FASTA files")


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
