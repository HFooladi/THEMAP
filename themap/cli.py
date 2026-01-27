"""
Command-line interface for THEMAP.

This module provides a click-based CLI for running the THEMAP pipeline
and related utilities.

Usage:
    themap run config.yaml              # Run pipeline with config
    themap run config.yaml -o output/   # Run with custom output
    themap init                         # Create sample config file
    themap convert input.csv CHEMBL123  # Convert CSV to JSONL.GZ
    themap featurize datasets/ -f ecfp  # Featurize datasets (no distance computation)
    themap list-featurizers             # List available featurizers
"""

from pathlib import Path
from typing import List, Optional, Tuple

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


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option(
    "--featurizer",
    "-f",
    multiple=True,
    default=["ecfp"],
    help="Featurizer(s) to use (can be specified multiple times)",
)
@click.option("--cache-dir", "-c", default="feature_cache", help="Directory to cache features")
@click.option(
    "--fold",
    type=click.Choice(["train", "test", "valid", "all"]),
    default="all",
    help="Which fold(s) to featurize",
)
@click.option("--n-jobs", "-j", default=1, help="Number of parallel jobs for featurization")
@click.option("--force", is_flag=True, help="Recompute features even if cached")
@click.pass_context
def featurize(
    ctx: click.Context,
    data_path: str,
    featurizer: Tuple[str, ...],
    cache_dir: str,
    fold: str,
    n_jobs: int,
    force: bool,
) -> None:
    """Compute and cache molecular features without distance computation.

    DATA_PATH can be either:
    - A directory with train/test/valid folders containing datasets
    - A single dataset file (.jsonl.gz or .csv)

    Features are cached to disk and can be reused by other commands.

    Examples:
        # Featurize all datasets in a directory with ECFP
        themap featurize datasets/ -f ecfp

        # Featurize with multiple featurizers
        themap featurize datasets/ -f ecfp -f maccs -f desc2D

        # Featurize only training data
        themap featurize datasets/ -f ecfp --fold train

        # Featurize a single file
        themap featurize datasets/train/CHEMBL123.jsonl.gz -f ecfp

        # Force recomputation (ignore cache)
        themap featurize datasets/ -f ecfp --force

        # Custom cache directory
        themap featurize datasets/ -f ecfp --cache-dir my_cache/
    """
    from .data.loader import DatasetLoader
    from .data.molecule_dataset import MoleculeDataset
    from .pipeline.featurization import FeaturizationPipeline

    data_path_obj = Path(data_path)
    cache_path = Path(cache_dir)
    featurizer_list: List[str] = list(featurizer)

    click.echo(f"Featurizing data from: {data_path}")
    click.echo(f"Featurizers: {', '.join(featurizer_list)}")
    click.echo(f"Cache directory: {cache_path}")

    try:
        # Determine if input is a file or directory
        if data_path_obj.is_file():
            # Single file mode
            click.echo(f"\nProcessing single file: {data_path_obj.name}")
            task_id = data_path_obj.stem.replace(".jsonl", "")

            dataset = MoleculeDataset.load_from_file(data_path_obj)
            datasets = [dataset]
            dataset_names = [task_id]

            click.echo(f"  Loaded {len(dataset)} molecules")
        else:
            # Directory mode
            loader = DatasetLoader(data_path_obj)
            stats = loader.get_statistics()

            click.echo(f"\nDataset directory: {stats['data_dir']}")

            datasets = []
            dataset_names = []

            # Determine which folds to process
            folds_to_process = ["train", "test", "valid"] if fold == "all" else [fold]

            for fold_name in folds_to_process:
                if fold_name not in stats.get("folds", {}):
                    continue

                fold_stats = stats["folds"][fold_name]
                click.echo(f"\n{fold_name.capitalize()} fold: {fold_stats['task_count']} tasks")

                fold_datasets = loader.load_datasets(fold_name)
                for task_id, ds in fold_datasets.items():
                    datasets.append(ds)
                    dataset_names.append(f"{fold_name}_{task_id}")

            if not datasets:
                click.echo("No datasets found to featurize.", err=True)
                raise SystemExit(1)

            click.echo(f"\nTotal datasets to featurize: {len(datasets)}")

        # Process each featurizer
        for feat_name in featurizer_list:
            click.echo(f"\n{'=' * 50}")
            click.echo(f"Featurizer: {feat_name}")
            click.echo(f"{'=' * 50}")

            pipeline = FeaturizationPipeline(
                cache_dir=cache_path,
                molecule_featurizer=feat_name,
            )

            # Check cache status
            if not force:
                cached_count = 0
                for ds in datasets:
                    if pipeline.store.has_molecule_features(ds.task_id, feat_name):
                        cached_count += 1

                if cached_count > 0:
                    click.echo(f"  Found {cached_count}/{len(datasets)} datasets already cached")
                    if cached_count == len(datasets):
                        click.echo("  All datasets already cached. Use --force to recompute.")
                        continue

            # Clear cache if force flag is set
            if force:
                click.echo("  Clearing existing cache...")
                pipeline.store.clear_cache(feat_name)

            # Featurize all datasets
            click.echo(f"  Computing features for {len(datasets)} datasets...")

            with click.progressbar(
                zip(datasets, dataset_names),
                length=len(datasets),
                label="  Featurizing",
            ) as bar:
                success_count = 0
                fail_count = 0

                for ds, name in bar:
                    try:
                        # Check if already cached
                        if not force and pipeline.store.has_molecule_features(ds.task_id, feat_name):
                            success_count += 1
                            continue

                        # Featurize
                        pipeline.featurize_all_datasets([ds])
                        success_count += 1
                    except Exception as e:
                        fail_count += 1
                        if ctx.obj.get("verbose"):
                            click.echo(f"\n  Error featurizing {name}: {e}", err=True)

            click.echo(f"  Completed: {success_count} succeeded, {fail_count} failed")

            # Show cache location
            cache_subdir = cache_path / "molecules" / feat_name
            if cache_subdir.exists():
                n_cached = len(list(cache_subdir.glob("*.npz")))
                click.echo(f"  Cached features: {cache_subdir} ({n_cached} files)")

        click.echo("\nFeaturization complete!")
        click.echo(f"Features cached at: {cache_path}")
        click.echo("\nTo use cached features in distance computation:")
        click.echo(f"  themap quick {data_path} --featurizer {featurizer_list[0]}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        raise SystemExit(1)


def main() -> None:
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
