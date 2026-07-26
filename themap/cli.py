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
from typing import Any, List, Optional, Tuple

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
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default=None,
    help="Device for OTDD ('auto' picks cuda if available). Overrides config.",
)
@click.pass_context
def run(
    ctx: click.Context,
    config: str,
    output: Optional[str],
    molecule_only: bool,
    protein_only: bool,
    n_jobs: Optional[int],
    device: Optional[str],
) -> None:
    """Run the distance computation pipeline.

    CONFIG is the path to a YAML configuration file.

    Examples:
        themap run config.yaml
        themap run config.yaml --output results/
        themap run config.yaml --molecule-only
        themap run config.yaml --device cuda
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

    if device:
        cfg.compute.device = device

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
        COUNT_FINGERPRINT_FEATURIZERS,
        DESCRIPTOR_FEATURIZERS,
        FINGERPRINT_FEATURIZERS,
        NEURAL_FEATURIZERS,
    )
    from .features.protein import ESM2_MODELS, ESM3_MODELS

    click.echo("Molecule Featurizers:")
    click.echo("\n  Fingerprints (fast):")
    for f in FINGERPRINT_FEATURIZERS:
        click.echo(f"    - {f}")

    click.echo("\n  Count Fingerprints (fast):")
    for f in COUNT_FINGERPRINT_FEATURIZERS:
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
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    help="Device for OTDD ('auto' picks cuda if available).",
)
def quick(
    data_dir: str,
    output: str,
    featurizer: str,
    method: str,
    n_jobs: int,
    device: str,
) -> None:
    """Quick distance computation with minimal configuration.

    DATA_DIR is the path to a directory with train/ and test/ folders.

    Examples:
        themap quick datasets/TDC/
        themap quick datasets/TDC/ --featurizer maccs --method cosine
        themap quick datasets/TDC/ -m otdd --device cuda
    """
    from .pipeline import quick_distance

    click.echo(f"Computing distances from {data_dir}...")
    click.echo(f"  Featurizer: {featurizer}")
    click.echo(f"  Method: {method}")
    click.echo(f"  Workers: {n_jobs}")
    click.echo(f"  Device: {device}")

    try:
        results = quick_distance(
            data_dir=data_dir,
            output_dir=output,
            molecule_featurizer=featurizer,
            molecule_method=method,
            n_jobs=n_jobs,
            device=device,
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
@click.option(
    "--distance-file",
    required=True,
    type=click.Path(exists=True),
    help="Saved distance file (JSON/CSV/NPZ) mapping target -> source -> distance.",
)
@click.option("--target-id", required=True, help="Target task id to evaluate.")
@click.option("--k", default=5, help="Number of nearest source datasets to meta-train on.")
@click.option(
    "--algorithm", type=click.Choice(["proto", "maml"]), default="proto", help="Meta-learning algorithm."
)
@click.option("--featurizer", "-f", default="ecfp", help="Molecule featurizer.")
@click.option(
    "--support-sizes", default="16,32,64,128", help="Comma-separated target support-set sizes to sweep."
)
@click.option("--seeds", default=5, help="Repeated seeds per support size.")
@click.option(
    "--train-shot-mode",
    type=click.Choice(["match", "fixed"]),
    default="match",
    help="'match' trains a fresh model per support size (shot tracks N); "
    "'fixed' is the FS-Mol single-model protocol (one model, eval all sizes; set --n-support 64).",
)
@click.option(
    "--query-fraction",
    default=0.5,
    help="Fraction of the target held out as a fixed query set shared across support sizes.",
)
@click.option("--n-support", default=10, help="Support examples per meta-training episode.")
@click.option("--n-query", default=15, help="Query examples per meta-training episode.")
@click.option("--inner-lr", default=0.01, help="MAML inner-loop learning rate.")
@click.option("--inner-steps", default=5, help="MAML inner-loop adaptation steps.")
@click.option("--outer-lr", default=0.001, help="Meta (outer-loop) learning rate.")
@click.option("--num-epochs", default=50, help="Meta-training epochs.")
@click.option("--episodes-per-epoch", default=100, help="Meta-training steps per epoch.")
@click.option("--meta-batch-size", default=8, help="Episodes per outer optimizer step.")
@click.option("--source-fold", default="train", help="Fold the source datasets live in.")
@click.option("--target-fold", default="test", help="Fold the target dataset lives in.")
@click.option("--n-jobs", "-j", default=8, help="Parallel jobs for featurization.")
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    help="Compute device ('auto' picks cuda if available).",
)
@click.option("--output", "-o", default="metalearn_out", help="Output directory.")
@click.pass_context
def metalearn(
    ctx: click.Context,
    data_dir: str,
    distance_file: str,
    target_id: str,
    k: int,
    algorithm: str,
    featurizer: str,
    support_sizes: str,
    seeds: int,
    train_shot_mode: str,
    query_fraction: float,
    n_support: int,
    n_query: int,
    inner_lr: float,
    inner_steps: int,
    outer_lr: float,
    num_epochs: int,
    episodes_per_epoch: int,
    meta_batch_size: int,
    source_fold: str,
    target_fold: str,
    n_jobs: int,
    device: str,
    output: str,
) -> None:
    """Distance-guided meta-learning for a target dataset.

    Picks the K nearest source datasets to TARGET_ID from a saved distance file,
    meta-trains a Prototypical Network or MAML on them, then measures the low-data
    AUROC gain on the target versus a from-scratch baseline.

    DATA_DIR is the path to a directory with train/test/valid folders.

    Examples:
        themap metalearn datasets/ --distance-file output/molecule_distances.csv \\
            --target-id CHEMBL1963831 --k 3 --algorithm proto
        themap metalearn datasets/ --distance-file dist.json --target-id T1 \\
            --algorithm maml --support-sizes 16,32,64
    """
    from .metalearning.config import (
        ExperimentConfig,
        MAMLConfig,
        TrainConfig,
    )
    from .metalearning.evaluation import LowDataEvaluator
    from .metalearning.runner import MetaLearnExperiment

    try:
        sizes = [int(s) for s in support_sizes.split(",") if s.strip()]
    except ValueError:
        click.echo(
            f"Error: invalid --support-sizes '{support_sizes}' (expected comma-separated ints).", err=True
        )
        raise SystemExit(1)

    config = ExperimentConfig(
        data_dir=data_dir,
        distance_file=distance_file,
        target_id=target_id,
        k=k,
        algorithm=algorithm,  # type: ignore[arg-type]
        featurizer=featurizer,
        support_sizes=sizes,
        train_shot_mode=train_shot_mode,  # type: ignore[arg-type]
        query_fraction=query_fraction,
        seeds=seeds,
        n_jobs=n_jobs,
        output_dir=output,
        source_fold=source_fold,
        target_fold=target_fold,
        maml=MAMLConfig(inner_lr=inner_lr, inner_steps=inner_steps),
        train=TrainConfig(
            n_support=n_support,
            n_query=n_query,
            num_epochs=num_epochs,
            episodes_per_epoch=episodes_per_epoch,
            meta_batch_size=meta_batch_size,
            outer_lr=outer_lr,
            device=device,
        ),
    )

    click.echo(f"Meta-learning ({algorithm}) for target '{target_id}' using k={k} nearest sources...")
    try:
        results = MetaLearnExperiment(config).run()
        summary = LowDataEvaluator.summarize(results)

        for title, col in (("AUROC", "auroc_mean"), ("ΔAUPRC", "delta_auprc_mean")):
            click.echo(f"\n{title} by support size (mean):")
            pivot = summary.pivot_table(index="support_size", columns="method", values=col)
            for n in sorted(pivot.index):
                meta = pivot.loc[n].get("meta", float("nan"))
                base = pivot.loc[n].get("baseline", float("nan"))
                click.echo(f"  N={n:>4}:  meta={meta:+.3f}  baseline={base:+.3f}  gain={meta - base:+.3f}")

        click.echo(f"\nOutput saved to: {output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        raise SystemExit(1)


@cli.command("metalearn-compare")
@click.argument("data_dir", type=click.Path(exists=True))
@click.option(
    "--distance-file",
    type=click.Path(),
    default=None,
    help="Saved distance file (JSON/CSV/NPZ). If omitted, distances are auto-computed "
    "with --distance-method.",
)
@click.option(
    "--distance-method",
    default="otdd",
    help="Distance to auto-compute when --distance-file is omitted ('otdd' is the paper's "
    "headline distance and needs the [otdd] extra; 'euclidean'/'cosine' are lightweight).",
)
@click.option("--target-id", default=None, help="Target task id to evaluate (required unless --demo).")
@click.option(
    "--demo",
    is_flag=True,
    help="Run a self-contained close-vs-distant scenario on the bundled datasets so the "
    "distance>random effect is reliably visible with one command (no --target-id needed).",
)
@click.option("--k", default=3, help="Number of source datasets each arm selects.")
@click.option(
    "--random-seeds", default=5, help="Number of random-selection draws to average over (with 95% CI)."
)
@click.option(
    "--algorithm", type=click.Choice(["proto", "maml"]), default="proto", help="Meta-learning algorithm."
)
@click.option("--featurizer", "-f", default="ecfp", help="Molecule featurizer.")
@click.option(
    "--support-sizes", default="16,32,64,128", help="Comma-separated target support-set sizes to sweep."
)
@click.option("--seeds", default=3, help="Evaluation seeds per support size (within each arm).")
@click.option(
    "--train-shot-mode",
    type=click.Choice(["match", "fixed"]),
    default="match",
    help="'match' trains a fresh model per support size; 'fixed' is the FS-Mol single-model protocol.",
)
@click.option(
    "--query-fraction",
    default=0.5,
    help="Fraction of the target held out as a fixed query set shared across support sizes.",
)
@click.option("--n-support", default=10, help="Support examples per meta-training episode.")
@click.option("--n-query", default=15, help="Query examples per meta-training episode.")
@click.option("--inner-lr", default=0.01, help="MAML inner-loop learning rate.")
@click.option("--inner-steps", default=5, help="MAML inner-loop adaptation steps.")
@click.option("--outer-lr", default=0.001, help="Meta (outer-loop) learning rate.")
@click.option("--num-epochs", default=50, help="Meta-training epochs.")
@click.option("--episodes-per-epoch", default=100, help="Meta-training steps per epoch.")
@click.option("--meta-batch-size", default=8, help="Episodes per outer optimizer step.")
@click.option("--source-fold", default="train", help="Fold the source datasets live in.")
@click.option("--target-fold", default="test", help="Fold the target dataset lives in.")
@click.option("--n-jobs", "-j", default=8, help="Parallel jobs for featurization/distances.")
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    help="Compute device ('auto' picks cuda if available).",
)
@click.option("--output", "-o", default="metalearn_compare_out", help="Output directory.")
@click.pass_context
def metalearn_compare(
    ctx: click.Context,
    data_dir: str,
    distance_file: Optional[str],
    distance_method: str,
    target_id: Optional[str],
    demo: bool,
    k: int,
    random_seeds: int,
    algorithm: str,
    featurizer: str,
    support_sizes: str,
    seeds: int,
    train_shot_mode: str,
    query_fraction: float,
    n_support: int,
    n_query: int,
    inner_lr: float,
    inner_steps: int,
    outer_lr: float,
    num_epochs: int,
    episodes_per_epoch: int,
    meta_batch_size: int,
    source_fold: str,
    target_fold: str,
    n_jobs: int,
    device: str,
    output: str,
) -> None:
    """Test THEMAP's headline hypothesis: distance-selected vs random-selected sources.

    For a target dataset, this selects the K nearest source datasets by distance and,
    separately, K random source datasets, meta-trains the same model on each, and
    reports whether the distance-selected sources give a better low-data target model.
    The random arm is averaged over several draws (--random-seeds) for a fair baseline.

    DATA_DIR is the path to a directory with train/test/valid folders.

    Examples:
        # One command, self-contained demonstration (no distance file needed):
        themap metalearn-compare datasets/ --demo

        # Real experiment on your own target (auto-computes OTDD distances):
        themap metalearn-compare datasets/ --target-id CHEMBL1963831 --k 3

        # Reuse a precomputed distance file and the MAML learner:
        themap metalearn-compare datasets/ --distance-file output/molecule_distances.csv \\
            --target-id CHEMBL1963831 --algorithm maml
    """
    from .metalearning.compare import CompareConfig, SelectionComparison
    from .metalearning.config import MAMLConfig, TrainConfig

    if not demo and not target_id:
        click.echo("Error: --target-id is required unless --demo is set.", err=True)
        raise SystemExit(1)

    try:
        sizes = [int(s) for s in support_sizes.split(",") if s.strip()]
    except ValueError:
        click.echo(
            f"Error: invalid --support-sizes '{support_sizes}' (expected comma-separated ints).", err=True
        )
        raise SystemExit(1)

    config = CompareConfig(
        data_dir=data_dir,
        target_id=target_id,
        distance_file=distance_file,
        distance_method=distance_method,
        demo=demo,
        k=k,
        random_seeds=random_seeds,
        algorithm=algorithm,  # type: ignore[arg-type]
        featurizer=featurizer,
        support_sizes=sizes,
        train_shot_mode=train_shot_mode,  # type: ignore[arg-type]
        query_fraction=query_fraction,
        seeds=seeds,
        n_jobs=n_jobs,
        output_dir=output,
        source_fold=source_fold,
        target_fold=target_fold,
        maml=MAMLConfig(inner_lr=inner_lr, inner_steps=inner_steps),
        train=TrainConfig(
            n_support=n_support,
            n_query=n_query,
            num_epochs=num_epochs,
            episodes_per_epoch=episodes_per_epoch,
            meta_batch_size=meta_batch_size,
            outer_lr=outer_lr,
            device=device,
        ),
    )

    label = "demo (close vs distant sources)" if demo else f"target '{target_id}'"
    click.echo(
        f"Comparing distance-selected vs {random_seeds} random draws of k={k} sources "
        f"for {label} ({algorithm})..."
    )
    try:
        comparison = SelectionComparison(config)
        results = comparison.run()
        summary = comparison.summarize(results)

        for title, prefix in (("AUROC", "auroc"), ("ΔAUPRC", "delta_auprc")):
            click.echo(f"\n{title} — distance vs random (meta-learned, on target) by support size:")
            for _, r in summary.iterrows():
                n = int(r["support_size"])
                dist = r[f"distance_{prefix}"]
                rmean = r[f"random_{prefix}_mean"]
                rci = r[f"random_{prefix}_ci95"]
                gap = r[f"gap_{prefix}"]
                click.echo(f"  N={n:>4}:  distance={dist:+.3f}  random={rmean:+.3f}±{rci:.3f}  Δ={gap:+.3f}")

        mean_gap = SelectionComparison.verdict(summary)
        outcome = "SUPPORTED" if mean_gap > 0 else "not supported"
        click.echo(
            f"\nMean AUROC advantage of distance-based over random selection: "
            f"{mean_gap:+.3f}  →  hypothesis {outcome}."
        )

        _plot_selection_comparison(summary, Path(output) / "comparison.png", label)
        click.echo(f"\nOutput saved to: {output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        raise SystemExit(1)


def _plot_selection_comparison(summary: Any, path: Path, label: str) -> None:
    """Plot distance vs random AUROC across support sizes (best-effort; never fatal)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plotting is optional
        return

    n = summary["support_size"].to_numpy()
    dist = summary["distance_auroc"].to_numpy()
    rmean = summary["random_auroc_mean"].to_numpy()
    rci = summary["random_auroc_ci95"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n, dist, marker="o", color="#3288bd", label="distance-selected sources")
    ax.plot(n, rmean, marker="s", color="#fc8d62", label="random-selected sources (mean)")
    ax.fill_between(n, rmean - rci, rmean + rci, color="#fc8d62", alpha=0.2, label="random 95% CI")
    ax.set_xlabel("Target support set size (N)")
    ax.set_ylabel("Meta-learned AUROC on held-out target")
    ax.set_title(f"Distance-based vs random source selection\n{label}")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


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


@cli.command("fsmol-subset")
@click.argument("data_dir", type=click.Path(exists=True))
@click.option(
    "--reference-dir",
    default="benchmarking_datasets/fsmol_reference",
    help="Directory holding (or to receive) FS-Mol's baseline summary CSVs.",
)
@click.option(
    "--proteins-csv",
    default=None,
    help="FS-Mol test protein metadata CSV. Defaults to DATA_DIR/fsmol_test_proteins.csv.",
)
@click.option("--n-tasks", default=20, help="Number of test tasks to select.")
@click.option("--min-large", default=6, help="Minimum picks large enough for the biggest support sizes.")
@click.option("--offline", is_flag=True, help="Never fetch reference CSVs from the network.")
@click.option(
    "-o",
    "--output",
    default="benchmarking_datasets/fsmol_subset_20.json",
    help="Where to write the subset file.",
)
@click.pass_context
def fsmol_subset(ctx, data_dir, reference_dir, proteins_csv, n_tasks, min_large, offline, output):
    """Pick a representative subset of the FS-Mol test tasks.

    Stratifies by EC super-class, task size and FS-Mol ProtoNet difficulty, deterministically,
    and reports how closely the subset's reference means track the full 157-task benchmark.

    \b
    Examples:
        themap fsmol-subset benchmarking_datasets/fsmol_datasets
        themap fsmol-subset benchmarking_datasets/fsmol_datasets --n-tasks 30 --offline
    """
    from .metalearning.fsmol_reference import load_reference_table, reference_checksums
    from .metalearning.subset import (
        SubsetSpec,
        save_subset,
        select_benchmark_subset,
        subset_representativeness,
    )

    try:
        proteins = proteins_csv or str(Path(data_dir) / "fsmol_test_proteins.csv")
        reference = load_reference_table(reference_dir, offline=offline)
        spec = SubsetSpec(n_tasks=n_tasks, min_large=min_large)
        selected = select_benchmark_subset(data_dir, reference, proteins, spec)
        save_subset(
            selected,
            output,
            spec,
            {"reference_sha256": reference_checksums(reference_dir), "data_dir": str(data_dir)},
        )

        click.echo(f"\nSelected {len(selected)} task(s) -> {output}")
        click.echo(f"  EC composition: {selected['ec_class'].value_counts().to_dict()}")
        click.echo(
            f"  tasks larger than {spec.large_task_threshold}: {int((selected['n'] > spec.large_task_threshold).sum())}"
        )
        rep = subset_representativeness(reference, selected["task_id"].tolist())
        click.echo(f"  max |subset mean - full mean| across methods/sizes: {rep['abs_delta'].max():.3f}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        raise SystemExit(1)


@cli.command("fsmol-benchmark")
@click.argument("data_dir", type=click.Path(exists=True))
@click.option(
    "--algorithm",
    "algorithms",
    multiple=True,
    type=click.Choice(["proto", "maml"]),
    help="Meta-learner to benchmark; repeat for several. Default: proto and maml.",
)
@click.option(
    "--subset-file",
    default=None,
    type=click.Path(),
    help="Subset JSON from `themap fsmol-subset`. Omit to evaluate every test task.",
)
@click.option("--task-id", "task_ids", multiple=True, help="Explicit target task id; repeat to add more.")
@click.option("--support-sizes", default="16,32,64,128", help="Comma-separated evaluation support sizes.")
@click.option("--seeds", default=10, help="Repeats per (task, support size). FS-Mol uses 10.")
@click.option("--train-shot", default=64, help="Support size for meta-training episodes.")
@click.option("--n-query", default=32, help="Query size for meta-training episodes.")
@click.option(
    "--no-adaptive",
    is_flag=True,
    help="Require the full meta-training shot instead of clipping it per task. "
    "On FS-Mol this drops ~83%% of the training assays at a 64-shot request.",
)
@click.option(
    "--query-mode",
    type=click.Choice(["fsmol", "holdout"]),
    default="fsmol",
    help="'fsmol': support of size N, query = the rest. 'holdout': THEMAP's shared query set.",
)
@click.option("--n-source-tasks", default=0, help="Cap on meta-training source tasks (0 = all).")
@click.option("--featurizer", "-f", default="ecfp", help="Molecular featurizer.")
@click.option("--num-epochs", default=50, help="Meta-training epochs.")
@click.option("--episodes-per-epoch", default=100, help="Meta-training steps per epoch.")
@click.option("--meta-batch-size", default=16, help="Episodes per outer optimizer step.")
@click.option("--outer-lr", default=1e-4, help="Outer-loop learning rate.")
@click.option("--cache-dir", default="feature_cache/fsmol", help="Feature cache directory.")
@click.option(
    "--reference-dir",
    default="benchmarking_datasets/fsmol_reference",
    help="Directory holding FS-Mol's baseline summary CSVs.",
)
@click.option("--offline", is_flag=True, help="Never fetch reference CSVs from the network.")
@click.option("-j", "--n-jobs", default=1, help="Featurization parallelism (1 is fastest for fingerprints).")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto")
@click.option("-o", "--output", default="fsmol_benchmark_out", help="Output directory.")
@click.pass_context
def fsmol_benchmark(
    ctx,
    data_dir,
    algorithms,
    subset_file,
    task_ids,
    support_sizes,
    seeds,
    train_shot,
    n_query,
    no_adaptive,
    query_mode,
    n_source_tasks,
    featurizer,
    num_epochs,
    episodes_per_epoch,
    meta_batch_size,
    outer_lr,
    cache_dir,
    reference_dir,
    offline,
    n_jobs,
    device,
    output,
):
    """Check THEMAP's meta-learners against FS-Mol's published results.

    Featurizes once, meta-trains once per algorithm on the full FS-Mol training fold, then
    evaluates every target task under FS-Mol's own protocol and writes a side-by-side
    comparison with the published per-task baselines.

    \b
    Examples:
        themap fsmol-benchmark benchmarking_datasets/fsmol_datasets \\
            --subset-file benchmarking_datasets/fsmol_subset_20.json
        themap fsmol-benchmark benchmarking_datasets/fsmol_datasets --algorithm proto --seeds 3
    """
    from .metalearning.benchmark import BenchmarkConfig, FSMolBenchmark
    from .metalearning.config import TrainConfig

    try:
        sizes = [int(x) for x in str(support_sizes).split(",") if x.strip()]
        config = BenchmarkConfig(
            data_dir=data_dir,
            algorithms=list(algorithms) or ["proto", "maml"],
            featurizer=featurizer,
            support_sizes=sizes,
            seeds=seeds,
            target_ids=list(task_ids) or None,
            subset_file=subset_file,
            adaptive_episodes=not no_adaptive,
            train_shot=train_shot,
            query_mode=query_mode,
            n_source_tasks=n_source_tasks,
            n_jobs=n_jobs,
            cache_dir=cache_dir,
            reference_dir=reference_dir,
            offline=offline,
            output_dir=output,
            train=TrainConfig(
                n_support=train_shot,
                n_query=n_query,
                num_epochs=num_epochs,
                episodes_per_epoch=episodes_per_epoch,
                meta_batch_size=meta_batch_size,
                outer_lr=outer_lr,
                device=device,
            ),
        )
        benchmark = FSMolBenchmark(config)
        results = benchmark.run()
        if results.empty:
            click.echo("No results were produced.", err=True)
            raise SystemExit(1)

        click.echo("\n" + (Path(output) / "report.md").read_text())
        click.echo(f"Artifacts written to: {output}")
    except SystemExit:
        raise
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
