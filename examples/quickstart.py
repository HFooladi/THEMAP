#!/usr/bin/env python
"""
THEMAP Quickstart Example
=========================

This example shows how to compute distances between molecular datasets using THEMAP.

Dataset Structure
-----------------
Organize your data in this structure:

    datasets/
    ├── train/                    # Source datasets
    │   ├── CHEMBL123456.jsonl.gz
    │   ├── CHEMBL789012.jsonl.gz
    │   └── ...
    ├── test/                     # Target datasets
    │   ├── CHEMBL111111.jsonl.gz
    │   └── ...
    └── tasks.json               # Optional: specify which tasks to use

Each .jsonl.gz file contains molecules in JSON lines format:
    {"SMILES": "CCO", "Property": 1}
    {"SMILES": "CCCO", "Property": 0}
    ...

Usage
-----
1. Quick one-liner:
   python quickstart.py

2. With config file:
   python quickstart.py --config config.yaml

3. Custom options:
   python quickstart.py --data datasets/ --featurizer maccs --method cosine
"""

import argparse
from pathlib import Path


def example_quick_distance():
    """Simplest way to compute distances - one function call."""
    from themap import quick_distance

    # Compute molecule distances between train and test datasets
    results = quick_distance(
        data_dir="datasets",  # Directory with train/ and test/ folders
        output_dir="output",  # Where to save results
        molecule_featurizer="ecfp",  # Molecular fingerprint type
        molecule_method="euclidean",  # Distance metric
        n_jobs=8,  # Parallel workers
    )

    # Results is a dict: {"molecule": {target_id: {source_id: distance}}}
    print("\nDistance Matrix:")
    for target_id, distances in results.get("molecule", {}).items():
        print(f"\n  Target: {target_id}")
        for source_id, dist in distances.items():
            print(f"    <- {source_id}: {dist:.4f}")

    print("\nResults saved to output/molecule_distances.csv")
    return results


def example_with_config():
    """Run pipeline from a YAML config file."""
    from themap import run_pipeline

    # Create a sample config
    config_content = """
# THEMAP Pipeline Configuration
data:
  directory: "datasets"
  task_list: null  # Auto-discover all files

molecule:
  enabled: true
  featurizer: "ecfp"    # Options: ecfp, maccs, desc2D, etc.
  method: "euclidean"   # Options: euclidean, cosine, otdd

protein:
  enabled: false        # Set to true if you have protein data

output:
  directory: "output"
  format: "csv"
  save_features: true   # Cache features for reuse

compute:
  n_jobs: 8
  device: "auto"        # auto, cpu, or cuda
"""
    # Save config
    config_path = Path("config.yaml")
    config_path.write_text(config_content)
    print(f"Created config file: {config_path}")

    # Run pipeline
    results = run_pipeline(str(config_path))
    return results


def example_programmatic():
    """Full programmatic control over the pipeline."""
    from themap import Pipeline, PipelineConfig
    from themap.config import (
        ComputeConfig,
        DataConfig,
        MoleculeDistanceConfig,
        OutputConfig,
    )

    # Build configuration programmatically
    config = PipelineConfig(
        data=DataConfig(
            directory=Path("datasets"),
            task_list=None,  # Auto-discover
        ),
        molecule=MoleculeDistanceConfig(
            enabled=True,
            featurizer="ecfp",
            method="euclidean",
        ),
        output=OutputConfig(
            directory=Path("output"),
            format="csv",
            save_features=True,
        ),
        compute=ComputeConfig(
            n_jobs=8,
            device="auto",
        ),
    )

    # Create and run pipeline
    pipeline = Pipeline(config)
    results = pipeline.run()

    return results


def example_analyzing_results():
    """Show how to analyze the distance results."""
    import pandas as pd

    # Load results from CSV
    distances = pd.read_csv("output/molecule_distances.csv", index_col=0)

    print("\nDistance Matrix Shape:", distances.shape)
    print("  Rows (sources):", list(distances.index))
    print("  Columns (targets):", list(distances.columns))

    # Find closest source for each target
    print("\nClosest source for each target:")
    for target in distances.columns:
        closest = distances[target].idxmin()
        dist = distances[target].min()
        print(f"  {target} <- {closest} (distance: {dist:.4f})")

    # Compute task hardness (average distance to k-nearest sources)
    k = 3
    print(f"\nTask Hardness (avg distance to {k}-nearest sources):")
    for target in distances.columns:
        k_nearest = distances[target].nsmallest(k).mean()
        print(f"  {target}: {k_nearest:.4f}")


def main():
    parser = argparse.ArgumentParser(description="THEMAP Distance Computation")
    parser.add_argument("--data", default="datasets", help="Data directory")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--featurizer", default="ecfp", help="Molecule featurizer")
    parser.add_argument("--method", default="euclidean", help="Distance method")
    parser.add_argument("--config", help="Path to YAML config file")
    args = parser.parse_args()

    print("=" * 60)
    print("THEMAP - Molecular Distance Computation")
    print("=" * 60)

    if args.config:
        # Use config file
        from themap import run_pipeline

        print(f"\nUsing config file: {args.config}")
        results = run_pipeline(args.config)
    else:
        # Use quick_distance
        from themap import quick_distance

        print(f"\nData directory: {args.data}")
        print(f"Featurizer: {args.featurizer}")
        print(f"Method: {args.method}")

        results = quick_distance(
            data_dir=args.data,
            output_dir=args.output,
            molecule_featurizer=args.featurizer,
            molecule_method=args.method,
        )

    # Show results summary
    if "molecule" in results:
        n_pairs = sum(len(v) for v in results["molecule"].values())
        print(f"\nComputed {n_pairs} pairwise distances")
        print(f"Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
