#!/usr/bin/env python3
"""
Example script demonstrating how to use OTDD task distance computation.

This script shows how to:
1. Load tasks from a directory
2. Compute OTDD distances between N source and M target tasks
3. Analyze the results using the built-in analysis functions
4. Compare different approaches (OTDD vs standard distance metrics)
"""

from themap.data.metadata import DataFold
from themap.data.tasks import Tasks
from themap.distance import OTDDTaskDistance


def main():
    # Replace this with your actual data directory
    data_dir = "datasets/"

    # Load tasks from directory
    print("Loading tasks from directory...")
    tasks = Tasks.from_directory(
        directory=data_dir,
        load_molecules=True,
        load_proteins=False,  # OTDD only works with molecules
        load_metadata=False,
        cache_dir="./cache",
    )

    print(f"Loaded tasks: {tasks}")

    # Example 1: Quick OTDD computation using convenience function
    print("\n=== Example 1: Quick OTDD computation ===")

    try:
        matrix, source_names, target_names, calculator = compute_task_distance_matrix(
            tasks=tasks,
            distance_type="otdd",
            source_fold=DataFold.TRAIN,
            target_folds=[DataFold.VALIDATION, DataFold.TEST],
            chunk_size=100,  # Smaller chunks for OTDD
            n_jobs=4,  # Use 4 parallel jobs
            maxsamples=500,  # Limit samples for faster computation
            cache_dir="./cache",
            force_recompute=False,
        )

        print(f"OTDD distance matrix shape: {matrix.shape}")
        print(f"Number of source tasks: {len(source_names)}")
        print(f"Number of target tasks: {len(target_names)}")
        print(f"Distance range: {matrix.min():.4f} - {matrix.max():.4f}")

        # Analyze task hardness using OTDD distances
        hardness = calculator.compute_task_hardness(
            distance_matrix=matrix, target_names=target_names, k=5, aggregation="mean"
        )

        print("\nTop 5 hardest tasks (highest average OTDD distance):")
        sorted_hardness = sorted(hardness.items(), key=lambda x: x[1], reverse=True)
        for task_name, hardness_score in sorted_hardness[:5]:
            print(f"  {task_name}: {hardness_score:.4f}")

        # Find nearest neighbors using OTDD
        nearest = calculator.get_k_nearest_tasks(
            distance_matrix=matrix, source_names=source_names, target_names=target_names, k=3
        )

        print("\nNearest source tasks for first 3 target tasks:")
        for i, target_name in enumerate(target_names[:3]):
            print(f"  {target_name}:")
            for source_name, distance in nearest[target_name]:
                print(f"    {source_name}: {distance:.4f}")

    except Exception as e:
        print(f"Error in OTDD computation: {e}")
        print("This might be due to missing OTDD dependencies or insufficient data")

    # Example 2: Advanced OTDD computation with custom settings
    print("\n=== Example 2: Advanced OTDD computation ===")

    try:
        # Create OTDD calculator with custom settings
        otdd_calc = OTDDTaskDistance(
            tasks=tasks,
            maxsamples=1000,  # Use more samples for better accuracy
            chunk_size=50,  # Smaller chunks for memory efficiency
            n_jobs=2,  # Conservative parallelization
            cache_dir="./cache",
        )

        # Compute distances between training and validation sets only
        matrix, source_names, target_names = otdd_calc.compute_distance_matrix(
            source_fold=DataFold.TRAIN,
            target_folds=[DataFold.VALIDATION],
            force_recompute=False,
            save_cache=True,
        )

        print(f"Advanced OTDD matrix shape: {matrix.shape}")
        print(f"Cache info: {otdd_calc.get_cache_info()}")

    except Exception as e:
        print(f"Error in advanced OTDD computation: {e}")

    # Example 3: Compare OTDD with standard molecular features
    print("\n=== Example 3: Comparison with standard molecular features ===")

    try:
        # Compute using standard molecular features (Morgan fingerprints + Euclidean)
        matrix_std, source_names_std, target_names_std, calc_std = compute_task_distance_matrix(
            tasks=tasks,
            distance_type="molecule",
            molecule_featurizer="morgan_fingerprints",
            distance_metric="euclidean",
            source_fold=DataFold.TRAIN,
            target_folds=[DataFold.VALIDATION],
            cache_dir="./cache",
        )

        print(f"Standard molecular distance matrix shape: {matrix_std.shape}")
        print(f"Standard distance range: {matrix_std.min():.4f} - {matrix_std.max():.4f}")

        # Compare task rankings
        if matrix.shape == matrix_std.shape:
            from scipy.stats import spearmanr

            # Flatten matrices and compute correlation
            otdd_flat = matrix.flatten()
            std_flat = matrix_std.flatten()

            correlation, p_value = spearmanr(otdd_flat, std_flat)
            print(
                f"Spearman correlation between OTDD and standard distances: {correlation:.4f} (p={p_value:.4e})"
            )

    except Exception as e:
        print(f"Error in comparison: {e}")


if __name__ == "__main__":
    main()
