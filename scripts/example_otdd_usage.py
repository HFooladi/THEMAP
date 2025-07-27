#!/usr/bin/env python3
"""
Example script demonstrating how to use task distance computation with the tasks_distance module.

This script shows how to:
1. Load tasks from a directory
2. Compute various types of distances between tasks (OTDD, Euclidean, Cosine)
3. Analyze the results using the built-in analysis functions
4. Compare different approaches (molecule vs protein vs combined distances)
"""

import numpy as np

from themap.data.tasks import Tasks
from themap.distance import MoleculeDatasetDistance, ProteinDatasetDistance, TaskDistance
from themap.utils.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    # Replace this with your actual data directory
    data_dir = "datasets/"
    sample_tasks_list = "datasets/sample_tasks_list.json"

    # Load tasks from directory
    logger.info("Loading tasks from directory...")
    tasks = Tasks.from_directory(
        directory=data_dir,
        load_molecules=True,
        load_proteins=True,  # Also load proteins for comprehensive comparison
        load_metadata=False,
        cache_dir="./cache",
        task_list_file=sample_tasks_list,
    )

    logger.info(f"Loaded tasks: {tasks}")

    # Example 1: OTDD distance computation using MoleculeDatasetDistance
    logger.info("\n=== Example 1: OTDD distance computation ===")

    try:
        # Create OTDD calculator for molecule datasets
        otdd_calc = MoleculeDatasetDistance(tasks=tasks, molecule_method="otdd")

        # Compute OTDD distances
        otdd_distances = otdd_calc.get_distance()

        if otdd_distances:
            print("OTDD distances computed successfully")
            print(f"Number of target tasks: {len(otdd_distances)}")

            # Calculate matrix statistics
            all_distances = []
            for target_id, source_distances in otdd_distances.items():
                print(f"Target {target_id}: {len(source_distances)} source tasks")
                all_distances.extend(source_distances.values())

            if all_distances:
                print(f"Distance range: {min(all_distances):.4f} - {max(all_distances):.4f}")
                print(f"Mean distance: {np.mean(all_distances):.4f}")

        else:
            print("No OTDD distances computed - check if molecule data is available")

    except Exception as e:
        print(f"Error in OTDD computation: {e}")
        print("This might be due to missing OTDD dependencies or insufficient data")

    # Example 2: Protein distance computation
    print("\n=== Example 2: Protein distance computation ===")

    try:
        # Create protein distance calculator
        protein_calc = ProteinDatasetDistance(tasks=tasks, protein_method="euclidean")

        protein_distances = protein_calc.get_distance()

        if protein_distances:
            print("Protein distances computed successfully")
            all_distances = []
            for target_id, source_distances in protein_distances.items():
                all_distances.extend(source_distances.values())

            if all_distances:
                print(f"Distance range: {min(all_distances):.4f} - {max(all_distances):.4f}")
                print(f"Mean distance: {np.mean(all_distances):.4f}")

        else:
            print("No protein distances computed - check if protein data is available")

    except Exception as e:
        print(f"Error in protein distance computation: {e}")

    # Example 3: Combined task distance computation
    print("\n=== Example 3: Combined task distance computation ===")

    try:
        # Create unified task distance calculator
        task_calc = TaskDistance(
            tasks=tasks,
            molecule_method="otdd",  # Use OTDD for faster computation
            protein_method="euclidean",
        )

        # Compute all distance types
        all_distances = task_calc.compute_all_distances(
            molecule_method="otdd", protein_method="euclidean", combination_strategy="average"
        )

        print(f"Available distance types: {list(all_distances.keys())}")

        for distance_type, distances in all_distances.items():
            if distances:
                all_vals = []
                for target_id, source_distances in distances.items():
                    all_vals.extend(source_distances.values())

                if all_vals:
                    print(f"{distance_type.title()} distances:")
                    print(f"  Range: {min(all_vals):.4f} - {max(all_vals):.4f}")
                    print(f"  Mean: {np.mean(all_vals):.4f}")

    except Exception as e:
        print(f"Error in combined distance computation: {e}")

    # Example 5: Task analysis using computed distances
    print("\n=== Example 5: Task analysis ===")

    try:
        # Use the TaskDistance for comprehensive analysis
        task_calc = TaskDistance(tasks=tasks, molecule_method="otdd", protein_method="euclidean")

        # Get the default distance (will try combined, then molecule, then protein)
        distances = task_calc.get_distance()

        if distances:
            print(f"Analysis using {len(distances)} target tasks")

            # Simple task hardness analysis
            hardness_scores = {}
            for target_id, source_distances in distances.items():
                if source_distances:
                    # Task hardness = average distance to all source tasks
                    hardness_scores[target_id] = np.mean(list(source_distances.values()))

            if hardness_scores:
                print("\nTop 5 most challenging tasks (highest average distance):")
                sorted_hardness = sorted(hardness_scores.items(), key=lambda x: x[1], reverse=True)
                for i, (task_name, hardness_score) in enumerate(sorted_hardness[:5]):
                    print(f"  {i + 1}. {task_name}: {hardness_score:.4f}")

                # Simple nearest neighbor analysis
                print("\nNearest source tasks for first 3 target tasks:")
                for i, (target_id, source_distances) in enumerate(list(distances.items())[:3]):
                    print(f"  {target_id}:")
                    # Sort by distance and show top 3 nearest
                    sorted_sources = sorted(source_distances.items(), key=lambda x: x[1])
                    for j, (source_id, distance) in enumerate(sorted_sources[:3]):
                        print(f"    {j + 1}. {source_id}: {distance:.4f}")

    except Exception as e:
        print(f"Error in task analysis: {e}")

    # Example 6: Convert to pandas DataFrame for further analysis
    print("\n=== Example 6: DataFrame conversion ===")

    try:
        task_calc = TaskDistance(tasks=tasks, molecule_method="otdd")

        # Compute molecule distances
        task_calc.compute_molecule_distance()

        # Convert to pandas DataFrame
        df = task_calc.to_pandas(distance_type="molecule")

        if df is not None and not df.empty:
            print(f"Distance matrix shape: {df.shape}")
            print("Sample of the distance matrix:")
            print(df.iloc[: min(5, len(df)), : min(5, len(df.columns))])
        else:
            print("Could not create distance matrix DataFrame")

    except Exception as e:
        print(f"Error in DataFrame conversion: {e}")


if __name__ == "__main__":
    main()
