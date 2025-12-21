#!/usr/bin/env python3
"""
MoleculeDatasets Usage Example

This example demonstrates how to use the MoleculeDatasets class for:
- Loading datasets from a directory
- Accessing task information
- Preparing data for distance computation
"""

from pathlib import Path

import numpy as np

from themap.data import DataFold
from themap.data.molecule_datasets import MoleculeDatasets
from themap.utils.logging import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Main function demonstrating MoleculeDatasets usage."""

    print("=" * 60)
    print("THEMAP MoleculeDatasets Usage Example")
    print("=" * 60)

    # Configuration
    data_dir = Path("datasets")
    task_list_file = Path("datasets/sample_tasks_list.json")

    # Step 1: Initialize MoleculeDatasets from directory
    print("\n" + "=" * 60)
    print("Step 1: Initialize MoleculeDatasets")
    print("=" * 60)

    molecule_datasets = MoleculeDatasets.from_directory(
        directory=data_dir,
        task_list_file=task_list_file,
    )

    print(f"Initialized: {molecule_datasets}")

    # Step 2: Get task information
    print("\n" + "=" * 60)
    print("Step 2: Get Task Information")
    print("=" * 60)

    train_count = molecule_datasets.get_num_fold_tasks(DataFold.TRAIN)
    test_count = molecule_datasets.get_num_fold_tasks(DataFold.TEST)
    valid_count = molecule_datasets.get_num_fold_tasks(DataFold.VALIDATION)

    print(f"Train tasks: {train_count}")
    print(f"Test tasks: {test_count}")
    print(f"Validation tasks: {valid_count}")

    # Get task names
    train_names = molecule_datasets.get_task_names(DataFold.TRAIN)
    if len(train_names) > 5:
        print(f"\nTrain task names: {train_names[:5]}...")
    else:
        print(f"\nTrain task names: {train_names}")

    # Step 3: Load a specific dataset
    print("\n" + "=" * 60)
    print("Step 3: Load a Specific Dataset")
    print("=" * 60)

    if train_names:
        task_name = train_names[0]
        dataset = molecule_datasets.load_dataset_by_name(task_name, DataFold.TRAIN)

        print(f"Loaded dataset: {dataset}")
        print(f"  Number of molecules: {len(dataset)}")
        print(f"  Positive ratio: {dataset.positive_ratio:.2%}")

        # Access dataset properties
        print(f"  First 3 SMILES: {dataset.smiles_list[:3]}")
        print(f"  Labels shape: {dataset.labels.shape}")

        # Get statistics
        stats = dataset.get_statistics()
        print(f"  Statistics: {stats}")

    # Step 4: Prepare for distance computation
    print("\n" + "=" * 60)
    print("Step 4: Prepare for Distance Computation")
    print("=" * 60)

    source_datasets, target_datasets, source_names, target_names = (
        molecule_datasets.get_datasets_for_distance_computation()
    )

    print(f"Source datasets (train): {len(source_datasets)}")
    print(f"Target datasets (test+valid): {len(target_datasets)}")
    print(f"Distance matrix dimensions: {len(source_datasets)} x {len(target_datasets)}")

    # Step 5: Example distance computation (simple prototype-based)
    print("\n" + "=" * 60)
    print("Step 5: Example Distance Computation")
    print("=" * 60)

    if source_datasets and target_datasets:
        # Use first 2 datasets for demo
        n_demo = min(2, len(source_datasets), len(target_datasets))

        print(f"Computing {n_demo}x{n_demo} distance matrix...")

        # For this demo, we'll use random features
        # In practice, you'd use FeaturizationPipeline to compute real features
        feature_dim = 100

        for i in range(n_demo):
            src = source_datasets[i]
            # Set random features for demo (normally use FeaturizationPipeline)
            src.set_features(np.random.randn(len(src), feature_dim).astype(np.float32), "random_demo")

        for j in range(n_demo):
            tgt = target_datasets[j]
            tgt.set_features(np.random.randn(len(tgt), feature_dim).astype(np.float32), "random_demo")

        # Compute distances using prototypes
        distance_matrix = np.zeros((n_demo, n_demo))

        for i in range(n_demo):
            src_pos, src_neg = source_datasets[i].get_prototype()
            for j in range(n_demo):
                tgt_pos, tgt_neg = target_datasets[j].get_prototype()

                # Distance between prototypes
                pos_dist = np.linalg.norm(src_pos - tgt_pos)
                neg_dist = np.linalg.norm(src_neg - tgt_neg)
                distance_matrix[i, j] = (pos_dist + neg_dist) / 2

        print(f"Distance matrix:\n{distance_matrix}")
        print(f"\nSource names: {source_names[:n_demo]}")
        print(f"Target names: {target_names[:n_demo]}")

    # Step 6: Get all unique SMILES
    print("\n" + "=" * 60)
    print("Step 6: Get All Unique SMILES")
    print("=" * 60)

    all_smiles = molecule_datasets.get_all_smiles()
    print(f"Total unique SMILES across all datasets: {len(all_smiles)}")

    # Step 7: Access individual molecules via datapoints
    print("\n" + "=" * 60)
    print("Step 7: Access Individual Molecules")
    print("=" * 60)

    if train_names:
        dataset = molecule_datasets.load_dataset_by_name(train_names[0], DataFold.TRAIN)
        # datapoints returns list of dicts
        if len(dataset) > 0:
            first_molecule = dataset.datapoints[0]
            print(f"First molecule (dict): {first_molecule}")
            print(f"  SMILES: {first_molecule['smiles']}")
            print(f"  Label: {first_molecule['bool_label']}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
