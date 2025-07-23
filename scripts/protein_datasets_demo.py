#!/usr/bin/env python3
"""Comprehensive demonstration of enhanced ProteinDatasets functionality.

This script demonstrates the complete workflow:
1. Download FASTA files from CHEMBL task lists
2. Load protein datasets with caching
3. Compute features with deduplication
4. Save/load precomputed features
5. Compute distance matrices for task hardness analysis

Usage:
    python scripts/protein_datasets_demo.py
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add repository root to path
REPO_PATH = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(REPO_PATH))

import numpy as np

from themap.data.protein_datasets import DataFold, ProteinDatasets
from themap.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Complete demonstration of ProteinDatasets functionality."""

    print("=" * 60)
    print("🧬 THEMAP ProteinDatasets Comprehensive Demo")
    print("=" * 60)

    # Configuration
    task_list_file = REPO_PATH / "datasets" / "sample_tasks_list.json"
    output_dir = REPO_PATH / "datasets"
    uniprot_mapping_file = REPO_PATH / "datasets" / "uniprot_mapping.csv"
    features_cache_file = REPO_PATH / "datasets" / "protein_features_demo.pkl"

    print(f"📋 Task list: {task_list_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🗂️  UniProt mapping: {uniprot_mapping_file}")

    # Step 1: Create FASTA files from CHEMBL task list
    print("\n" + "=" * 50)
    print("📥 STEP 1: Creating FASTA files from task list")
    print("=" * 50)

    if not all(
        [
            (output_dir / "train").exists() and len(list((output_dir / "train").glob("*.fasta"))) > 1,
            (output_dir / "test").exists() and len(list((output_dir / "test").glob("*.fasta"))) > 1,
        ]
    ):
        print("🔄 Downloading FASTA files for all tasks...")
        protein_datasets = ProteinDatasets.create_fasta_files_from_task_list(
            task_list_file=task_list_file, output_dir=output_dir, uniprot_mapping_file=uniprot_mapping_file
        )
        print(f"✅ Downloaded FASTA files: {protein_datasets}")
    else:
        print("✅ FASTA files already exist")

    # Step 2: Load datasets from directory
    print("\n" + "=" * 50)
    print("📂 STEP 2: Loading protein datasets")
    print("=" * 50)

    protein_datasets = ProteinDatasets.from_directory(
        directory=str(output_dir),
        task_list_file=str(task_list_file),
        uniprot_mapping_file=str(uniprot_mapping_file),
        cache_dir=output_dir / "cache",
    )

    print(f"✅ Loaded: {protein_datasets}")
    print(f"   🎯 Train tasks: {protein_datasets.get_num_fold_tasks(DataFold.TRAIN)}")
    print(f"   🎯 Test tasks: {protein_datasets.get_num_fold_tasks(DataFold.TEST)}")

    # Show task names
    train_names = protein_datasets.get_task_names(DataFold.TRAIN)
    test_names = protein_datasets.get_task_names(DataFold.TEST)
    print(f"   📋 Train tasks: {train_names[:5]}{'...' if len(train_names) > 5 else ''}")
    print(f"   📋 Test tasks: {test_names}")

    # Step 3: Load individual protein datasets
    print("\n" + "=" * 50)
    print("🔄 STEP 3: Loading individual protein datasets")
    print("=" * 50)

    start_time = time.time()
    datasets = protein_datasets.load_datasets([DataFold.TRAIN, DataFold.TEST])
    load_time = time.time() - start_time

    print(f"✅ Loaded {len(datasets)} protein datasets in {load_time:.2f}s")

    # Show sample datasets
    print("📊 Sample datasets:")
    for name, dataset in list(datasets.items())[:5]:
        print(f"   {name}: {dataset}")

    # Step 4: Demonstrate deduplication analysis
    print("\n" + "=" * 50)
    print("🔍 STEP 4: Protein deduplication analysis")
    print("=" * 50)

    # Analyze protein uniqueness
    unique_proteins: Dict[str, Any] = {}
    protein_usage: Dict[str, List[str]] = {}

    for dataset_name, dataset in datasets.items():
        uniprot_id = dataset.uniprot_id
        if uniprot_id not in unique_proteins:
            unique_proteins[uniprot_id] = {
                "sequence_length": len(dataset.sequence),
                "sequence": dataset.sequence[:50] + "..." if len(dataset.sequence) > 50 else dataset.sequence,
            }
            protein_usage[uniprot_id] = []
        protein_usage[uniprot_id].append(dataset_name)

    total_datasets = len(datasets)
    unique_count = len(unique_proteins)
    dedup_savings = ((total_datasets - unique_count) / total_datasets * 100) if total_datasets > 0 else 0

    print("📊 Deduplication analysis:")
    print(f"   Total datasets: {total_datasets}")
    print(f"   Unique proteins: {unique_count}")
    print(f"   Potential savings: {dedup_savings:.1f}%")

    print("\n🧬 Unique proteins found:")
    for uniprot_id, info in list(unique_proteins.items())[:5]:
        usage_count = len(protein_usage[uniprot_id])
        print(f"   {uniprot_id}: {info['sequence_length']} aa, used by {usage_count} dataset(s)")

    # Step 5: Mock feature computation (replace with real ESM when needed)
    print("\n" + "=" * 50)
    print("🧠 STEP 5: Protein feature computation (MOCK)")
    print("=" * 50)

    print("ℹ️  Using mock features for demonstration (replace with real ESM model)")

    # Create mock features for demonstration
    feature_dim = 512
    mock_features = {}

    print("🎭 Creating mock protein features...")
    start_time = time.time()

    for dataset_name, dataset in datasets.items():
        # Create reproducible mock features based on sequence
        np.random.seed(hash(dataset.sequence) % 2**32)
        mock_features[dataset_name] = np.random.randn(feature_dim).astype(np.float32)
        dataset.features = mock_features[dataset_name]

    compute_time = time.time() - start_time
    print(f"✅ Computed features for {len(mock_features)} datasets in {compute_time:.2f}s")

    # Step 6: Save features
    print("\n" + "=" * 50)
    print("💾 STEP 6: Saving computed features")
    print("=" * 50)

    # Simulate the save functionality
    save_data = {
        "features": mock_features,
        "featurizer_name": "mock_esm2_t33_650M_UR50D",
        "layer": 33,
        "timestamp": time.time(),
        "num_datasets": len(mock_features),
        "train_task_ids": [
            name.replace("train_", "") for name in mock_features.keys() if name.startswith("train_")
        ],
        "test_task_ids": [
            name.replace("test_", "") for name in mock_features.keys() if name.startswith("test_")
        ],
    }

    import pickle

    with open(features_cache_file, "wb") as f:
        pickle.dump(save_data, f)

    file_size = features_cache_file.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Saved features to {features_cache_file}")
    print(f"   📊 File size: {file_size:.2f} MB")
    print(f"   🎯 Datasets: {save_data['num_datasets']}")
    print(f"   🏋️ Feature dimension: {feature_dim}")

    # Step 7: Load features and verify
    print("\n" + "=" * 50)
    print("📤 STEP 7: Loading and verifying features")
    print("=" * 50)

    loaded_features = ProteinDatasets.load_features_from_file(features_cache_file)

    print(f"✅ Loaded features for {len(loaded_features)} datasets")

    # Verify data integrity
    verification_passed = 0
    for name, original_features in list(mock_features.items())[:3]:
        if name in loaded_features:
            if np.allclose(original_features, loaded_features[name]):
                verification_passed += 1

    print(f"🔍 Verification: {verification_passed}/3 samples match exactly")

    # Step 8: Distance matrix computation
    print("\n" + "=" * 50)
    print("📏 STEP 8: Distance matrix computation")
    print("=" * 50)

    # Organize features for distance computation
    source_features = []
    source_names = []
    target_features = []
    target_names = []

    for dataset_name, features in loaded_features.items():
        if dataset_name.startswith("train_"):
            source_features.append(features)
            source_names.append(dataset_name)
        elif dataset_name.startswith("test_"):
            target_features.append(features)
            target_names.append(dataset_name)

    print("📊 Distance computation setup:")
    print(f"   Source (train) proteins: {len(source_features)}")
    print(f"   Target (test) proteins: {len(target_features)}")
    print(f"   Distance matrix size: {len(source_features)}×{len(target_features)}")

    if source_features and target_features:
        from scipy.spatial.distance import cdist

        # Compute distance matrix
        print("🔢 Computing distance matrix...")
        start_time = time.time()

        source_matrix = np.stack(source_features)
        target_matrix = np.stack(target_features)
        distance_matrix = cdist(source_matrix, target_matrix, metric="euclidean")

        compute_time = time.time() - start_time

        print(f"✅ Distance matrix computed in {compute_time:.3f}s")
        print(f"   📊 Shape: {distance_matrix.shape}")
        print(f"   📈 Distance range: {distance_matrix.min():.3f} - {distance_matrix.max():.3f}")
        print(f"   📍 Mean distance: {distance_matrix.mean():.3f}")

        # Show some example distances
        print("\n🔍 Example protein distances:")
        for i, source_name in enumerate(source_names[:3]):
            for j, target_name in enumerate(target_names):
                distance = distance_matrix[i, j]
                source_task = source_name.replace("train_", "")
                target_task = target_name.replace("test_", "")
                print(f"   {source_task} → {target_task}: {distance:.3f}")

        # Step 9: Task hardness simulation
        print("\n" + "=" * 50)
        print("🎯 STEP 9: Task hardness analysis")
        print("=" * 50)

        # Simulate task hardness calculation (k-nearest neighbors)
        k = min(3, len(source_features))  # Use k=3 or fewer if not enough source tasks

        print(f"📊 Computing task hardness using {k}-nearest neighbors...")

        # For each target task, find k nearest source tasks
        task_hardness = {}
        for j, target_name in enumerate(target_names):
            distances_to_target = distance_matrix[:, j]
            k_nearest_distances = np.partition(distances_to_target, k)[:k]
            hardness = np.mean(k_nearest_distances)

            task_id = target_name.replace("test_", "")
            task_hardness[task_id] = hardness

            print(f"   {task_id}: hardness = {hardness:.3f}")

        # Normalize hardness scores
        hardness_values = list(task_hardness.values())
        min_hardness = min(hardness_values)
        max_hardness = max(hardness_values)

        print("\n📈 Hardness statistics:")
        print(f"   Range: {min_hardness:.3f} - {max_hardness:.3f}")
        print(f"   Mean: {np.mean(hardness_values):.3f}")
        print(f"   Std: {np.std(hardness_values):.3f}")

    # Step 10: Cleanup and summary
    print("\n" + "=" * 50)
    print("🧹 STEP 10: Cleanup and summary")
    print("=" * 50)

    # Clean up demo files
    if features_cache_file.exists():
        features_cache_file.unlink()
        print("✅ Cleaned up demo cache file")

    print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("📋 Summary of capabilities demonstrated:")
    print("   ✅ FASTA file downloading from CHEMBL task lists")
    print("   ✅ Protein dataset loading and management")
    print("   ✅ UniProt ID deduplication analysis")
    print("   ✅ Feature computation infrastructure")
    print("   ✅ Feature caching and persistence")
    print("   ✅ Distance matrix computation")
    print("   ✅ Task hardness analysis workflow")

    print("\n🚀 Ready for production use with real ESM models!")
    print("   Replace mock features with: protein_datasets.compute_all_features_with_deduplication()")
    print("   Use models like: 'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', etc.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
