#!/usr/bin/env python3
"""Example script demonstrating the enhanced ProteinDatasets functionality.

This script shows how to:
1. Create FASTA files from a task list (CHEMBLID -> UNIPROT -> FASTA)
2. Load protein datasets from directories
3. Compute protein features with caching and deduplication
4. Save and load precomputed features
5. Get distance computation ready features

Usage:
    python scripts/protein_features_example.py
"""

import sys
from pathlib import Path

from themap.data.protein_datasets import DataFold, ProteinMetadataDatasets
from themap.utils.logging import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Main function demonstrating ProteinDatasets functionality."""

    logger.info("🧬 Starting protein datasets example")

    # Add repository root to path
    REPO_PATH = Path(__file__).parent.parent.parent.absolute()
    sys.path.insert(0, str(REPO_PATH))  # noqa: E402

    # Define paths
    task_list_file = REPO_PATH / "datasets" / "sample_tasks_list.json"
    output_dir = REPO_PATH / "datasets"
    uniprot_mapping_file = REPO_PATH / "datasets" / "uniprot_mapping.csv"
    features_cache_file = REPO_PATH / "datasets" / "protein_features_cache.pkl"

    # Step 1: Create FASTA files from task list (if they don't exist)
    logger.info("📥 Step 1: Creating FASTA files from task list")

    if not all([(output_dir / "train").exists(), (output_dir / "test").exists()]):
        logger.info("Creating FASTA files from task list...")
        protein_datasets = ProteinMetadataDatasets.create_fasta_files_from_task_list(
            task_list_file=task_list_file, output_dir=output_dir, uniprot_mapping_file=uniprot_mapping_file
        )
        logger.info(f"Created ProteinDatasets: {protein_datasets}")
    else:
        logger.info("FASTA directories already exist, skipping download")

    # Step 2: Load datasets from directory
    logger.info("📂 Step 2: Loading datasets from directory")

    protein_datasets = ProteinMetadataDatasets.from_directory(
        directory=output_dir,
        task_list_file=task_list_file,
        uniprot_mapping_file=uniprot_mapping_file,
        cache_dir=output_dir / "cache",
    )

    logger.info(f"Loaded: {protein_datasets}")
    logger.info(f"Train tasks: {protein_datasets.get_num_fold_tasks(DataFold.TRAIN)}")
    logger.info(f"Test tasks: {protein_datasets.get_num_fold_tasks(DataFold.TEST)}")

    # Step 3: Load individual datasets
    logger.info("🔄 Step 3: Loading individual protein datasets")

    datasets = protein_datasets.load_datasets([DataFold.TRAIN, DataFold.TEST])
    logger.info(f"Loaded {len(datasets)} individual protein datasets")

    for name, dataset in list(datasets.items())[:3]:  # Show first 3
        logger.info(f"  {name}: {dataset}")

    # Step 4: Compute features with deduplication
    logger.info("🧠 Step 4: Computing protein features with deduplication")

    # Use a smaller/faster model for demonstration
    featurizer_name = "esm2_t12_35M_UR50D"  # Smaller model for faster demo
    layer = 12

    if features_cache_file.exists():
        logger.info("Loading precomputed features from cache...")
        features = ProteinMetadataDatasets.load_features_from_file(features_cache_file)
    else:
        logger.info("Computing features (this may take a while for larger models)...")
        features = protein_datasets.compute_all_features_with_deduplication(
            featurizer_name=featurizer_name, layer=layer, folds=[DataFold.TRAIN, DataFold.TEST]
        )

        # Save features for future use
        logger.info("Saving computed features to cache...")
        protein_datasets.save_features_to_file(
            output_path=features_cache_file,
            featurizer_name=featurizer_name,
            layer=layer,
            folds=[DataFold.TRAIN, DataFold.TEST],
        )

    logger.info(f"Computed features for {len(features)} datasets")
    for name, feat in list(features.items())[:3]:  # Show first 3
        logger.info(f"  {name}: shape {feat.shape}")

    # Step 5: Get distance computation ready features
    logger.info("📏 Step 5: Preparing features for distance computation")

    source_features, target_features, source_names, target_names = (
        protein_datasets.get_distance_computation_ready_features(
            featurizer_name=featurizer_name,
            layer=layer,
            source_fold=DataFold.TRAIN,
            target_folds=[DataFold.TEST],
        )
    )

    logger.info(f"Source (train) datasets: {len(source_features)}")
    logger.info(f"Target (test) datasets: {len(target_features)}")
    logger.info(f"Ready for {len(source_features)}×{len(target_features)} distance matrix computation")

    # Step 6: Example distance computation
    logger.info("🔢 Step 6: Example distance matrix computation")

    if source_features and target_features:
        import numpy as np
        from scipy.spatial.distance import cdist

        # Stack features for distance computation
        source_matrix = np.stack(source_features)  # Shape: (n_source, feature_dim)
        target_matrix = np.stack(target_features)  # Shape: (n_target, feature_dim)

        # Compute distance matrix
        distance_matrix = cdist(source_matrix, target_matrix, metric="euclidean")

        logger.info(f"Distance matrix shape: {distance_matrix.shape}")
        logger.info(f"Distance range: {distance_matrix.min():.3f} - {distance_matrix.max():.3f}")

        # Show some example distances
        logger.info("Example distances:")
        for i, source_name in enumerate(source_names[:3]):
            for j, target_name in enumerate(target_names[:3]):
                if i < len(source_features) and j < len(target_features):
                    distance = distance_matrix[i, j]
                    logger.info(f"  {source_name} -> {target_name}: {distance:.3f}")

    # Step 7: Cache statistics
    logger.info("📊 Step 7: Cache statistics")

    cache_stats = protein_datasets.get_global_cache_stats()
    if cache_stats:
        logger.info(f"Cache statistics: {cache_stats}")
    else:
        logger.info("No cache statistics available")

    logger.info("✅ Protein datasets example completed successfully!")


if __name__ == "__main__":
    main()
