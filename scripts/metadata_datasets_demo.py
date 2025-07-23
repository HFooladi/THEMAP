#!/usr/bin/env python3
"""
Demo script showing how to use the MetadataDatasets blueprint system.

This script demonstrates:
1. How to create different types of metadata datasets (text, numerical, categorical)
2. How to load and featurize metadata
3. How to prepare features for distance computation
4. How to extend the system for custom metadata types

Run from project root:
    python scripts/metadata_datasets_demo.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from themap.data.metadata import (
    CategoricalMetadataDataset,
    CustomMetadataDataset,
    MetadataDatasets,
    NumericalMetadataDataset,
    TextMetadataDataset,
)
from themap.utils.logging import get_logger

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = get_logger(__name__)


def create_sample_metadata_files(temp_dir: Path) -> None:
    """Create sample metadata files for demonstration."""
    logger.info("Creating sample metadata files...")

    # Sample task IDs (matching our protein dataset tasks)
    train_tasks = ["CHEMBL2219236", "CHEMBL3991889", "CHEMBL4439485"]
    test_tasks = ["CHEMBL235", "CHEMBL244", "CHEMBL264"]

    # Create directory structure
    for fold in ["train", "test"]:
        (temp_dir / fold).mkdir(parents=True, exist_ok=True)

    # 1. Text metadata: Assay descriptions
    assay_descriptions = {
        "CHEMBL2219236": "High-throughput binding assay for protein kinase inhibition",
        "CHEMBL3991889": "Cell-based viability assay measuring cytotoxic effects",
        "CHEMBL4439485": "Fluorescence polarization assay for protein-protein interactions",
        "CHEMBL235": "Enzymatic activity assay using colorimetric detection",
        "CHEMBL244": "Radioligand binding competition assay",
        "CHEMBL264": "Calcium flux assay for GPCR activation",
    }

    for task_id, description in assay_descriptions.items():
        fold = "train" if task_id in train_tasks else "test"
        file_path = temp_dir / fold / f"{task_id}_assay.json"
        with open(file_path, "w") as f:
            json.dump({"description": description, "assay_type": "binding"}, f)

    # 2. Numerical metadata: Bioactivity values
    bioactivity_data = {
        "CHEMBL2219236": [7.5, 8.2, 6.9],  # IC50 values in -log10(M)
        "CHEMBL3991889": [6.8, 7.1, 6.5],
        "CHEMBL4439485": [8.9, 9.2, 8.7],
        "CHEMBL235": [5.5, 5.8, 5.2],
        "CHEMBL244": [7.2, 7.5, 6.9],
        "CHEMBL264": [6.1, 6.4, 5.9],
    }

    for task_id, values in bioactivity_data.items():
        fold = "train" if task_id in train_tasks else "test"
        file_path = temp_dir / fold / f"{task_id}_bioactivity.json"
        with open(file_path, "w") as f:
            json.dump({"ic50_values": values, "unit": "nM"}, f)

    # 3. Categorical metadata: Target organisms and assay types
    target_info = {
        "CHEMBL2219236": {"organism": "Homo sapiens", "target_class": "Kinase"},
        "CHEMBL3991889": {"organism": "Homo sapiens", "target_class": "GPCR"},
        "CHEMBL4439485": {"organism": "Rattus norvegicus", "target_class": "Enzyme"},
        "CHEMBL235": {"organism": "Homo sapiens", "target_class": "Ion Channel"},
        "CHEMBL244": {"organism": "Mus musculus", "target_class": "Kinase"},
        "CHEMBL264": {"organism": "Homo sapiens", "target_class": "GPCR"},
    }

    for task_id, info in target_info.items():
        fold = "train" if task_id in train_tasks else "test"
        file_path = temp_dir / fold / f"{task_id}_target.json"
        with open(file_path, "w") as f:
            json.dump(info, f)

    logger.info(f"Created metadata files in {temp_dir}")


def demo_text_metadata(temp_dir: Path) -> None:
    """Demonstrate text metadata handling."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 1: Text Metadata (Assay Descriptions)")
    logger.info("=" * 60)

    # Create text metadata datasets
    text_datasets = MetadataDatasets.from_directory(
        directory=str(temp_dir),
        metadata_type="assay_description",
        dataset_class=TextMetadataDataset,
        file_pattern="*_assay.json",
    )

    logger.info(f"Text metadata datasets: {text_datasets}")

    # Load datasets
    datasets = text_datasets.load_datasets()
    logger.info(f"Loaded {len(datasets)} text metadata datasets")

    # Show sample dataset
    sample_name = list(datasets.keys())[0]
    sample_dataset = datasets[sample_name]
    logger.info(f"Sample dataset: {sample_dataset}")
    logger.info(f"Raw data: {sample_dataset.raw_data}")

    # Compute features
    logger.info("Computing text features...")
    all_features = text_datasets.compute_all_features_with_deduplication(
        featurizer_name="sentence-transformers", model_name="all-MiniLM-L6-v2"
    )

    for name, features in list(all_features.items())[:2]:
        logger.info(f"{name}: features shape {features.shape}")

    # Prepare for distance computation
    source_features, target_features, source_names, target_names = (
        text_datasets.get_distance_computation_ready_features(featurizer_name="sentence-transformers")
    )

    logger.info(f"Distance computation ready: {len(source_features)} source × {len(target_features)} target")


def demo_numerical_metadata(temp_dir: Path) -> None:
    """Demonstrate numerical metadata handling."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Numerical Metadata (Bioactivity Values)")
    logger.info("=" * 60)

    # Custom dataset class for bioactivity data
    class BioactivityDataset(NumericalMetadataDataset):
        def preprocess_data(self) -> np.ndarray:
            # Extract IC50 values from JSON
            if isinstance(self.raw_data, dict) and "ic50_values" in self.raw_data:
                values = self.raw_data["ic50_values"]
                return np.array(values, dtype=np.float32)
            return super().preprocess_data()

    # Create numerical metadata datasets
    numerical_datasets = MetadataDatasets.from_directory(
        directory=str(temp_dir),
        metadata_type="bioactivity",
        dataset_class=BioactivityDataset,
        file_pattern="*_bioactivity.json",
    )

    logger.info(f"Numerical metadata datasets: {numerical_datasets}")

    # Load and show sample
    datasets = numerical_datasets.load_datasets()
    sample_name = list(datasets.keys())[0]
    sample_dataset = datasets[sample_name]
    logger.info(f"Sample dataset: {sample_dataset}")
    logger.info(f"Raw data: {sample_dataset.raw_data}")
    logger.info(f"Preprocessed: {sample_dataset.preprocess_data()}")

    # Compute features with different normalizations
    for featurizer in ["standardize", "normalize", "log_transform"]:
        logger.info(f"\nComputing features with {featurizer}...")
        all_features = numerical_datasets.compute_all_features_with_deduplication(featurizer_name=featurizer)

        for name, features in list(all_features.items())[:2]:
            logger.info(f"{name}: {features} (shape: {features.shape})")


def demo_categorical_metadata(temp_dir: Path) -> None:
    """Demonstrate categorical metadata handling."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Categorical Metadata (Target Information)")
    logger.info("=" * 60)

    # Custom dataset class for target information
    class TargetInfoDataset(CategoricalMetadataDataset):
        def preprocess_data(self) -> list[str]:
            # Extract organism and target class from JSON
            if isinstance(self.raw_data, dict):
                organism = self.raw_data.get("organism", "unknown")
                target_class = self.raw_data.get("target_class", "unknown")
                return [organism.lower(), target_class.lower()]
            return super().preprocess_data()

    # Create categorical metadata datasets
    categorical_datasets = MetadataDatasets.from_directory(
        directory=str(temp_dir),
        metadata_type="target_info",
        dataset_class=TargetInfoDataset,
        file_pattern="*_target.json",
    )

    logger.info(f"Categorical metadata datasets: {categorical_datasets}")

    # Load and show sample
    datasets = categorical_datasets.load_datasets()
    sample_name = list(datasets.keys())[0]
    sample_dataset = datasets[sample_name]
    logger.info(f"Sample dataset: {sample_dataset}")
    logger.info(f"Raw data: {sample_dataset.raw_data}")
    logger.info(f"Preprocessed: {sample_dataset.preprocess_data()}")

    # Create vocabulary from all datasets
    all_categories = set()
    for dataset in datasets.values():
        all_categories.update(dataset.preprocess_data())
    vocabulary = sorted(list(all_categories))
    logger.info(f"Vocabulary: {vocabulary}")

    # Compute one-hot features
    logger.info("Computing one-hot features...")
    all_features = categorical_datasets.compute_all_features_with_deduplication(
        featurizer_name="one_hot", vocabulary=vocabulary
    )

    for name, features in list(all_features.items())[:2]:
        logger.info(f"{name}: {features} (shape: {features.shape})")


def demo_convenience_functions() -> None:
    """Demonstrate convenience functions for common metadata types."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Convenience Functions")
    logger.info("=" * 60)

    # These would work if you have actual metadata directories
    logger.info("Example convenience function calls:")
    logger.info("1. create_assay_description_datasets(directory, task_list_file)")
    logger.info("2. create_bioactivity_datasets(directory, task_list_file)")
    logger.info("3. create_target_organism_datasets(directory, task_list_file)")

    # Show how to save/load features
    logger.info("\nFeature persistence:")
    logger.info("- datasets.save_features_to_file(path, featurizer_name)")
    logger.info("- MetadataDatasets.load_features_from_file(path)")


def demo_custom_metadata_type() -> None:
    """Demonstrate how to create custom metadata types."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 5: Custom Metadata Type Extension")
    logger.info("=" * 60)

    # Example of creating a custom metadata dataset
    custom_dataset = CustomMetadataDataset(
        task_id="CHEMBL123",
        metadata_type="custom_complex_data",
        raw_data={"complex_field": [1, 2, 3], "text": "some description"},
    )

    logger.info(f"Custom dataset: {custom_dataset}")

    # Show how features would be computed
    logger.info("Custom featurization would extract features from complex data structure")
    logger.info("Extend the CustomMetadataDataset.get_features() method for your specific needs")


def demo_distance_computation_workflow(temp_dir: Path) -> None:
    """Demonstrate the complete workflow for distance computation."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 6: Complete Distance Computation Workflow")
    logger.info("=" * 60)

    # Load text metadata
    text_datasets = MetadataDatasets.from_directory(
        directory=str(temp_dir),
        metadata_type="assay_description",
        dataset_class=TextMetadataDataset,
        file_pattern="*_assay.json",
    )

    # Get features ready for distance computation
    source_features, target_features, source_names, target_names = (
        text_datasets.get_distance_computation_ready_features(featurizer_name="sentence-transformers")
    )

    logger.info(f"Source datasets: {source_names}")
    logger.info(f"Target datasets: {target_names}")

    # Compute distance matrix (example)
    if source_features and target_features:
        logger.info("Computing pairwise distances...")

        # Stack features for vectorized computation
        source_matrix = np.stack(source_features)  # (N, D)
        target_matrix = np.stack(target_features)  # (M, D)

        # Compute cosine distances
        from sklearn.metrics.pairwise import cosine_distances

        distance_matrix = cosine_distances(source_matrix, target_matrix)

        logger.info(f"Distance matrix shape: {distance_matrix.shape}")
        logger.info(f"Sample distances:\n{distance_matrix}")

        # Find most similar pairs
        min_indices = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        logger.info(f"Most similar pair: {source_names[min_indices[0]]} <-> {target_names[min_indices[1]]}")
        logger.info(f"Distance: {distance_matrix[min_indices]:.4f}")


def main():
    """Run all metadata demos."""
    logger.info("Starting MetadataDatasets Blueprint Demo")

    # Create temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample metadata files
        create_sample_metadata_files(temp_path)

        # Run demonstrations
        demo_text_metadata(temp_path)
        demo_numerical_metadata(temp_path)
        demo_categorical_metadata(temp_path)
        demo_convenience_functions()
        demo_custom_metadata_type()
        demo_distance_computation_workflow(temp_path)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: How to Use the Metadata Blueprint")
    logger.info("=" * 60)

    summary = """
1. Choose the appropriate dataset class:
   - TextMetadataDataset: For text descriptions, assay protocols, etc.
   - NumericalMetadataDataset: For IC50 values, molecular weights, etc.
   - CategoricalMetadataDataset: For assay types, organisms, etc.
   - CustomMetadataDataset: For complex/specialized metadata

2. Create MetadataDatasets using from_directory():
   - Specify metadata_type and dataset_class
   - Use file_pattern to match your files
   - Optionally filter by task_list_file

3. Load and featurize:
   - load_datasets() to load metadata from files
   - compute_all_features_with_deduplication() for efficient feature computation
   - Use appropriate featurizer for your data type

4. Prepare for distance computation:
   - get_distance_computation_ready_features() organizes features by fold
   - Returns source/target features ready for N×M distance matrix

5. Save/load for persistence:
   - save_features_to_file() and load_features_from_file()

6. Extend for new metadata types:
   - Subclass BaseMetadataDataset
   - Implement preprocess_data() and get_features()
   - Use with MetadataDatasets class
    """

    logger.info(summary)


if __name__ == "__main__":
    main()
