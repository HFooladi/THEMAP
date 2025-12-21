#!/usr/bin/env python3
"""
Example: Computing Molecule Distances with Different Featurizers

This example demonstrates how to compute molecular dataset distances using different
featurizers (ECFP, MACCS, descriptors, etc.) with the MoleculeDatasetDistance class.
"""

import sys
from pathlib import Path

# Add the project root to Python path
REPO_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_PATH))

from themap.data.molecule_dataset import MoleculeDataset  # noqa: E402
from themap.data.tasks import Task, Tasks  # noqa: E402
from themap.distance.molecule_distance import MoleculeDatasetDistance  # noqa: E402
from themap.utils.logging import get_logger, setup_logging  # noqa: E402

# Initialize logging
logger = get_logger(__name__)
setup_logging()


def load_sample_tasks():
    """Load sample molecular tasks for distance computation."""
    # Load some sample datasets - adjust paths to your data
    train_datasets = [
        MoleculeDataset.load_from_file("datasets/train/CHEMBL1613776.jsonl.gz"),
        MoleculeDataset.load_from_file("datasets/train/CHEMBL2219012.jsonl.gz"),
    ]

    test_datasets = [
        MoleculeDataset.load_from_file("datasets/test/CHEMBL2219236.jsonl.gz"),
        MoleculeDataset.load_from_file("datasets/test/CHEMBL2219358.jsonl.gz"),
    ]

    # Create tasks
    train_tasks = [
        Task(task_id="CHEMBL1613776", molecule_dataset=train_datasets[0]),
        Task(task_id="CHEMBL2219012", molecule_dataset=train_datasets[1]),
    ]

    test_tasks = [
        Task(task_id="CHEMBL2219236", molecule_dataset=test_datasets[0]),
        Task(task_id="CHEMBL2219358", molecule_dataset=test_datasets[1]),
    ]

    return Tasks(train_tasks=train_tasks, test_tasks=test_tasks)


def demonstrate_different_featurizers():
    """Demonstrate distance computation with different featurizers."""
    logger.info("Loading sample tasks...")
    tasks = load_sample_tasks()

    # Available featurizers to test
    featurizers = [
        "ecfp",  # Extended Connectivity Fingerprints (default)
        "maccs",  # MACCS Keys
        "desc2D",  # 2D Molecular Descriptors
        # "mordred",     # Mordred descriptors (if available)
        # "ChemBERTa-77M-MLM",  # Neural embeddings (if available)
    ]

    distance_methods = ["euclidean", "cosine"]

    results = {}

    for featurizer in featurizers:
        logger.info(f"\\n--- Testing featurizer: {featurizer} ---")
        results[featurizer] = {}

        for method in distance_methods:
            logger.info(f"Computing {method} distances with {featurizer} features...")

            try:
                # Create distance computer
                distance_computer = MoleculeDatasetDistance(tasks=tasks, molecule_method=method)

                # Compute distances with specific featurizer
                distances = distance_computer.get_distance(featurizer_name=featurizer)

                # Store results
                results[featurizer][method] = distances

                # Print sample results
                for target_id, source_distances in distances.items():
                    for source_id, distance_value in source_distances.items():
                        logger.info(f"  {source_id} → {target_id}: {distance_value:.4f}")

            except Exception as e:
                logger.error(f"Failed to compute {method} distances with {featurizer}: {e}")
                results[featurizer][method] = None

    return results


def compare_featurizer_results(results):
    """Compare distance results across different featurizers."""
    logger.info("\\n=== Featurizer Comparison ===")

    # Extract a specific pair for comparison
    target_id = list(results[list(results.keys())[0]]["euclidean"].keys())[0]
    source_id = list(results[list(results.keys())[0]]["euclidean"][target_id].keys())[0]

    logger.info(f"Comparing distances for pair: {source_id} → {target_id}")
    logger.info("-" * 60)

    for featurizer, methods in results.items():
        logger.info(f"Featurizer: {featurizer}")
        for method, distances in methods.items():
            if distances:
                distance_value = distances[target_id][source_id]
                logger.info(f"  {method}: {distance_value:.6f}")
        logger.info("")


if __name__ == "__main__":
    logger.info("🧪 Molecule Distance Featurizers Demo")
    logger.info("=" * 50)

    try:
        # Run the demonstration
        results = demonstrate_different_featurizers()

        # Compare results
        compare_featurizer_results(results)

        logger.info("\\n✅ Demo completed successfully!")
        logger.info("\\nKey takeaways:")
        logger.info("- Different featurizers capture different molecular properties")
        logger.info("- ECFP focuses on structural fragments")
        logger.info("- MACCS uses predefined pharmacophore keys")
        logger.info("- desc2D captures physicochemical properties")
        logger.info("- Distance values will vary significantly between featurizers")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        sys.exit(1)
