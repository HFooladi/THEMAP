#!/usr/bin/env python3
"""
Demo script showing how to use the unified Task and Tasks system.

This script demonstrates:
1. Loading tasks with integrated molecule, protein, and metadata data
2. Computing combined features from multiple data types
3. Distance computation across tasks
4. Task feature caching and persistence
5. Multi-modal task analysis

Run from project root:
    python scripts/tasks_demo.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from themap.data.protein_datasets import DataFold
from themap.data.tasks import Tasks
from themap.utils.logging import get_logger

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = get_logger(__name__)


def demo_individual_task():
    """Demonstrate individual Task functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("🎯 DEMO 1: Individual Task Operations")
    logger.info("=" * 60)

    # Note: This demo focuses on the API structure since we don't have actual data loaded
    logger.info("📋 Task API Capabilities:")
    logger.info("1. 🧪 Task(task_id, molecule_dataset, protein_dataset, metadata_datasets)")
    logger.info("2. ⚗️ task.get_molecule_features(featurizer_name)")
    logger.info("3. 🧬 task.get_protein_features(featurizer_name, layer)")
    logger.info("4. 📊 task.get_metadata_features(metadata_type, featurizer_name)")
    logger.info("5. 🔗 task.get_combined_features(molecule_featurizer, protein_featurizer, metadata_configs)")

    # Example task creation (would work with real data)
    logger.info("\n📝 Example Task creation pattern:")
    logger.info("""
    task = Task(
        task_id="CHEMBL2219236",
        molecule_dataset=loaded_molecule_dataset,  # ⚗️ Molecules
        protein_dataset=loaded_protein_dataset,    # 🧬 Protein
        metadata_datasets={                        # 📊 Metadata
            "assay_description": text_metadata_dataset,
            "bioactivity": numerical_metadata_dataset
        }
    )
    """)

    logger.info("\n🔧 Example combined feature extraction:")
    logger.info("""
    combined_features = task.get_combined_features(
        molecule_featurizer="ecfp",     # ⚗️ Molecular features
        protein_featurizer="esm2_t33_650M_UR50D",     # 🧬 Protein features
        metadata_configs={                             # 📊 Metadata features
            "assay_description": {"featurizer_name": "sentence-transformers"},
            "bioactivity": {"featurizer_name": "standardize"}
        },
        combination_method="concatenate"               # 🔗 Fusion strategy
    )
    """)


def demo_tasks_from_directory():
    """Demonstrate Tasks.from_directory functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("📂 DEMO 2: Loading Tasks from Directory")
    logger.info("=" * 60)

    try:
        # Try to load tasks from the actual datasets directory
        logger.info("🔍 Attempting to load tasks from datasets/ directory...")

        tasks = Tasks.from_directory(
            directory="datasets/",
            task_list_file="datasets/sample_tasks_list.json",
            load_molecules=True,
            load_proteins=True,
            load_metadata=False,  # Skip metadata for now since it may not exist
            cache_dir="cache/",
        )

        logger.info(f"✅ Successfully loaded tasks: {tasks}")

        # Show task distribution
        logger.info("\n📊 Task distribution by fold:")
        for fold in [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]:
            num_tasks = tasks.get_num_fold_tasks(fold)
            task_ids = tasks.get_task_ids(fold)
            fold_name = {DataFold.TRAIN: "train", DataFold.VALIDATION: "validation", DataFold.TEST: "test"}[
                fold
            ]
            fold_emoji = {"train": "🏋️", "validation": "🔍", "test": "🧪"}[fold_name]
            logger.info(f"{fold_emoji} {fold_name.title()} tasks ({num_tasks}): {task_ids}")

        # Show sample task details
        train_tasks = tasks.get_tasks(DataFold.TRAIN)
        if train_tasks:
            sample_task = train_tasks[0]
            logger.info(f"\n🔬 Sample task: {sample_task}")

            # Test individual feature extraction (if data is available)
            try:
                if sample_task.molecule_dataset:
                    logger.info("✅ ⚗️ Molecule dataset available for feature extraction")
                if sample_task.protein_dataset:
                    logger.info("✅ 🧬 Protein dataset available for feature extraction")
                if sample_task.metadata_datasets:
                    logger.info(f"✅ 📊 Metadata available: {list(sample_task.metadata_datasets.keys())}")
            except Exception as e:
                logger.warning(f"⚠️ Could not access task data: {e}")

        return tasks

    except Exception as e:
        logger.warning(f"❌ Could not load tasks from directory: {e}")
        logger.info("💡 This is expected if datasets/ directory is not set up")
        return None


def demo_combined_feature_computation(tasks):
    """Demonstrate combined feature computation across tasks."""
    logger.info("\n" + "=" * 60)
    logger.info("🧮 DEMO 3: Combined Feature Computation")
    logger.info("=" * 60)

    if tasks is None:
        logger.info("⏭️ Skipping combined feature computation - no tasks loaded")
        return None

    try:
        logger.info("🔄 Computing combined features for all tasks...")

        # Example configuration for combined features
        metadata_configs = {
            "assay_description": {
                "featurizer_name": "sentence-transformers",
                "kwargs": {"model_name": "all-MiniLM-L6-v2"},
            },
            "bioactivity": {"featurizer_name": "standardize", "kwargs": {}},
        }

        logger.info("🎛️ Feature extraction configuration:")
        logger.info("   ⚗️ Molecules: morgan_fingerprints")
        logger.info("   🧬 Proteins: esm2_t33_650M_UR50D")
        logger.info("   📊 Metadata: sentence-transformers + standardization")
        logger.info("   🔗 Combination: concatenate")

        # Compute features (this may take time with real data)
        all_features = tasks.compute_all_task_features(
            molecule_featurizer="ecfp",
            protein_featurizer="esm2_t33_650M_UR50D",
            metadata_configs=metadata_configs,
            combination_method="concatenate",
            folds=[DataFold.TRAIN, DataFold.TEST],  # Focus on train and test
        )

        logger.info(f"✅ Computed features for {len(all_features)} tasks")

        # Show feature shapes
        logger.info("\n📐 Sample feature shapes:")
        for name, features in list(all_features.items())[:3]:
            logger.info(f"  🎯 {name}: feature shape {features.shape}")

        return all_features

    except Exception as e:
        logger.error(f"❌ Failed to compute combined features: {e}")
        logger.info("💡 This is expected if the required data/models are not available")
        return None


def demo_distance_computation(tasks):
    """Demonstrate distance computation between tasks."""
    logger.info("\n" + "=" * 60)
    logger.info("📏 DEMO 4: Task Distance Computation")
    logger.info("=" * 60)

    if tasks is None:
        logger.info("⏭️ Skipping distance computation - no tasks loaded")
        return

    try:
        logger.info("🔧 Preparing features for distance computation...")

        # Get features organized for N×M distance matrix
        source_features, target_features, source_names, target_names = (
            tasks.get_distance_computation_ready_features(
                molecule_featurizer="ecfp",
                protein_featurizer="esm2_t33_650M_UR50D",
                combination_method="concatenate",
                source_fold=DataFold.TRAIN,
                target_folds=[DataFold.TEST],
            )
        )

        logger.info(f"🏋️ Source tasks: {source_names}")
        logger.info(f"🧪 Target tasks: {target_names}")

        if source_features and target_features:
            logger.info("\n📊 Computing task distance matrix...")

            # Stack features for vectorized computation
            source_matrix = np.stack(source_features)  # (N, D)
            target_matrix = np.stack(target_features)  # (M, D)

            logger.info(f"📐 Source matrix shape: {source_matrix.shape}")
            logger.info(f"📐 Target matrix shape: {target_matrix.shape}")

            # Compute cosine distances
            distance_matrix = cosine_distances(source_matrix, target_matrix)

            logger.info(f"🎯 Distance matrix shape: {distance_matrix.shape}")
            logger.info(f"📊 Sample distances:\n{distance_matrix}")

            # Find most similar task pairs
            min_indices = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            logger.info(
                f"🔗 Most similar pair: {source_names[min_indices[0]]} <-> {target_names[min_indices[1]]}"
            )
            logger.info(f"📏 Distance: {distance_matrix[min_indices]:.4f}")

            # Analyze task similarities
            analyze_task_similarities(distance_matrix, source_names, target_names)
        else:
            logger.warning("⚠️ No features available for distance computation")

    except Exception as e:
        logger.error(f"❌ Failed to compute task distances: {e}")
        logger.info("💡 This is expected if the required data/models are not available")


def analyze_task_similarities(distance_matrix, source_names, target_names):
    """Analyze task similarity patterns."""
    logger.info("\n🔍 --- Task Similarity Analysis ---")

    # Find nearest neighbors for each source task
    logger.info("🎯 Nearest neighbors for each source task:")
    for i, source_name in enumerate(source_names):
        distances = distance_matrix[i]
        nearest_idx = np.argmin(distances)
        nearest_name = target_names[nearest_idx]
        nearest_distance = distances[nearest_idx]

        logger.info(f"  🏋️ {source_name} → 🧪 {nearest_name} (📏 distance: {nearest_distance:.4f})")

    # Overall statistics
    mean_distance = np.mean(distance_matrix)
    std_distance = np.std(distance_matrix)
    min_distance = np.min(distance_matrix)
    max_distance = np.max(distance_matrix)

    logger.info("\n📊 Distance statistics:")
    logger.info(f"  📈 Mean: {mean_distance:.4f}")
    logger.info(f"  📏 Std:  {std_distance:.4f}")
    logger.info(f"  🔽 Min:  {min_distance:.4f}")
    logger.info(f"  🔼 Max:  {max_distance:.4f}")


def demo_feature_persistence(tasks):
    """Demonstrate feature saving and loading."""
    logger.info("\n" + "=" * 60)
    logger.info("💾 DEMO 5: Feature Persistence")
    logger.info("=" * 60)

    if tasks is None:
        logger.info("⏭️ Skipping feature persistence - no tasks loaded")
        return

    try:
        # Create temporary directory for demo
        with tempfile.TemporaryDirectory() as temp_dir:
            feature_file = Path(temp_dir) / "task_features.pkl"

            logger.info(f"💾 Saving task features to {feature_file}")

            # Save features
            tasks.save_task_features_to_file(
                output_path=feature_file,
                molecule_featurizer="ecfp",
                protein_featurizer="esm2_t33_650M_UR50D",
                combination_method="concatenate",
                folds=[DataFold.TRAIN, DataFold.TEST],
            )

            # Load features
            logger.info("📂 Loading task features from file...")
            loaded_features = Tasks.load_task_features_from_file(feature_file)

            logger.info(f"✅ Loaded {len(loaded_features)} task features")

            # Verify consistency
            logger.info("🔍 Verifying loaded features:")
            for name, features in list(loaded_features.items())[:2]:
                logger.info(f"  🎯 {name}: loaded feature shape {features.shape}")

    except Exception as e:
        logger.error(f"❌ Failed to save/load features: {e}")


def demo_cache_statistics(tasks):
    """Demonstrate cache statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("📈 DEMO 6: Cache Statistics")
    logger.info("=" * 60)

    if tasks is None:
        logger.info("⏭️ No tasks loaded - cannot show cache statistics")
        return

    cache_stats = tasks.get_cache_stats()
    logger.info("📊 Cache statistics:")
    for key, value in cache_stats.items():
        if "cache" in key.lower():
            emoji = "🗄️"
        elif "dir" in key.lower():
            emoji = "📂"
        else:
            emoji = "📈"
        logger.info(f"  {emoji} {key}: {value}")


def demo_multi_modal_analysis():
    """Demonstrate multi-modal analysis concepts."""
    logger.info("\n" + "=" * 60)
    logger.info("🌐 DEMO 7: Multi-Modal Analysis Concepts")
    logger.info("=" * 60)

    logger.info("🎭 Multi-modal task analysis enables:")

    logger.info("\n1. 🔗 Feature Fusion Strategies:")
    logger.info("   📎 concatenate: [mol_features | protein_features | metadata_features]")
    logger.info("   📊 average: element-wise average of padded features")
    logger.info("   ⚖️ weighted_average: importance-weighted combination")

    logger.info("\n2. 🔍 Similarity Analysis:")
    logger.info("   🎯 Task-to-task similarity based on combined representations")
    logger.info("   🔄 Cross-modal analysis (molecule vs protein vs metadata contributions)")
    logger.info("   💪 Hardness prediction based on multi-modal features")

    logger.info("\n3. 🚀 Transfer Learning Applications:")
    logger.info("   🎯 Find similar tasks for few-shot learning")
    logger.info("   🧠 Identify task relationships for meta-learning")
    logger.info("   🌉 Cross-domain knowledge transfer")

    logger.info("\n4. 📊 Data Integration Patterns:")
    logger.info("   ⚗️ Molecules: structural and chemical properties")
    logger.info("   🧬 Proteins: sequence and structural features")
    logger.info("   📋 Metadata: experimental conditions, assay descriptions")


def main():
    """Run all task system demos."""
    logger.info("🚀 Starting Task and Tasks System Demo")

    # Demo individual task functionality
    demo_individual_task()

    # Try to load actual tasks
    tasks = demo_tasks_from_directory()

    # Demo feature computation (if tasks loaded)
    all_features = demo_combined_feature_computation(tasks)

    # Demo distance computation
    demo_distance_computation(tasks)

    # Demo feature persistence
    demo_feature_persistence(tasks)

    # Demo cache statistics
    demo_cache_statistics(tasks)

    # Demo multi-modal concepts
    demo_multi_modal_analysis()

    logger.info("\n" + "=" * 60)
    logger.info("📋 SUMMARY: Task System Capabilities")
    logger.info("=" * 60)

    summary = """
🎉 The Task and Tasks system provides:

1. 🧪 UNIFIED TASK REPRESENTATION:
   • Single Task objects containing molecules, proteins, and metadata
   • Flexible data combinations (any subset can be present)
   • Backward compatibility with legacy metadata

2. 🔗 MULTI-MODAL FEATURE EXTRACTION:
   • Individual feature extraction per data type
   • Combined feature computation with multiple strategies
   • Configurable featurizer parameters per modality

3. 📂 COLLECTION MANAGEMENT:
   • Load tasks from organized directory structures
   • Fold-based organization (train/validation/test)
   • Task filtering via task list files

4. 📏 DISTANCE COMPUTATION:
   • N×M distance matrices between task sets
   • Multi-modal similarity analysis
   • Efficient feature organization for batch computation

5. ⚡ PERFORMANCE OPTIMIZATIONS:
   • Feature caching for expensive computations
   • Persistent storage for computed features
   • Lazy loading and efficient memory usage

6. 🔌 INTEGRATION READY:
   • Works with existing MoleculeDatasets and ProteinDatasets
   • Supports new MetadataDatasets system
   • Extensible for additional data types

💡 Usage Pattern:
```python
# 📂 Load integrated tasks
tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=True,
    load_metadata=True
)

# 🧮 Compute multi-modal features
features = tasks.compute_all_task_features(
    molecule_featurizer="morgan_fingerprints",     # ⚗️
    protein_featurizer="esm2_t33_650M_UR50D",     # 🧬
    metadata_configs={                             # 📊
        "assay_description": {"featurizer_name": "sentence-transformers"}
    }
)

# 📏 Compute task distances
source_features, target_features, source_names, target_names = (
    tasks.get_distance_computation_ready_features(...)
)
```

🎯 This system is now ready for production use with your 1000+ tasks! 🚀
    """

    logger.info(summary)


if __name__ == "__main__":
    main()
