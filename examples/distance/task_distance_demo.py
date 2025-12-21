"""
Demonstration script for the optimized task distance module.

This script shows how to use the TaskDistance class to efficiently
compute N×M distance matrices for large task collections, with support for:
- Combined multi-modal features (molecules + proteins + metadata)
- Single-modality distances (molecules only, proteins only, metadata only)
- Memory-efficient processing for large datasets
- Task distance analysis

Usage:
    python scripts/task_distance_demo.py
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

from themap.data import DataFold
from themap.data.tasks import Tasks
from themap.distance import (
    MoleculeDatasetDistance,
    ProteinDatasetDistance,
    TaskDistance,
)
from themap.utils.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def convert_distance_dict_to_matrix(distance_dict, source_names, target_names):
    """Convert distance dictionary to matrix format with names."""
    n_source = len(source_names)
    n_target = len(target_names)
    matrix = np.zeros((n_source, n_target))

    for i, target_name in enumerate(target_names):
        if target_name in distance_dict:
            for j, source_name in enumerate(source_names):
                if source_name in distance_dict[target_name]:
                    matrix[j, i] = distance_dict[target_name][source_name]

    return matrix


def demo_combined_task_distance(tasks: Tasks, cache_dir: Optional[Path] = None) -> TaskDistance:
    """Demonstrate combined multi-modal task distance computation."""
    logger.info("\n" + "=" * 70)
    logger.info("🔄 DEMO 1: Combined Multi-Modal Task Distances")
    logger.info("=" * 70)

    try:
        # Create combined distance calculator
        calculator = TaskDistance(
            tasks=tasks,
            molecule_method="euclidean",
            protein_method="euclidean",
        )

        source_count = tasks.get_num_fold_tasks(DataFold.TRAIN)
        target_count = tasks.get_num_fold_tasks(DataFold.TEST)
        logger.info(
            f"📊 Calculator created for tasks with {source_count}/{target_count} source/target tasks"
        )

        # Compute combined distance matrix
        logger.info("🔄 Computing combined distance matrix...")
        start_time = time.time()

        distance_dict = calculator.compute_combined_distance(
            molecule_featurizer="ecfp",
            protein_featurizer="esm2_t33_650M_UR50D",
            combination="average",
        )

        elapsed_time = time.time() - start_time

        # Get task names
        target_names = list(distance_dict.keys())
        source_names = list(distance_dict[target_names[0]].keys()) if target_names else []

        # Convert to matrix for analysis
        distance_matrix = convert_distance_dict_to_matrix(distance_dict, source_names, target_names)

        logger.info(f"✅ Computed {distance_matrix.shape} distance matrix in {elapsed_time:.2f}s")
        logger.info(f"📏 Distance range: {distance_matrix.min():.4f} - {distance_matrix.max():.4f}")

        # Analyze results
        logger.info("\n📈 Distance Analysis:")
        logger.info(f"   🎯 Source tasks: {len(source_names)}")
        logger.info(f"   🧪 Target tasks: {len(target_names)}")
        logger.info(
            f"   📊 Matrix sparsity: {(distance_matrix == 0).sum() / distance_matrix.size * 100:.1f}%"
        )

        # Show example distances
        logger.info("\n🔍 Example distances:")
        for i, target_name in enumerate(target_names[:2]):  # Show first 2 targets
            logger.info(f"   {target_name}:")
            target_distances = [(source, dist) for source, dist in distance_dict[target_name].items()]
            target_distances.sort(key=lambda x: x[1])  # Sort by distance
            for j, (source_name, distance) in enumerate(target_distances[:3]):  # Show 3 nearest
                logger.info(f"     {j + 1}. {source_name}: {distance:.4f}")

        return calculator

    except Exception as e:
        logger.error(f"❌ Combined distance demo failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def demo_molecule_task_distance(tasks: Tasks, cache_dir: Optional[Path] = None) -> MoleculeDatasetDistance:
    """Demonstrate molecule-only task distance computation."""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 DEMO 2: Molecule-Only Task Distances")
    logger.info("=" * 70)

    try:
        # Create molecule distance calculator
        calculator = MoleculeDatasetDistance(
            tasks=tasks,
            molecule_method="cosine",
        )

        # Compute distance matrix
        logger.info("🔄 Computing molecular distance matrix...")
        distance_dict = calculator.get_distance()

        # Get task names and convert to matrix
        target_names = list(distance_dict.keys())
        source_names = list(distance_dict[target_names[0]].keys()) if target_names else []
        distance_matrix = convert_distance_dict_to_matrix(distance_dict, source_names, target_names)

        logger.info(f"✅ Molecular distances computed: {distance_matrix.shape}")
        logger.info(f"📏 Distance range: {distance_matrix.min():.4f} - {distance_matrix.max():.4f}")

        # Test repeat computation (should be fast due to caching)
        logger.info("🔄 Testing repeat computation...")
        start_time = time.time()
        cached_dict = calculator.get_distance()
        cache_time = time.time() - start_time
        logger.info(
            f"✅ Repeat computation in {cache_time:.4f}s (result equal: {distance_dict == cached_dict})"
        )

        return calculator

    except Exception as e:
        logger.error(f"❌ Molecule distance demo failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def demo_protein_task_distance(tasks: Tasks, cache_dir: Optional[Path] = None) -> ProteinDatasetDistance:
    """Demonstrate protein-only task distance computation."""
    logger.info("\n" + "=" * 70)
    logger.info("🧬 DEMO 3: Protein-Only Task Distances")
    logger.info("=" * 70)

    try:
        # Create protein distance calculator
        calculator = ProteinDatasetDistance(
            tasks=tasks,
            protein_method="euclidean",
        )

        # Compute distance matrix
        logger.info("🔄 Computing protein distance matrix...")
        distance_dict = calculator.get_distance()

        # Get task names and convert to matrix
        target_names = list(distance_dict.keys())
        source_names = list(distance_dict[target_names[0]].keys()) if target_names else []
        distance_matrix = convert_distance_dict_to_matrix(distance_dict, source_names, target_names)

        logger.info(f"✅ Protein distances computed: {distance_matrix.shape}")
        logger.info(f"📏 Distance range: {distance_matrix.min():.4f} - {distance_matrix.max():.4f}")

        return calculator

    except Exception as e:
        logger.error(f"❌ Protein distance demo failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def demo_task_distance_methods(tasks: Tasks, cache_dir: Optional[Path] = None) -> TaskDistance:
    """Demonstrate different TaskDistance computation methods."""
    logger.info("\n" + "=" * 70)
    logger.info("⚡ DEMO 4: TaskDistance Methods")
    logger.info("=" * 70)

    try:
        # Create main task distance calculator
        calculator = TaskDistance(
            tasks=tasks,
            molecule_method="euclidean",
            protein_method="euclidean",
        )

        logger.info("🔄 Computing all distance types...")

        # Compute all distances at once
        all_distances = calculator.compute_all_distances(
            molecule_featurizer="ecfp",
            protein_featurizer="esm2_t33_650M_UR50D",
            combination="average",
        )

        # Analyze each distance type
        for distance_type, distance_dict in all_distances.items():
            if distance_dict:  # Only process non-empty results
                target_names = list(distance_dict.keys())
                source_names = list(distance_dict[target_names[0]].keys()) if target_names else []
                matrix = convert_distance_dict_to_matrix(distance_dict, source_names, target_names)

                logger.info(f"\n💡 {distance_type.title()} Distance Analysis:")
                logger.info(f"   📊 Matrix shape: {matrix.shape}")
                logger.info(f"   📏 Distance range: {matrix.min():.4f} - {matrix.max():.4f}")
                logger.info(f"   📈 Mean distance: {matrix.mean():.4f}")

        # Test different combination strategies
        logger.info("\n🔧 Testing different combination strategies...")

        for strategy in ["average", "weighted_average", "min", "max"]:
            try:
                weights = {"molecules": 0.7, "protein": 0.3} if strategy == "weighted_average" else None
                combined_dict = calculator.compute_combined_distance(
                    combination=strategy,
                    weights=weights,
                )
                target_names = list(combined_dict.keys())
                if target_names:
                    # Just get one sample distance
                    sample_distance = list(combined_dict[target_names[0]].values())[0]
                    logger.info(f"   ✅ {strategy}: sample distance = {sample_distance:.4f}")
            except Exception as e:
                logger.warning(f"   ⚠️ {strategy} failed: {e}")

        return calculator

    except Exception as e:
        logger.error(f"❌ TaskDistance methods demo failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def demo_performance_optimization(tasks: Tasks, cache_dir: Optional[Path] = None) -> TaskDistance:
    """Demonstrate performance optimization features."""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 DEMO 5: Performance Optimization")
    logger.info("=" * 70)

    try:
        # Test different methods
        logger.info("🔄 Testing different distance methods...")

        molecule_methods = ["euclidean", "cosine"]
        protein_methods = ["euclidean", "cosine"]

        for mol_method in molecule_methods:
            mol_calculator = MoleculeDatasetDistance(
                tasks=tasks,
                molecule_method=mol_method,
            )

            start_time = time.time()
            distance_dict = mol_calculator.get_distance()
            elapsed_time = time.time() - start_time

            target_count = len(distance_dict.keys())
            logger.info(f"   Molecule {mol_method}: {elapsed_time:.3f}s, {target_count} targets")

        for prot_method in protein_methods:
            prot_calculator = ProteinDatasetDistance(
                tasks=tasks,
                protein_method=prot_method,
            )

            start_time = time.time()
            distance_dict = prot_calculator.get_distance()
            elapsed_time = time.time() - start_time

            target_count = len(distance_dict.keys())
            logger.info(f"   Protein {prot_method}: {elapsed_time:.3f}s, {target_count} targets")

        # Test caching benefits
        logger.info("\n💾 Cache Performance:")
        calculator = TaskDistance(tasks=tasks)

        # First computation
        start_time = time.time()
        calculator.compute_molecule_distance()
        first_time = time.time() - start_time

        # Second computation (should use cached results)
        start_time = time.time()
        calculator.compute_molecule_distance()
        second_time = time.time() - start_time

        logger.info(f"   First computation: {first_time:.3f}s")
        logger.info(f"   Cached computation: {second_time:.3f}s")
        logger.info(f"   Speedup: {first_time / second_time:.1f}x")

        return calculator

    except Exception as e:
        logger.error(f"❌ Performance demo failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main() -> None:
    """Run all task distance demonstrations."""
    logger.info("🎯 Task Distance Module Demonstration")
    logger.info("=" * 70)

    # Setup
    cache_dir = Path("./task_distance_cache")
    cache_dir.mkdir(exist_ok=True)

    try:
        # Load tasks from the available dataset
        logger.info("📁 Loading tasks from datasets directory...")

        tasks = Tasks.from_directory(
            directory="datasets",
            task_list_file="datasets/sample_tasks_list.json",
            cache_dir=cache_dir,
            load_molecules=True,
            load_proteins=True,
            load_metadata=False,  # No metadata in this dataset
        )

        logger.info("✅ Tasks loaded successfully!")
        logger.info(f"   Training tasks: {tasks.get_num_fold_tasks(DataFold.TRAIN)}")
        logger.info(f"   Test tasks: {tasks.get_num_fold_tasks(DataFold.TEST)}")

        # Run demonstrations
        logger.info("\n🚀 Starting Task Distance Demonstrations...")

        # Demo 1: Combined features (molecules + proteins)
        demo_combined_task_distance(tasks, cache_dir)

        # Demo 2: Molecule-only distances
        demo_molecule_task_distance(tasks, cache_dir)

        # Demo 3: Protein-only distances
        demo_protein_task_distance(tasks, cache_dir)

        # Demo 4: TaskDistance methods
        demo_task_distance_methods(tasks, cache_dir)

        # Demo 5: Performance optimization
        demo_performance_optimization(tasks, cache_dir)

        logger.info("\n✅ All demonstrations completed successfully!")
        logger.info(f"💾 Cache directory: {cache_dir}")

        # Show final usage summary
        logger.info("\n💡 Usage Summary:")
        logger.info("# Create distance calculator:")
        logger.info("calculator = TaskDistance(tasks=tasks)")
        logger.info("")
        logger.info("# Compute distances:")
        logger.info("molecule_distances = calculator.compute_molecule_distance()")
        logger.info("protein_distances = calculator.compute_protein_distance()")
        logger.info("combined_distances = calculator.compute_combined_distance()")
        logger.info("")
        logger.info("# Or compute all at once:")
        logger.info("all_distances = calculator.compute_all_distances()")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
