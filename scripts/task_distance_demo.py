"""
Demonstration script for the optimized task distance module.

This script shows how to use the new TaskDistanceCalculator classes to efficiently
compute N×M distance matrices for large task collections, with support for:
- Combined multi-modal features (molecules + proteins + metadata)
- Single-modality distances (molecules only, proteins only, metadata only)
- Memory-efficient chunking for large datasets (>1000 tasks)
- Parallel processing and caching
- Task hardness analysis and k-nearest neighbor searches

Usage:
    python scripts/task_distance_demo.py
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np

from themap.data.metadata import DataFold
from themap.data.tasks import Tasks
from themap.distance import (
    CombinedTaskDistance,
    MoleculeTaskDistance,
    ProteinTaskDistance,
    compute_task_distance_matrix,
    create_task_distance_calculator,
)
from themap.utils.logging import get_logger

logger = get_logger(__name__)


def demo_combined_task_distance(tasks: Tasks, cache_dir: Optional[Path] = None) -> None:
    """Demonstrate combined multi-modal task distance computation."""
    logger.info("\n" + "=" * 70)
    logger.info("🔄 DEMO 1: Combined Multi-Modal Task Distances")
    logger.info("=" * 70)

    try:
        # Create combined distance calculator
        calculator = CombinedTaskDistance(
            tasks=tasks,
            molecule_featurizer="ecfp",
            protein_featurizer="esm2_t33_650M_UR50D",
            combination_method="concatenate",
            distance_metric="euclidean",
            cache_dir=cache_dir,
            chunk_size=50,  # Small for demo
            n_jobs=2,
        )

        logger.info(f"📊 Calculator info: {calculator.get_cache_info()}")

        # Compute distance matrix
        logger.info("🔄 Computing combined distance matrix...")
        start_time = time.time()

        distance_matrix, source_names, target_names = calculator.compute_distance_matrix(
            source_fold=DataFold.TRAIN,
            target_folds=[DataFold.TEST],
            save_cache=True,
        )

        elapsed_time = time.time() - start_time
        logger.info(f"✅ Computed {distance_matrix.shape} distance matrix in {elapsed_time:.2f}s")
        logger.info(f"📏 Distance range: {distance_matrix.min():.4f} - {distance_matrix.max():.4f}")

        # Analyze results
        logger.info("\n📈 Distance Analysis:")
        logger.info(f"   🎯 Source tasks: {len(source_names)}")
        logger.info(f"   🧪 Target tasks: {len(target_names)}")
        logger.info(
            f"   📊 Matrix sparsity: {(distance_matrix == 0).sum() / distance_matrix.size * 100:.1f}%"
        )

        # Compute task hardness
        hardness_scores = calculator.compute_task_hardness(
            distance_matrix, target_names, k=min(5, len(source_names)), aggregation="mean"
        )
        logger.info(f"   💪 Average task hardness: {np.mean(list(hardness_scores.values())):.4f}")

        # Find k-nearest neighbors
        k_nearest = calculator.get_k_nearest_tasks(
            distance_matrix, source_names, target_names, k=min(3, len(source_names))
        )

        # Show example nearest neighbors
        logger.info("\n🔍 Example K-Nearest Neighbors:")
        for i, (target_name, nearest_list) in enumerate(k_nearest.items()):
            if i >= 2:  # Show only first 2 examples
                break
            logger.info(f"   {target_name}:")
            for rank, (source_name, distance) in enumerate(nearest_list, 1):
                logger.info(f"     {rank}. {source_name}: {distance:.4f}")

        return calculator

    except Exception as e:
        logger.error(f"❌ Combined distance demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_molecule_task_distance(tasks: Tasks, cache_dir: Optional[Path] = None) -> None:
    """Demonstrate molecule-only task distance computation."""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 DEMO 2: Molecule-Only Task Distances")
    logger.info("=" * 70)

    try:
        # Create molecule distance calculator
        calculator = MoleculeTaskDistance(
            tasks=tasks,
            molecule_featurizer="ecfp",
            distance_metric="cosine",
            cache_dir=cache_dir,
            chunk_size=50,
            n_jobs=1,
        )

        # Compute distance matrix
        logger.info("🔄 Computing molecular distance matrix...")
        distance_matrix, source_names, target_names = calculator.compute_distance_matrix()

        logger.info(f"✅ Molecular distances computed: {distance_matrix.shape}")
        logger.info(f"📏 Distance range: {distance_matrix.min():.4f} - {distance_matrix.max():.4f}")

        # Test caching
        logger.info("🔄 Testing cache performance...")
        start_time = time.time()
        cached_matrix, _, _ = calculator.compute_distance_matrix()
        cache_time = time.time() - start_time
        logger.info(
            f"✅ Cache retrieval in {cache_time:.4f}s (matrix equal: {np.array_equal(distance_matrix, cached_matrix)})"
        )

        return calculator

    except Exception as e:
        logger.error(f"❌ Molecule distance demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_protein_task_distance(tasks: Tasks, cache_dir: Optional[Path] = None) -> None:
    """Demonstrate protein-only task distance computation."""
    logger.info("\n" + "=" * 70)
    logger.info("🧬 DEMO 3: Protein-Only Task Distances")
    logger.info("=" * 70)

    try:
        # Create protein distance calculator
        calculator = ProteinTaskDistance(
            tasks=tasks,
            protein_featurizer="esm2_t33_650M_UR50D",
            layer=33,
            distance_metric="euclidean",
            cache_dir=cache_dir,
        )

        # Compute distance matrix
        logger.info("🔄 Computing protein distance matrix...")
        distance_matrix, source_names, target_names = calculator.compute_distance_matrix()

        logger.info(f"✅ Protein distances computed: {distance_matrix.shape}")
        logger.info(f"📏 Distance range: {distance_matrix.min():.4f} - {distance_matrix.max():.4f}")

        return calculator

    except Exception as e:
        logger.error(f"❌ Protein distance demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_convenience_functions(tasks: Tasks, cache_dir: Optional[Path] = None) -> None:
    """Demonstrate convenience functions for quick distance computation."""
    logger.info("\n" + "=" * 70)
    logger.info("⚡ DEMO 4: Convenience Functions")
    logger.info("=" * 70)

    try:
        # Quick distance calculation using convenience function
        logger.info("🔄 Using compute_task_distance_matrix convenience function...")

        matrix, source_names, target_names, calculator = compute_task_distance_matrix(
            tasks=tasks,
            distance_type="molecule",  # Use molecule only for faster demo
            distance_metric="euclidean",
            chunk_size=50,
            cache_dir=cache_dir,
            molecule_featurizer="ecfp",
        )

        logger.info(f"✅ Quick computation complete: {matrix.shape}")

        # Analyze task hardness
        hardness = calculator.compute_task_hardness(matrix, target_names, k=min(5, len(source_names)))
        sorted_hardness = sorted(hardness.items(), key=lambda x: x[1], reverse=True)

        logger.info("\n💪 Task Hardness Analysis:")
        for i, (task_name, hardness_score) in enumerate(sorted_hardness):
            logger.info(f"   {i + 1}. {task_name}: {hardness_score:.4f}")

        # Test different distance calculators
        logger.info("\n🔧 Testing different calculator types...")

        for distance_type in ["molecule", "protein"]:
            try:
                calc = create_task_distance_calculator(
                    tasks,
                    distance_type=distance_type,
                    cache_dir=cache_dir,
                )
                logger.info(f"   ✅ {distance_type} calculator created successfully")
            except Exception as e:
                logger.warning(f"   ⚠️ {distance_type} calculator failed: {e}")

        return calculator

    except Exception as e:
        logger.error(f"❌ Convenience functions demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def demo_performance_optimization(tasks: Tasks, cache_dir: Optional[Path] = None) -> None:
    """Demonstrate performance optimization features."""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 DEMO 5: Performance Optimization")
    logger.info("=" * 70)

    try:
        # Test different chunk sizes
        logger.info("🔄 Testing different chunk sizes...")

        for chunk_size in [25, 50]:  # Smaller for demo
            calculator = MoleculeTaskDistance(
                tasks=tasks,
                molecule_featurizer="ecfp",
                chunk_size=chunk_size,
                cache_dir=cache_dir,
            )

            start_time = time.time()
            distance_matrix, _, _ = calculator.compute_distance_matrix(force_recompute=True)
            elapsed_time = time.time() - start_time

            logger.info(f"   Chunk size {chunk_size}: {elapsed_time:.3f}s, shape: {distance_matrix.shape}")

        # Test parallel vs sequential
        logger.info("\n⚡ Testing parallel processing...")

        for n_jobs in [1, 2]:
            calculator = MoleculeTaskDistance(
                tasks=tasks,
                molecule_featurizer="ecfp",
                n_jobs=n_jobs,
                cache_dir=cache_dir,
            )

            start_time = time.time()
            distance_matrix, _, _ = calculator.compute_distance_matrix(force_recompute=True)
            elapsed_time = time.time() - start_time

            job_type = "sequential" if n_jobs == 1 else f"parallel ({n_jobs} jobs)"
            logger.info(f"   {job_type}: {elapsed_time:.3f}s")

        # Cache statistics
        logger.info("\n💾 Cache Statistics:")
        cache_info = calculator.get_cache_info()
        for key, value in cache_info.items():
            logger.info(f"   {key}: {value}")

        return calculator

    except Exception as e:
        logger.error(f"❌ Performance demo failed: {e}")
        import traceback

        traceback.print_exc()
        return None


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

        # Demo 4: Convenience functions
        demo_convenience_functions(tasks, cache_dir)

        # Demo 5: Performance optimization
        demo_performance_optimization(tasks, cache_dir)

        logger.info("\n✅ All demonstrations completed successfully!")
        logger.info(f"💾 Cache directory: {cache_dir}")

        # Show final usage summary
        logger.info("\n💡 Usage Summary:")
        logger.info("# Quick distance computation:")
        logger.info("matrix, source_names, target_names, calc = compute_task_distance_matrix(")
        logger.info("    tasks=tasks,")
        logger.info("    distance_type='combined',")
        logger.info("    molecule_featurizer='ecfp',")
        logger.info("    protein_featurizer='esm2_t33_650M_UR50D'")
        logger.info(")")
        logger.info("")
        logger.info("# Analysis:")
        logger.info("hardness = calc.compute_task_hardness(matrix, target_names)")
        logger.info("nearest = calc.get_k_nearest_tasks(matrix, source_names, target_names)")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
