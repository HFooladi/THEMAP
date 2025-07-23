#!/usr/bin/env python3
"""
MoleculeDatasets Usage Example

This example demonstrates how to use the MoleculeDatasets class with:
- Persistent caching for features
- Global SMILES deduplication across datasets
- Memory-efficient storage
- Efficient N×M distance matrix computation

Author: Hosein Fooladi
Date: 2025-07-15
"""

import logging
import time
from pathlib import Path

import numpy as np

from themap.data.molecule_datasets import DataFold, MoleculeDatasets
from themap.utils.memory_utils import MemoryEfficientFeatureStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating molecules feature computation."""

    # Configuration
    data_dir = Path("datasets")  # Replace with your actual data directory
    cache_dir = Path("cache/molecular_features")
    featurizer_name = "ecfp"  # or "mordred", "fcfp", etc.

    print("🧬 THEMAP MoleculeDatasets Usage Example")
    print("=" * 50)

    # 1. Initialize MoleculeDatasets with caching enabled
    print("\n📂 Step 1: Initialize MoleculeDatasets with optimizations")
    molecule_datasets = MoleculeDatasets.from_directory(
        directory=data_dir,
        cache_dir=cache_dir,  # Enable persistent caching
        num_workers=4,
    )

    print(f"✅ Initialized: {molecule_datasets}")
    print(f"📁 Cache directory: {cache_dir}")

    # 2. Demonstrate global caching statistics (before computation)
    print("\n📊 Step 2: Check initial cache status")
    cache_stats = molecule_datasets.get_global_cache_stats()
    if cache_stats:
        print(f"💾 Cache stats: {cache_stats}")
    else:
        print("💾 No cache statistics available")

    # 3. Compute features for all datasets with global deduplication
    print(f"\n⚡ Step 3: Compute features with global deduplication using {featurizer_name}")
    start_time = time.time()

    all_features = molecule_datasets.compute_all_features_with_deduplication(
        featurizer_name=featurizer_name,
        folds=[DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST],
        batch_size=1000,
        n_jobs=4,
        force_recompute=False,  # Use cache if available
    )

    computation_time = time.time() - start_time
    print(f"⏱️  Total computation time: {computation_time:.2f} seconds")
    print(f"📝 Computed features for {len(all_features)} datasets")

    # 4. Show efficiency gains from deduplication
    print("\n🔍 Step 4: Analyze efficiency gains")
    total_molecules = 0
    for dataset_name, features in all_features.items():
        print(f"  {dataset_name}: {features.shape}")
        total_molecules += features.shape[0]

    print(f"🧮 Total molecules processed: {total_molecules}")

    # Get cache statistics after computation
    cache_stats_after = molecule_datasets.get_global_cache_stats()
    if cache_stats_after:
        persistent_stats = cache_stats_after.get("persistent_cache_stats", {})
        cache_size = cache_stats_after.get("persistent_cache_size", {})

        print(f"📈 Cache hit rate: {persistent_stats.get('hit_rate', 0):.2%}")
        print(f"💾 Cache size: {cache_size.get('disk_usage_mb', 0):.1f} MB")
        print(f"🗃️  Cached files: {cache_size.get('stored_files', 0)}")

    # 5. Prepare features for N×M distance matrix computation
    print("\n🎯 Step 5: Prepare features for N×M distance computation")
    source_features, target_features, source_names, target_names = (
        molecule_datasets.get_distance_computation_ready_features(
            featurizer_name=featurizer_name,
            source_fold=DataFold.TRAIN,  # N source datasets
            target_folds=[DataFold.VALIDATION, DataFold.TEST],  # M target datasets
        )
    )

    print(f"🎯 Source datasets (N): {len(source_features)}")
    print(f"🎯 Target datasets (M): {len(target_features)}")
    print(f"📐 Distance matrix dimensions: {len(source_features)} × {len(target_features)}")

    # 6. Demonstrate memory-efficient operations
    print("\n💾 Step 6: Memory-efficient storage demonstration")
    if molecule_datasets.global_cache and molecule_datasets.global_cache.persistent_cache:
        storage_stats = molecule_datasets.global_cache.persistent_cache.get_cache_size_info()
        print(f"💾 Memory usage: {storage_stats.get('memory_usage_mb', 0):.1f} MB")
        print(f"💿 Disk usage: {storage_stats.get('disk_usage_mb', 0):.1f} MB")

        # Demonstrate memory-efficient loading
        if hasattr(molecule_datasets.global_cache.persistent_cache, "_memory_storage"):
            memory_storage = molecule_datasets.global_cache.persistent_cache._memory_storage
            if memory_storage:
                print(f"🔧 Memory mapping enabled: {memory_storage.use_memory_mapping}")
                print(f"🗜️  Compression level: {memory_storage.compression_level}")

    # 7. Example: Computing a simple distance matrix
    print("\n📏 Step 7: Example distance matrix computation")
    if source_features and target_features:
        # Compute a simple Euclidean distance matrix for the first few datasets
        max_datasets = min(3, len(source_features), len(target_features))

        print(f"Computing {max_datasets}×{max_datasets} distance matrix...")
        distance_matrix = np.zeros((max_datasets, max_datasets))

        for i in range(max_datasets):
            for j in range(max_datasets):
                # Use dataset prototypes (mean features) for distance computation
                source_prototype = np.mean(source_features[i], axis=0)
                target_prototype = np.mean(target_features[j], axis=0)

                # Compute Euclidean distance
                distance = np.linalg.norm(source_prototype - target_prototype)
                distance_matrix[i, j] = distance

        print(f"📏 Sample distance matrix shape: {distance_matrix.shape}")
        print(f"📏 Distance matrix:\n{distance_matrix}")

    # 8. Performance summary
    print("\n📊 Step 8: Performance Summary")
    print("=" * 50)
    print(f"⏱️  Total runtime: {time.time() - start_time:.2f} seconds")
    print(f"🧮 Molecules processed: {total_molecules}")
    print(f"🎯 N×M matrix size: {len(source_features)} × {len(target_features)}")

    if cache_stats_after:
        persistent_stats = cache_stats_after.get("persistent_cache_stats", {})
        print(f"📈 Cache efficiency: {persistent_stats.get('hit_rate', 0):.2%} hit rate")
        print("💾 Memory saved through deduplication")
        print("💿 Persistent caching enabled for future runs")

    print("\n✅ Example completed successfully!")
    print("\n💡 Key Benefits Demonstrated:")
    print("   - Persistent caching prevents recomputation across sessions")
    print("   - Global SMILES deduplication reduces redundant calculations")
    print("   - Memory-efficient storage handles large datasets")
    print("   - Optimized for N×M distance matrix computations")
    print("   - Significant performance improvements for repeated operations")


def advanced_usage_example():
    """Advanced usage example with custom configurations."""

    print("\n🚀 Advanced Usage Example")
    print("=" * 50)

    # Custom cache configuration
    cache_dir = Path("cache/advanced_features")

    # Initialize with custom memory storage settings
    memory_storage = MemoryEfficientFeatureStorage(
        storage_dir=cache_dir / "memory_efficient",
        use_memory_mapping=True,
        compression_level=9,  # Maximum compression
        max_memory_cache_mb=2048,  # 2GB memory cache
    )

    print("💾 Advanced memory storage initialized")
    print("🗜️  Compression level: 9 (maximum)")
    print("🧠 Memory cache: 2GB")
    print("🔗 Memory mapping: Enabled")

    # Example: Store and retrieve features manually
    print("\n🔧 Manual feature storage example")

    # Generate sample features
    sample_features = np.random.rand(1000, 2048).astype(np.float32)
    sample_metadata = {"featurizer": "ecfp", "created_at": time.time(), "shape": sample_features.shape}

    # Store features
    storage_path = memory_storage.store_features(
        identifier="sample_molecule_set", features=sample_features, metadata=sample_metadata
    )
    print(f"✅ Stored sample features: {storage_path}")

    # Load features back
    loaded_features = memory_storage.load_features("sample_molecule_set")
    if loaded_features is not None:
        print(f"✅ Loaded features shape: {loaded_features.shape}")
        print(f"🔍 Features match: {np.array_equal(sample_features, loaded_features)}")

    # Get storage statistics
    storage_stats = memory_storage.get_storage_stats()
    print("\n📊 Storage Statistics:")
    for key, value in storage_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    # Cleanup
    memory_storage.cleanup()
    print("\n🧹 Cleanup completed")


if __name__ == "__main__":
    try:
        main()
        advanced_usage_example()
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
