# Performance Optimization

This tutorial covers best practices for optimizing THEMAP performance when working with large datasets and complex distance computations.

## Overview

THEMAP offers several strategies to optimize performance:

1. **Method selection**: Choose appropriate distance metrics
2. **Caching**: Leverage feature caching for repeated computations
3. **Memory management**: Handle large datasets efficiently
4. **Parallel processing**: Utilize multiple cores when possible
5. **Data preprocessing**: Optimize data loading and storage

## Distance Method Performance

### Speed Comparison

| Method | Speed | Memory | Accuracy | Best For |
|--------|-------|--------|----------|----------|
| **Euclidean** | ⚡⚡⚡ | 💾 | ⭐⭐ | Initial exploration, large datasets |
| **Cosine** | ⚡⚡⚡ | 💾 | ⭐⭐⭐ | High-dimensional features |
| **OTDD** | ⚡ | 💾💾💾 | ⭐⭐⭐⭐⭐ | Detailed analysis, small-medium datasets |

### Choosing the Right Method

```python
import time
from themap.distance import MoleculeDatasetDistance
from themap.data import MoleculeDataset

def benchmark_methods(datasets, methods=["euclidean", "cosine", "otdd"]):
    """Benchmark different distance methods."""

    results = {}

    for method in methods:
        print(f"Benchmarking {method}...")

        start_time = time.time()

        try:
            distance_calc = MoleculeDatasetDistance(
                tasks=None,
                molecule_method=method
            )

            distance_calc.source_molecule_datasets = datasets[:2]
            distance_calc.target_molecule_datasets = datasets[:2]
            distance_calc.source_task_ids = [d.task_id for d in datasets[:2]]
            distance_calc.target_task_ids = [d.task_id for d in datasets[:2]]

            distances = distance_calc.get_distance()

            elapsed = time.time() - start_time
            results[method] = {
                'time': elapsed,
                'success': True,
                'distances': distances
            }

            print(f"  ✅ {method}: {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            results[method] = {
                'time': elapsed,
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ {method}: Failed after {elapsed:.2f}s")

    return results

# Load sample datasets
datasets = [
    MoleculeDataset.load_from_file(f"datasets/train/{task_id}.jsonl.gz")
    for task_id in ["CHEMBL1023359", "CHEMBL1613776"]
]

# Run benchmark
benchmark_results = benchmark_methods(datasets)
```

## Caching Strategies

### Feature Caching

```python
from themap.data.tasks import Tasks

# Enable comprehensive caching
tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=True,
    cache_dir="cache/",  # All computed features cached here
    force_reload=False   # Use existing cache when available
)

print("Features will be cached for future use")
```

### Custom Cache Management

```python
import os
import shutil
from pathlib import Path

def manage_cache(cache_dir="cache/"):
    """Manage THEMAP cache directory."""

    cache_path = Path(cache_dir)

    if cache_path.exists():
        # Get cache size
        total_size = sum(
            f.stat().st_size for f in cache_path.rglob('*') if f.is_file()
        )

        print(f"Cache directory: {cache_dir}")
        print(f"Cache size: {total_size / (1024**3):.2f} GB")
        print(f"Cached files: {len(list(cache_path.rglob('*')))}")

        # List cache contents
        for subdir in cache_path.iterdir():
            if subdir.is_dir():
                n_files = len(list(subdir.rglob('*')))
                print(f"  {subdir.name}: {n_files} files")
    else:
        print(f"Cache directory {cache_dir} does not exist")

def clear_cache(cache_dir="cache/", confirm=True):
    """Clear the cache directory."""

    if confirm:
        response = input(f"Clear cache directory {cache_dir}? (y/N): ")
        if response.lower() != 'y':
            return

    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"Cache cleared: {cache_dir}")
    else:
        print(f"Cache directory {cache_dir} does not exist")

# Example usage
manage_cache()
```

## Memory Management

### Large Dataset Handling

```python
def process_large_datasets(dataset_dir, batch_size=5, method="euclidean"):
    """Process large numbers of datasets in batches."""

    from pathlib import Path
    import gc

    # Find all dataset files
    dataset_files = list(Path(dataset_dir).glob("*.jsonl.gz"))
    print(f"Found {len(dataset_files)} datasets")

    # Process in batches
    all_results = {}

    for i in range(0, len(dataset_files), batch_size):
        batch_files = dataset_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch_files)} files")

        # Load batch
        batch_datasets = []
        for file_path in batch_files:
            try:
                dataset = MoleculeDataset.load_from_file(str(file_path))
                batch_datasets.append(dataset)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        if len(batch_datasets) < 2:
            continue

        # Compute distances for this batch
        try:
            distance_calc = MoleculeDatasetDistance(
                tasks=None,
                molecule_method=method
            )

            distance_calc.source_molecule_datasets = batch_datasets
            distance_calc.target_molecule_datasets = batch_datasets
            distance_calc.source_task_ids = [d.task_id for d in batch_datasets]
            distance_calc.target_task_ids = [d.task_id for d in batch_datasets]

            batch_results = distance_calc.get_distance()
            all_results.update(batch_results)

            print(f"  ✅ Computed {len(batch_results)} distances")

        except Exception as e:
            print(f"  ❌ Batch failed: {e}")

        # Clean up memory
        del batch_datasets
        gc.collect()

    return all_results

# Process large dataset directory
results = process_large_datasets("datasets/train/", batch_size=3)
```

### Memory Monitoring

```python
import psutil
import os

def monitor_memory():
    """Monitor current memory usage."""

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    print(f"RSS Memory: {memory_info.rss / (1024**3):.2f} GB")
    print(f"VMS Memory: {memory_info.vms / (1024**3):.2f} GB")

    # System memory
    system_memory = psutil.virtual_memory()
    print(f"System Memory: {system_memory.percent}% used")

    return memory_info

def memory_efficient_processing(tasks, max_memory_gb=8):
    """Process tasks with memory monitoring."""

    initial_memory = monitor_memory()

    # Estimate memory per task
    sample_memory = initial_memory.rss / (1024**3)

    # Calculate safe batch size
    available_memory = max_memory_gb - sample_memory
    estimated_memory_per_task = 0.1  # GB, adjust based on your data
    safe_batch_size = max(1, int(available_memory / estimated_memory_per_task))

    print(f"Using batch size: {safe_batch_size}")

    # Process in safe batches
    task_ids = tasks.get_task_ids()

    for i in range(0, len(task_ids), safe_batch_size):
        batch_ids = task_ids[i:i + safe_batch_size]

        print(f"Processing batch {i//safe_batch_size + 1}")
        monitor_memory()

        # Process batch here
        # ... your processing code ...

        # Force garbage collection
        import gc
        gc.collect()

# Example usage
# memory_efficient_processing(tasks, max_memory_gb=16)
```

## Parallel Processing

### Multi-Processing for Distance Computation

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def compute_single_distance(args):
    """Compute distance between two datasets (for multiprocessing)."""

    source_path, target_path, method = args

    try:
        # Load datasets
        source = MoleculeDataset.load_from_file(source_path)
        target = MoleculeDataset.load_from_file(target_path)

        # Compute distance
        distance_calc = MoleculeDatasetDistance(
            tasks=None,
            molecule_method=method
        )

        distance_calc.source_molecule_datasets = [source]
        distance_calc.target_molecule_datasets = [target]
        distance_calc.source_task_ids = [source.task_id]
        distance_calc.target_task_ids = [target.task_id]

        result = distance_calc.get_distance()

        return {
            'source': source.task_id,
            'target': target.task_id,
            'distance': list(result.values())[0][source.task_id],
            'status': 'success'
        }

    except Exception as e:
        return {
            'source': str(source_path),
            'target': str(target_path),
            'distance': None,
            'status': f'error: {str(e)}'
        }

def parallel_distance_computation(
    source_files,
    target_files,
    method="euclidean",
    max_workers=None
):
    """Compute distances in parallel."""

    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Don't use all cores

    # Create all combinations
    tasks = [
        (source, target, method)
        for source in source_files
        for target in target_files
    ]

    print(f"Computing {len(tasks)} distances using {max_workers} workers")

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(compute_single_distance, task): task
            for task in tasks
        }

        # Collect results
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed += 1

            if completed % 10 == 0:
                print(f"Completed: {completed}/{len(tasks)}")

    return results

# Example usage
source_files = ["datasets/train/CHEMBL1023359.jsonl.gz"]
target_files = ["datasets/test/CHEMBL2219358.jsonl.gz", "datasets/test/CHEMBL1963831.jsonl.gz"]

parallel_results = parallel_distance_computation(
    source_files,
    target_files,
    method="euclidean",
    max_workers=4
)
```

## Data Preprocessing Optimization

### Efficient Data Loading

```python
def optimize_data_loading(data_dir, cache_dir="cache/"):
    """Optimize data loading with preprocessing."""

    from pathlib import Path
    import pickle

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # Check for preprocessed data
    preprocessed_file = cache_path / "preprocessed_datasets.pkl"

    if preprocessed_file.exists():
        print("Loading preprocessed datasets...")
        with open(preprocessed_file, 'rb') as f:
            return pickle.load(f)

    print("Preprocessing datasets...")

    # Load and preprocess all datasets
    datasets = {}
    dataset_files = list(Path(data_dir).glob("*.jsonl.gz"))

    for i, file_path in enumerate(dataset_files):
        try:
            dataset = MoleculeDataset.load_from_file(str(file_path))

            # Precompute features if needed
            # This could include molecular descriptors, embeddings, etc.

            datasets[dataset.task_id] = dataset

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset_files)} datasets")

        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    # Cache preprocessed data
    with open(preprocessed_file, 'wb') as f:
        pickle.dump(datasets, f)

    print(f"Preprocessed {len(datasets)} datasets")
    return datasets

# Load optimized datasets
datasets = optimize_data_loading("datasets/train/")
```

### Feature Pre-computation

```python
def precompute_features(tasks, force_recompute=False):
    """Precompute and cache molecular/protein features."""

    print("Precomputing features...")

    # This will trigger feature computation and caching
    task_distance = TaskDistance(
        tasks=tasks,
        molecule_method="euclidean",  # Fast method for precomputation
        protein_method="euclidean"
    )

    # Force feature computation by attempting distance calculation
    try:
        # This will compute and cache all features
        sample_distances = task_distance.get_molecule_distances()
        print("✅ Molecular features precomputed")

        sample_distances = task_distance.get_protein_distances()
        print("✅ Protein features precomputed")

    except Exception as e:
        print(f"⚠️ Feature precomputation failed: {e}")

    print("Feature precomputation complete")

# Precompute features for faster subsequent operations
# precompute_features(tasks)
```

## OTDD Optimization

### OTDD Parameter Tuning

```python
def optimize_otdd_parameters(datasets, max_samples_range=[100, 500, 1000]):
    """Find optimal OTDD parameters for your datasets."""

    import time

    results = {}

    for max_samples in max_samples_range:
        print(f"Testing OTDD with max_samples={max_samples}")

        start_time = time.time()

        try:
            distance_calc = MoleculeDatasetDistance(
                tasks=None,
                molecule_method="otdd"
            )

            # OTDD parameters are handled internally
            # But you can influence performance by dataset size

            distance_calc.source_molecule_datasets = datasets[:2]
            distance_calc.target_molecule_datasets = datasets[:2]
            distance_calc.source_task_ids = [d.task_id for d in datasets[:2]]
            distance_calc.target_task_ids = [d.task_id for d in datasets[:2]]

            distances = distance_calc.get_distance()

            elapsed = time.time() - start_time

            results[max_samples] = {
                'time': elapsed,
                'success': True,
                'distances': distances
            }

            print(f"  ✅ Completed in {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            results[max_samples] = {
                'time': elapsed,
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ Failed after {elapsed:.2f}s: {e}")

    return results

# Find optimal parameters
# otdd_results = optimize_otdd_parameters(datasets)
```

## Performance Monitoring

### Benchmarking Suite

```python
import time
import psutil
import os

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for THEMAP operations."""

    def __init__(self):
        self.results = {}

    def benchmark_distance_methods(self, datasets, methods=None):
        """Benchmark different distance methods."""

        if methods is None:
            methods = ["euclidean", "cosine"]  # Skip OTDD for speed

        for method in methods:
            print(f"\n🔍 Benchmarking {method}...")

            # Memory before
            memory_before = self._get_memory_usage()
            start_time = time.time()

            try:
                distance_calc = MoleculeDatasetDistance(
                    tasks=None,
                    molecule_method=method
                )

                distance_calc.source_molecule_datasets = datasets
                distance_calc.target_molecule_datasets = datasets
                distance_calc.source_task_ids = [d.task_id for d in datasets]
                distance_calc.target_task_ids = [d.task_id for d in datasets]

                distances = distance_calc.get_distance()

                # Calculate metrics
                elapsed = time.time() - start_time
                memory_after = self._get_memory_usage()
                memory_used = memory_after - memory_before

                n_comparisons = sum(len(d) for d in distances.values())

                self.results[method] = {
                    'time': elapsed,
                    'memory_mb': memory_used,
                    'comparisons': n_comparisons,
                    'comparisons_per_second': n_comparisons / elapsed,
                    'status': 'success'
                }

                print(f"  ✅ Time: {elapsed:.2f}s")
                print(f"  💾 Memory: {memory_used:.1f}MB")
                print(f"  📊 {n_comparisons} comparisons ({n_comparisons/elapsed:.1f}/s)")

            except Exception as e:
                elapsed = time.time() - start_time
                self.results[method] = {
                    'time': elapsed,
                    'memory_mb': 0,
                    'comparisons': 0,
                    'comparisons_per_second': 0,
                    'status': f'failed: {str(e)}'
                }
                print(f"  ❌ Failed: {e}")

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def print_summary(self):
        """Print benchmark summary."""

        print("\n📊 Benchmark Summary:")
        print("-" * 80)
        print(f"{'Method':<12} {'Time (s)':<10} {'Memory (MB)':<12} {'Comp/sec':<10} {'Status'}")
        print("-" * 80)

        for method, results in self.results.items():
            print(f"{method:<12} {results['time']:<10.2f} {results['memory_mb']:<12.1f} "
                  f"{results['comparisons_per_second']:<10.1f} {results['status']}")

        # Recommendations
        successful = {k: v for k, v in self.results.items() if v['status'] == 'success'}

        if successful:
            fastest = min(successful.items(), key=lambda x: x[1]['time'])
            most_efficient = min(successful.items(), key=lambda x: x[1]['memory_mb'])

            print(f"\n🏆 Recommendations:")
            print(f"⚡ Fastest: {fastest[0]} ({fastest[1]['time']:.2f}s)")
            print(f"💾 Most memory efficient: {most_efficient[0]} ({most_efficient[1]['memory_mb']:.1f}MB)")

# Run comprehensive benchmark
benchmark = PerformanceBenchmark()
# benchmark.benchmark_distance_methods(datasets[:3])  # Test with 3 datasets
# benchmark.print_summary()
```

## Best Practices Summary

### Quick Optimization Checklist

1. **✅ Use caching**: Always specify a cache directory
2. **✅ Choose appropriate methods**: Start with euclidean for exploration
3. **✅ Process in batches**: Handle large datasets in smaller chunks
4. **✅ Monitor memory**: Keep track of memory usage during processing
5. **✅ Parallel processing**: Use multiple cores for independent computations
6. **✅ Precompute features**: Cache expensive feature computations
7. **✅ Profile your workflow**: Identify bottlenecks with benchmarking

### Method Selection Guidelines

```python
def choose_distance_method(n_datasets, dataset_sizes, time_budget_minutes=60):
    """Helper function to choose appropriate distance method."""

    total_comparisons = n_datasets * (n_datasets - 1) // 2
    avg_dataset_size = sum(dataset_sizes) / len(dataset_sizes)

    print(f"Analysis: {total_comparisons} comparisons, avg {avg_dataset_size:.0f} molecules/dataset")

    if total_comparisons > 100 or avg_dataset_size > 1000:
        recommendation = "euclidean"
        reason = "Large number of comparisons or large datasets"
    elif total_comparisons > 50:
        recommendation = "cosine"
        reason = "Moderate number of comparisons"
    elif time_budget_minutes > 30:
        recommendation = "otdd"
        reason = "High accuracy needed and sufficient time budget"
    else:
        recommendation = "euclidean"
        reason = "Limited time budget"

    print(f"Recommended method: {recommendation}")
    print(f"Reason: {reason}")

    return recommendation

# Example usage
# method = choose_distance_method(
#     n_datasets=10,
#     dataset_sizes=[250, 180, 500, 300],
#     time_budget_minutes=30
# )
```

This comprehensive guide should help you optimize THEMAP performance for your specific use case. Start with the basic optimizations and gradually apply more advanced techniques as needed.
