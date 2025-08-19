# Examples

This section provides practical, runnable examples for common THEMAP use cases. Each example includes complete code, sample data, and explanations.

## Quick Examples

### 1. Basic Molecular Distance

```python
"""
Compute distance between two molecular datasets using OTDD.
"""
from themap.data import MoleculeDataset
from themap.distance import MoleculeDatasetDistance
from dpu_utils.utils.richpath import RichPath

# Load datasets
source = MoleculeDataset.load_from_file(
    RichPath.create("datasets/train/CHEMBL1023359.jsonl.gz")
)
target = MoleculeDataset.load_from_file(
    RichPath.create("datasets/test/CHEMBL2219358.jsonl.gz")
)

# Compute OTDD distance
distance_calc = MoleculeDatasetDistance(
    tasks=None,
    molecule_method="otdd"
)
distance_calc.source_molecule_datasets = [source]
distance_calc.target_molecule_datasets = [target]
distance_calc.source_task_ids = [source.task_id]
distance_calc.target_task_ids = [target.task_id]

result = distance_calc.get_distance()
print(f"OTDD distance: {result}")
```

### 2. Protein Similarity Analysis

```python
"""
Analyze protein similarity using sequence embeddings.
"""
from themap.data import ProteinMetadataDatasets
from themap.distance import ProteinDatasetDistance

# Load protein sequences
proteins = ProteinMetadataDatasets.from_directory("datasets/train/")

# Compute pairwise distances
prot_distance = ProteinDatasetDistance(
    tasks=None,
    protein_method="euclidean"
)

# Set up for self-comparison
prot_distance.source_protein_datasets = proteins
prot_distance.target_protein_datasets = proteins

distances = prot_distance.get_distance()
print("Protein distance matrix:")
for target_id, source_distances in distances.items():
    print(f"{target_id}: {source_distances}")
```

### 3. Unified Task System

```python
"""
Work with the unified task system for multi-modal analysis.
"""
from themap.data.tasks import Tasks
from themap.distance import TaskDistance

# Load integrated tasks
tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=True,
    cache_dir="cache/"
)

# Compute combined distances
task_distance = TaskDistance(
    tasks=tasks,
    molecule_method="cosine",
    protein_method="euclidean"
)

# Get all distance types
all_distances = task_distance.compute_all_distances(
    combination_strategy="weighted_average",
    molecule_weight=0.7,
    protein_weight=0.3
)

print("Distance computation complete:")
print(f"Molecule distances: {len(all_distances['molecule'])} tasks")
print(f"Protein distances: {len(all_distances['protein'])} tasks")
print(f"Combined distances: {len(all_distances['combined'])} tasks")
```

## Complete Workflow Examples

### Transfer Learning Pipeline

```python
"""
Complete transfer learning workflow with task hardness estimation.
"""
import numpy as np
import pandas as pd
from themap.data.tasks import Tasks
from themap.distance import TaskDistance
from themap.hardness import TaskHardness

def transfer_learning_pipeline(source_tasks_dir, target_tasks_dir):
    """
    Complete pipeline for transfer learning task selection.
    """

    # 1. Load source and target tasks
    print("📂 Loading tasks...")
    source_tasks = Tasks.from_directory(
        directory=source_tasks_dir,
        load_molecules=True,
        load_proteins=True
    )

    target_tasks = Tasks.from_directory(
        directory=target_tasks_dir,
        load_molecules=True,
        load_proteins=True
    )

    print(f"Source tasks: {len(source_tasks)}")
    print(f"Target tasks: {len(target_tasks)}")

    # 2. Compute task distances
    print("📏 Computing task distances...")
    task_distance = TaskDistance(
        tasks=source_tasks + target_tasks,  # Combined for comparison
        molecule_method="cosine",
        protein_method="euclidean"
    )

    distances = task_distance.compute_all_distances()

    # 3. Estimate task hardness
    print("💪 Estimating task hardness...")
    hardness_estimator = TaskHardness(
        tasks=target_tasks,
        method="combined_distance"
    )

    hardness_scores = hardness_estimator.estimate_hardness(distances['combined'])

    # 4. Create transferability matrix
    print("🗺️ Creating transferability map...")
    transfer_matrix = pd.DataFrame(distances['combined'])

    # 5. Recommend source tasks for each target
    recommendations = {}
    for target_task in target_tasks.get_task_ids():
        if target_task in transfer_matrix.index:
            # Get closest source tasks
            source_distances = transfer_matrix.loc[target_task]
            top_sources = source_distances.nsmallest(3)

            recommendations[target_task] = {
                'recommended_sources': top_sources.index.tolist(),
                'distances': top_sources.values.tolist(),
                'estimated_hardness': hardness_scores.get(target_task, 'N/A')
            }

    return {
        'transfer_matrix': transfer_matrix,
        'hardness_scores': hardness_scores,
        'recommendations': recommendations,
        'distances': distances
    }

# Run pipeline
if __name__ == "__main__":
    results = transfer_learning_pipeline(
        source_tasks_dir="datasets/train/",
        target_tasks_dir="datasets/test/"
    )

    # Print recommendations
    print("\n🎯 Transfer Learning Recommendations:")
    for target, rec in results['recommendations'].items():
        print(f"\nTarget: {target}")
        print(f"Hardness: {rec['estimated_hardness']:.3f}")
        print("Best source tasks:")
        for source, dist in zip(rec['recommended_sources'], rec['distances']):
            print(f"  {source}: distance={dist:.3f}")
```

### Batch Processing Pipeline

```python
"""
Process large numbers of datasets efficiently.
"""
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from themap.distance import MoleculeDatasetDistance
from themap.data import MoleculeDataset

def process_single_comparison(source_path, target_path, method="euclidean"):
    """Process a single dataset comparison."""
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
            'method': method,
            'status': 'success'
        }

    except Exception as e:
        return {
            'source': Path(source_path).stem,
            'target': Path(target_path).stem,
            'distance': None,
            'method': method,
            'status': f'error: {str(e)}'
        }

def batch_distance_computation(
    source_dir,
    target_dir,
    method="euclidean",
    max_workers=4,
    output_file="distance_results.csv"
):
    """
    Compute distances between all source-target dataset pairs.
    """

    # Find all dataset files
    source_files = list(Path(source_dir).glob("*.jsonl.gz"))
    target_files = list(Path(target_dir).glob("*.jsonl.gz"))

    print(f"Found {len(source_files)} source and {len(target_files)} target datasets")

    # Create all combinations
    comparisons = [
        (str(source), str(target))
        for source in source_files
        for target in target_files
    ]

    print(f"Total comparisons: {len(comparisons)}")

    # Process in parallel
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_comparison = {
            executor.submit(process_single_comparison, source, target, method): (source, target)
            for source, target in comparisons
        }

        # Collect results
        for future in as_completed(future_to_comparison):
            result = future.result()
            results.append(result)
            completed += 1

            if completed % 10 == 0:
                print(f"Completed: {completed}/{len(comparisons)}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)

    print(f"\n✅ Results saved to {output_file}")
    print(f"Successful computations: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed computations: {sum(1 for r in results if r['status'] != 'success')}")

    return df_results

# Run batch processing
if __name__ == "__main__":
    results_df = batch_distance_computation(
        source_dir="datasets/train/",
        target_dir="datasets/test/",
        method="euclidean",
        max_workers=4
    )

    # Basic analysis
    successful_results = results_df[results_df['status'] == 'success']
    if len(successful_results) > 0:
        print(f"\nDistance statistics:")
        print(f"Mean distance: {successful_results['distance'].mean():.3f}")
        print(f"Std distance: {successful_results['distance'].std():.3f}")
        print(f"Min distance: {successful_results['distance'].min():.3f}")
        print(f"Max distance: {successful_results['distance'].max():.3f}")
```

### Performance Benchmarking

```python
"""
Benchmark different distance methods for performance and accuracy.
"""
import time
import psutil
import os
from themap.distance import MoleculeDatasetDistance
from themap.data.tasks import Tasks

def benchmark_distance_methods(tasks, methods=["euclidean", "cosine", "otdd"]):
    """
    Benchmark different distance computation methods.
    """

    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    results = {}

    for method in methods:
        print(f"\n🔍 Benchmarking {method} method...")

        # Memory before
        memory_before = get_memory_usage()

        # Time the computation
        start_time = time.time()

        try:
            # Create distance calculator
            distance_calc = MoleculeDatasetDistance(
                tasks=tasks,
                molecule_method=method
            )

            # Compute distances
            distances = distance_calc.get_distance()

            end_time = time.time()
            memory_after = get_memory_usage()

            # Calculate metrics
            computation_time = end_time - start_time
            memory_usage = memory_after - memory_before

            # Analyze results
            if distances:
                all_distances = [
                    dist for target_dists in distances.values()
                    for dist in target_dists.values()
                ]
                mean_distance = np.mean(all_distances)
                std_distance = np.std(all_distances)
            else:
                mean_distance = None
                std_distance = None

            results[method] = {
                'computation_time': computation_time,
                'memory_usage': memory_usage,
                'mean_distance': mean_distance,
                'std_distance': std_distance,
                'num_comparisons': sum(len(d) for d in distances.values()) if distances else 0,
                'status': 'success'
            }

            print(f"  ✅ Time: {computation_time:.2f}s")
            print(f"  💾 Memory: {memory_usage:.1f}MB")
            if mean_distance is not None:
                print(f"  📊 Mean distance: {mean_distance:.3f}")

        except Exception as e:
            end_time = time.time()
            results[method] = {
                'computation_time': end_time - start_time,
                'memory_usage': None,
                'mean_distance': None,
                'std_distance': None,
                'num_comparisons': 0,
                'status': f'failed: {str(e)}'
            }
            print(f"  ❌ Failed: {str(e)}")

    return results

def compare_methods(benchmark_results):
    """
    Compare benchmark results across methods.
    """
    print("\n📊 Benchmark Comparison:")
    print("-" * 80)
    print(f"{'Method':<10} {'Time (s)':<10} {'Memory (MB)':<12} {'Mean Dist':<10} {'Status':<15}")
    print("-" * 80)

    for method, results in benchmark_results.items():
        time_str = f"{results['computation_time']:.2f}" if results['computation_time'] else "N/A"
        memory_str = f"{results['memory_usage']:.1f}" if results['memory_usage'] else "N/A"
        dist_str = f"{results['mean_distance']:.3f}" if results['mean_distance'] else "N/A"

        print(f"{method:<10} {time_str:<10} {memory_str:<12} {dist_str:<10} {results['status']:<15}")

    # Performance ranking
    successful_methods = {
        method: results for method, results in benchmark_results.items()
        if results['status'] == 'success'
    }

    if successful_methods:
        print(f"\n🏆 Rankings:")

        # Speed ranking
        speed_ranking = sorted(
            successful_methods.items(),
            key=lambda x: x[1]['computation_time']
        )
        print(f"⚡ Fastest: {' > '.join([method for method, _ in speed_ranking])}")

        # Memory ranking
        memory_ranking = sorted(
            successful_methods.items(),
            key=lambda x: x[1]['memory_usage']
        )
        print(f"💾 Most memory efficient: {' > '.join([method for method, _ in memory_ranking])}")

# Run benchmark
if __name__ == "__main__":
    # Load sample tasks
    tasks = Tasks.from_directory(
        directory="datasets/",
        task_list_file="datasets/sample_tasks_list.json",
        load_molecules=True
    )

    # Run benchmark
    results = benchmark_distance_methods(
        tasks=tasks,
        methods=["euclidean", "cosine", "otdd"]
    )

    # Compare results
    compare_methods(results)
```

## Visualization Examples

### Distance Matrix Heatmap

```python
"""
Create publication-ready distance matrix visualizations.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from themap.distance import TaskDistance

def plot_distance_matrix(distance_dict, title="Task Distance Matrix", figsize=(10, 8)):
    """
    Create a heatmap visualization of task distances.
    """
    # Convert to DataFrame
    df = pd.DataFrame(distance_dict)

    # Create figure
    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        df,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'Distance'},
        square=True
    )

    plt.title(title)
    plt.xlabel('Source Tasks')
    plt.ylabel('Target Tasks')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    return plt.gcf()

def plot_distance_comparison(molecule_distances, protein_distances, combined_distances):
    """
    Compare different distance types in a multi-panel plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    distance_types = [
        (molecule_distances, "Molecule Distances", axes[0]),
        (protein_distances, "Protein Distances", axes[1]),
        (combined_distances, "Combined Distances", axes[2])
    ]

    for distances, title, ax in distance_types:
        df = pd.DataFrame(distances)

        sns.heatmap(
            df,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            ax=ax,
            cbar_kws={'label': 'Distance'}
        )

        ax.set_title(title)
        ax.set_xlabel('Source Tasks')
        ax.set_ylabel('Target Tasks')

    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Load tasks and compute distances
    tasks = Tasks.from_directory("datasets/", load_molecules=True, load_proteins=True)

    task_distance = TaskDistance(tasks=tasks)
    all_distances = task_distance.compute_all_distances()

    # Create visualizations
    fig1 = plot_distance_matrix(
        all_distances['molecule'],
        title="Molecular Dataset Distances"
    )

    fig2 = plot_distance_comparison(
        all_distances['molecule'],
        all_distances['protein'],
        all_distances['combined']
    )

    # Save figures
    fig1.savefig('molecule_distances.png', dpi=300, bbox_inches='tight')
    fig2.savefig('distance_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()
```

## Script Templates

### Command-Line Interface

```python
#!/usr/bin/env python3
"""
Command-line interface for THEMAP distance computation.

Usage:
    python distance_cli.py --source datasets/train/ --target datasets/test/ --method otdd
"""

import argparse
import json
from pathlib import Path
from themap.data.tasks import Tasks
from themap.distance import TaskDistance

def main():
    parser = argparse.ArgumentParser(description="THEMAP Distance Computation CLI")

    parser.add_argument("--source", required=True, help="Source tasks directory")
    parser.add_argument("--target", required=True, help="Target tasks directory")
    parser.add_argument("--method", default="euclidean",
                       choices=["euclidean", "cosine", "otdd"],
                       help="Distance computation method")
    parser.add_argument("--output", default="distances.json",
                       help="Output file for distances")
    parser.add_argument("--cache-dir", default="cache/",
                       help="Cache directory for features")
    parser.add_argument("--molecule-featurizer", default="ecfp",
                       help="Molecular featurizer to use")
    parser.add_argument("--protein-featurizer", default="esm2_t33_650M_UR50D",
                       help="Protein featurizer to use")

    args = parser.parse_args()

    print(f"🔍 Computing {args.method} distances")
    print(f"📂 Source: {args.source}")
    print(f"🎯 Target: {args.target}")

    # Load tasks
    source_tasks = Tasks.from_directory(
        directory=args.source,
        load_molecules=True,
        load_proteins=True,
        cache_dir=args.cache_dir
    )

    target_tasks = Tasks.from_directory(
        directory=args.target,
        load_molecules=True,
        load_proteins=True,
        cache_dir=args.cache_dir
    )

    print(f"📊 Loaded {len(source_tasks)} source and {len(target_tasks)} target tasks")

    # Compute distances
    all_tasks = source_tasks + target_tasks
    task_distance = TaskDistance(
        tasks=all_tasks,
        molecule_method=args.method,
        protein_method=args.method
    )

    distances = task_distance.get_distance()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(distances, f, indent=2)

    print(f"💾 Results saved to {output_path}")
    print(f"✅ Computed {sum(len(d) for d in distances.values())} pairwise distances")

if __name__ == "__main__":
    main()
```

These examples provide a solid foundation for working with THEMAP in various scenarios. Each example is self-contained and can be adapted to your specific use case. For more detailed explanations, see our [tutorials](../tutorials/index.md).
