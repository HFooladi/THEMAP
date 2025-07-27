# Basic Distance Computation

This tutorial covers the fundamentals of computing distances between molecular and protein datasets using THEMAP.

## Overview

Distance computation is at the core of THEMAP's functionality. This tutorial will show you how to:

1. Choose appropriate distance metrics
2. Configure distance calculations
3. Handle different data types
4. Optimize performance

## Distance Methods Comparison

### Euclidean Distance
- **Fast** and memory-efficient
- Good for initial exploration
- Works with any embedding

```python
from themap.distance import MoleculeDatasetDistance

distance_calc = MoleculeDatasetDistance(
    tasks=None,
    molecule_method="euclidean"
)
```

### Cosine Distance
- Focuses on feature orientation
- Good for high-dimensional embeddings
- Normalized by vector magnitude

```python
distance_calc = MoleculeDatasetDistance(
    tasks=None,
    molecule_method="cosine"
)
```

### OTDD (Optimal Transport Dataset Distance)
- Most comprehensive but computationally expensive
- Considers both features and labels
- Best for detailed analysis

```python
distance_calc = MoleculeDatasetDistance(
    tasks=None,
    molecule_method="otdd"
)
```

## Working with Different Data Types

### Molecular Datasets

```python
from themap.data import MoleculeDataset
from themap.distance import MoleculeDatasetDistance

# Load datasets
datasets = [
    MoleculeDataset.load_from_file(f"datasets/train/{task_id}.jsonl.gz")
    for task_id in ["CHEMBL1023359", "CHEMBL1613776"]
]

# Compute all pairwise distances
distance_calc = MoleculeDatasetDistance(
    tasks=None,
    molecule_method="cosine"
)

# Set up for pairwise comparison
distance_calc.source_molecule_datasets = datasets
distance_calc.target_molecule_datasets = datasets
distance_calc.source_task_ids = [d.task_id for d in datasets]
distance_calc.target_task_ids = [d.task_id for d in datasets]

distances = distance_calc.get_distance()
```

### Protein Datasets

```python
from themap.data import ProteinDataset
from themap.distance import ProteinDatasetDistance

# Load protein sequences
proteins = ProteinDataset.load_from_file("datasets/train/train_proteins.fasta")

# Compute protein distances
protein_distance = ProteinDatasetDistance(
    tasks=None,
    protein_method="euclidean"
)

protein_distance.source_protein_datasets = proteins
protein_distance.target_protein_datasets = proteins

protein_distances = protein_distance.get_distance()
```

## Batch Processing

For multiple datasets:

```python
import os
from pathlib import Path

def compute_all_distances(data_dir, method="euclidean"):
    """Compute distances between all datasets in a directory."""

    # Find all dataset files
    dataset_files = list(Path(data_dir).glob("*.jsonl.gz"))

    # Load all datasets
    datasets = []
    for file_path in dataset_files:
        try:
            dataset = MoleculeDataset.load_from_file(str(file_path))
            datasets.append(dataset)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    # Compute distances
    if datasets:
        distance_calc = MoleculeDatasetDistance(
            tasks=None,
            molecule_method=method
        )

        distance_calc.source_molecule_datasets = datasets
        distance_calc.target_molecule_datasets = datasets
        distance_calc.source_task_ids = [d.task_id for d in datasets]
        distance_calc.target_task_ids = [d.task_id for d in datasets]

        return distance_calc.get_distance()

    return {}

# Example usage
distances = compute_all_distances("datasets/train/")
```

## Performance Optimization

### Memory Management

```python
# For large datasets, use euclidean instead of OTDD
distance_calc = MoleculeDatasetDistance(
    tasks=None,
    molecule_method="euclidean"  # Much faster than OTDD
)
```

### OTDD Configuration

```python
# Limit samples for OTDD to reduce computation time
distance_calc = MoleculeDatasetDistance(
    tasks=None,
    molecule_method="otdd"
)

# OTDD parameters are handled internally
# but you can monitor progress
```

## Interpreting Results

Distance results are returned as nested dictionaries:

```python
# Result structure: {target_id: {source_id: distance}}
distances = distance_calc.get_distance()

for target_id, source_distances in distances.items():
    print(f"Target: {target_id}")
    for source_id, distance in source_distances.items():
        print(f"  From {source_id}: {distance:.4f}")
```

### Distance Interpretation

- **0.0**: Identical datasets
- **Low values (< 1.0)**: Very similar datasets
- **Medium values (1.0-10.0)**: Moderately similar
- **High values (> 10.0)**: Very different datasets

Note: Actual ranges depend on the distance method and data characteristics.

## Common Patterns

### Finding Most Similar Datasets

```python
def find_most_similar(distances, target_task_id, top_k=3):
    """Find the most similar source tasks for a target task."""

    if target_task_id not in distances:
        return []

    source_distances = distances[target_task_id]

    # Sort by distance (ascending = most similar first)
    sorted_sources = sorted(
        source_distances.items(),
        key=lambda x: x[1]
    )

    return sorted_sources[:top_k]

# Example usage
similar_tasks = find_most_similar(distances, "CHEMBL2219358", top_k=3)
for source_id, distance in similar_tasks:
    print(f"Similar task: {source_id} (distance: {distance:.3f})")
```

### Creating Distance Matrix

```python
import pandas as pd

def create_distance_matrix(distances):
    """Convert nested distance dict to pandas DataFrame."""

    # Get all unique task IDs
    all_tasks = set()
    for target_id, source_dict in distances.items():
        all_tasks.add(target_id)
        all_tasks.update(source_dict.keys())

    all_tasks = sorted(list(all_tasks))

    # Create matrix
    matrix = pd.DataFrame(index=all_tasks, columns=all_tasks)

    for target_id, source_dict in distances.items():
        for source_id, distance in source_dict.items():
            matrix.loc[target_id, source_id] = distance

    return matrix

# Create and display matrix
distance_matrix = create_distance_matrix(distances)
print(distance_matrix)
```

## Troubleshooting

### Common Errors

1. **Memory Error**: Use euclidean distance or reduce dataset size
2. **Import Error**: Install required dependencies
3. **File Not Found**: Check file paths and data structure

### Performance Tips

1. Start with euclidean distance for exploration
2. Use OTDD only for final analysis
3. Cache results for repeated computations
4. Process in batches for large datasets

## Next Steps

- Learn about [working with tasks](working-with-tasks.md)
- Explore [performance optimization](performance-optimization.md)
- Try the unified task system
