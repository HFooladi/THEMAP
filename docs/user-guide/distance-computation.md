# Distance Computation Guide

This guide provides comprehensive information about computing distances between datasets, tasks, and molecular/protein spaces in THEMAP.

## Overview

Distance computation is central to THEMAP's functionality, enabling:

- **Dataset similarity assessment**: Compare chemical spaces between datasets
- **Transfer learning guidance**: Identify similar tasks for knowledge transfer
- **Task hardness estimation**: Quantify prediction difficulty
- **Multi-modal analysis**: Combine molecular, protein, and metadata information

## Distance Types

### 1. Molecular Dataset Distances

#### OTDD (Optimal Transport Dataset Distance)

OTDD provides the most comprehensive comparison by considering both feature distributions and label relationships.

```python
from themap.distance import MoleculeDatasetDistance

# Initialize with OTDD
mol_distance = MoleculeDatasetDistance(
    tasks=tasks,
    molecule_method="otdd"
)

# Compute distances
distances = mol_distance.get_distance()
```

**When to use OTDD:**
- ✅ High accuracy requirements
- ✅ Moderate dataset sizes (< 10,000 molecules)
- ✅ Both features and labels are important
- ❌ Large-scale computations (memory intensive)

**Configuration options:**
```python
# Customize OTDD parameters
hopts = mol_distance.get_hopts("molecule")
print(hopts)
# {'maxsamples': 1000, 'device': 'auto', ...}

# Modify parameters through configuration file
# themap/models/distance_configures/otdd.json
```

#### Euclidean Distance

Fast and interpretable distance based on feature vector similarity.

```python
mol_distance = MoleculeDatasetDistance(
    tasks=tasks,
    molecule_method="euclidean"
)

distances = mol_distance.get_distance()
```

**When to use Euclidean:**
- ✅ Large datasets (> 10,000 molecules)
- ✅ Fast computation requirements
- ✅ Feature magnitude is important
- ❌ High-dimensional sparse features

#### Cosine Distance

Measures angular similarity, good for high-dimensional feature spaces.

```python
mol_distance = MoleculeDatasetDistance(
    tasks=tasks,
    molecule_method="cosine"
)

distances = mol_distance.get_distance()
```

**When to use Cosine:**
- ✅ High-dimensional features
- ✅ Sparse feature vectors
- ✅ Feature orientation matters more than magnitude
- ❌ When magnitude differences are important

### 2. Protein Dataset Distances

Protein distances focus on sequence and structural similarity.

```python
from themap.distance import ProteinDatasetDistance

# Euclidean distance for protein features
prot_distance = ProteinDatasetDistance(
    tasks=tasks,
    protein_method="euclidean"
)

distances = prot_distance.get_distance()
```

**Available methods:**
- `"euclidean"`: Standard L2 distance
- `"cosine"`: Angular similarity
- `"sequence_identity"`: (Future) Direct sequence comparison

### 3. Combined Task Distances

Integrate multiple data modalities for comprehensive task comparison.

```python
from themap.distance import TaskDistance

# Initialize with multiple methods
task_distance = TaskDistance(
    tasks=tasks,
    molecule_method="cosine",
    protein_method="euclidean",
    metadata_method="jaccard"
)

# Compute all distance types
all_distances = task_distance.compute_all_distances(
    combination_strategy="weighted_average",
    molecule_weight=0.5,
    protein_weight=0.3,
    metadata_weight=0.2
)
```

**Combination strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `"average"` | Simple arithmetic mean | Equal importance of modalities |
| `"weighted_average"` | Weighted combination | Different modality importance |
| `"min"` | Minimum distance | Conservative similarity |
| `"max"` | Maximum distance | Liberal dissimilarity |

## Working with Features

### Feature Computation

THEMAP provides unified feature extraction across data modalities:

```python
# Molecular features
mol_features = tasks.compute_all_task_features(
    molecule_featurizer="ecfp",
    combination_method="concatenate",
    folds=["TRAIN", "TEST"]
)

# Protein features
prot_features = tasks.compute_all_task_features(
    protein_featurizer="esm2_t33_650M_UR50D",
    combination_method="concatenate"
)

# Combined multi-modal features
combined_features = tasks.compute_all_task_features(
    molecule_featurizer="morgan",
    protein_featurizer="esm2_t33_650M_UR50D",
    metadata_configs={
        "assay_description": {"featurizer_name": "sentence-transformers"},
        "bioactivity": {"featurizer_name": "standardize"}
    },
    combination_method="concatenate"
)
```

### Feature Caching

Expensive feature computations can be cached for reuse:

```python
# Save computed features
tasks.save_task_features_to_file(
    output_path="cache/task_features.pkl",
    molecule_featurizer="ecfp",
    protein_featurizer="esm2_t33_650M_UR50D"
)

# Load cached features
cached_features = Tasks.load_task_features_from_file("cache/task_features.pkl")
```

### Distance Matrix Organization

THEMAP organizes distance computations for N×M comparisons:

```python
# Get features ready for distance computation
source_features, target_features, source_names, target_names = (
    tasks.get_distance_computation_ready_features(
        molecule_featurizer="ecfp",
        protein_featurizer="esm2_t33_650M_UR50D",
        combination_method="concatenate",
        source_fold="TRAIN",      # N source tasks
        target_folds=["TEST"]     # M target tasks
    )
)

print(f"Computing {len(source_features)}×{len(target_features)} distance matrix")
```

## External Distance Matrices

Work with pre-computed distance matrices from external sources:

### Loading External Matrices

```python
# Load chemical space distances
chem_distances = TaskDistance.load_ext_chem_distance("external_chem_dist.pkl")

# Load protein space distances
prot_distances = TaskDistance.load_ext_prot_distance("external_prot_dist.pkl")

# Initialize with external matrices
import numpy as np

external_matrix = np.random.rand(10, 8)  # 10 source × 8 target
task_distance = TaskDistance(
    tasks=None,
    source_task_ids=["train_1", "train_2", ...],
    target_task_ids=["test_1", "test_2", ...],
    external_chemical_space=external_matrix
)
```

### Expected File Format

External distance files should contain:

```python
{
    "source_task_ids": ["CHEMBL001", "CHEMBL002", ...],    # or train_chembl_ids
    "target_task_ids": ["CHEMBL100", "CHEMBL101", ...],    # or test_chembl_ids
    "distance_matrices": numpy_array_or_tensor              # Shape: (n_targets, n_sources)
}
```

## Performance Optimization

### Memory Management

```python
# Monitor memory usage
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

check_memory()

# Compute distances in batches for large datasets
def compute_batched_distances(tasks, batch_size=100):
    results = {}

    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i+batch_size]
        batch_distance = MoleculeDatasetDistance(
            tasks=batch_tasks,
            molecule_method="euclidean"  # Faster method
        )
        batch_results = batch_distance.get_distance()
        results.update(batch_results)

    return results
```

### Computational Efficiency

```python
# Choose methods based on dataset size
def choose_distance_method(num_molecules):
    if num_molecules < 1000:
        return "otdd"           # Most accurate
    elif num_molecules < 10000:
        return "cosine"         # Good balance
    else:
        return "euclidean"      # Fastest

# Parallel computation for independent distances
from concurrent.futures import ProcessPoolExecutor

def compute_parallel_distances(task_pairs):
    def compute_single_distance(task_pair):
        source_task, target_task = task_pair
        distance_calc = MoleculeDatasetDistance(
            tasks=[source_task, target_task],
            molecule_method="euclidean"
        )
        return distance_calc.get_distance()

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(compute_single_distance, task_pairs))

    return results
```

### GPU Acceleration

For OTDD computations with GPU support:

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name()}")

    # OTDD will automatically use GPU if available
    mol_distance = MoleculeDatasetDistance(
        tasks=tasks,
        molecule_method="otdd"
    )

    # Monitor GPU memory
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
else:
    print("Using CPU computation")
```

## Error Handling and Debugging

### Common Issues and Solutions

```python
from themap.distance import DistanceComputationError, DataValidationError

try:
    distances = mol_distance.get_distance()
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install -e '.[otdd]'")

except DistanceComputationError as e:
    print(f"Distance computation failed: {e}")
    # Fallback to simpler method
    mol_distance.molecule_method = "euclidean"
    distances = mol_distance.get_distance()

except DataValidationError as e:
    print(f"Data validation failed: {e}")
    # Check data format and task IDs

except torch.cuda.OutOfMemoryError:
    print("GPU out of memory, falling back to CPU")
    torch.cuda.empty_cache()
    # Reduce batch size or use CPU
```

### Debugging Distance Computations

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate input data
def validate_distance_inputs(tasks):
    print(f"Number of tasks: {len(tasks)}")

    for i, task in enumerate(tasks[:3]):  # Check first 3 tasks
        print(f"Task {i}: {task.task_id}")
        if task.molecule_dataset:
            print(f"  Molecules: {len(task.molecule_dataset)}")
        if task.protein_dataset:
            print(f"  Proteins: {len(task.protein_dataset)}")

# Check feature computation
source_features, target_features, source_names, target_names = (
    tasks.get_distance_computation_ready_features(
        molecule_featurizer="ecfp",
        combination_method="concatenate",
        source_fold="TRAIN",
        target_folds=["TEST"]
    )
)

print(f"Source features: {len(source_features)} × {len(source_features[0]) if source_features else 0}")
print(f"Target features: {len(target_features)} × {len(target_features[0]) if target_features else 0}")
print(f"Source names: {source_names}")
print(f"Target names: {target_names}")
```

## Analysis and Visualization

### Converting to Analysis-Ready Formats

```python
# Convert to pandas DataFrame
df_distances = task_distance.to_pandas("molecule")
print(df_distances.head())

# Statistical analysis
print(f"Mean distance: {df_distances.values.mean():.3f}")
print(f"Distance std: {df_distances.values.std():.3f}")

# Find most similar tasks
min_distance_idx = df_distances.values.argmin()
row, col = np.unravel_index(min_distance_idx, df_distances.shape)
print(f"Most similar: {df_distances.index[row]} ↔ {df_distances.columns[col]}")
```

### Visualization Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distance matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_distances, annot=True, cmap='viridis', fmt='.2f')
plt.title('Task Distance Matrix')
plt.show()

# Distance distribution
plt.figure(figsize=(8, 6))
distances_flat = df_distances.values.flatten()
plt.hist(distances_flat, bins=20, alpha=0.7)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.title('Distance Distribution')
plt.show()

# Hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Convert to condensed distance matrix
condensed_distances = squareform(df_distances.values)
linkage_matrix = linkage(condensed_distances, method='ward')

plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=df_distances.index, orientation='top')
plt.title('Task Similarity Dendrogram')
plt.xticks(rotation=45)
plt.show()
```

This comprehensive guide covers all aspects of distance computation in THEMAP. For specific use cases and advanced examples, see our [tutorials](../tutorials/index.md) and [API documentation](../api/distance.md).
