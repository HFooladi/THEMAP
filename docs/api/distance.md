# Distance Computation API

The distance module provides comprehensive functionality for computing distances between molecular datasets, protein datasets, and tasks. This module supports various distance metrics and can handle both single dataset comparisons and batch comparisons across multiple datasets.

## Overview

The distance computation system consists of three main classes:

- **`MoleculeDatasetDistance`** - Computes distances between molecule datasets
- **`ProteinDatasetDistance`** - Computes distances between protein datasets
- **`TaskDistance`** - Unified interface for computing combined task distances

## Core Classes

### AbstractTasksDistance

::: themap.distance.base.AbstractTasksDistance
    options:
      show_root_heading: true
      heading_level: 3

### MoleculeDatasetDistance

::: themap.distance.molecule_distance.MoleculeDatasetDistance
    options:
      show_root_heading: true
      heading_level: 3

### ProteinDatasetDistance

::: themap.distance.protein_distance.ProteinDatasetDistance
    options:
      show_root_heading: true
      heading_level: 3

### TaskDistance

::: themap.distance.task_distance.TaskDistance
    options:
      show_root_heading: true
      heading_level: 3

## Utility Functions

### Validation Functions

::: themap.distance.base._validate_and_extract_task_id
    options:
      show_root_heading: true
      heading_level: 4

## Exception Classes

### DistanceComputationError

::: themap.distance.exceptions.DistanceComputationError
    options:
      show_root_heading: true
      heading_level: 4

### DataValidationError

::: themap.distance.exceptions.DataValidationError
    options:
      show_root_heading: true
      heading_level: 4

## Constants

### Supported Methods

```python
# Available distance methods for molecule datasets
MOLECULE_DISTANCE_METHODS = ["otdd", "euclidean", "cosine"]

# Available distance methods for protein datasets
PROTEIN_DISTANCE_METHODS = ["euclidean", "cosine"]
```

## Usage Examples

### Basic Molecule Distance Computation

```python
from themap.data.tasks import Tasks
from themap.distance import MoleculeDatasetDistance

# Load tasks from directory
tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=False
)

# Compute molecule distances using OTDD
mol_distance = MoleculeDatasetDistance(
    tasks=tasks,
    molecule_method="otdd"
)

distances = mol_distance.get_distance()
print(distances)
# {'target_task': {'source_task': 0.75, ...}}
```

### Protein Distance Computation

```python
from themap.distance import ProteinDatasetDistance

# Compute protein distances using euclidean method
prot_distance = ProteinDatasetDistance(
    tasks=tasks,
    protein_method="euclidean"
)

distances = prot_distance.get_distance()
```

### Combined Task Distance

```python
from themap.distance import TaskDistance

# Compute combined distances from multiple modalities
task_distance = TaskDistance(
    tasks=tasks,
    molecule_method="cosine",
    protein_method="euclidean"
)

# Compute all distance types
all_distances = task_distance.compute_all_distances(
    combination_strategy="weighted_average",
    molecule_weight=0.7,
    protein_weight=0.3
)

# Access specific distance types
molecule_distances = all_distances["molecule"]
protein_distances = all_distances["protein"]
combined_distances = all_distances["combined"]
```

### Working with External Distance Matrices

```python
import numpy as np

# Load pre-computed distances
task_distance = TaskDistance.load_ext_chem_distance("path/to/chemical_distances.pkl")

# Or initialize with external matrices
external_chem = np.random.rand(10, 8)  # 10 source, 8 target tasks
task_distance = TaskDistance(
    tasks=None,
    source_task_ids=["task1", "task2", ...],
    target_task_ids=["test1", "test2", ...],
    external_chemical_space=external_chem
)

# Convert to pandas for analysis
df = task_distance.to_pandas("external_chemical")
```

### Error Handling

```python
from themap.distance import DistanceComputationError, DataValidationError

try:
    # This might fail if OTDD dependencies are missing
    distances = mol_distance.otdd_distance()
except ImportError as e:
    print(f"OTDD not available: {e}")
    # Fall back to euclidean distance
    distances = mol_distance.euclidean_distance()
except DistanceComputationError as e:
    print(f"Distance computation failed: {e}")
except DataValidationError as e:
    print(f"Data validation failed: {e}")
```

## Performance Considerations

### Memory Usage

- **OTDD**: Most memory-intensive, especially for large datasets
- **Euclidean/Cosine**: More memory-efficient, suitable for large-scale computations
- **External matrices**: Memory usage depends on matrix size

### Computational Complexity

- **OTDD**: O(n²m²) where n,m are dataset sizes
- **Euclidean/Cosine**: O(nm) for feature extraction + O(kl) for distance matrix where k,l are number of tasks
- **Combined distances**: Sum of individual method complexities

### Optimization Tips

```python
# 1. Use appropriate max_samples for OTDD
hopts = {"maxsamples": 500}  # Reduce for faster computation

# 2. Cache features for repeated computations
tasks.save_task_features_to_file("cached_features.pkl")
cached_features = Tasks.load_task_features_from_file("cached_features.pkl")

# 3. Use appropriate distance method based on data size
if num_molecules > 10000:
    method = "euclidean"  # Faster for large datasets
else:
    method = "otdd"       # More accurate for smaller datasets
```

## Configuration

### Distance Method Configuration

Configuration files for distance methods are stored in `themap/models/distance_configures/`:

```json
// otdd.json
{
    "method": "otdd",
    "maxsamples": 1000,
    "device": "auto",
    "parallel": true
}
```

### Custom Configuration

```python
from themap.utils.distance_utils import get_configure

# Get default configuration
config = get_configure("otdd")

# Modify configuration
config["maxsamples"] = 500
config["device"] = "cpu"

# Use in distance computation
mol_distance = MoleculeDatasetDistance(tasks=tasks, molecule_method="otdd")
# Configuration is automatically loaded and can be overridden
```
