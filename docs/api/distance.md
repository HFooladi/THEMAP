# Distance Module

The distance module provides classes for computing distances between molecular datasets, metadata vectors, and combined task distances.

## Core Classes

### DatasetDistance

::: themap.distance.dataset_distance.DatasetDistance
    options:
      show_root_heading: true
      heading_level: 3

### MetadataDistance

::: themap.distance.metadata_distance.MetadataDistance
    options:
      show_root_heading: true
      heading_level: 3

### TaskDistanceCalculator

::: themap.distance.task_distance.TaskDistanceCalculator
    options:
      show_root_heading: true
      heading_level: 3

## Convenience Functions

### compute_dataset_distance_matrix

::: themap.distance.dataset_distance.compute_dataset_distance_matrix
    options:
      show_root_heading: true
      heading_level: 4

### compute_metadata_distance_matrix

::: themap.distance.metadata_distance.compute_metadata_distance_matrix
    options:
      show_root_heading: true
      heading_level: 4

### combine_distance_matrices

::: themap.distance.metadata_distance.combine_distance_matrices
    options:
      show_root_heading: true
      heading_level: 4

## Base Class

### AbstractTasksDistance

::: themap.distance.base.AbstractTasksDistance
    options:
      show_root_heading: true
      heading_level: 3

## Legacy Classes

These classes are kept for backward compatibility. Prefer `DatasetDistance` and `MetadataDistance` for new code.

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

## Exceptions

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

```python
DATASET_DISTANCE_METHODS = ["otdd", "euclidean", "cosine"]
METADATA_DISTANCE_METHODS = ["euclidean", "cosine", "manhattan", "jaccard"]
```

See the [Distance Computation Guide](../user-guide/distance-computation.md) for usage examples and method comparisons.
