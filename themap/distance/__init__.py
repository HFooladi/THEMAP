"""
Distance computation module for THEMAP.

This module provides efficient N×M distance matrix computation between tasks,
supporting both dataset distances (molecules) and metadata distances (protein, etc.).

Main Classes:
- DatasetDistance: Compute distances between molecule datasets (OTDD, Euclidean, Cosine)
- MetadataDistance: Compute distances between single-vector metadata (Euclidean, Cosine)
- TaskDistanceCalculator: High-level orchestrator combining both

Convenience Functions:
- compute_dataset_distance_matrix: Quick dataset distance computation
- compute_metadata_distance_matrix: Quick metadata distance computation
- combine_distance_matrices: Combine multiple distance matrices

Example:
    >>> from themap.distance import (
    ...     DatasetDistance,
    ...     MetadataDistance,
    ...     compute_dataset_distance_matrix,
    ...     TaskDistanceCalculator
    ... )
    >>> # Compute molecule distance matrix
    >>> distances = compute_dataset_distance_matrix(
    ...     source_features, source_labels,
    ...     target_features, target_labels,
    ...     source_ids, target_ids,
    ...     method="euclidean"
    ... )
"""

# Base class
from .base import AbstractTasksDistance

# New unified distance classes
from .dataset_distance import (
    DatasetDistance,
    compute_dataset_distance_matrix,
)

# Exceptions
from .exceptions import DataValidationError, DistanceComputationError
from .metadata_distance import (
    MetadataDistance,
    combine_distance_matrices,
    compute_metadata_distance_matrix,
)

# Legacy classes for backward compatibility with examples
from .molecule_distance import MoleculeDatasetDistance
from .protein_distance import ProteinDatasetDistance
from .task_distance import TaskDistance, TaskDistanceCalculator

# Constants for backward compatibility
DATASET_DISTANCE_METHODS = ["otdd", "euclidean", "cosine"]
METADATA_DISTANCE_METHODS = ["euclidean", "cosine", "manhattan"]

# Legacy aliases
MOLECULE_DISTANCE_METHODS = DATASET_DISTANCE_METHODS
PROTEIN_DISTANCE_METHODS = METADATA_DISTANCE_METHODS

__all__ = [
    # Base class
    "AbstractTasksDistance",
    # New unified classes
    "DatasetDistance",
    "MetadataDistance",
    "TaskDistanceCalculator",
    # Convenience functions
    "compute_dataset_distance_matrix",
    "compute_metadata_distance_matrix",
    "combine_distance_matrices",
    # Legacy classes
    "TaskDistance",
    "MoleculeDatasetDistance",
    "ProteinDatasetDistance",
    # Exceptions
    "DistanceComputationError",
    "DataValidationError",
    # Constants
    "DATASET_DISTANCE_METHODS",
    "METADATA_DISTANCE_METHODS",
    "MOLECULE_DISTANCE_METHODS",
    "PROTEIN_DISTANCE_METHODS",
]
