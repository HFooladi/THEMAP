"""
Combined task distance computation orchestrator.

This module provides the high-level interface for computing N×M distance matrices
between tasks, coordinating dataset distances (molecules) and metadata distances
(protein, etc.).

Usage:
    # Basic usage with Tasks collection
    calculator = TaskDistanceCalculator(
        tasks=tasks,
        dataset_method="euclidean",
        metadata_method="cosine"
    )
    all_distances = calculator.compute_all_distances(
        molecule_featurizer="ecfp",
        protein_featurizer="esm2_t33_650M_UR50D"
    )

    # Or use the simplified pipeline approach
    from themap.pipeline import FeaturizationPipeline
    from themap.distance import compute_dataset_distance_matrix

    pipeline = FeaturizationPipeline(cache_dir="./cache", molecule_featurizer="ecfp")
    pipeline.featurize_all_datasets(datasets)
    features, labels, ids = pipeline.load_dataset_features(datasets, names)
    distances = compute_dataset_distance_matrix(...)
"""

from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
from numpy.typing import NDArray

from ..data.enums import DataFold
from ..data.tasks import Task, Tasks
from ..utils.logging import get_logger
from .dataset_distance import DatasetDistance
from .metadata_distance import MetadataDistance, combine_distance_matrices

logger = get_logger(__name__)

# Type aliases
DistanceMatrix = Dict[str, Dict[str, float]]
CombinationStrategy = Literal["weighted_average", "average", "min", "max", "sum"]


class TaskDistanceCalculator:
    """High-level orchestrator for computing task distance matrices.

    This class coordinates the computation of distance matrices for different
    aspects of tasks (molecules, proteins, other metadata) and provides
    methods to combine them.

    Args:
        tasks: Optional Tasks collection (if provided, can auto-extract data)
        dataset_method: Distance method for molecule datasets ('otdd', 'euclidean', 'cosine')
        metadata_method: Distance method for metadata ('euclidean', 'cosine', 'manhattan')

    Attributes:
        molecule_distances: Computed molecule distance matrix
        protein_distances: Computed protein distance matrix
        combined_distances: Combined distance matrix

    Example:
        >>> calculator = TaskDistanceCalculator(
        ...     tasks=tasks,
        ...     dataset_method="euclidean",
        ...     metadata_method="cosine"
        ... )
        >>> # Compute all distance types
        >>> all_dist = calculator.compute_all_distances(
        ...     molecule_featurizer="ecfp",
        ...     n_jobs=8
        ... )
        >>> # Access individual matrices
        >>> mol_dist = all_dist["molecules"]
        >>> prot_dist = all_dist["protein"]
        >>> combined = all_dist["combined"]
    """

    def __init__(
        self,
        tasks: Optional[Tasks] = None,
        dataset_method: str = "euclidean",
        metadata_method: str = "euclidean",
        # Legacy parameters for backward compatibility
        molecule_method: Optional[str] = None,
        protein_method: Optional[str] = None,
        method: Optional[str] = None,
    ):
        """Initialize the task distance calculator.

        Args:
            tasks: Tasks collection containing source and target tasks
            dataset_method: Method for dataset distances (molecules)
            metadata_method: Method for metadata distances (protein, etc.)
            molecule_method: Legacy alias for dataset_method
            protein_method: Legacy alias for metadata_method
            method: Legacy default method (applied to both)
        """
        self.tasks = tasks

        # Handle legacy parameters
        if method is not None:
            dataset_method = method
            metadata_method = method
        if molecule_method is not None:
            dataset_method = molecule_method
        if protein_method is not None:
            metadata_method = protein_method

        self.dataset_method = dataset_method
        self.metadata_method = metadata_method

        # Initialize distance calculators
        # Cast to Literal types for mypy
        self._dataset_distance = DatasetDistance(
            method=cast(Literal["otdd", "euclidean", "cosine"], dataset_method)
        )
        self._metadata_distance = MetadataDistance(
            method=cast(Literal["euclidean", "cosine", "manhattan"], metadata_method)
        )

        # Storage for computed distances
        self.molecule_distances: Optional[DistanceMatrix] = None
        self.protein_distances: Optional[DistanceMatrix] = None
        self.combined_distances: Optional[DistanceMatrix] = None

        logger.info(
            f"TaskDistanceCalculator initialized: "
            f"dataset_method={dataset_method}, metadata_method={metadata_method}"
        )

    def compute_molecule_distance(
        self,
        molecule_featurizer: str = "ecfp",
        n_jobs: int = 1,
        source_fold: DataFold = DataFold.TRAIN,
        target_folds: Optional[List[DataFold]] = None,
        **kwargs: Any,
    ) -> DistanceMatrix:
        """Compute molecule dataset distance matrix.

        Args:
            molecule_featurizer: Name of molecular featurizer
            n_jobs: Number of parallel jobs
            source_fold: Fold to use as source
            target_folds: Folds to use as targets
            **kwargs: Additional arguments for distance computation

        Returns:
            Distance matrix Dict[target_id][source_id] = distance
        """
        if self.tasks is None:
            raise ValueError("Tasks collection required for molecule distance computation")

        if target_folds is None:
            target_folds = [DataFold.VALIDATION, DataFold.TEST]

        # Get features from Tasks
        source_features, target_features, source_names, target_names = (
            self.tasks.get_distance_computation_ready_features(
                molecule_featurizer=molecule_featurizer,
                source_fold=source_fold,
                target_folds=target_folds,
            )
        )

        if not source_features or not target_features:
            logger.warning("No features available for molecule distance computation")
            return {}

        # Get labels from datasets
        source_labels = self._get_labels_for_features(source_names)
        target_labels = self._get_labels_for_features(target_names)

        # Compute distance matrix
        self.molecule_distances = self._dataset_distance.compute_matrix(
            source_features=source_features,
            source_labels=source_labels,
            target_features=target_features,
            target_labels=target_labels,
            source_ids=source_names,
            target_ids=target_names,
            n_jobs=n_jobs,
            **kwargs,
        )

        logger.info(f"Computed molecule distance matrix: {len(target_names)}×{len(source_names)}")
        return self.molecule_distances

    def compute_protein_distance(
        self,
        protein_featurizer: str = "esm2_t33_650M_UR50D",
        source_fold: DataFold = DataFold.TRAIN,
        target_folds: Optional[List[DataFold]] = None,
    ) -> DistanceMatrix:
        """Compute protein metadata distance matrix.

        Args:
            protein_featurizer: Name of protein featurizer
            source_fold: Fold to use as source
            target_folds: Folds to use as targets

        Returns:
            Distance matrix Dict[target_id][source_id] = distance
        """
        if self.tasks is None:
            raise ValueError("Tasks collection required for protein distance computation")

        if target_folds is None:
            target_folds = [DataFold.VALIDATION, DataFold.TEST]

        # Get protein features from Tasks
        source_features, target_features, source_names, target_names = (
            self.tasks.get_distance_computation_ready_features(
                protein_featurizer=protein_featurizer,
                source_fold=source_fold,
                target_folds=target_folds,
            )
        )

        if not source_features or not target_features:
            logger.warning("No features available for protein distance computation")
            return {}

        # Stack features into arrays (each task has one protein vector)
        source_array = np.vstack(source_features).astype(np.float32)
        target_array = np.vstack(target_features).astype(np.float32)

        # Compute distance matrix
        self.protein_distances = self._metadata_distance.compute_matrix(
            source_vectors=source_array,
            target_vectors=target_array,
            source_ids=source_names,
            target_ids=target_names,
        )

        logger.info(f"Computed protein distance matrix: {len(target_names)}×{len(source_names)}")
        return self.protein_distances

    def compute_combined_distance(
        self,
        molecule_featurizer: str = "ecfp",
        protein_featurizer: str = "esm2_t33_650M_UR50D",
        weights: Optional[Dict[str, float]] = None,
        combination: CombinationStrategy = "weighted_average",
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> DistanceMatrix:
        """Compute combined distance matrix from molecules and proteins.

        Args:
            molecule_featurizer: Name of molecular featurizer
            protein_featurizer: Name of protein featurizer
            weights: Optional weights for combination (default: equal)
            combination: Combination strategy
            n_jobs: Number of parallel jobs
            **kwargs: Additional arguments

        Returns:
            Combined distance matrix
        """
        # Compute individual distances if not already computed
        if self.molecule_distances is None:
            try:
                self.compute_molecule_distance(
                    molecule_featurizer=molecule_featurizer,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"Failed to compute molecule distances: {e}")

        if self.protein_distances is None:
            try:
                self.compute_protein_distance(
                    protein_featurizer=protein_featurizer,
                )
            except Exception as e:
                logger.warning(f"Failed to compute protein distances: {e}")

        # Combine available distances
        matrices: Dict[str, DistanceMatrix] = {}
        if self.molecule_distances:
            matrices["molecules"] = self.molecule_distances
        if self.protein_distances:
            matrices["protein"] = self.protein_distances

        if not matrices:
            logger.warning("No distance matrices available to combine")
            return {}

        self.combined_distances = combine_distance_matrices(
            matrices, weights=weights, combination=combination
        )

        logger.info(f"Computed combined distance matrix using {combination}")
        return self.combined_distances

    def compute_all_distances(
        self,
        molecule_featurizer: str = "ecfp",
        protein_featurizer: str = "esm2_t33_650M_UR50D",
        weights: Optional[Dict[str, float]] = None,
        combination: CombinationStrategy = "weighted_average",
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> Dict[str, DistanceMatrix]:
        """Compute all distance types and return as dictionary.

        This is the main entry point for computing complete task distances.

        Args:
            molecule_featurizer: Name of molecular featurizer
            protein_featurizer: Name of protein featurizer
            weights: Optional weights for combination
            combination: Combination strategy
            n_jobs: Number of parallel jobs
            **kwargs: Additional arguments

        Returns:
            Dictionary with keys: 'molecules', 'protein', 'combined'
        """
        results: Dict[str, DistanceMatrix] = {}

        # Compute molecule distances
        try:
            results["molecules"] = self.compute_molecule_distance(
                molecule_featurizer=molecule_featurizer,
                n_jobs=n_jobs,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"Failed to compute molecule distances: {e}")
            results["molecules"] = {}

        # Compute protein distances
        try:
            results["protein"] = self.compute_protein_distance(
                protein_featurizer=protein_featurizer,
            )
        except Exception as e:
            logger.warning(f"Failed to compute protein distances: {e}")
            results["protein"] = {}

        # Compute combined distances
        try:
            results["combined"] = self.compute_combined_distance(
                molecule_featurizer=molecule_featurizer,
                protein_featurizer=protein_featurizer,
                weights=weights,
                combination=combination,
                n_jobs=n_jobs,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"Failed to compute combined distances: {e}")
            results["combined"] = {}

        return results

    def _get_labels_for_features(self, dataset_names: List[str]) -> List[NDArray[np.int32]]:
        """Extract labels for datasets given their names.

        Args:
            dataset_names: List of dataset names

        Returns:
            List of label arrays
        """
        labels = []
        for name in dataset_names:
            # Try to find the task by name
            task = self._find_task_by_name(name)
            if task is not None and task.molecule_dataset is not None:
                labels.append(task.molecule_dataset.labels)
            else:
                # Default to empty array if not found
                labels.append(np.array([], dtype=np.int32))
        return labels

    def _find_task_by_name(self, name: str) -> Optional[Task]:
        """Find a task by its name in the Tasks collection.

        Args:
            name: Task name (may include fold prefix like 'train_CHEMBL123')

        Returns:
            Task if found, None otherwise
        """
        if self.tasks is None:
            return None

        # Strip fold prefix if present
        for prefix in ["train_", "valid_", "test_"]:
            if name.startswith(prefix):
                task_id = name[len(prefix) :]
                break
        else:
            task_id = name

        # Search through all folds
        for fold in [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]:
            for task in self.tasks.get_tasks(fold):
                if task.task_id == task_id or task.task_id == name:
                    return task

        return None

    # Legacy method aliases for backward compatibility
    def get_distance(self) -> DistanceMatrix:
        """Legacy method: get default distance matrix."""
        if self.combined_distances is not None:
            return self.combined_distances
        if self.molecule_distances is not None:
            return self.molecule_distances
        if self.protein_distances is not None:
            return self.protein_distances
        return self.compute_all_distances().get("combined", {})


# Legacy alias for backward compatibility
TaskDistance = TaskDistanceCalculator
