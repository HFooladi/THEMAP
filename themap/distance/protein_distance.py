"""
Protein dataset distance computation for the THEMAP framework.

This module provides functionality to compute various types of distances between
protein datasets using Euclidean or cosine distance methods.
"""

import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ..data.metadata import DataFold
from ..data.protein_datasets import ProteinMetadataDataset
from ..data.tasks import Tasks
from ..utils.distance_utils import get_configure
from ..utils.logging import get_logger
from .base import (
    MOLECULE_DISTANCE_METHODS,
    PROTEIN_DISTANCE_METHODS,
    AbstractTasksDistance,
    _validate_and_extract_task_id,
)

logger = get_logger(__name__)


class ProteinDatasetDistance(AbstractTasksDistance):
    """Calculate distances between protein datasets using various methods.

    This class implements distance computation between protein datasets using:
    - Euclidean distance
    - Cosine distance

    The class supports both single dataset comparisons and batch comparisons
    across multiple datasets.

    Args:
        tasks: Tasks collection containing protein datasets for distance computation
        method: Distance computation method ('euclidean' or 'cosine')

    Raises:
        ValueError: If the specified method is not supported for protein datasets
    """

    def __init__(
        self,
        tasks: Optional[Tasks] = None,
        protein_method: str = "euclidean",
        method: Optional[str] = None,
    ):
        super().__init__(
            tasks=tasks,
            protein_method=protein_method,
            method=method,
        )

        # Validate protein method
        if self.protein_method not in PROTEIN_DISTANCE_METHODS:
            raise ValueError(
                f"Method {self.protein_method} not supported for protein datasets. "
                f"Supported methods are: {PROTEIN_DISTANCE_METHODS}"
            )

        # Extract protein datasets from tasks
        self._extract_protein_datasets()
        self.distance: Optional[Dict[str, Dict[str, float]]] = None

    def _extract_protein_datasets(self) -> None:
        """Extract protein datasets from source and target tasks."""
        if self.source is None or self.target is None:
            self.source_protein_datasets: List[ProteinMetadataDataset] = []
            self.target_protein_datasets: List[ProteinMetadataDataset] = []
            self.source_task_ids: List[str] = []
            self.target_task_ids: List[str] = []
            return

        # Extract protein datasets from source tasks
        self.source_protein_datasets = []
        self.source_task_ids = []
        for task in self.source:
            if task.protein_dataset is not None:
                self.source_protein_datasets.append(task.protein_dataset)
                self.source_task_ids.append(task.task_id)

        # Extract protein datasets from target tasks
        self.target_protein_datasets = []
        self.target_task_ids = []
        for task in self.target:
            if task.protein_dataset is not None:
                self.target_protein_datasets.append(task.protein_dataset)
                self.target_task_ids.append(task.task_id)

    def _compute_features(
        self, protein_featurizer: str = "esm2_t33_650M_UR50D", combination_method: str = "concatenate"
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
        """Compute features for source and target tasks using Tasks.get_distance_computation_ready_features.

        Args:
            protein_featurizer: Name of protein featurizer to use
            combination_method: Method to combine features

        Returns:
            Tuple of (source_features, target_features, source_names, target_names)
        """
        if self.tasks is None:
            return [], [], [], []

        # Determine source and target folds
        source_fold = DataFold.TRAIN
        target_folds = []

        # Check what folds are available in target
        if self.tasks.get_num_fold_tasks(DataFold.VALIDATION) > 0:
            target_folds.append(DataFold.VALIDATION)
        if self.tasks.get_num_fold_tasks(DataFold.TEST) > 0:
            target_folds.append(DataFold.TEST)

        # If no validation or test, use train as target
        if not target_folds:
            target_folds = [DataFold.TRAIN]

        return self.tasks.get_distance_computation_ready_features(
            protein_featurizer=protein_featurizer,
            combination_method=combination_method,
            source_fold=source_fold,
            target_folds=target_folds,
        )

    def get_hopts(self, data_type: str = "protein") -> Optional[Dict[str, Any]]:
        """Get hyperparameters for the distance computation method.

        Args:
            data_type: Type of data ("molecule", "protein", "metadata")

        Returns:
            Dictionary of hyperparameters specific to the chosen distance method for the data type.
        """
        if data_type == "molecule":
            return get_configure(self.molecule_method)
        elif data_type == "protein":
            return get_configure(self.protein_method)
        elif data_type == "metadata":
            return get_configure(self.metadata_method)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def get_supported_methods(self, data_type: str) -> List[str]:
        """Get list of supported methods for a specific data type.

        Args:
            data_type: Type of data ("molecule", "protein", "metadata")

        Returns:
            List of supported method names for the data type
        """
        if data_type == "molecule":
            return MOLECULE_DISTANCE_METHODS.copy()
        elif data_type == "protein":
            return PROTEIN_DISTANCE_METHODS.copy()
        elif data_type == "metadata":
            return ["euclidean", "cosine", "jaccard", "hamming"]
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def euclidean_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute Euclidean distance between protein datasets.

        This method calculates the pairwise Euclidean distances between protein
        feature vectors in the datasets.

        Returns:
            Dictionary containing Euclidean distances between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        # Compute features for all tasks
        source_features, target_features, source_names, target_names = self._compute_features()

        if not source_features or not target_features:
            return {}

        # Convert to numpy arrays for efficient computation
        source_features_array = np.array(source_features)
        target_features_array = np.array(target_features)

        # Compute pairwise euclidean distances
        distances = cdist(target_features_array, source_features_array, metric="euclidean")

        # Organize results in the expected format
        result: Dict[str, Dict[str, float]] = {}
        for i, target_name in enumerate(target_names):
            target_task_id = _validate_and_extract_task_id(target_name)
            result[target_task_id] = {}
            for j, source_name in enumerate(source_names):
                source_task_id = _validate_and_extract_task_id(source_name)
                result[target_task_id][source_task_id] = float(distances[i, j])

        return result

    def cosine_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute cosine distance between protein datasets.

        This method calculates the pairwise cosine distances between protein
        feature vectors in the datasets.

        Returns:
            Dictionary containing cosine distances between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        # Compute features for all tasks
        source_features, target_features, source_names, target_names = self._compute_features()

        if not source_features or not target_features:
            return {}

        # Convert to numpy arrays for efficient computation
        source_features_array = np.array(source_features)
        target_features_array = np.array(target_features)

        # Compute pairwise cosine distances
        distances = cdist(target_features_array, source_features_array, metric="cosine")

        # Organize results in the expected format
        result: Dict[str, Dict[str, float]] = {}
        for i, target_name in enumerate(target_names):
            target_task_id = _validate_and_extract_task_id(target_name)
            result[target_task_id] = {}
            for j, source_name in enumerate(source_names):
                source_task_id = _validate_and_extract_task_id(source_name)
                result[target_task_id][source_task_id] = float(distances[i, j])

        return result

    def sequence_identity_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute sequence identity-based distance between protein datasets.

        This method calculates distances based on protein sequence identity.

        Returns:
            Dictionary containing sequence identity-based distances between datasets.

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "Sequence identity distance computation is not yet implemented. "
            "Use 'euclidean' or 'cosine' methods instead."
        )

    def get_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute the distance between protein datasets using the specified method.

        Returns:
            Dictionary containing distance matrix between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        if self.protein_method == "euclidean":
            self.distance = self.euclidean_distance()
        elif self.protein_method == "cosine":
            self.distance = self.cosine_distance()
        else:
            raise ValueError(f"Unknown protein method: {self.protein_method}")
        return self.distance

    def load_distance(self, path: str) -> None:
        """Load pre-computed distances from a file.

        Args:
            path: Path to the file containing pre-computed distances

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict):
                raise ValueError("Distance file must contain a dictionary")

            # Validate the structure
            for target_id, source_distances in data.items():
                if not isinstance(source_distances, dict):
                    raise ValueError(f"Invalid format for target {target_id}")

            self.distance = data
            logger.info(f"Successfully loaded distances from {path}")

        except FileNotFoundError:
            logger.error(f"Distance file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load distances from {path}: {e}")
            raise ValueError(f"Invalid distance file format: {e}") from e

    def to_pandas(self) -> pd.DataFrame:
        """Convert the distance matrix to a pandas DataFrame.

        Returns:
            DataFrame with source task IDs as index and target task IDs as columns,
            containing the distance values.
        """
        return pd.DataFrame(self.distance)

    def __repr__(self) -> str:
        """Return a string representation of the ProteinDatasetDistance instance.

        Returns:
            String containing the class name and initialization parameters.
        """
        num_source = len(self.source_protein_datasets) if hasattr(self, "source_protein_datasets") else 0
        num_target = len(self.target_protein_datasets) if hasattr(self, "target_protein_datasets") else 0
        return f"ProteinDatasetDistance(source_tasks={num_source}, target_tasks={num_target}, method={self.protein_method})"
