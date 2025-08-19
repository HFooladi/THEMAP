"""
Molecule dataset distance computation for the THEMAP framework.

This module provides functionality to compute various types of distances between
molecule datasets using OTDD, Euclidean, or cosine distance methods.
"""

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ..data.metadata import DataFold
from ..data.molecule_dataset import MoleculeDataset
from ..data.tasks import Tasks
from ..data.torch_dataset import MoleculeDataloader
from ..utils.distance_utils import get_configure
from .base import (
    DEFAULT_MAX_SAMPLES,
    MOLECULE_DISTANCE_METHODS,
    PROTEIN_DISTANCE_METHODS,
    AbstractTasksDistance,
    _get_dataset_distance,
)
from .exceptions import DistanceComputationError

# Configure logging
logger = logging.getLogger(__name__)


class MoleculeDatasetDistance(AbstractTasksDistance):
    """Calculate distances between molecule datasets using various methods.

    This class implements distance computation between molecule datasets using:
    - Optimal Transport Dataset Distance (OTDD)
    - Euclidean distance
    - Cosine distance

    The class supports both single dataset comparisons and batch comparisons
    across multiple datasets.

    Args:
        tasks: Tasks collection containing molecule datasets for distance computation
        method: Distance computation method ('otdd', 'euclidean', or 'cosine')
        **kwargs: Additional arguments passed to the distance computation method

    Raises:
        ValueError: If the specified method is not supported for molecule datasets
    """

    def __init__(
        self,
        tasks: Optional[Tasks] = None,
        molecule_method: str = "euclidean",
        method: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            tasks=tasks,
            molecule_method=molecule_method,
            method=method,
        )

        # Validate molecule method
        if self.molecule_method not in MOLECULE_DISTANCE_METHODS:
            raise ValueError(
                f"Method {self.molecule_method} not supported for molecule datasets. "
                f"Supported methods are: {MOLECULE_DISTANCE_METHODS}"
            )

        # Extract molecule datasets from tasks
        self._extract_molecule_datasets()
        self.distance: Optional[Dict[str, Dict[str, float]]] = None
        self._current_featurizer: str = "ecfp"  # default featurizer

    def _extract_molecule_datasets(self) -> None:
        """Extract molecule datasets from source and target tasks."""
        if self.source is None or self.target is None:
            self.source_molecule_datasets: List[MoleculeDataset] = []
            self.target_molecule_datasets: List[MoleculeDataset] = []
            self.source_task_ids: List[str] = []
            self.target_task_ids: List[str] = []
            return

        # Extract molecule datasets from source tasks
        self.source_molecule_datasets = []
        self.source_task_ids = []
        for task in self.source:
            if task.molecule_dataset is not None:
                self.source_molecule_datasets.append(task.molecule_dataset)
                self.source_task_ids.append(task.task_id)

        # Extract molecule datasets from target tasks
        self.target_molecule_datasets = []
        self.target_task_ids = []
        for task in self.target:
            if task.molecule_dataset is not None:
                self.target_molecule_datasets.append(task.molecule_dataset)
                self.target_task_ids.append(task.task_id)

    def _compute_features(
        self, molecule_featurizer: str = "ecfp", combination_method: str = "concatenate"
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
        """Compute features for source and target tasks using Tasks.get_distance_computation_ready_features.

        Args:
            molecule_featurizer: Name of molecular featurizer to use
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
            molecule_featurizer=molecule_featurizer,
            combination_method=combination_method,
            source_fold=source_fold,
            target_folds=target_folds,
        )

    def get_hopts(self, data_type: str = "molecule") -> Optional[Dict[str, Any]]:
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

    def otdd_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute Optimal Transport Dataset Distance between molecule datasets.

        This method uses the OTDD implementation to compute distances between
        molecule datasets, which takes into account both the feature space
        and label space of the datasets.

        Returns:
            Dictionary containing OTDD distances between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        source_features, target_features, source_names, target_names = self._compute_features()

        if not source_features or not target_features:
            return {}

        chem_distances: Dict[str, Dict[str, float]] = {}
        hopts = self.get_hopts(data_type="molecule")
        loaders_src = [MoleculeDataloader(d) for d in self.source_molecule_datasets]
        loaders_tgt = [MoleculeDataloader(d) for d in self.target_molecule_datasets]
        for i, tgt in enumerate(loaders_tgt):
            chem_distance: Dict[str, float] = {}
            for j, src in enumerate(loaders_src):
                try:
                    DatasetDistanceClass = _get_dataset_distance()
                    dist = DatasetDistanceClass(src, tgt, **hopts)
                    d = dist.distance(maxsamples=hopts.get("maxsamples", DEFAULT_MAX_SAMPLES))  # type: ignore

                    # Safe device handling for tensor conversion
                    if hasattr(d, "device"):
                        d_cpu = d.cpu() if d.device.type != "cpu" else d
                    else:
                        d_cpu = d

                    distance_value = float(d_cpu.item() if hasattr(d_cpu, "item") else d_cpu)
                    chem_distance[self.source_task_ids[j]] = distance_value

                except Exception as e:
                    logger.error(
                        f"Failed to compute OTDD distance between {self.source_task_ids[j]} and {self.target_task_ids[i]}: {e}"
                    )
                    # Use a default high distance value as fallback
                    chem_distance[self.source_task_ids[j]] = 1.0
            chem_distances[self.target_task_ids[i]] = chem_distance
        return chem_distances

    def euclidean_distance(self, featurizer_name: str = "ecfp") -> Dict[str, Dict[str, float]]:
        """Compute Euclidean distance between molecule datasets.

        This method computes the dataset-level Euclidean distance by comparing
        individual molecules between datasets, similar to how OTDD works.

        Args:
            featurizer_name: Name of the molecular featurizer to use (e.g., "ecfp", "maccs", "desc2D")

        Returns:
            Dictionary containing Euclidean distances between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.

        Raises:
            DistanceComputationError: If feature computation fails
        """
        try:
            if not self.source_molecule_datasets or not self.target_molecule_datasets:
                logger.warning("No datasets available for euclidean distance computation")
                return {}

            chem_distances: Dict[str, Dict[str, float]] = {}

            # Compute distances between each target and source dataset pair
            for i, tgt_dataset in enumerate(self.target_molecule_datasets):
                chem_distance: Dict[str, float] = {}

                for j, src_dataset in enumerate(self.source_molecule_datasets):
                    try:
                        src_features = src_dataset.get_features(featurizer_name=featurizer_name)
                        tgt_features = tgt_dataset.get_features(featurizer_name=featurizer_name)

                        if src_features.size == 0 or tgt_features.size == 0:
                            logger.warning(
                                f"Empty features for {self.source_task_ids[j]} or {self.target_task_ids[i]}"
                            )
                            chem_distance[self.source_task_ids[j]] = 1.0
                            continue

                        # Compute pairwise distances between all molecules in the two datasets
                        pairwise_distances = cdist(tgt_features, src_features, metric="euclidean")

                        # Use mean of all pairwise distances as the dataset distance
                        # This is a common approach for dataset-level distance computation
                        dataset_distance = float(np.mean(pairwise_distances))

                        # Validate distance value
                        if np.isnan(dataset_distance) or np.isinf(dataset_distance):
                            logger.warning(
                                f"Invalid distance value between {self.source_task_ids[j]} and {self.target_task_ids[i]}: {dataset_distance}"
                            )
                            dataset_distance = 1.0  # Use default high distance

                        chem_distance[self.source_task_ids[j]] = dataset_distance

                    except Exception as e:
                        logger.error(
                            f"Failed to compute euclidean distance between {self.source_task_ids[j]} and {self.target_task_ids[i]}: {e}"
                        )
                        # Use a default high distance value as fallback
                        chem_distance[self.source_task_ids[j]] = 1.0

                chem_distances[self.target_task_ids[i]] = chem_distance

            return chem_distances

        except Exception as e:
            logger.error(f"Failed to compute euclidean distances: {e}")
            raise DistanceComputationError(f"Euclidean distance computation failed: {e}") from e

    def cosine_distance(self, featurizer_name: str = "ecfp") -> Dict[str, Dict[str, float]]:
        """Compute cosine distance between molecule datasets.

        This method computes the dataset-level cosine distance by comparing
        individual molecules between datasets, similar to how OTDD works.

        Args:
            featurizer_name: Name of the molecular featurizer to use (e.g., "ecfp", "maccs", "desc2D")

        Returns:
            Dictionary containing cosine distances between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        try:
            if not self.source_molecule_datasets or not self.target_molecule_datasets:
                logger.warning("No datasets available for cosine distance computation")
                return {}

            chem_distances: Dict[str, Dict[str, float]] = {}

            # Compute distances between each target and source dataset pair
            for i, tgt_dataset in enumerate(self.target_molecule_datasets):
                chem_distance: Dict[str, float] = {}

                for j, src_dataset in enumerate(self.source_molecule_datasets):
                    try:
                        src_features = src_dataset.get_features(featurizer_name=featurizer_name)
                        tgt_features = tgt_dataset.get_features(featurizer_name=featurizer_name)

                        if src_features.size == 0 or tgt_features.size == 0:
                            logger.warning(
                                f"Empty features for {self.source_task_ids[j]} or {self.target_task_ids[i]}"
                            )
                            chem_distance[self.source_task_ids[j]] = 1.0
                            continue

                        # Compute pairwise cosine distances between all molecules in the two datasets
                        pairwise_distances = cdist(tgt_features, src_features, metric="cosine")

                        # Use mean of all pairwise distances as the dataset distance
                        dataset_distance = float(np.mean(pairwise_distances))

                        # Validate distance value
                        if np.isnan(dataset_distance) or np.isinf(dataset_distance):
                            logger.warning(
                                f"Invalid distance value between {self.source_task_ids[j]} and {self.target_task_ids[i]}: {dataset_distance}"
                            )
                            dataset_distance = 1.0  # Use default high distance

                        chem_distance[self.source_task_ids[j]] = dataset_distance

                    except Exception as e:
                        logger.error(
                            f"Failed to compute cosine distance between {self.source_task_ids[j]} and {self.target_task_ids[i]}: {e}"
                        )
                        # Use a default high distance value as fallback
                        chem_distance[self.source_task_ids[j]] = 1.0

                chem_distances[self.target_task_ids[i]] = chem_distance

            return chem_distances

        except Exception as e:
            logger.error(f"Failed to compute cosine distances: {e}")
            return {}

    def get_distance(self, featurizer_name: str = "ecfp") -> Dict[str, Dict[str, float]]:
        """Compute the distance between molecule datasets using the specified method.

        Args:
            featurizer_name: Name of the molecular featurizer to use (e.g., "ecfp", "maccs", "desc2D")

        Returns:
            Dictionary containing distance matrix between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        if self.molecule_method == "otdd":
            self.distance = self.otdd_distance()
        elif self.molecule_method == "euclidean":
            self.distance = self.euclidean_distance(featurizer_name=featurizer_name)
        elif self.molecule_method == "cosine":
            self.distance = self.cosine_distance(featurizer_name=featurizer_name)
        else:
            raise ValueError(f"Unknown molecule method: {self.molecule_method}")
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
        """Return a string representation of the MoleculeDatasetDistance instance.

        Returns:
            String containing the class name and initialization parameters.
        """
        num_source = len(self.source_molecule_datasets) if hasattr(self, "source_molecule_datasets") else 0
        num_target = len(self.target_molecule_datasets) if hasattr(self, "target_molecule_datasets") else 0
        return f"MoleculeDatasetDistance(source_tasks={num_source}, target_tasks={num_target}, method={self.molecule_method})"
