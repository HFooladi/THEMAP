"""
Module for calculating distances between datasets in the THEMAP framework.

This module provides functionality to compute various types of distances between:
- Molecule datasets (using OTDD, Euclidean, or cosine distances)
- Protein datasets (using Euclidean or cosine distances)
- Task distances (using external chemical or protein space)

The module supports both single dataset comparisons and batch comparisons
across multiple datasets.
"""

import logging
import multiprocessing
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Import directly from their respective modules to avoid circular imports
from themap.data.metadata import DataFold
from themap.data.molecule_dataset import MoleculeDataset
from themap.data.protein_datasets import ProteinDataset
from themap.data.tasks import Tasks
from themap.data.torch_dataset import MoleculeDataloader
from themap.utils.distance_utils import get_configure

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MAX_SAMPLES = 1000
DEFAULT_N_JOBS = min(8, multiprocessing.cpu_count())


class DistanceComputationError(Exception):
    """Custom exception for distance computation errors."""

    pass


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


def _validate_and_extract_task_id(task_name: str) -> str:
    """Safely extract task ID from task name with validation.

    Args:
        task_name: Task name in format 'fold_task_id'

    Returns:
        Extracted task ID

    Raises:
        DataValidationError: If task name format is invalid
    """
    if not isinstance(task_name, str):
        raise DataValidationError(f"Task name must be string, got {type(task_name)}")

    parts = task_name.split("_", 1)
    if len(parts) < 2:
        logger.warning(f"Task name '{task_name}' doesn't follow expected format 'fold_task_id', using as-is")
        return task_name

    return parts[1]


def _get_dataset_distance() -> Any:
    """Lazy import of DatasetDistance with proper error handling."""
    try:
        from themap.models.otdd.src.distance import DatasetDistance

        return DatasetDistance
    except ImportError as e:
        logger.error(f"Failed to import OTDD DatasetDistance: {e}")
        raise ImportError(f"OTDD dependencies not available. Please install required packages: {e}") from e


MOLECULE_DISTANCE_METHODS = ["otdd", "euclidean", "cosine"]
PROTEIN_DISTANCE_METHODS = ["euclidean", "cosine"]


class AbstractTasksDistance:
    """Base class for computing distances between datasets.

    This abstract class defines the interface for dataset distance computation.
    It provides a common structure for both molecule and protein dataset distances.

    Args:
        tasks: Tasks collection for distance computation
        molecule_method: Distance computation method for molecules (default: "euclidean")
        protein_method: Distance computation method for proteins (default: "euclidean")
        metadata_method: Distance computation method for metadata (default: "euclidean")
        method: Global method (for backward compatibility, overrides individual methods if provided)
    """

    def __init__(
        self,
        tasks: Optional[Tasks] = None,
        molecule_method: str = "euclidean",
        protein_method: str = "euclidean",
        metadata_method: str = "euclidean",
        method: Optional[str] = None,
    ):
        self.tasks = tasks
        self._setup_source_target()

        # If global method is provided, use it for all data types (backward compatibility)
        if method is not None:
            self.molecule_method = method
            self.protein_method = method
            self.metadata_method = method
            self.method = method  # Keep for backward compatibility
        else:
            self.molecule_method = molecule_method
            self.protein_method = protein_method
            self.metadata_method = metadata_method
            self.method = molecule_method  # Default fallback for backward compatibility

    def _setup_source_target(self) -> None:
        """Setup source and target datasets based on tasks."""
        if self.tasks is None:
            self.source = None
            self.target = None
            self.symmetric_tasks = True
            return

        # Get train tasks as source (always)
        train_tasks = self.tasks.get_tasks(DataFold.TRAIN)
        self.source = train_tasks

        # Check for validation or test tasks as target
        valid_tasks = self.tasks.get_tasks(DataFold.VALIDATION)
        test_tasks = self.tasks.get_tasks(DataFold.TEST)

        # If both test and valid tasks are present, use both as target
        if test_tasks and valid_tasks:
            self.target = test_tasks + valid_tasks
            self.symmetric_tasks = False
        # If only test tasks are present, use them as target
        elif test_tasks:
            self.target = test_tasks
            self.symmetric_tasks = False
        # If only valid tasks are present, use them as target
        elif valid_tasks:
            self.target = valid_tasks
            self.symmetric_tasks = False
        # If no valid or test data, use source as target
        else:
            # No valid or test data, use source as target
            self.target = self.source
            self.symmetric_tasks = True

    def get_num_tasks(self) -> Tuple[int, int]:
        """Get the number of source and target tasks."""
        if self.source is None or self.target is None:
            return 0, 0

        source_num = len(self.source)
        target_num = len(self.target)
        return source_num, target_num

    def get_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute the distance between datasets.

        Each of the subclasses should implement this method.

        Returns:
            Dictionary containing distance matrix between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def get_hopts(self, data_type: str = "molecule") -> Optional[Dict[str, Any]]:
        """Get hyperparameters for distance computation.

        Each of the subclasses should implement this method.

        Args:
            data_type: Type of data ("molecule", "protein", "metadata")

        Returns:
            Dictionary of hyperparameters for the specified data type distance computation method.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def get_supported_methods(self, data_type: str) -> List[str]:
        """Get list of supported methods for a specific data type.

        Args:
            data_type: Type of data ("molecule", "protein", "metadata")

        Returns:
            List of supported method names for the data type

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Dict[str, Dict[str, float]]:
        """Allow the class to be called as a function.

        Each of the subclasses should implement this method.

        Returns:
            The computed distance matrix.
        """
        return self.get_distance()


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

    def euclidean_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute Euclidean distance between molecule datasets.

        This method computes the Euclidean distance between the feature vectors
        of the datasets. For each dataset, it computes the mean feature vector
        and then calculates the pairwise distances between these mean vectors.

        Returns:
            Dictionary containing Euclidean distances between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.

        Raises:
            DistanceComputationError: If feature computation fails
        """
        try:
            # Compute features for all tasks
            source_features, target_features, source_names, target_names = self._compute_features()

            if not source_features or not target_features:
                logger.warning("No features available for euclidean distance computation")
                return {}

            if len(source_features) != len(source_names) or len(target_features) != len(target_names):
                raise DistanceComputationError("Mismatch between features and names lengths")

            # Convert to numpy arrays for efficient computation
            source_features_array = np.array(source_features)
            target_features_array = np.array(target_features)

            # Validate array shapes
            if source_features_array.ndim != 2 or target_features_array.ndim != 2:
                raise DistanceComputationError("Feature arrays must be 2-dimensional")

            if source_features_array.shape[1] != target_features_array.shape[1]:
                raise DistanceComputationError(
                    f"Feature dimension mismatch: source={source_features_array.shape[1]}, "
                    f"target={target_features_array.shape[1]}"
                )

            # Compute pairwise euclidean distances
            distances = cdist(target_features_array, source_features_array, metric="euclidean")

            # Organize results in the expected format
            result: Dict[str, Dict[str, float]] = {}
            for i, target_name in enumerate(target_names):
                target_task_id = _validate_and_extract_task_id(target_name)
                result[target_task_id] = {}
                for j, source_name in enumerate(source_names):
                    source_task_id = _validate_and_extract_task_id(source_name)
                    distance_value = float(distances[i, j])

                    # Validate distance value
                    if np.isnan(distance_value) or np.isinf(distance_value):
                        logger.warning(
                            f"Invalid distance value between {source_task_id} and {target_task_id}: {distance_value}"
                        )
                        distance_value = 1.0  # Use default high distance

                    result[target_task_id][source_task_id] = distance_value

            return result

        except Exception as e:
            logger.error(f"Failed to compute euclidean distances: {e}")
            raise DistanceComputationError(f"Euclidean distance computation failed: {e}") from e

    def cosine_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute cosine distance between molecule datasets.

        This method computes the cosine distance between the feature vectors
        of the datasets.

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

    def get_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute the distance between molecule datasets using the specified method.

        Returns:
            Dictionary containing distance matrix between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        if self.molecule_method == "otdd":
            self.distance = self.otdd_distance()
        elif self.molecule_method == "euclidean":
            self.distance = self.euclidean_distance()
        elif self.molecule_method == "cosine":
            self.distance = self.cosine_distance()
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
            self.source_protein_datasets: List[ProteinDataset] = []
            self.target_protein_datasets: List[ProteinDataset] = []
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


class TaskDistance(AbstractTasksDistance):
    """Class for computing and managing distances between tasks.

    This class handles the computation and storage of distances between tasks,
    supporting both chemical and protein space distances. It can compute distances
    directly from Tasks collections or work with pre-computed distance matrices.

    Args:
        tasks: Tasks collection for distance computation (optional)
        method: Default distance computation method
        source_task_ids: List of task IDs for source tasks (legacy, optional)
        target_task_ids: List of task IDs for target tasks (legacy, optional)
        external_chemical_space: Pre-computed chemical space distance matrix (optional)
        external_protein_space: Pre-computed protein space distance matrix (optional)
    """

    def __init__(
        self,
        tasks: Optional[Tasks] = None,
        molecule_method: str = "euclidean",
        protein_method: str = "euclidean",
        metadata_method: str = "euclidean",
        method: Optional[str] = None,
        source_task_ids: Optional[List[str]] = None,
        target_task_ids: Optional[List[str]] = None,
        external_chemical_space: Optional[np.ndarray] = None,
        external_protein_space: Optional[np.ndarray] = None,
    ):
        # Initialize parent class
        super().__init__(
            tasks=tasks,
            molecule_method=molecule_method,
            protein_method=protein_method,
            metadata_method=metadata_method,
            method=method,
        )

        # Handle legacy mode - override parent setup if using legacy parameters
        if tasks is None and (source_task_ids is not None or target_task_ids is not None):
            self.source_task_ids = source_task_ids or []
            self.target_task_ids = target_task_ids or []
            self.source = None
            self.target = None

        self.external_chemical_space = external_chemical_space
        self.external_protein_space = external_protein_space

        # Storage for computed distances
        self.molecule_distances: Optional[Dict[str, Dict[str, float]]] = None
        self.protein_distances: Optional[Dict[str, Dict[str, float]]] = None
        self.combined_distances: Optional[Dict[str, Dict[str, float]]] = None

    def get_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute and return the default distance between tasks.

        Uses the combined distance if both molecule and protein data are available,
        otherwise uses molecule distance, then protein distance as fallback.

        Returns:
            Dictionary containing distance matrix between source and target tasks.
        """
        # Try to compute combined distance first
        if self.tasks is not None:
            try:
                if self.combined_distances is None:
                    self.compute_combined_distance()
                if self.combined_distances:
                    return self.combined_distances
            except Exception:
                pass

            # Fall back to molecule distance
            try:
                if self.molecule_distances is None:
                    self.compute_molecule_distance()
                if self.molecule_distances:
                    return self.molecule_distances
            except Exception:
                pass

            # Fall back to protein distance
            try:
                if self.protein_distances is None:
                    self.compute_protein_distance()
                if self.protein_distances:
                    return self.protein_distances
            except Exception:
                pass

        # If nothing worked, return empty dict
        return {}

    def get_hopts(self, data_type: str = "molecule") -> Optional[Dict[str, Any]]:
        """Get hyperparameters for distance computation.

        Args:
            data_type: Type of data ("molecule", "protein", "metadata")

        Returns:
            Dictionary of hyperparameters for the specified data type distance computation method.
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
            # TODO: Define metadata distance methods
            return ["euclidean", "cosine", "jaccard", "hamming"]
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def __repr__(self) -> str:
        """Return a string representation of the TaskDistance instance.

        Returns:
            String containing the number of source and target tasks and the mode.
        """
        mode = "tasks" if self.tasks is not None else "legacy"
        num_computed = 0
        if self.molecule_distances:
            num_computed += 1
        if self.protein_distances:
            num_computed += 1
        if self.combined_distances:
            num_computed += 1

        return (
            f"TaskDistance(mode={mode}, source_tasks={len(self.source_task_ids)}, "
            f"target_tasks={len(self.target_task_ids)}, computed_distances={num_computed}, "
            f"mol_method={self.molecule_method}, prot_method={self.protein_method}, meta_method={self.metadata_method})"
        )

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the distance matrix.

        Returns:
            Tuple containing (number of source tasks, number of target tasks).
        """
        return len(self.source_task_ids), len(self.target_task_ids)

    def compute_molecule_distance(
        self, method: Optional[str] = None, molecule_featurizer: str = "ecfp"
    ) -> Dict[str, Dict[str, float]]:
        """Compute distances between tasks using molecule data.

        Args:
            method: Distance computation method ('euclidean', 'cosine', or 'otdd').
                   If None, uses the molecule_method from initialization.
            molecule_featurizer: Molecular featurizer to use

        Returns:
            Dictionary containing molecule-based distances between tasks.
        """
        if self.tasks is None:
            raise ValueError("Tasks collection required for computing molecule distances")

        # Use provided method or fall back to instance method
        actual_method = method if method is not None else self.molecule_method

        # Use MoleculeDatasetDistance to compute distances
        mol_distance = MoleculeDatasetDistance(
            tasks=self.tasks,
            molecule_method=actual_method,
        )
        self.molecule_distances = mol_distance.get_distance()
        return self.molecule_distances

    def compute_protein_distance(
        self, method: Optional[str] = None, protein_featurizer: str = "esm2_t33_650M_UR50D"
    ) -> Dict[str, Dict[str, float]]:
        """Compute distances between tasks using protein data.

        Args:
            method: Distance computation method ('euclidean' or 'cosine').
                   If None, uses the protein_method from initialization.
            protein_featurizer: Protein featurizer to use

        Returns:
            Dictionary containing protein-based distances between tasks.
        """
        if self.tasks is None:
            raise ValueError("Tasks collection required for computing protein distances")

        # Use provided method or fall back to instance method
        actual_method = method if method is not None else self.protein_method

        # Use ProteinDatasetDistance to compute distances
        prot_distance = ProteinDatasetDistance(
            tasks=self.tasks,
            protein_method=actual_method,
        )
        self.protein_distances = prot_distance.get_distance()
        return self.protein_distances

    def compute_combined_distance(
        self,
        molecule_method: Optional[str] = None,
        protein_method: Optional[str] = None,
        combination_strategy: str = "average",
        molecule_weight: float = 0.5,
        protein_weight: float = 0.5,
        molecule_featurizer: str = "ecfp",
        protein_featurizer: str = "esm2_t33_650M_UR50D",
    ) -> Dict[str, Dict[str, float]]:
        """Compute combined distances using both molecule and protein data.

        Args:
            molecule_method: Method for molecule distance computation
            protein_method: Method for protein distance computation
            combination_strategy: How to combine distances ('average', 'weighted_average', 'min', 'max')
            molecule_weight: Weight for molecule distances (used with 'weighted_average')
            protein_weight: Weight for protein distances (used with 'weighted_average')
            molecule_featurizer: Molecular featurizer to use
            protein_featurizer: Protein featurizer to use

        Returns:
            Dictionary containing combined distances between tasks.
        """
        # Use provided methods or fall back to instance methods
        actual_molecule_method = molecule_method if molecule_method is not None else self.molecule_method
        actual_protein_method = protein_method if protein_method is not None else self.protein_method

        # Compute individual distances if not already computed
        if self.molecule_distances is None:
            self.compute_molecule_distance(
                method=actual_molecule_method, molecule_featurizer=molecule_featurizer
            )
        if self.protein_distances is None:
            self.compute_protein_distance(method=actual_protein_method, protein_featurizer=protein_featurizer)

        if self.molecule_distances is None or self.protein_distances is None:
            raise ValueError("Could not compute both molecule and protein distances")

        # Combine distances
        self.combined_distances = {}

        # Get all target task IDs present in both distance matrices
        mol_target_ids = set(self.molecule_distances.keys())
        prot_target_ids = set(self.protein_distances.keys())
        common_target_ids = mol_target_ids.intersection(prot_target_ids)

        for target_id in common_target_ids:
            self.combined_distances[target_id] = {}

            # Get all source task IDs present in both distance matrices for this target
            mol_source_ids = set(self.molecule_distances[target_id].keys())
            prot_source_ids = set(self.protein_distances[target_id].keys())
            common_source_ids = mol_source_ids.intersection(prot_source_ids)

            for source_id in common_source_ids:
                mol_dist = self.molecule_distances[target_id][source_id]
                prot_dist = self.protein_distances[target_id][source_id]

                if combination_strategy == "average":
                    combined_dist = (mol_dist + prot_dist) / 2.0
                elif combination_strategy == "weighted_average":
                    # Normalize weights
                    total_weight = molecule_weight + protein_weight
                    if total_weight == 0:
                        combined_dist = (mol_dist + prot_dist) / 2.0
                    else:
                        combined_dist = (
                            mol_dist * molecule_weight + prot_dist * protein_weight
                        ) / total_weight
                elif combination_strategy == "min":
                    combined_dist = min(mol_dist, prot_dist)
                elif combination_strategy == "max":
                    combined_dist = max(mol_dist, prot_dist)
                else:
                    raise ValueError(f"Unknown combination strategy: {combination_strategy}")

                self.combined_distances[target_id][source_id] = combined_dist

        return self.combined_distances

    def compute_all_distances(
        self,
        molecule_method: Optional[str] = None,
        protein_method: Optional[str] = None,
        combination_strategy: str = "average",
        molecule_weight: float = 0.5,
        protein_weight: float = 0.5,
        molecule_featurizer: str = "ecfp",
        protein_featurizer: str = "esm2_t33_650M_UR50D",
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute all distance types (molecule, protein, and combined).

        Args:
            molecule_method: Method for molecule distance computation
            protein_method: Method for protein distance computation
            combination_strategy: How to combine distances
            molecule_weight: Weight for molecule distances
            protein_weight: Weight for protein distances
            molecule_featurizer: Molecular featurizer to use
            protein_featurizer: Protein featurizer to use

        Returns:
            Dictionary with keys 'molecule', 'protein', 'combined' containing respective distance matrices.
        """
        # Use provided methods or fall back to instance methods
        actual_molecule_method = molecule_method if molecule_method is not None else self.molecule_method
        actual_protein_method = protein_method if protein_method is not None else self.protein_method

        results = {}

        # Compute molecule distances
        try:
            results["molecule"] = self.compute_molecule_distance(
                method=actual_molecule_method, molecule_featurizer=molecule_featurizer
            )
        except Exception as e:
            print(f"Warning: Could not compute molecule distances: {e}")
            results["molecule"] = {}

        # Compute protein distances
        try:
            results["protein"] = self.compute_protein_distance(
                method=actual_protein_method, protein_featurizer=protein_featurizer
            )
        except Exception as e:
            print(f"Warning: Could not compute protein distances: {e}")
            results["protein"] = {}

        # Compute combined distances if both are available
        if self.molecule_distances and self.protein_distances:
            try:
                results["combined"] = self.compute_combined_distance(
                    molecule_method=actual_molecule_method,
                    protein_method=actual_protein_method,
                    combination_strategy=combination_strategy,
                    molecule_weight=molecule_weight,
                    protein_weight=protein_weight,
                    molecule_featurizer=molecule_featurizer,
                    protein_featurizer=protein_featurizer,
                )
            except Exception as e:
                print(f"Warning: Could not compute combined distances: {e}")
                results["combined"] = {}
        else:
            results["combined"] = {}

        return results

    def compute_ext_chem_distance(self, method: str) -> Dict[str, Dict[str, float]]:
        """Compute chemical space distances between tasks using external matrices.

        Args:
            method: Distance computation method to use

        Returns:
            Dictionary containing chemical space distances between tasks.

        Raises:
            NotImplementedError: If external chemical space is not provided
        """
        if self.external_chemical_space is None:
            raise NotImplementedError(
                "External chemical space matrix not provided. "
                "Use compute_molecule_distance() for direct computation."
            )

        # Convert external matrix to expected format
        result: Dict[str, Dict[str, float]] = {}
        for i, target_id in enumerate(self.target_task_ids):
            result[target_id] = {}
            for j, source_id in enumerate(self.source_task_ids):
                result[target_id][source_id] = float(self.external_chemical_space[i, j])

        return result

    def compute_ext_prot_distance(self, method: str) -> Dict[str, Dict[str, float]]:
        """Compute protein space distances between tasks using external matrices.

        Args:
            method: Distance computation method to use

        Returns:
            Dictionary containing protein space distances between tasks.

        Raises:
            NotImplementedError: If external protein space is not provided
        """
        if self.external_protein_space is None:
            raise NotImplementedError(
                "External protein space matrix not provided. "
                "Use compute_protein_distance() for direct computation."
            )

        # Convert external matrix to expected format
        result: Dict[str, Dict[str, float]] = {}
        for i, target_id in enumerate(self.target_task_ids):
            result[target_id] = {}
            for j, source_id in enumerate(self.source_task_ids):
                result[target_id][source_id] = float(self.external_protein_space[i, j])

        return result

    @staticmethod
    def load_ext_chem_distance(path: str) -> "TaskDistance":
        """Load pre-computed chemical space distances from a file.

        Args:
            path: Path to the file containing pre-computed chemical space distances

        Returns:
            TaskDistance instance initialized with the loaded distances.

        Note:
            The file should contain a dictionary with keys:
            - 'train_chembl_ids' or 'train_pubchem_ids' or 'source_task_ids'
            - 'test_chembl_ids' or 'test_pubchem_ids' or 'target_task_ids'
            - 'distance_matrices'
        """
        with open(path, "rb") as f:
            x = pickle.load(f)

        if "train_chembl_ids" in x.keys():
            source_task_ids = x["train_chembl_ids"]
        elif "train_pubchem_ids" in x.keys():
            source_task_ids = x["train_pubchem_ids"]
        elif "source_task_ids" in x.keys():
            source_task_ids = x["source_task_ids"]
        else:
            raise ValueError("No source task IDs found in the loaded file")

        if "test_chembl_ids" in x.keys():
            target_task_ids = x["test_chembl_ids"]
        elif "test_pubchem_ids" in x.keys():
            target_task_ids = x["test_pubchem_ids"]
        elif "target_task_ids" in x.keys():
            target_task_ids = x["target_task_ids"]
        else:
            raise ValueError("No target task IDs found in the loaded file")

        return TaskDistance(
            tasks=None,
            source_task_ids=source_task_ids,
            target_task_ids=target_task_ids,
            external_chemical_space=x["distance_matrices"],
        )

    @staticmethod
    def load_ext_prot_distance(path: str) -> "TaskDistance":
        """Load pre-computed protein space distances from a file.

        Args:
            path: Path to the file containing pre-computed protein space distances

        Returns:
            TaskDistance instance initialized with the loaded distances.

        Note:
            The file should contain a dictionary with keys:
            - 'train_chembl_ids' or 'train_pubchem_ids' or 'source_task_ids'
            - 'test_chembl_ids' or 'test_pubchem_ids' or 'target_task_ids'
            - 'distance_matrices'
        """
        with open(path, "rb") as f:
            x = pickle.load(f)

        if "train_chembl_ids" in x.keys():
            source_task_ids = x["train_chembl_ids"]
        elif "train_pubchem_ids" in x.keys():
            source_task_ids = x["train_pubchem_ids"]
        elif "source_task_ids" in x.keys():
            source_task_ids = x["source_task_ids"]
        else:
            raise ValueError("No source task IDs found in the loaded file")

        if "test_chembl_ids" in x.keys():
            target_task_ids = x["test_chembl_ids"]
        elif "test_pubchem_ids" in x.keys():
            target_task_ids = x["test_pubchem_ids"]
        elif "target_task_ids" in x.keys():
            target_task_ids = x["target_task_ids"]
        else:
            raise ValueError("No target task IDs found in the loaded file")

        return TaskDistance(
            tasks=None,
            source_task_ids=source_task_ids,
            target_task_ids=target_task_ids,
            external_protein_space=x["distance_matrices"],
        )

    def get_computed_distance(self, distance_type: str = "combined") -> Optional[Dict[str, Dict[str, float]]]:
        """Get computed distances of the specified type.

        Args:
            distance_type: Type of distance to return ('molecule', 'protein', 'combined')

        Returns:
            Dictionary containing the requested distances, or None if not computed.
        """
        if distance_type == "molecule":
            return self.molecule_distances
        elif distance_type == "protein":
            return self.protein_distances
        elif distance_type == "combined":
            return self.combined_distances
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")

    def to_pandas(self, distance_type: str = "combined") -> pd.DataFrame:
        """Convert distance matrix to a pandas DataFrame.

        Args:
            distance_type: Type of distance to convert ('molecule', 'protein', 'combined', 'external_chemical')

        Returns:
            DataFrame with source task IDs as index and target task IDs as columns,
            containing the distance values.

        Raises:
            ValueError: If no distances of the specified type are available
        """
        if distance_type == "external_chemical":
            if self.external_chemical_space is None:
                raise ValueError("No external chemical space distances available")
            df = pd.DataFrame(
                self.external_chemical_space, index=self.source_task_ids, columns=self.target_task_ids
            )
            return df
        elif distance_type == "external_protein":
            if self.external_protein_space is None:
                raise ValueError("No external protein space distances available")
            df = pd.DataFrame(
                self.external_protein_space, index=self.source_task_ids, columns=self.target_task_ids
            )
            return df
        else:
            distances = self.get_computed_distance(distance_type)
            if distances is None:
                raise ValueError(f"No {distance_type} distances available")
            return pd.DataFrame(distances)
