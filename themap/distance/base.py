"""
Base classes and utilities for distance computation in the THEMAP framework.

This module provides the abstract base class for all distance computation classes
and common utility functions used across different distance computation methods.
"""

import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

from ..data.metadata import DataFold
from ..data.tasks import Tasks
from ..distance.exceptions import DataValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Configuration constants
DEFAULT_MAX_SAMPLES = 1000
DEFAULT_N_JOBS = min(8, multiprocessing.cpu_count())

# Supported distance methods by data type
DATASET_DISTANCE_METHODS = ["otdd", "euclidean", "cosine"]  # For actual datasets (sets of molecules)
METADATA_DISTANCE_METHODS = [
    "euclidean",
    "cosine",
    "manhattan",
    "jaccard",
]  # For metadata (single vectors per task)

# Backward compatibility
MOLECULE_DISTANCE_METHODS = DATASET_DISTANCE_METHODS
PROTEIN_DISTANCE_METHODS = METADATA_DISTANCE_METHODS  # Deprecated: protein is metadata, not dataset


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
        from ..models.otdd.src.distance import DatasetDistance

        return DatasetDistance
    except ImportError as e:
        logger.error(f"Failed to import OTDD DatasetDistance: {e}")
        raise ImportError(f"OTDD dependencies not available. Please install required packages: {e}") from e


class AbstractTasksDistance:
    """Base class for computing distances between tasks.

    This abstract class defines the interface for task distance computation.
    It distinguishes between:
    - Dataset distances: Between sets of molecules (OTDD, set-based Euclidean/Cosine)
    - Metadata distances: Between single vectors per task (vector-based Euclidean/Cosine)

    Args:
        tasks: Tasks collection for distance computation
        dataset_method: Distance computation method for datasets (molecules) (default: "euclidean")
        metadata_method: Distance computation method for metadata including protein (default: "euclidean")
        molecule_method: Deprecated alias for dataset_method
        protein_method: Deprecated - protein is metadata, use metadata_method
        method: Global method (for backward compatibility, overrides individual methods if provided)
    """

    def __init__(
        self,
        tasks: Optional[Tasks] = None,
        dataset_method: str = "euclidean",
        metadata_method: str = "euclidean",
        molecule_method: Optional[str] = None,  # Deprecated alias
        protein_method: Optional[str] = None,  # Deprecated - protein is metadata
        method: Optional[str] = None,
    ):
        self.tasks = tasks
        self._setup_source_target()

        # Handle backward compatibility
        if molecule_method is not None:
            logger.warning("molecule_method is deprecated, use dataset_method")
            dataset_method = molecule_method
        if protein_method is not None:
            logger.warning("protein_method is deprecated, protein is metadata - use metadata_method")
            metadata_method = protein_method

        # If global method is provided, use it for all data types (backward compatibility)
        if method is not None:
            self.dataset_method = method
            self.metadata_method = method
            self.method = method  # Keep for backward compatibility
            # Backward compatibility aliases
            self.molecule_method = method
            self.protein_method = method
        else:
            self.dataset_method = dataset_method
            self.metadata_method = metadata_method
            self.method = dataset_method  # Default fallback for backward compatibility
            # Backward compatibility aliases
            self.molecule_method = dataset_method
            self.protein_method = metadata_method

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

    def get_hopts(self, data_type: str = "dataset") -> Optional[Dict[str, Any]]:
        """Get hyperparameters for distance computation.

        Each of the subclasses should implement this method.

        Args:
            data_type: Type of data ("dataset", "metadata")
                      Legacy: "molecule" (alias for "dataset"), "protein" (alias for "metadata")

        Returns:
            Dictionary containing hyperparameters for the distance computation method
            or None if no hyperparameters are needed.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def get_supported_methods(self, data_type: str) -> List[str]:
        """Get list of supported methods for a specific data type.

        Args:
            data_type: Type of data ("dataset", "metadata")
                      Legacy: "molecule" (alias for "dataset"), "protein" (alias for "metadata")

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
