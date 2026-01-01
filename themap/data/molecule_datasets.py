"""
Molecule datasets module.

This module contains classes for managing collections of molecule datasets,
including loading from directories, computing features, and organizing them
for N×M distance matrix computation.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from dpu_utils.utils import RichPath

from ..utils.logging import get_logger, setup_logging
from .enums import DataFold
from .exceptions import DatasetValidationError
from .molecule_dataset import MoleculeDataset, get_task_name_from_path

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 1000
DEFAULT_NUM_WORKERS = None
FOLD_NAME_MAPPING = {
    DataFold.TRAIN: "train",
    DataFold.VALIDATION: "valid",
    DataFold.TEST: "test",
}
SUPPORTED_FILE_EXTENSIONS = [".jsonl.gz"]


class MoleculeDatasets:
    """Manager for molecular datasets organized into train/validation/test folds.

    This class provides a high-level interface for working with collections of molecular
    datasets, supporting efficient loading and organization for distance computation.

    Attributes:
        cache_dir: Directory for persistent feature caching

    Examples:
        Basic usage:

        >>> datasets = MoleculeDatasets.from_directory("data/")
        >>> train_data = datasets.load_datasets([DataFold.TRAIN])
        >>> task_names = datasets.get_task_names(DataFold.TRAIN)
    """

    def __init__(
        self,
        train_data_paths: Optional[List[RichPath]] = None,
        valid_data_paths: Optional[List[RichPath]] = None,
        test_data_paths: Optional[List[RichPath]] = None,
        num_workers: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize MoleculeDatasets.

        Args:
            train_data_paths: List of paths to training data files
            valid_data_paths: List of paths to validation data files
            test_data_paths: List of paths to test data files
            num_workers: Number of workers for data loading
            cache_dir: Directory for persistent caching
        """
        logger.info("Initializing MoleculeDatasets")

        self._validate_init_params(num_workers, cache_dir)

        self._fold_to_data_paths: Dict[DataFold, List[RichPath]] = {
            DataFold.TRAIN: train_data_paths or [],
            DataFold.VALIDATION: valid_data_paths or [],
            DataFold.TEST: test_data_paths or [],
        }
        self._num_workers = num_workers
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Cache for loaded datasets
        self._loaded_datasets: Dict[str, MoleculeDataset] = {}

        logger.info(
            f"Initialized with {len(self._fold_to_data_paths[DataFold.TRAIN])} training, "
            f"{len(self._fold_to_data_paths[DataFold.VALIDATION])} validation, and "
            f"{len(self._fold_to_data_paths[DataFold.TEST])} test paths"
        )

    def _validate_init_params(
        self, num_workers: Optional[int], cache_dir: Optional[Union[str, Path]]
    ) -> None:
        """Validate initialization parameters."""
        if num_workers is not None:
            if not isinstance(num_workers, int):
                raise DatasetValidationError(
                    "initialization", f"num_workers must be int or None, got {type(num_workers)}"
                )
            if num_workers == 0:
                raise DatasetValidationError("initialization", "num_workers cannot be 0")
            if num_workers < -1:
                raise DatasetValidationError(
                    "initialization", f"num_workers must be >= -1, got {num_workers}"
                )

        if cache_dir is not None:
            try:
                cache_path = Path(cache_dir)
                if cache_path.exists() and not cache_path.is_dir():
                    raise DatasetValidationError(
                        "initialization", f"cache_dir exists but is not a directory: {cache_dir}"
                    )
            except (TypeError, ValueError) as e:
                raise DatasetValidationError(
                    "initialization", f"Invalid cache_dir path: {cache_dir}. Error: {e}"
                ) from e

    @staticmethod
    def _validate_fold(fold: DataFold) -> None:
        """Validate that fold is a valid DataFold enum value."""
        if not isinstance(fold, DataFold):
            raise DatasetValidationError("fold_validation", f"fold must be a DataFold enum, got {type(fold)}")

    @staticmethod
    def _validate_folds_list(folds: Optional[List[DataFold]]) -> None:
        """Validate list of folds."""
        if folds is None:
            return
        if not isinstance(folds, list):
            raise DatasetValidationError(
                "folds_validation", f"folds must be a list or None, got {type(folds)}"
            )
        if not folds:
            raise DatasetValidationError("folds_validation", "folds list cannot be empty")
        for i, fold in enumerate(folds):
            if not isinstance(fold, DataFold):
                raise DatasetValidationError(
                    "folds_validation", f"folds[{i}] must be a DataFold enum, got {type(fold)}"
                )

    def _validate_directory_path(self, directory: Union[str, RichPath, Path]) -> RichPath:
        """Validate and convert directory path to RichPath."""
        if directory is None:
            raise DatasetValidationError("directory_validation", "directory cannot be None")

        try:
            if isinstance(directory, str):
                rich_path = RichPath.create(directory)
            elif isinstance(directory, Path):
                rich_path = RichPath.create(str(directory))
            elif isinstance(directory, RichPath):
                rich_path = directory
            else:
                raise DatasetValidationError(
                    "directory_validation", f"directory must be str, Path, or RichPath, got {type(directory)}"
                )
            return rich_path
        except Exception as e:
            raise DatasetValidationError(
                "directory_validation", f"Failed to process directory path {directory}: {e}"
            ) from e

    def __repr__(self) -> str:
        return (
            f"MoleculeDatasets(train={len(self._fold_to_data_paths[DataFold.TRAIN])}, "
            f"valid={len(self._fold_to_data_paths[DataFold.VALIDATION])}, "
            f"test={len(self._fold_to_data_paths[DataFold.TEST])})"
        )

    def get_num_fold_tasks(self, fold: DataFold) -> int:
        """Get number of tasks in a specific fold."""
        self._validate_fold(fold)
        return len(self._fold_to_data_paths[fold])

    @staticmethod
    def from_directory(
        directory: Union[str, RichPath, Path],
        task_list_file: Optional[Union[str, RichPath]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> "MoleculeDatasets":
        """Create MoleculeDatasets from a directory.

        Args:
            directory: Directory containing train/valid/test subdirectories
            task_list_file: File containing list of tasks to include
            cache_dir: Directory for persistent caching
            **kwargs: Additional arguments to pass to constructor

        Returns:
            Created MoleculeDatasets instance
        """
        logger.info(f"Loading datasets from directory {directory}")

        # Create temporary instance for validation methods
        temp_instance = MoleculeDatasets.__new__(MoleculeDatasets)
        directory = temp_instance._validate_directory_path(directory)

        # Handle task list file
        fold_task_lists, task_list = temp_instance._load_task_list_file(task_list_file)

        def get_fold_file_names(data_fold_name: str) -> List[RichPath]:
            return temp_instance._get_fold_file_names(directory, data_fold_name, fold_task_lists, task_list)

        train_data_paths = get_fold_file_names("train")
        valid_data_paths = get_fold_file_names("valid")
        test_data_paths = get_fold_file_names("test")

        logger.info(
            f"Found {len(train_data_paths)} training, {len(valid_data_paths)} valid, "
            f"and {len(test_data_paths)} test tasks"
        )
        return MoleculeDatasets(
            train_data_paths=train_data_paths,
            valid_data_paths=valid_data_paths,
            test_data_paths=test_data_paths,
            cache_dir=cache_dir,
            **kwargs,
        )

    @contextmanager
    def _safe_file_read(self, file_path: Union[str, RichPath]) -> Generator[str, None, None]:
        """Context manager for safe file reading."""
        try:
            with open(str(file_path), "r", encoding="utf-8") as f:
                yield f.read().strip()
        except (OSError, IOError) as e:
            raise DatasetValidationError("file_reading", f"Failed to read file {file_path}: {e}") from e
        except UnicodeDecodeError as e:
            raise DatasetValidationError(
                "file_reading", f"Failed to decode file {file_path} as UTF-8: {e}"
            ) from e

    def _load_task_list_file(
        self, task_list_file: Optional[Union[str, RichPath]]
    ) -> Tuple[Dict[str, List[str]], Optional[List[str]]]:
        """Load and parse task list file."""
        if task_list_file is None:
            return {}, None

        if isinstance(task_list_file, str):
            task_list_file = RichPath.create(task_list_file)

        logger.info(f"Using task list file: {task_list_file}")

        try:
            with self._safe_file_read(task_list_file) as content:
                try:
                    task_data = json.loads(content)
                    if isinstance(task_data, dict):
                        fold_task_lists = {
                            "train": task_data.get("train", []),
                            "valid": task_data.get("valid", []),
                            "test": task_data.get("test", []),
                        }
                        logger.info(
                            f"Loaded JSON task list: {len(fold_task_lists['train'])} train, "
                            f"{len(fold_task_lists['valid'])} valid, {len(fold_task_lists['test'])} test tasks"
                        )
                        return fold_task_lists, None
                    else:
                        task_list = task_data if isinstance(task_data, list) else [str(task_data)]
                        return {}, task_list
                except json.JSONDecodeError:
                    task_list = [line.strip() for line in content.split("\n") if line.strip()]
                    logger.info(f"Loaded text task list with {len(task_list)} tasks")
                    return {}, task_list
        except Exception as e:
            raise DatasetValidationError(
                "task_list_loading", f"Failed to load task list from {task_list_file}: {e}"
            ) from e

    def _get_fold_file_names(
        self,
        directory: RichPath,
        data_fold_name: str,
        fold_task_lists: Dict[str, List[str]],
        task_list: Optional[List[str]],
    ) -> List[RichPath]:
        """Get file names for a specific data fold with task filtering."""
        fold_dir = directory.join(data_fold_name)
        if not fold_dir.exists():
            logger.warning(f"Directory {fold_dir} does not exist")
            return []

        try:
            fold_path = Path(str(fold_dir))
            all_files: List[Path] = []
            for ext in SUPPORTED_FILE_EXTENSIONS:
                all_files.extend(fold_path.glob(f"*{ext}"))

            rich_paths = []
            for file_path in all_files:
                try:
                    rich_path = RichPath.create(str(file_path))
                    task_name = get_task_name_from_path(rich_path)

                    if fold_task_lists:
                        applicable_tasks = fold_task_lists.get(data_fold_name, [])
                        if task_name in applicable_tasks:
                            rich_paths.append(rich_path)
                    elif task_list is None or task_name in task_list:
                        rich_paths.append(rich_path)
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    continue

            return rich_paths
        except Exception as e:
            logger.error(f"Failed to scan directory {fold_dir}: {e}")
            return []

    def get_task_names(self, data_fold: DataFold) -> List[str]:
        """Get list of task names in a specific fold."""
        self._validate_fold(data_fold)
        return [get_task_name_from_path(path) for path in self._fold_to_data_paths[data_fold]]

    def get_all_task_names(self) -> Dict[DataFold, List[str]]:
        """Get all task names organized by fold."""
        return {
            fold: self.get_task_names(fold) for fold in [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]
        }

    def load_datasets(self, folds: Optional[List[DataFold]] = None) -> Dict[str, MoleculeDataset]:
        """Load all datasets from specified folds.

        Args:
            folds: List of folds to load. If None, loads all folds

        Returns:
            Dictionary mapping dataset names to loaded datasets
        """
        if folds is None:
            folds = [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]

        self._validate_folds_list(folds)

        datasets = {}
        for fold in folds:
            fold_name = FOLD_NAME_MAPPING[fold]
            for path in self._fold_to_data_paths[fold]:
                dataset_name = f"{fold_name}_{get_task_name_from_path(path)}"

                if dataset_name not in self._loaded_datasets:
                    try:
                        logger.info(f"Loading dataset {dataset_name}")
                        self._loaded_datasets[dataset_name] = MoleculeDataset.load_from_file(path)
                    except Exception as e:
                        logger.error(f"Failed to load dataset {dataset_name}: {e}")
                        raise DatasetValidationError(
                            dataset_name, f"Failed to load dataset from {path}: {e}"
                        ) from e

                datasets[dataset_name] = self._loaded_datasets[dataset_name]

        logger.info(f"Successfully loaded {len(datasets)} datasets")
        return datasets

    def load_dataset_by_name(self, task_name: str, fold: DataFold) -> Optional[MoleculeDataset]:
        """Load a specific dataset by name and fold.

        Args:
            task_name: Name of the task (e.g., "CHEMBL123456")
            fold: Data fold to search in

        Returns:
            MoleculeDataset if found, None otherwise
        """
        self._validate_fold(fold)
        fold_name = FOLD_NAME_MAPPING[fold]
        full_name = f"{fold_name}_{task_name}"

        if full_name in self._loaded_datasets:
            return self._loaded_datasets[full_name]

        for path in self._fold_to_data_paths[fold]:
            if get_task_name_from_path(path) == task_name:
                try:
                    self._loaded_datasets[full_name] = MoleculeDataset.load_from_file(path)
                    return self._loaded_datasets[full_name]
                except Exception as e:
                    logger.error(f"Failed to load dataset {task_name}: {e}")
                    return None

        return None

    def get_datasets_for_distance_computation(
        self,
        source_fold: DataFold = DataFold.TRAIN,
        target_folds: Optional[List[DataFold]] = None,
    ) -> Tuple[List[MoleculeDataset], List[MoleculeDataset], List[str], List[str]]:
        """Get datasets organized for N×M distance matrix computation.

        Args:
            source_fold: Fold to use as source datasets (N)
            target_folds: Folds to use as target datasets (M)

        Returns:
            Tuple containing:
            - source_datasets: List of source MoleculeDataset objects
            - target_datasets: List of target MoleculeDataset objects
            - source_names: List of source task names
            - target_names: List of target task names
        """
        self._validate_fold(source_fold)
        if target_folds is None:
            target_folds = [DataFold.VALIDATION, DataFold.TEST]
        self._validate_folds_list(target_folds)

        # Load all relevant datasets
        all_folds = [source_fold] + target_folds
        all_datasets = self.load_datasets(all_folds)

        # Separate by fold
        source_datasets = []
        source_names = []
        target_datasets = []
        target_names = []

        source_prefix = FOLD_NAME_MAPPING[source_fold]
        target_prefixes = [FOLD_NAME_MAPPING[f] for f in target_folds]

        for name, dataset in all_datasets.items():
            prefix = name.split("_")[0]
            if prefix == source_prefix:
                source_datasets.append(dataset)
                source_names.append(name)
            elif prefix in target_prefixes:
                target_datasets.append(dataset)
                target_names.append(name)

        logger.info(
            f"Prepared {len(source_datasets)} source and {len(target_datasets)} target datasets "
            f"for distance computation"
        )

        return source_datasets, target_datasets, source_names, target_names

    def get_all_smiles(self, folds: Optional[List[DataFold]] = None) -> List[str]:
        """Get all unique SMILES strings across specified folds.

        Useful for batch featurization with deduplication.

        Args:
            folds: List of folds to include. If None, includes all folds.

        Returns:
            List of unique SMILES strings
        """
        datasets = self.load_datasets(folds)
        all_smiles = set()
        for dataset in datasets.values():
            all_smiles.update(dataset.smiles_list)
        return list(all_smiles)

    def get_smiles_to_datasets_mapping(self, folds: Optional[List[DataFold]] = None) -> Dict[str, List[str]]:
        """Get mapping from SMILES to dataset names containing them.

        Useful for understanding SMILES overlap between datasets.

        Args:
            folds: List of folds to include. If None, includes all folds.

        Returns:
            Dictionary mapping SMILES to list of dataset names
        """
        datasets = self.load_datasets(folds)
        smiles_to_datasets: Dict[str, List[str]] = {}

        for name, dataset in datasets.items():
            for smiles in dataset.smiles_list:
                if smiles not in smiles_to_datasets:
                    smiles_to_datasets[smiles] = []
                smiles_to_datasets[smiles].append(name)

        return smiles_to_datasets

    def clear_loaded_datasets(self) -> None:
        """Clear all loaded datasets from memory."""
        self._loaded_datasets.clear()
        logger.info("Cleared all loaded datasets from memory")
