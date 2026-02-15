"""
Protein datasets module.

This module contains classes for managing protein datasets, including loading,
downloading, and computing features.
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio import SeqIO
from dpu_utils.utils import RichPath
from numpy.typing import NDArray

from ..utils.cache_utils import GlobalMoleculeCache
from ..utils.logging import get_logger
from ..utils.protein_utils import (
    convert_fasta_to_dict,
    get_protein_accession,
    get_protein_features,
    get_protein_sequence,
)
from .enums import DataFold
from .exceptions import CacheError, DatasetValidationError, FeaturizationError

logger = get_logger(__name__)

# Constants
DEFAULT_FEATURIZER_NAME = "esm3_sm_open_v1"
DEFAULT_BATCH_SIZE = 100
DEFAULT_NUM_WORKERS = None
FOLD_NAME_MAPPING = {
    DataFold.TRAIN: "train",
    DataFold.VALIDATION: "valid",
    DataFold.TEST: "test",
}
SUPPORTED_PROTEIN_EXTENSIONS = [".fasta"]
UNIPROT_MAPPING_DEFAULT = "datasets/uniprot_mapping.csv"


@dataclass
class ProteinFeatureComputationConfig:
    """Configuration for protein feature computation operations.

    Attributes:
        featurizer_name: Name of the protein featurizer to use
        layer: Layer number for ESM models (optional)
        batch_size: Batch size for feature computation
        force_recompute: Whether to force recomputation even if cached
        enable_deduplication: Whether to enable UniProt ID deduplication
    """

    featurizer_name: str = DEFAULT_FEATURIZER_NAME
    layer: Optional[int] = None
    batch_size: int = DEFAULT_BATCH_SIZE
    force_recompute: bool = False
    enable_deduplication: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.featurizer_name, str) or not self.featurizer_name.strip():
            raise ValueError("featurizer_name must be a non-empty string")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.layer is not None and self.layer < 0:
            raise ValueError(f"layer must be non-negative, got {self.layer}")


@dataclass
class ProteinMetadataDataset:
    """Single protein metadata dataset representing one task.

    This class represents a single protein target with its associated metadata
    and computed features for use in molecular property prediction tasks.

    Attributes:
        task_id: Unique identifier for the task (typically ChEMBL ID)
        uniprot_id: UniProt accession ID for the protein
        sequence: Protein amino acid sequence
        features: Optional pre-computed protein features

    Examples:
        Basic usage:

        >>> protein = ProteinMetadataDataset(
        ...     task_id="CHEMBL123",
        ...     uniprot_id="P12345",
        ...     sequence="MKLLVLGLG..."
        ... )
        >>> features = protein.get_features("esm3_sm_open_v1")
    """

    task_id: str
    uniprot_id: str
    sequence: str
    features: Optional[NDArray[np.float32]] = None

    def __post_init__(self) -> None:
        """Validate initialization data.

        Raises:
            DatasetValidationError: If validation fails
        """
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise DatasetValidationError("protein_dataset", "task_id must be a non-empty string")
        if not isinstance(self.uniprot_id, str) or not self.uniprot_id.strip():
            raise DatasetValidationError("protein_dataset", "uniprot_id must be a non-empty string")
        if not isinstance(self.sequence, str) or not self.sequence.strip():
            raise DatasetValidationError("protein_dataset", "sequence must be a non-empty string")
        if self.features is not None and not isinstance(self.features, np.ndarray):
            raise DatasetValidationError("protein_dataset", "features must be numpy array or None")

    def __repr__(self) -> str:
        return f"ProteinMetadataDataset(task_id={self.task_id}, uniprot_id={self.uniprot_id}, seq_len={len(self.sequence)})"

    def get_features(
        self, featurizer_name: str = DEFAULT_FEATURIZER_NAME, layer: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Get protein features using the specified featurizer.

        Args:
            featurizer_name: Name of the protein featurizer to use
            layer: Layer number for ESM models

        Returns:
            Computed protein features

        Raises:
            FeaturizationError: If feature computation fails
        """
        if self.features is None:
            try:
                protein_dict = {self.uniprot_id: self.sequence}
                self.features = get_protein_features(protein_dict, featurizer_name, layer)
            except Exception as e:
                raise FeaturizationError(
                    self.uniprot_id, featurizer_name, f"Failed to compute protein features: {e}"
                ) from e

        return self.features

    @property
    def get_computed_features(self) -> Optional[NDArray[np.float32]]:
        """Get computed protein features."""
        return self.features


class ProteinMetadataDatasets:
    """Manager for protein datasets organized into train/validation/test folds.

    This class provides a high-level interface for working with collections of protein
    datasets, supporting efficient feature computation with UniProt ID deduplication,
    FASTA file management, and distance matrix preparation.

    Key Features:
        - Automatic discovery of protein FASTA files from directory structure
        - UniProt ID deduplication across datasets for efficiency
        - FASTA file downloading from UniProt using ChEMBL-to-UniProt mapping
        - Persistent caching for computed protein features
        - Organized feature extraction for distance matrix computation
        - Support for selective dataset loading based on task lists

    Attributes:
        cache_dir: Directory for persistent feature caching
        global_cache: Global cache instance for feature storage
        uniprot_mapping_file: Path to ChEMBL ID to UniProt ID mapping file

    Examples:
        Basic usage:

        >>> # Create from directory structure
        >>> datasets = ProteinMetadataDatasets.from_directory("protein_data/")
        >>>
        >>> # Load datasets from specific folds
        >>> train_data = datasets.load_datasets([DataFold.TRAIN])
        >>>
        >>> # Compute features with deduplication
        >>> config = ProteinFeatureComputationConfig("esm3_sm_open_v1")
        >>> features = datasets.compute_all_features(config)
        >>>
        >>> # Prepare for distance computation
        >>> src_feat, tgt_feat, src_names, tgt_names = (
        ...     datasets.get_distance_computation_ready_features("esm3_sm_open_v1")
        ... )

        Advanced usage with FASTA downloading:

        >>> # Download FASTA files from task list
        >>> datasets = ProteinMetadataDatasets.create_fasta_files_from_task_list(
        ...     "task_list.json",
        ...     "output_dir/",
        ...     uniprot_mapping_file="mapping.csv"
        ... )
        >>>
        >>> # Enable caching for large datasets
        >>> datasets = ProteinMetadataDatasets.from_directory(
        ...     "protein_data/",
        ...     cache_dir="./cache"
        ... )
    """

    def __init__(
        self,
        train_data_paths: Optional[List[RichPath]] = None,
        valid_data_paths: Optional[List[RichPath]] = None,
        test_data_paths: Optional[List[RichPath]] = None,
        num_workers: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        uniprot_mapping_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize ProteinMetadataDatasets.

        Args:
            train_data_paths: List of paths to training FASTA files
            valid_data_paths: List of paths to validation FASTA files
            test_data_paths: List of paths to test FASTA files
            num_workers: Number of workers for data loading
            cache_dir: Directory for persistent caching
            uniprot_mapping_file: Path to ChEMBL ID to UniProt ID mapping file

        Raises:
            DatasetValidationError: If validation of input parameters fails
        """
        logger.info("Initializing ProteinMetadataDatasets")

        # Validate inputs
        self._validate_init_params(num_workers, cache_dir, uniprot_mapping_file)

        self._fold_to_data_paths: Dict[DataFold, List[RichPath]] = {
            DataFold.TRAIN: train_data_paths or [],
            DataFold.VALIDATION: valid_data_paths or [],
            DataFold.TEST: test_data_paths or [],
        }
        self._num_workers = num_workers

        # Initialize global caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.global_cache = self._initialize_cache(cache_dir)

        # Cache for loaded datasets
        self._loaded_datasets: Dict[str, ProteinMetadataDataset] = {}

        # UniProt mapping
        self.uniprot_mapping_file = uniprot_mapping_file or UNIPROT_MAPPING_DEFAULT
        self._uniprot_mapping: Optional[pd.DataFrame] = None

        logger.info(
            f"Initialized with {len(train_data_paths or [])} training, {len(valid_data_paths or [])} validation, and {len(test_data_paths or [])} test paths"
        )
        if cache_dir:
            logger.info(f"Global caching enabled at {cache_dir}")

    def __repr__(self) -> str:
        return f"ProteinMetadataDatasets(train={len(self._fold_to_data_paths[DataFold.TRAIN])}, valid={len(self._fold_to_data_paths[DataFold.VALIDATION])}, test={len(self._fold_to_data_paths[DataFold.TEST])})"

    def _validate_init_params(
        self,
        num_workers: Optional[int],
        cache_dir: Optional[Union[str, Path]],
        uniprot_mapping_file: Optional[Union[str, Path]],
    ) -> None:
        """Validate initialization parameters.

        Args:
            num_workers: Number of workers for data loading
            cache_dir: Directory for persistent caching
            uniprot_mapping_file: Path to UniProt mapping file

        Raises:
            DatasetValidationError: If validation fails
        """
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

        if uniprot_mapping_file is not None:
            try:
                Path(uniprot_mapping_file)  # Just validate the path
            except (TypeError, ValueError) as e:
                raise DatasetValidationError(
                    "initialization", f"Invalid uniprot_mapping_file path: {uniprot_mapping_file}. Error: {e}"
                ) from e

    def _initialize_cache(self, cache_dir: Optional[Union[str, Path]]) -> Optional[GlobalMoleculeCache]:
        """Initialize global cache with proper error handling.

        Args:
            cache_dir: Directory for persistent caching

        Returns:
            GlobalMoleculeCache instance or None if caching disabled

        Raises:
            CacheError: If cache initialization fails
        """
        if not cache_dir:
            return None

        try:
            return GlobalMoleculeCache(cache_dir)
        except Exception as e:
            raise CacheError(f"Failed to initialize cache at {cache_dir}: {e}") from e

    @staticmethod
    def _validate_fold(fold: DataFold) -> None:
        """Validate that fold is a valid DataFold enum value.

        Args:
            fold: Data fold to validate

        Raises:
            DatasetValidationError: If fold is invalid
        """
        if not isinstance(fold, DataFold):
            raise DatasetValidationError("fold_validation", f"fold must be a DataFold enum, got {type(fold)}")

    @staticmethod
    def _validate_folds_list(folds: Optional[List[DataFold]]) -> None:
        """Validate list of folds.

        Args:
            folds: List of folds to validate

        Raises:
            DatasetValidationError: If folds list is invalid
        """
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
        """Validate and convert directory path to RichPath.

        Args:
            directory: Directory path to validate

        Returns:
            Validated RichPath object

        Raises:
            DatasetValidationError: If directory path is invalid
        """
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

    @contextmanager
    def _safe_file_read(self, file_path: Union[str, RichPath, Path]) -> Generator[str, None, None]:
        """Context manager for safe file reading with proper error handling.

        Args:
            file_path: Path to file to read

        Yields:
            File content as string

        Raises:
            DatasetValidationError: If file cannot be read
        """
        try:
            with open(str(file_path), "r", encoding="utf-8") as f:
                yield f.read().strip()
        except (OSError, IOError) as e:
            raise DatasetValidationError("file_reading", f"Failed to read file {file_path}: {e}") from e
        except UnicodeDecodeError as e:
            raise DatasetValidationError(
                "file_reading", f"Failed to decode file {file_path} as UTF-8: {e}"
            ) from e

    @property
    def uniprot_mapping(self) -> pd.DataFrame:
        """Lazy load UniProt mapping dataframe.

        Returns:
            DataFrame containing ChEMBL ID to UniProt ID mappings

        Raises:
            DatasetValidationError: If mapping file cannot be loaded
        """
        if self._uniprot_mapping is None:
            try:
                self._uniprot_mapping = pd.read_csv(self.uniprot_mapping_file)
                logger.info(f"Loaded UniProt mapping with {len(self._uniprot_mapping)} entries")
            except Exception as e:
                raise DatasetValidationError(
                    "uniprot_mapping", f"Failed to load UniProt mapping from {self.uniprot_mapping_file}: {e}"
                ) from e
        return self._uniprot_mapping

    def get_uniprot_id_from_chembl(self, chembl_id: str) -> Optional[str]:
        """Get UniProt ID from ChEMBL ID using mapping file.

        Args:
            chembl_id: ChEMBL task ID

        Returns:
            UniProt accession ID if found, None otherwise

        Raises:
            DatasetValidationError: If chembl_id is invalid
        """
        if not isinstance(chembl_id, str) or not chembl_id.strip():
            raise DatasetValidationError("chembl_lookup", "chembl_id must be a non-empty string")

        try:
            mapping_row = self.uniprot_mapping[self.uniprot_mapping["chembl_id"] == chembl_id]
            if not mapping_row.empty:
                return str(mapping_row.iloc[0]["target_accession_id"])
        except Exception as e:
            logger.warning(f"Failed to get UniProt ID for {chembl_id} from mapping file: {e}")

        # Fallback: try API
        try:
            logger.info(f"Attempting API fallback for {chembl_id}")
            result = get_protein_accession(chembl_id)
            return result
        except Exception as e:
            logger.error(f"API fallback failed for {chembl_id}: {e}")
            return None

    def download_fasta_for_task(self, chembl_id: str, output_path: Path) -> bool:
        """Download FASTA file for a single task.

        Args:
            chembl_id: ChEMBL task ID
            output_path: Path where to save the FASTA file

        Returns:
            True if successful, False otherwise
        """
        uniprot_id = self.get_uniprot_id_from_chembl(chembl_id)
        if not uniprot_id:
            logger.error(f"Could not get UniProt ID for {chembl_id}")
            return False

        try:
            # Download protein sequence
            seq_records = get_protein_sequence(uniprot_id)
            if not seq_records:
                logger.error(f"Could not download sequence for UniProt ID {uniprot_id}")
                return False

            # Save to FASTA file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                SeqIO.write(seq_records, f, "fasta")

            logger.info(f"Downloaded FASTA for {chembl_id} -> {uniprot_id} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download FASTA for {chembl_id}: {e}")
            return False

    @staticmethod
    def create_fasta_files_from_task_list(
        task_list_file: Union[str, Path],
        output_dir: Union[str, Path],
        uniprot_mapping_file: Optional[Union[str, Path]] = None,
    ) -> "ProteinMetadataDatasets":
        """Create FASTA files from a task list and return ProteinMetadataDatasets.

        Args:
            task_list_file: Path to JSON file containing fold-specific task lists
            output_dir: Base directory where to create train/test subdirectories
            uniprot_mapping_file: Path to CHEMBLID -> UNIPROT mapping file

        Returns:
            ProteinMetadataDatasets instance with paths to created FASTA files
        """
        logger.info(f"Creating FASTA files from task list {task_list_file}")

        # Load task list
        with open(task_list_file, "r") as f:
            task_data = json.load(f)

        output_dir = Path(output_dir)

        # Create ProteinDatasets instance for downloading
        protein_datasets = ProteinMetadataDatasets(uniprot_mapping_file=uniprot_mapping_file)

        # Process each fold
        fold_paths: Dict[DataFold, List[RichPath]] = {}
        fold_names = {"train": DataFold.TRAIN, "validation": DataFold.VALIDATION, "test": DataFold.TEST}

        for fold_name, fold_tasks in task_data.items():
            if fold_name in fold_names and fold_tasks:
                fold_dir = output_dir / fold_name
                fold_dir.mkdir(parents=True, exist_ok=True)

                fold_file_paths: List[RichPath] = []
                logger.info(f"Processing {len(fold_tasks)} tasks for {fold_name} fold")

                for task_id in fold_tasks:
                    fasta_path = fold_dir / f"{task_id}.fasta"

                    # Skip if file already exists
                    if fasta_path.exists():
                        logger.info(f"FASTA file already exists: {fasta_path}")
                        fold_file_paths.append(RichPath.create(str(fasta_path)))
                        continue

                    # Download FASTA file
                    if protein_datasets.download_fasta_for_task(task_id, fasta_path):
                        fold_file_paths.append(RichPath.create(str(fasta_path)))
                    else:
                        logger.warning(f"Failed to create FASTA for {task_id}")

                fold_paths[fold_names[fold_name]] = fold_file_paths
                logger.info(f"Created {len(fold_file_paths)} FASTA files for {fold_name}")

        # Create final ProteinDatasets instance
        return ProteinMetadataDatasets(
            train_data_paths=fold_paths.get(DataFold.TRAIN, []),
            valid_data_paths=fold_paths.get(DataFold.VALIDATION, []),
            test_data_paths=fold_paths.get(DataFold.TEST, []),
            uniprot_mapping_file=uniprot_mapping_file,
        )

    @staticmethod
    def from_directory(
        directory: Union[str, RichPath],
        task_list_file: Optional[Union[str, RichPath]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        uniprot_mapping_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> "ProteinMetadataDatasets":
        """Create ProteinMetadataDatasets from a directory containing FASTA files.

        Args:
            directory: Directory containing train/valid/test subdirectories with FASTA files
            task_list_file: File containing list of tasks to include
            cache_dir: Directory for persistent caching
            uniprot_mapping_file: Path to CHEMBLID -> UNIPROT mapping file
            **kwargs: Additional arguments

        Returns:
            ProteinMetadataDatasets instance
        """
        logger.info(f"Loading protein datasets from directory {directory}")
        if isinstance(directory, str):
            directory = RichPath.create(directory)
        elif isinstance(directory, Path):
            directory = RichPath.create(str(directory))

        # Handle task list file
        fold_task_lists: Dict[str, List[str]] = {}
        task_list: Optional[List[str]] = None
        if task_list_file is not None:
            if isinstance(task_list_file, str):
                task_list_file = RichPath.create(task_list_file)
            logger.info(f"Using task list file: {task_list_file}")

            with open(str(task_list_file), "r") as f:
                content = f.read().strip()

            try:
                task_data = json.loads(content)
                if isinstance(task_data, dict):
                    fold_task_lists = {
                        "train": task_data.get("train", []),
                        "valid": task_data.get("validation", []),
                        "test": task_data.get("test", []),
                    }
                    logger.info(
                        f"Loaded JSON task list: {len(fold_task_lists['train'])} train, "
                        f"{len(fold_task_lists['valid'])} validation, {len(fold_task_lists['test'])} test tasks"
                    )
                else:
                    task_list = task_data if isinstance(task_data, list) else [str(task_data)]
            except json.JSONDecodeError:
                task_list = [line.strip() for line in content.split("\n") if line.strip()]
                logger.info(f"Loaded text task list with {len(task_list)} tasks")

        def get_fold_file_names(data_fold_name: str) -> List[RichPath]:
            fold_dir = directory.join(data_fold_name)
            if not fold_dir.exists():
                logger.warning(f"Directory {fold_dir} does not exist")
                return []

            fold_path = Path(str(fold_dir))
            fasta_files = list(fold_path.glob("*.fasta"))

            rich_paths: List[RichPath] = []
            for file_path in fasta_files:
                rich_path = RichPath.create(str(file_path))
                task_name = file_path.stem  # Get filename without extension

                # Use fold-specific task list if available
                if fold_task_lists:
                    applicable_tasks = fold_task_lists.get(data_fold_name, [])
                    if not applicable_tasks or task_name in applicable_tasks:
                        rich_paths.append(rich_path)
                elif task_list is None or task_name in task_list:
                    rich_paths.append(rich_path)

            return rich_paths

        train_data_paths = get_fold_file_names("train")
        valid_data_paths = get_fold_file_names("valid")
        test_data_paths = get_fold_file_names("test")

        logger.info(
            f"Found {len(train_data_paths)} training, {len(valid_data_paths)} validation, and {len(test_data_paths)} test protein files"
        )

        return ProteinMetadataDatasets(
            train_data_paths=train_data_paths,
            valid_data_paths=valid_data_paths,
            test_data_paths=test_data_paths,
            cache_dir=cache_dir,
            uniprot_mapping_file=uniprot_mapping_file,
            **kwargs,
        )

    def get_num_fold_tasks(self, fold: DataFold) -> int:
        """Get number of tasks in a specific fold.

        Args:
            fold: The fold to get number of tasks for

        Returns:
            Number of tasks in the fold

        Raises:
            DatasetValidationError: If fold is invalid
        """
        self._validate_fold(fold)
        return len(self._fold_to_data_paths[fold])

    def get_task_names(self, data_fold: DataFold) -> List[str]:
        """Get list of task names in a specific fold.

        Args:
            data_fold: The fold to get task names for

        Returns:
            List of task names in the fold

        Raises:
            DatasetValidationError: If data_fold is invalid
        """
        self._validate_fold(data_fold)
        task_names: List[str] = []
        for path in self._fold_to_data_paths[data_fold]:
            # Extract task name from file path (assume CHEMBL format)
            file_name = path.basename()
            task_name = file_name.replace(".fasta", "")
            task_names.append(task_name)
        return task_names

    def load_datasets(self, folds: Optional[List[DataFold]] = None) -> Dict[str, ProteinMetadataDataset]:
        """Load all protein datasets from specified folds.

        Args:
            folds: List of folds to load. If None, loads all folds

        Returns:
            Dictionary mapping dataset names to loaded ProteinMetadataDataset objects

        Raises:
            DatasetValidationError: If folds list is invalid
        """
        if folds is None:
            folds = [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]

        self._validate_folds_list(folds)

        datasets: Dict[str, ProteinMetadataDataset] = {}
        for fold in folds:
            fold_name = FOLD_NAME_MAPPING[fold]
            for path in self._fold_to_data_paths[fold]:
                file_name = path.basename()
                task_name = file_name.replace(".fasta", "")
                dataset_name = f"{fold_name}_{task_name}"

                # Check if already loaded
                if dataset_name not in self._loaded_datasets:
                    try:
                        logger.info(f"Loading protein dataset {dataset_name}")

                        # Load FASTA file
                        fasta_dict = convert_fasta_to_dict(str(path))

                        if not fasta_dict:
                            logger.warning(f"Empty FASTA file: {path}")
                            continue

                        # Get first protein (assuming one protein per task)
                        uniprot_id = list(fasta_dict.keys())[0]
                        if "|" in uniprot_id:
                            uniprot_id = uniprot_id.split("|")[1]  # Extract UniProt ID from header

                        sequence = list(fasta_dict.values())[0]

                        protein_dataset = ProteinMetadataDataset(
                            task_id=task_name, uniprot_id=uniprot_id, sequence=sequence
                        )

                        self._loaded_datasets[dataset_name] = protein_dataset

                    except Exception as e:
                        logger.error(f"Failed to load protein dataset {dataset_name}: {e}")
                        raise DatasetValidationError(
                            dataset_name, f"Failed to load dataset from {path}: {e}"
                        ) from e

                datasets[dataset_name] = self._loaded_datasets[dataset_name]

        logger.info(f"Loaded {len(datasets)} protein datasets")
        return datasets

    def compute_all_features(
        self,
        config: ProteinFeatureComputationConfig,
        folds: Optional[List[DataFold]] = None,
    ) -> Dict[str, NDArray[np.float32]]:
        """Compute features for all protein datasets using the provided configuration.

        This method provides significant efficiency gains by:
        1. Finding all unique UniProt IDs across all datasets
        2. Computing features only once per unique protein (if deduplication enabled)
        3. Distributing computed features back to all datasets
        4. Using persistent caching to avoid recomputation

        Args:
            config: Protein feature computation configuration
            folds: List of folds to process. If None, processes all folds

        Returns:
            Dictionary mapping dataset names to computed features

        Raises:
            FeaturizationError: If feature computation fails
            DatasetValidationError: If input validation fails
        """
        start_time = time.time()

        if folds is None:
            folds = [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]

        self._validate_folds_list(folds)

        try:
            # Load all datasets
            datasets = self.load_datasets(folds)
            dataset_list = list(datasets.values())
            dataset_names = list(datasets.keys())

            logger.info(
                f"Computing protein features for {len(dataset_list)} datasets using {config.featurizer_name}"
            )

            # Choose computation strategy
            if config.enable_deduplication:
                results, unique_proteins_count = self._compute_features_with_deduplication(
                    config, datasets, dataset_list, dataset_names
                )
            else:
                results = self._compute_features_separately(config, datasets)
                unique_proteins_count = len(dataset_list)

            # Log performance metrics
            elapsed_time = time.time() - start_time
            self._log_computation_stats(
                elapsed_time, dataset_list, unique_proteins_count, config.enable_deduplication
            )

            return results

        except Exception as e:
            logger.error(f"Protein feature computation failed: {e}")
            raise FeaturizationError("", config.featurizer_name, str(e)) from e

    def compute_all_features_with_deduplication(
        self,
        featurizer_name: str = DEFAULT_FEATURIZER_NAME,
        layer: Optional[int] = None,
        folds: Optional[List[DataFold]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray[np.float32]]:
        """Legacy method for computing protein features with deduplication.

        This method is deprecated. Use compute_all_features() with ProteinFeatureComputationConfig instead.

        Args:
            featurizer_name: Name of protein featurizer to use
            layer: Layer number for ESM models
            folds: List of folds to process. If None, processes all folds
            batch_size: Batch size for feature computation
            force_recompute: Whether to force recomputation even if cached

        Returns:
            Dictionary mapping dataset names to computed features
        """
        import warnings

        warnings.warn(
            "compute_all_features_with_deduplication is deprecated. "
            "Use compute_all_features() with ProteinFeatureComputationConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        config = ProteinFeatureComputationConfig(
            featurizer_name=featurizer_name,
            layer=layer,
            batch_size=batch_size,
            force_recompute=force_recompute,
            enable_deduplication=True,
        )
        return self.compute_all_features(config, folds)

    def _compute_features_with_deduplication(
        self,
        config: ProteinFeatureComputationConfig,
        datasets: Dict[str, ProteinMetadataDataset],
        dataset_list: List[ProteinMetadataDataset],
        dataset_names: List[str],
    ) -> Tuple[Dict[str, NDArray[np.float32]], int]:
        """Compute features using UniProt ID deduplication.

        Args:
            config: Feature computation configuration
            datasets: Dictionary of loaded datasets
            dataset_list: List of dataset objects
            dataset_names: List of dataset names

        Returns:
            Tuple of (computed features dict, unique proteins count)

        Raises:
            FeaturizationError: If feature computation fails
        """
        try:
            # Collect unique proteins by UniProt ID
            unique_proteins, protein_to_datasets = self._collect_unique_proteins(dataset_names, dataset_list)

            logger.info(f"Found {len(unique_proteins)} unique proteins across {len(dataset_list)} datasets")

            # Compute features for unique proteins
            results: Dict[str, NDArray[np.float32]] = {}
            if len(unique_proteins) > 0:
                logger.info(f"Computing features for {len(unique_proteins)} unique proteins")
                all_features = get_protein_features(unique_proteins, config.featurizer_name, config.layer)

                # Map features back to datasets
                results = self._distribute_features_to_datasets(protein_to_datasets, all_features)
            else:
                logger.warning("No proteins found to compute features for")

            return results, len(unique_proteins)

        except Exception as e:
            logger.error(f"Deduplication-based feature computation failed: {e}")
            raise FeaturizationError("", config.featurizer_name, f"Deduplication failed: {e}") from e

    def _compute_features_separately(
        self,
        config: ProteinFeatureComputationConfig,
        datasets: Dict[str, ProteinMetadataDataset],
    ) -> Dict[str, NDArray[np.float32]]:
        """Compute features for each dataset separately (fallback method).

        Args:
            config: Feature computation configuration
            datasets: Dictionary of loaded datasets

        Returns:
            Dictionary mapping dataset names to computed features

        Raises:
            FeaturizationError: If feature computation fails for any dataset
        """
        logger.warning("Computing features separately for each dataset (deduplication disabled)")
        results: Dict[str, NDArray[np.float32]] = {}

        for dataset_name, dataset in datasets.items():
            try:
                logger.info(f"Computing features for protein dataset {dataset_name}")
                features = dataset.get_features(config.featurizer_name, config.layer)
                results[dataset_name] = features

            except Exception as e:
                logger.error(f"Feature computation failed for dataset {dataset_name}: {e}")
                raise FeaturizationError(
                    "", config.featurizer_name, f"Failed for dataset {dataset_name}: {e}"
                ) from e

        return results

    def _collect_unique_proteins(
        self,
        dataset_names: List[str],
        dataset_list: List[ProteinMetadataDataset],
    ) -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, ProteinMetadataDataset]]]]:
        """Collect unique proteins by UniProt ID from all datasets.

        Args:
            dataset_names: List of dataset names
            dataset_list: List of dataset objects

        Returns:
            Tuple of (unique_proteins dict, protein_to_datasets mapping)
        """
        unique_proteins: Dict[str, str] = {}
        protein_to_datasets: Dict[str, List[Tuple[str, ProteinMetadataDataset]]] = {}

        for dataset_name, dataset in zip(dataset_names, dataset_list):
            uniprot_id = dataset.uniprot_id

            if uniprot_id not in unique_proteins:
                unique_proteins[uniprot_id] = dataset.sequence
                protein_to_datasets[uniprot_id] = []

            protein_to_datasets[uniprot_id].append((dataset_name, dataset))

        return unique_proteins, protein_to_datasets

    def _distribute_features_to_datasets(
        self,
        protein_to_datasets: Dict[str, List[Tuple[str, ProteinMetadataDataset]]],
        all_features: NDArray[np.float32],
    ) -> Dict[str, NDArray[np.float32]]:
        """Distribute computed features back to individual datasets.

        Args:
            protein_to_datasets: Mapping from UniProt ID to dataset tuples
            all_features: Array of computed features for all unique proteins

        Returns:
            Dictionary mapping dataset names to feature arrays
        """
        results: Dict[str, NDArray[np.float32]] = {}

        for i, (uniprot_id, datasets_list) in enumerate(protein_to_datasets.items()):
            protein_features = all_features[i]

            for dataset_name, dataset in datasets_list:
                dataset.features = protein_features
                results[dataset_name] = protein_features

        return results

    def _log_computation_stats(
        self,
        elapsed_time: float,
        dataset_list: List[ProteinMetadataDataset],
        unique_proteins_count: int,
        deduplication_enabled: bool,
    ) -> None:
        """Log performance statistics for feature computation.

        Args:
            elapsed_time: Time taken for computation
            dataset_list: List of processed datasets
            unique_proteins_count: Number of unique proteins processed
            deduplication_enabled: Whether deduplication was used
        """
        total_datasets = len(dataset_list)

        if total_datasets > 0:
            if deduplication_enabled:
                deduplication_percentage = (total_datasets - unique_proteins_count) / total_datasets * 100
                logger.info(
                    f"Computed protein features for {total_datasets} datasets "
                    f"({unique_proteins_count} unique proteins) in {elapsed_time:.2f} seconds. "
                    f"Deduplication saved {deduplication_percentage:.1f}% computation"
                )
            else:
                logger.info(
                    f"Computed protein features for {total_datasets} datasets in {elapsed_time:.2f} seconds "
                    "(deduplication disabled)"
                )
        else:
            logger.info(f"No datasets to process - completed in {elapsed_time:.2f} seconds")

    def get_distance_computation_ready_features(
        self,
        featurizer_name: str = DEFAULT_FEATURIZER_NAME,
        layer: Optional[int] = None,
        source_fold: DataFold = DataFold.TRAIN,
        target_folds: Optional[List[DataFold]] = None,
    ) -> Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]], List[str], List[str]]:
        """Get protein features organized for efficient N×M distance matrix computation.

        Args:
            featurizer_name: Name of protein featurizer to use
            layer: Layer number for ESM models
            source_fold: Fold to use as source datasets (N)
            target_folds: Folds to use as target datasets (M)

        Returns:
            Tuple containing:
            - source_features: List of feature arrays for source datasets
            - target_features: List of feature arrays for target datasets
            - source_names: List of source dataset names
            - target_names: List of target dataset names

        Raises:
            DatasetValidationError: If fold validation fails
            FeaturizationError: If feature computation fails

        Note:
            If target_folds is not provided, validation and test folds are used as defaults.
        """
        # Validate inputs
        self._validate_fold(source_fold)
        if target_folds is None:
            target_folds = [DataFold.VALIDATION, DataFold.TEST]
        self._validate_folds_list(target_folds)

        try:
            # Compute features for all relevant datasets using new method
            all_folds = [source_fold] + target_folds
            config = ProteinFeatureComputationConfig(featurizer_name=featurizer_name, layer=layer)
            all_features = self.compute_all_features(config, folds=all_folds)

            # Separate source and target features
            source_features, source_names, target_features, target_names = (
                self._separate_source_target_features(all_features, source_fold, target_folds)
            )

            logger.info(
                f"Prepared {len(source_features)} source and {len(target_features)} target protein datasets "
                f"for {len(source_features)}×{len(target_features)} distance matrix computation"
            )

            return source_features, target_features, source_names, target_names

        except Exception as e:
            logger.error(f"Failed to prepare features for distance computation: {e}")
            raise

    def _separate_source_target_features(
        self,
        all_features: Dict[str, NDArray[np.float32]],
        source_fold: DataFold,
        target_folds: List[DataFold],
    ) -> Tuple[List[NDArray[np.float32]], List[str], List[NDArray[np.float32]], List[str]]:
        """Separate computed features into source and target groups.

        Args:
            all_features: Dictionary mapping dataset names to computed features
            source_fold: Source fold for feature separation
            target_folds: Target folds for feature separation

        Returns:
            Tuple of (source_features, source_names, target_features, target_names)
        """

        # Separate source and target features
        source_features: List[NDArray[np.float32]] = []
        source_names: List[str] = []
        target_features: List[NDArray[np.float32]] = []
        target_names: List[str] = []

        for dataset_name, features in all_features.items():
            fold_name = dataset_name.split("_")[0].lower()

            if fold_name == FOLD_NAME_MAPPING[source_fold]:
                source_features.append(features)
                source_names.append(dataset_name)
            elif any(fold_name == FOLD_NAME_MAPPING[fold] for fold in target_folds):
                target_features.append(features)
                target_names.append(dataset_name)

        return source_features, source_names, target_features, target_names

    def save_features_to_file(
        self,
        output_path: Union[str, Path],
        featurizer_name: str = "esm3_sm_open_v1",
        layer: Optional[int] = None,
        folds: Optional[List[DataFold]] = None,
    ) -> None:
        """Save computed features to a pickle file for efficient loading.

        Args:
            output_path: Path where to save the features
            featurizer_name: Name of protein featurizer used
            layer: Layer number for ESM models
            folds: List of folds to save. If None, saves all folds
        """
        features = self.compute_all_features_with_deduplication(
            featurizer_name=featurizer_name, layer=layer, folds=folds
        )

        # Prepare data for saving
        save_data: Dict[str, Any] = {
            "features": features,
            "featurizer_name": featurizer_name,
            "layer": layer,
            "timestamp": time.time(),
            "num_datasets": len(features),
        }

        # Extract task IDs by fold
        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "validation", DataFold.TEST: "test"}
        for fold, fold_name in fold_names.items():
            fold_tasks = [
                name.replace(f"{fold_name}_", "")
                for name in features.keys()
                if name.startswith(f"{fold_name}_")
            ]
            save_data[f"{fold_name}_task_ids"] = fold_tasks

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import pickle

        with open(output_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Saved protein features for {len(features)} datasets to {output_path}")

    @staticmethod
    def load_features_from_file(file_path: Union[str, Path]) -> Dict[str, NDArray[np.float32]]:
        """Load precomputed features from a pickle file.

        Args:
            file_path: Path to the saved features file

        Returns:
            Dictionary mapping dataset names to features
        """
        import pickle

        with open(file_path, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)

        logger.info(f"Loaded protein features for {data.get('num_datasets', 0)} datasets")
        logger.info(
            f"Featurizer: {data.get('featurizer_name', 'unknown')}, Layer: {data.get('layer', 'unknown')}"
        )

        features: Dict[str, NDArray[np.float32]] = data["features"]
        return features

    def get_global_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the global cache usage.

        Returns:
            Cache statistics if global cache is enabled, None otherwise
        """
        if self.global_cache is None or self.global_cache.persistent_cache is None:
            return None

        try:
            return {
                "persistent_cache_stats": self.global_cache.persistent_cache.get_stats(),
                "persistent_cache_size": self.global_cache.persistent_cache.get_cache_size_info(),
                "loaded_datasets_count": len(self._loaded_datasets),
                "cache_enabled": True,
                "cache_directory": str(self.cache_dir) if self.cache_dir else None,
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {
                "cache_enabled": True,
                "cache_directory": str(self.cache_dir) if self.cache_dir else None,
                "error": str(e),
            }


# Legacy compatibility: users can still import ProteinMetadataDatasets as old ProteinMetadataDataset
# But now we have both ProteinMetadataDataset (single) and ProteinMetadataDatasets (collection)
