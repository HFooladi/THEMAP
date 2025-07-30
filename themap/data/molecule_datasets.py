import time
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dpu_utils.utils import RichPath

from ..utils.cache_utils import CacheKey, GlobalMoleculeCache, get_global_feature_cache
from ..utils.logging import get_logger, setup_logging
from .molecule_dataset import MoleculeDataset, get_task_name_from_path

# Setup logging
setup_logging()
logger = get_logger(__name__)


class DataFold(IntEnum):
    """Enum for data fold types.

    This enum represents the different data splits used in machine learning:
    - TRAIN (0): Training/source tasks
    - VALIDATION (1): Validation/development tasks
    - TEST (2): Test/target tasks

    By inheriting from IntEnum, each fold type is assigned an integer value
    which allows for easy indexing and comparison operations.
    """

    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class MoleculeDatasets:
    """Dataset of related tasks, provided as individual files split into meta-train, meta-valid and
    meta-test sets."""

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
            train_data_paths (List[RichPath]): List of paths to training data files.
            valid_data_paths (List[RichPath]): List of paths to validation data files.
            test_data_paths (List[RichPath]): List of paths to test data files.
            num_workers (Optional[int]): Number of workers for data loading.
            cache_dir (Optional[Union[str, Path]]): Directory for persistent caching.
        """
        logger.info("Initializing MoleculeDatasets")
        self._fold_to_data_paths: Dict[DataFold, List[RichPath]] = {
            DataFold.TRAIN: train_data_paths or [],
            DataFold.VALIDATION: valid_data_paths or [],
            DataFold.TEST: test_data_paths or [],
        }
        self._num_workers = num_workers

        # Initialize global caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.global_cache = GlobalMoleculeCache(cache_dir) if cache_dir else None

        # Cache for loaded datasets
        self._loaded_datasets: Dict[str, MoleculeDataset] = {}

        logger.info(
            f"Initialized with {len(self._fold_to_data_paths[DataFold.TRAIN])} training, "
            f"{len(self._fold_to_data_paths[DataFold.VALIDATION])} validation, and "
            f"{len(self._fold_to_data_paths[DataFold.TEST])} test paths"
        )
        if cache_dir:
            logger.info(f"Global caching enabled at {cache_dir}")

    def __repr__(self) -> str:
        return f"MoleculeDatasets(train={len(self._fold_to_data_paths[DataFold.TRAIN])}, valid={len(self._fold_to_data_paths[DataFold.VALIDATION])}, test={len(self._fold_to_data_paths[DataFold.TEST])})"

    def get_num_fold_tasks(self, fold: DataFold) -> int:
        """Get number of tasks in a specific fold.

        Args:
            fold (DataFold): The fold to get number of tasks for.

        Returns:
            int: Number of tasks in the fold.
        """
        return len(self._fold_to_data_paths[fold])

    @staticmethod
    def from_directory(
        directory: Union[str, RichPath],
        task_list_file: Optional[Union[str, RichPath]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> "MoleculeDatasets":
        """Create MoleculeDatasets from a directory.

        Args:
            directory (Union[str, RichPath]): Directory containing train/valid/test subdirectories.
            task_list_file (Optional[Union[str, RichPath]]): File containing list of tasks to include.
                Can be either a text file (one task per line) or JSON file with fold-specific task lists.
            cache_dir (Optional[Union[str, Path]]): Directory for persistent caching.
            **kwargs (any): Additional arguments to pass to MoleculeDatasets constructor.

        Returns:
            MoleculeDatasets: Created dataset.
        """
        logger.info(f"Loading datasets from directory {directory}")
        if isinstance(directory, str):
            directory = RichPath.create(directory)
        elif isinstance(directory, Path):
            directory = RichPath.create(str(directory))
        else:
            directory = directory

        # Handle task list file
        fold_task_lists = {}
        if task_list_file is not None:
            if isinstance(task_list_file, str):
                task_list_file = RichPath.create(task_list_file)
            logger.info(f"Using task list file: {task_list_file}")

            with open(str(task_list_file), "r") as f:
                content = f.read().strip()

            # Try to parse as JSON first
            try:
                import json

                task_data = json.loads(content)
                if isinstance(task_data, dict):
                    # JSON format with fold-specific lists
                    fold_task_lists = {
                        "train": task_data.get("train", []),
                        "valid": task_data.get("valid", []),
                        "test": task_data.get("test", []),
                    }
                    logger.info(
                        f"Loaded JSON task list: {len(fold_task_lists['train'])} train, "
                        f"{len(fold_task_lists['valid'])} valid, {len(fold_task_lists['test'])} test tasks"
                    )
                else:
                    # JSON format but not the expected structure
                    logger.warning(
                        "JSON task list file does not have expected structure, treating as general task list"
                    )
                    task_list = task_data if isinstance(task_data, list) else [str(task_data)]
            except json.JSONDecodeError:
                # Not JSON, treat as text file with one task per line
                task_list = [line.strip() for line in content.split("\n") if line.strip()]
                logger.info(f"Loaded text task list with {len(task_list)} tasks")
        else:
            task_list = None

        def get_fold_file_names(data_fold_name: str) -> List[RichPath]:
            fold_dir = directory.join(data_fold_name)
            if not fold_dir.exists():
                logger.warning(f"Directory {fold_dir} does not exist")
                return []

            # Convert to Path for file listing since RichPath doesn't have glob
            from pathlib import Path

            fold_path = Path(str(fold_dir))

            # Get all .jsonl.gz files in the directory
            jsonl_files = list(fold_path.glob("*.jsonl.gz"))

            # Convert back to RichPath objects and filter by task list
            rich_paths = []
            for file_path in jsonl_files:
                rich_path = RichPath.create(str(file_path))
                task_name = get_task_name_from_path(rich_path)

                # Use fold-specific task list if available, otherwise use general task list
                if fold_task_lists:
                    applicable_tasks = fold_task_lists.get(data_fold_name, [])
                    if task_name in applicable_tasks:
                        rich_paths.append(rich_path)
                elif task_list is None or task_name in task_list:
                    rich_paths.append(rich_path)

            return rich_paths

        train_data_paths = get_fold_file_names("train")
        valid_data_paths = get_fold_file_names("valid")
        test_data_paths = get_fold_file_names("test")

        logger.info(
            f"Found {len(train_data_paths)} training, {len(valid_data_paths)} valid, and {len(test_data_paths)} test tasks"
        )
        return MoleculeDatasets(
            train_data_paths=train_data_paths,
            valid_data_paths=valid_data_paths,
            test_data_paths=test_data_paths,
            cache_dir=cache_dir,
            **kwargs,
        )

    def get_task_names(self, data_fold: DataFold) -> List[str]:
        """Get list of task names in a specific fold.

        Args:
            data_fold (DataFold): The fold to get task names for.

        Returns:
            List[str]: List of task names in the fold.
        """
        return [get_task_name_from_path(path) for path in self._fold_to_data_paths[data_fold]]

    def load_datasets(self, folds: Optional[List[DataFold]] = None) -> Dict[str, MoleculeDataset]:
        """Load all datasets from specified folds.

        Args:
            folds: List of folds to load. If None, loads all folds.

        Returns:
            Dictionary mapping dataset names to loaded datasets
        """
        if folds is None:
            folds = [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]

        # Create mapping from fold values to names
        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "valid", DataFold.TEST: "test"}

        datasets = {}
        for fold in folds:
            fold_name = fold_names[fold]
            for path in self._fold_to_data_paths[fold]:
                dataset_name = f"{fold_name}_{get_task_name_from_path(path)}"

                # Check if already loaded
                if dataset_name not in self._loaded_datasets:
                    logger.info(f"Loading dataset {dataset_name}")
                    self._loaded_datasets[dataset_name] = MoleculeDataset.load_from_file(path)

                datasets[dataset_name] = self._loaded_datasets[dataset_name]

        logger.info(f"Loaded {len(datasets)} datasets")
        return datasets

    def compute_all_features_with_deduplication(
        self,
        featurizer_name: str,
        folds: Optional[List[DataFold]] = None,
        batch_size: int = 1000,
        n_jobs: int = -1,
        force_recompute: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute features for all datasets with global SMILES deduplication.

        This method provides significant efficiency gains by:
        1. Finding all unique SMILES across all datasets
        2. Computing features only once per unique SMILES
        3. Distributing computed features back to all datasets
        4. Using persistent caching to avoid recomputation

        Args:
            featurizer_name: Name of featurizer to use
            folds: List of folds to process. If None, processes all folds
            batch_size: Batch size for feature computation
            n_jobs: Number of parallel jobs
            force_recompute: Whether to force recomputation even if cached

        Returns:
            Dictionary mapping dataset names to computed features

        """
        start_time = time.time()

        if folds is None:
            folds = [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]

        # Load all datasets
        datasets = self.load_datasets(folds)
        dataset_list = list(datasets.values())
        dataset_names = list(datasets.keys())

        logger.info(f"Computing features for {len(dataset_list)} datasets using {featurizer_name}")

        # Global SMILES deduplication
        if self.global_cache:
            unique_smiles_map = self.global_cache.get_unique_smiles_across_datasets(dataset_list)
            unique_smiles = list(unique_smiles_map.keys())

            # Compute features for unique SMILES
            smiles_to_features = self.global_cache.batch_compute_features(
                unique_smiles=unique_smiles,
                featurizer_name=featurizer_name,
                batch_size=batch_size,
                n_jobs=n_jobs,
            )

            # Distribute features back to datasets using proper interfaces
            results = {}
            global_cache = get_global_feature_cache()

            for dataset_idx, (dataset_name, dataset) in enumerate(zip(dataset_names, dataset_list)):
                logger.info(f"Distributing features to dataset {dataset_name}")

                # Create feature array for this dataset and store in global cache
                dataset_features = []
                for mol in dataset.data:
                    canonical_smiles = self.global_cache._canonicalize_smiles(mol.smiles)
                    if canonical_smiles in smiles_to_features:
                        feature_vector = smiles_to_features[canonical_smiles]
                        dataset_features.append(feature_vector)

                        # Store in global cache using proper interface
                        cache_key = CacheKey(smiles=mol.smiles, featurizer_name=featurizer_name)
                        global_cache.store(cache_key, feature_vector)
                    else:
                        # Fallback: zero features if SMILES not found
                        feature_dim = len(next(iter(smiles_to_features.values())))
                        zero_features = np.zeros(feature_dim)
                        dataset_features.append(zero_features)

                        cache_key = CacheKey(smiles=mol.smiles, featurizer_name=featurizer_name)
                        global_cache.store(cache_key, zero_features)

                dataset_features_array = np.array(dataset_features)

                # Update dataset to track current featurizer (proper interface)
                dataset._current_featurizer = featurizer_name

                results[dataset_name] = dataset_features_array

        else:
            # Fallback: compute features for each dataset separately
            logger.warning("Global cache not available, computing features separately for each dataset")
            results = {}
            for dataset_name, dataset in datasets.items():
                logger.info(f"Computing features for dataset {dataset_name}")
                features = dataset.get_dataset_embedding(
                    featurizer_name=featurizer_name,
                    n_jobs=n_jobs,
                    force_recompute=force_recompute,
                    batch_size=batch_size,
                )
                results[dataset_name] = features

        elapsed_time = time.time() - start_time
        total_molecules = sum(len(dataset.data) for dataset in dataset_list)
        unique_molecules = len(unique_smiles) if self.global_cache else total_molecules

        if total_molecules > 0:
            deduplication_percentage = (total_molecules - unique_molecules) / total_molecules * 100
            logger.info(
                f"Computed features for {total_molecules} total molecules "
                f"({unique_molecules} unique) in {elapsed_time:.2f} seconds. "
                f"Deduplication saved {deduplication_percentage:.1f}% computation"
            )
        else:
            logger.info(f"No molecules to process (empty datasets) - completed in {elapsed_time:.2f} seconds")

        return results

    def get_distance_computation_ready_features(
        self,
        featurizer_name: str,
        source_fold: DataFold = DataFold.TRAIN,
        target_folds: Optional[List[DataFold]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[str]]:
        """Get features organized for efficient N×M distance matrix computation.

        Args:
            featurizer_name: Name of featurizer to use
            source_fold: Fold to use as source datasets (N)
            target_folds: Folds to use as target datasets (M)

        Returns:
            Tuple containing:
            - source_features: List of feature arrays for source datasets
            - target_features: List of feature arrays for target datasets
            - source_names: List of source dataset names
            - target_names: List of target dataset names

        Notes:
            if you dont provide target_folds, it will use the validation and test folds as default target folds.
        """
        if target_folds is None:
            target_folds = [DataFold.VALIDATION, DataFold.TEST]

        # Compute features for all relevant datasets
        all_folds = [source_fold] + target_folds
        all_features = self.compute_all_features_with_deduplication(
            featurizer_name=featurizer_name, folds=all_folds
        )

        # Create mapping from fold values to names
        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "valid", DataFold.TEST: "test"}

        # Separate source and target features
        source_features = []
        source_names = []
        target_features = []
        target_names = []

        for dataset_name, features in all_features.items():
            fold_name = dataset_name.split("_")[0].lower()

            if fold_name == fold_names[source_fold]:
                source_features.append(features)
                source_names.append(dataset_name)
            elif any(fold_name == fold_names[fold] for fold in target_folds):
                target_features.append(features)
                target_names.append(dataset_name)

        logger.info(
            f"Prepared {len(source_features)} source and {len(target_features)} target datasets "
            f"for {len(source_features)}×{len(target_features)} distance matrix computation"
        )

        return source_features, target_features, source_names, target_names

    def get_global_cache_stats(self) -> Optional[Dict]:
        """Get statistics about the global cache usage.

        Returns:
            Cache statistics if global cache is enabled, None otherwise
        """
        if self.global_cache is None or self.global_cache.persistent_cache is None:
            return None

        return {
            "persistent_cache_stats": self.global_cache.persistent_cache.get_stats(),
            "persistent_cache_size": self.global_cache.persistent_cache.get_cache_size_info(),
            "loaded_datasets": len(self._loaded_datasets),
        }
