import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from dpu_utils.utils import RichPath
from numpy.typing import NDArray

from ..utils.cache_utils import GlobalMoleculeCache
from ..utils.logging import get_logger
from .protein_datasets import DataFold

logger = get_logger(__name__)


@dataclass
class BaseMetadataDataset(ABC):
    """Abstract base class for individual metadata datasets.

    This provides a blueprint for different types of metadata (text, numerical, etc.)

    Args:
        task_id (str): Unique identifier for the task (CHEMBL ID)
        metadata_type (str): Type of metadata (e.g., 'assay_description', 'target_info', 'bioactivity')
        raw_data (Any): Raw metadata content
        features (Optional[NDArray[np.float32]]): Optional pre-computed features
    """

    task_id: str
    metadata_type: str
    raw_data: Any
    features: Optional[NDArray[np.float32]] = None

    def __post_init__(self) -> None:
        """Validate initialization data."""
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")
        if not isinstance(self.metadata_type, str):
            raise TypeError("metadata_type must be a string")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task_id={self.task_id}, type={self.metadata_type})"

    @abstractmethod
    def get_features(self, featurizer_name: str, **kwargs: Any) -> NDArray[np.float32]:
        """Get features using the specified featurizer.

        Args:
            featurizer_name: Name of the featurizer to use
            **kwargs: Additional featurizer arguments

        Returns:
            Computed features
        """
        pass

    @abstractmethod
    def preprocess_data(self) -> Any:
        """Preprocess raw data for featurization.

        Returns:
            Preprocessed data ready for featurization
        """
        pass


@dataclass
class TextMetadataDataset(BaseMetadataDataset):
    """Metadata dataset for text-based data (e.g., assay descriptions, target descriptions).

    Args:
        task_id: Unique identifier for the task
        metadata_type: Type of text metadata
        raw_data: Text content (str)
        features: Optional pre-computed text features
    """

    def preprocess_data(self) -> str:
        """Preprocess text data (cleaning, normalization)."""
        if not isinstance(self.raw_data, str):
            return str(self.raw_data)

        # Basic text preprocessing
        text = self.raw_data.strip()
        # Add more preprocessing as needed (lowercase, remove special chars, etc.)
        return text

    def get_features(
        self,
        featurizer_name: str = "sentence-transformers",
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Get text features using the specified featurizer.

        Args:
            featurizer_name: Name of the text featurizer ('sentence-transformers', 'tfidf', 'bert', etc.)
            model_name: Specific model name for the featurizer
            **kwargs: Additional featurizer arguments

        Returns:
            Computed text features
        """
        if self.features is None:
            processed_text = self.preprocess_data()

            if featurizer_name == "sentence-transformers":
                try:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(model_name)
                    self.features = model.encode([processed_text])[0].astype(np.float32)
                except ImportError:
                    logger.error("sentence-transformers not available, falling back to dummy features")
                    self.features = np.random.randn(384).astype(np.float32)

            elif featurizer_name == "tfidf":
                # Placeholder for TF-IDF implementation
                logger.warning("TF-IDF featurizer not implemented, using dummy features")
                self.features = np.random.randn(100).astype(np.float32)

            else:
                raise ValueError(f"Unknown text featurizer: {featurizer_name}")

        return self.features


@dataclass
class NumericalMetadataDataset(BaseMetadataDataset):
    """Metadata dataset for numerical data (e.g., molecular weight, IC50 values).

    Args:
        task_id: Unique identifier for the task
        metadata_type: Type of numerical metadata
        raw_data: Numerical values (float, int, or list/array of numbers)
        features: Optional pre-computed numerical features
    """

    def preprocess_data(self) -> NDArray[np.float32]:
        """Preprocess numerical data."""
        if isinstance(self.raw_data, (int, float)):
            return np.array([self.raw_data], dtype=np.float32)
        elif isinstance(self.raw_data, (list, tuple)):
            return np.array(self.raw_data, dtype=np.float32)
        elif isinstance(self.raw_data, np.ndarray):
            return self.raw_data.astype(np.float32)
        else:
            raise ValueError(f"Cannot convert {type(self.raw_data)} to numerical data")

    def get_features(self, featurizer_name: str = "standardize", **kwargs: Any) -> NDArray[np.float32]:
        """Get numerical features using the specified featurizer.

        Args:
            featurizer_name: Name of the numerical featurizer ('standardize', 'normalize', 'log_transform')
            **kwargs: Additional featurizer arguments

        Returns:
            Computed numerical features
        """
        if self.features is None:
            processed_data = self.preprocess_data()

            if featurizer_name == "standardize":
                # Z-score standardization
                mean = np.mean(processed_data)
                std = np.std(processed_data)
                if std > 0:
                    self.features = (processed_data - mean) / std
                else:
                    self.features = processed_data

            elif featurizer_name == "normalize":
                # Min-max normalization
                min_val = np.min(processed_data)
                max_val = np.max(processed_data)
                if max_val > min_val:
                    self.features = (processed_data - min_val) / (max_val - min_val)
                else:
                    self.features = processed_data

            elif featurizer_name == "log_transform":
                # Log transformation (handle negative values)
                self.features = np.log1p(np.abs(processed_data))

            else:
                # Default: use as-is
                self.features = processed_data

        return self.features


@dataclass
class CategoricalMetadataDataset(BaseMetadataDataset):
    """Metadata dataset for categorical data (e.g., assay type, organism).

    Args:
        task_id: Unique identifier for the task
        metadata_type: Type of categorical metadata
        raw_data: Categorical value (str or list of strings)
        features: Optional pre-computed categorical features
    """

    def preprocess_data(self) -> List[str]:
        """Preprocess categorical data."""
        if isinstance(self.raw_data, str):
            return [self.raw_data.strip().lower()]
        elif isinstance(self.raw_data, (list, tuple)):
            return [str(item).strip().lower() for item in self.raw_data]
        else:
            return [str(self.raw_data).strip().lower()]

    def get_features(
        self, featurizer_name: str = "one_hot", vocabulary: Optional[List[str]] = None, **kwargs: Any
    ) -> NDArray[np.float32]:
        """Get categorical features using the specified featurizer.

        Args:
            featurizer_name: Name of the categorical featurizer ('one_hot', 'label_encode', 'embedding')
            vocabulary: Known vocabulary for encoding
            **kwargs: Additional featurizer arguments

        Returns:
            Computed categorical features
        """
        if self.features is None:
            processed_data = self.preprocess_data()

            if featurizer_name == "one_hot":
                if vocabulary is None:
                    # Create vocabulary from current data
                    vocabulary = list(set(processed_data))

                # One-hot encoding
                feature_vector = np.zeros(len(vocabulary), dtype=np.float32)
                for item in processed_data:
                    if item in vocabulary:
                        feature_vector[vocabulary.index(item)] = 1.0

                self.features = feature_vector

            elif featurizer_name == "label_encode":
                if vocabulary is None:
                    vocabulary = list(set(processed_data))

                # Label encoding (single value)
                if len(processed_data) == 1:
                    self.features = np.array([vocabulary.index(processed_data[0])], dtype=np.float32)
                else:
                    # Multi-label case
                    labels = [vocabulary.index(item) if item in vocabulary else -1 for item in processed_data]
                    self.features = np.array(labels, dtype=np.float32)

            else:
                raise ValueError(f"Unknown categorical featurizer: {featurizer_name}")

        return self.features


class MetadataDatasets:
    """Collection of metadata datasets for different folds (train/validation/test).

    This class provides a unified interface for managing various types of metadata
    across different task folds, similar to MoleculeDatasets and ProteinDatasets.
    """

    def __init__(
        self,
        train_data_paths: Optional[List[RichPath]] = None,
        valid_data_paths: Optional[List[RichPath]] = None,
        test_data_paths: Optional[List[RichPath]] = None,
        metadata_type: str = "text",
        dataset_class: Type[BaseMetadataDataset] = TextMetadataDataset,
        num_workers: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize MetadataDatasets.

        Args:
            train_data_paths: List of paths to training metadata files
            valid_data_paths: List of paths to validation metadata files
            test_data_paths: List of paths to test metadata files
            metadata_type: Type of metadata being handled
            dataset_class: Class to use for individual metadata datasets
            num_workers: Number of workers for data loading
            cache_dir: Directory for persistent caching
        """
        logger.info(f"Initializing MetadataDatasets for {metadata_type}")
        self._fold_to_data_paths: Dict[DataFold, List[RichPath]] = {
            DataFold.TRAIN: train_data_paths or [],
            DataFold.VALIDATION: valid_data_paths or [],
            DataFold.TEST: test_data_paths or [],
        }
        self._metadata_type = metadata_type
        self._dataset_class = dataset_class
        self._num_workers = num_workers

        # Initialize caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.global_cache = GlobalMoleculeCache(cache_dir) if cache_dir else None

        # Cache for loaded datasets
        self._loaded_datasets: Dict[str, BaseMetadataDataset] = {}

        logger.info(
            f"Initialized with {len(train_data_paths or [])} training, {len(valid_data_paths or [])} validation, "
            f"and {len(test_data_paths or [])} test metadata files"
        )

    def __repr__(self) -> str:
        return f"MetadataDatasets(type={self._metadata_type}, train={len(self._fold_to_data_paths[DataFold.TRAIN])}, valid={len(self._fold_to_data_paths[DataFold.VALIDATION])}, test={len(self._fold_to_data_paths[DataFold.TEST])})"

    @staticmethod
    def from_directory(
        directory: Union[str, RichPath],
        metadata_type: str = "text",
        dataset_class: Type[BaseMetadataDataset] = TextMetadataDataset,
        task_list_file: Optional[Union[str, RichPath]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        file_pattern: str = "*.json",
        **kwargs: Any,
    ) -> "MetadataDatasets":
        """Create MetadataDatasets from a directory containing metadata files.

        Args:
            directory: Directory containing train/valid/test subdirectories
            metadata_type: Type of metadata being loaded
            dataset_class: Class to use for individual metadata datasets
            task_list_file: File containing list of tasks to include
            cache_dir: Directory for persistent caching
            file_pattern: File pattern to match (e.g., '*.json', '*.csv')
            **kwargs: Additional arguments

        Returns:
            MetadataDatasets instance
        """
        logger.info(f"Loading {metadata_type} metadata from directory {directory}")
        if isinstance(directory, str):
            directory = RichPath.create(directory)
        elif isinstance(directory, Path):
            directory = RichPath.create(str(directory))

        # Handle task list filtering (similar to protein_datasets)
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
                else:
                    task_list = task_data if isinstance(task_data, list) else [str(task_data)]
            except json.JSONDecodeError:
                task_list = [line.strip() for line in content.split("\n") if line.strip()]

        def get_fold_file_names(data_fold_name: str) -> List[RichPath]:
            fold_dir = directory.join(data_fold_name)
            if not fold_dir.exists():
                logger.warning(f"Directory {fold_dir} does not exist")
                return []

            fold_path = Path(str(fold_dir))

            # Extract file extension from pattern
            if file_pattern.startswith("*."):
                extension = file_pattern[2:]
                metadata_files = list(fold_path.glob(f"*.{extension}"))
            else:
                metadata_files = list(fold_path.glob(file_pattern))

            rich_paths: List[RichPath] = []
            for file_path in metadata_files:
                rich_path = RichPath.create(str(file_path))
                task_name = file_path.stem

                # Filter by task list
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
            f"Found {len(train_data_paths)} training, {len(valid_data_paths)} validation, "
            f"and {len(test_data_paths)} test {metadata_type} metadata files"
        )

        return MetadataDatasets(
            train_data_paths=train_data_paths,
            valid_data_paths=valid_data_paths,
            test_data_paths=test_data_paths,
            metadata_type=metadata_type,
            dataset_class=dataset_class,
            cache_dir=cache_dir,
            **kwargs,
        )

    def get_num_fold_tasks(self, fold: DataFold) -> int:
        """Get number of tasks in a specific fold."""
        return len(self._fold_to_data_paths[fold])

    def get_task_names(self, data_fold: DataFold) -> List[str]:
        """Get list of task names in a specific fold."""
        task_names: List[str] = []
        for path in self._fold_to_data_paths[data_fold]:
            file_name = path.basename()
            # Remove extension to get task name
            task_name = Path(file_name).stem
            task_names.append(task_name)
        return task_names

    def load_datasets(self, folds: Optional[List[DataFold]] = None) -> Dict[str, BaseMetadataDataset]:
        """Load all metadata datasets from specified folds.

        Args:
            folds: List of folds to load. If None, loads all folds.

        Returns:
            Dictionary mapping dataset names to loaded metadata datasets
        """
        if folds is None:
            folds = [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]

        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "validation", DataFold.TEST: "test"}

        datasets: Dict[str, BaseMetadataDataset] = {}
        for fold in folds:
            fold_name = fold_names[fold]
            for path in self._fold_to_data_paths[fold]:
                file_name = path.basename()
                task_name = Path(file_name).stem
                dataset_name = f"{fold_name}_{task_name}"

                # Check if already loaded
                if dataset_name not in self._loaded_datasets:
                    logger.info(f"Loading {self._metadata_type} metadata dataset {dataset_name}")

                    # Load metadata file (support JSON, CSV, etc.)
                    raw_data = self._load_metadata_file(str(path))

                    metadata_dataset = self._dataset_class(
                        task_id=task_name, metadata_type=self._metadata_type, raw_data=raw_data
                    )

                    self._loaded_datasets[dataset_name] = metadata_dataset

                datasets[dataset_name] = self._loaded_datasets[dataset_name]

        logger.info(f"Loaded {len(datasets)} {self._metadata_type} metadata datasets")
        return datasets

    def _load_metadata_file(self, file_path: str) -> Any:
        """Load metadata from a file based on its extension."""
        path = Path(file_path)

        if path.suffix.lower() == ".json":
            with open(file_path, "r") as f:
                return json.load(f)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
            # Return first row as dict if single row, otherwise return as dict
            if len(df) == 1:
                return df.iloc[0].to_dict()
            else:
                return df.to_dict("records")
        elif path.suffix.lower() == ".txt":
            with open(file_path, "r") as f:
                return f.read().strip()
        else:
            # Default: try JSON first, then text
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                with open(file_path, "r") as f:
                    return f.read().strip()

    def compute_all_features_with_deduplication(
        self,
        featurizer_name: str,
        folds: Optional[List[DataFold]] = None,
        force_recompute: bool = False,
        **featurizer_kwargs: Any,
    ) -> Dict[str, NDArray[np.float32]]:
        """Compute features for all metadata datasets with deduplication.

        Args:
            featurizer_name: Name of featurizer to use
            folds: List of folds to process. If None, processes all folds
            force_recompute: Whether to force recomputation even if cached
            **featurizer_kwargs: Additional arguments for featurizer

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

        logger.info(
            f"Computing {self._metadata_type} features for {len(dataset_list)} datasets using {featurizer_name}"
        )

        # Collect unique metadata by content hash
        unique_metadata: Dict[str, Tuple[BaseMetadataDataset, str]] = {}
        content_to_datasets: Dict[str, List[Tuple[str, BaseMetadataDataset]]] = {}

        for dataset_name, dataset in zip(dataset_names, dataset_list):
            # Create content hash for deduplication
            content_hash = hash(str(dataset.raw_data))
            content_key = f"{self._metadata_type}_{content_hash}"

            if content_key not in unique_metadata:
                unique_metadata[content_key] = (dataset, str(dataset.raw_data))
                content_to_datasets[content_key] = []

            content_to_datasets[content_key].append((dataset_name, dataset))

        logger.info(
            f"Found {len(unique_metadata)} unique {self._metadata_type} items across {len(dataset_list)} datasets"
        )

        # Compute features for unique metadata
        results: Dict[str, NDArray[np.float32]] = {}
        for content_key, (sample_dataset, content) in unique_metadata.items():
            logger.debug(f"Computing features for {content_key}")
            features = sample_dataset.get_features(featurizer_name, **featurizer_kwargs)

            # Map features back to all datasets with same content
            for dataset_name, dataset in content_to_datasets[content_key]:
                dataset.features = features
                results[dataset_name] = features

        elapsed_time = time.time() - start_time
        total_datasets = len(dataset_list)
        unique_items = len(unique_metadata)

        if total_datasets > 0:
            deduplication_percentage = (total_datasets - unique_items) / total_datasets * 100
            logger.info(
                f"Computed {self._metadata_type} features for {total_datasets} datasets "
                f"({unique_items} unique items) in {elapsed_time:.2f} seconds. "
                f"Deduplication saved {deduplication_percentage:.1f}% computation"
            )

        return results

    def get_distance_computation_ready_features(
        self,
        featurizer_name: str,
        source_fold: DataFold = DataFold.TRAIN,
        target_folds: Optional[List[DataFold]] = None,
        **featurizer_kwargs: Any,
    ) -> Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]], List[str], List[str]]:
        """Get metadata features organized for efficient N×M distance matrix computation.

        Args:
            featurizer_name: Name of featurizer to use
            source_fold: Fold to use as source datasets (N)
            target_folds: Folds to use as target datasets (M)
            **featurizer_kwargs: Additional arguments for featurizer

        Returns:
            Tuple containing:
            - source_features: List of feature arrays for source datasets
            - target_features: List of feature arrays for target datasets
            - source_names: List of source dataset names
            - target_names: List of target dataset names
        """
        if target_folds is None:
            target_folds = [DataFold.VALIDATION, DataFold.TEST]

        # Compute features for all relevant datasets
        all_folds = [source_fold] + target_folds
        all_features = self.compute_all_features_with_deduplication(
            featurizer_name=featurizer_name, folds=all_folds, **featurizer_kwargs
        )

        # Create mapping from fold values to names
        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "validation", DataFold.TEST: "test"}

        # Separate source and target features
        source_features: List[NDArray[np.float32]] = []
        source_names: List[str] = []
        target_features: List[NDArray[np.float32]] = []
        target_names: List[str] = []

        for dataset_name, features in all_features.items():
            fold_name = dataset_name.split("_")[0].lower()

            if fold_name == fold_names[source_fold]:
                source_features.append(features)
                source_names.append(dataset_name)
            elif any(fold_name == fold_names[fold] for fold in target_folds):
                target_features.append(features)
                target_names.append(dataset_name)

        logger.info(
            f"Prepared {len(source_features)} source and {len(target_features)} target {self._metadata_type} datasets "
            f"for {len(source_features)}×{len(target_features)} distance matrix computation"
        )

        return source_features, target_features, source_names, target_names

    def save_features_to_file(
        self,
        output_path: Union[str, Path],
        featurizer_name: str,
        folds: Optional[List[DataFold]] = None,
        **featurizer_kwargs: Any,
    ) -> None:
        """Save computed features to a pickle file for efficient loading."""
        features = self.compute_all_features_with_deduplication(
            featurizer_name=featurizer_name, folds=folds, **featurizer_kwargs
        )

        # Prepare data for saving
        save_data: Dict[str, Any] = {
            "features": features,
            "featurizer_name": featurizer_name,
            "metadata_type": self._metadata_type,
            "featurizer_kwargs": featurizer_kwargs,
            "timestamp": time.time(),
            "num_datasets": len(features),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import pickle

        with open(output_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Saved {self._metadata_type} features for {len(features)} datasets to {output_path}")

    @staticmethod
    def load_features_from_file(file_path: Union[str, Path]) -> Dict[str, NDArray[np.float32]]:
        """Load precomputed features from a pickle file."""
        import pickle

        with open(file_path, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)

        logger.info(
            f"Loaded {data.get('metadata_type', 'unknown')} features for {data.get('num_datasets', 0)} datasets"
        )
        logger.info(f"Featurizer: {data.get('featurizer_name', 'unknown')}")

        features: Dict[str, NDArray[np.float32]] = data["features"]
        return features
