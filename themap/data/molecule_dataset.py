import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from dpu_utils.utils import RichPath
from numpy.typing import NDArray

from themap.data.molecule_datapoint import MoleculeDatapoint
from themap.utils.cache_utils import PersistentFeatureCache
from themap.utils.featurizer_utils import get_featurizer
from themap.utils.logging import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

# Type definitions for better type hints
DatasetStats = Dict[str, Union[int, float]]  # Type for dataset statistics


def get_task_name_from_path(path: RichPath) -> str:
    """Extract task name from file path.

    Args:
        path (RichPath): Path-like object

    Returns:
        str: Extracted task name
    """
    try:
        name = path.basename()
        return name[: -len(".jsonl.gz")] if name.endswith(".jsonl.gz") else name
    except Exception:
        return "unknown_task"


@dataclass
class MoleculeDataset:
    """Data structure holding information for a dataset of molecules.

    This class represents a collection of molecule datapoints, providing methods for
    dataset manipulation, feature computation, and statistical analysis.

    Attributes:
        task_id (str): String describing the task this dataset is taken from.
        data (List[MoleculeDatapoint]): List of MoleculeDatapoint objects.
        _features (Optional[NDArray[np.float32]]): Cached features for the dataset.
        _cache_info (Dict[str, Any]): Information about the feature caching.
    """

    task_id: str
    data: List[MoleculeDatapoint] = field(default_factory=list)
    _features: Optional[NDArray[np.float32]] = None
    _cache_info: Dict[str, Any] = field(default_factory=dict)
    _persistent_cache: Optional[PersistentFeatureCache] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate dataset initialization."""
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")
        if not isinstance(self.data, list):
            raise TypeError("data must be a list of MoleculeDatapoints")
        if not all(isinstance(x, MoleculeDatapoint) for x in self.data):
            raise TypeError("All items in data must be MoleculeDatapoint instances")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MoleculeDatapoint:
        return self.data[idx]

    def __iter__(self) -> Iterator[MoleculeDatapoint]:
        return iter(self.data)

    def __repr__(self) -> str:
        return f"MoleculeDataset(task_id={self.task_id}, task_size={len(self.data)})"

    def get_dataset_embedding(
        self,
        featurizer_name: str,
        n_jobs: Optional[int] = None,
        force_recompute: bool = False,
        batch_size: int = 1000,
    ) -> NDArray[np.float32]:
        """Get the features for the entire dataset using a featurizer.

        Efficiently computes features for all molecules in the dataset using the
        specified featurizer, taking advantage of the featurizer's built-in
        parallelization capabilities and maintaining a two-level cache strategy.

        Args:
            featurizer_name (str): Name of the featurizer to use
            n_jobs (Optional[int]): Number of parallel jobs. If provided, temporarily
                                overrides the featurizer's own setting
            force_recompute (bool): Whether to force recomputation even if cached
            batch_size (int): Batch size for processing, used for memory efficiency
                            when handling large datasets

        Returns:
            NDArray[np.float32]: Features for the entire dataset, shape (n_samples, n_features)

        Raises:
            ValueError: If the generated features length doesn't match the dataset length
            TypeError: If featurizer_name is not a string
            RuntimeError: If featurization fails
            IndexError: If dataset is empty
        """
        # Input validation
        if not isinstance(featurizer_name, str):
            raise TypeError(f"featurizer_name must be a string, got {type(featurizer_name)}")

        if not featurizer_name.strip():
            raise ValueError("featurizer_name cannot be empty")

        if len(self.data) == 0:
            raise IndexError("Cannot compute features for empty dataset")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if n_jobs is not None and n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")

        # Start timing for performance measurement
        start_time = time.time()

        # Return cached features if available and not forcing recompute
        if self._features is not None and not force_recompute:
            logger.info(f"Using cached dataset features for {self.task_id}")
            return self._features

        logger.info(f"Generating embeddings for dataset {self.task_id} with {len(self.data)} molecules")

        # Get the featurizer with error handling
        try:
            featurizer = get_featurizer(featurizer_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load featurizer '{featurizer_name}': {e}") from e

        # Check if we're dealing with a BaseFeaturizer
        if not hasattr(featurizer, "n_jobs"):
            logger.warning(
                f"Featurizer {featurizer_name} does not appear to be a BaseFeaturizer. "
                f"Parallelization settings may not apply."
            )

        # Configure parallelization if needed and possible
        original_n_jobs: Optional[int] = None
        if n_jobs is not None and hasattr(featurizer, "n_jobs"):
            original_n_jobs = featurizer.n_jobs
            featurizer.n_jobs = n_jobs
            logger.info(f"Temporarily set featurizer to use {featurizer.n_jobs} jobs")

        try:
            # Get all SMILES strings with validation
            smiles_list = []
            for i, dp in enumerate(self.data):
                if not dp.smiles or not isinstance(dp.smiles, str):
                    raise ValueError(f"Invalid SMILES at index {i}: {dp.smiles}")
                smiles_list.append(dp.smiles)

            # Find unique SMILES for optimization (avoid computing duplicates)
            unique_smiles: List[str] = []
            smiles_indices: Dict[str, List[int]] = {}
            for i, smiles in enumerate(smiles_list):
                if smiles not in smiles_indices:
                    unique_smiles.append(smiles)
                    smiles_indices[smiles] = []
                smiles_indices[smiles].append(i)

            has_duplicates = len(unique_smiles) < len(smiles_list)
            if has_duplicates:
                logger.info(
                    f"Found {len(smiles_list) - len(unique_smiles)} duplicate SMILES - optimizing computation"
                )

            # Initialize features variable to satisfy mypy
            unique_features: NDArray[np.float32]

            # Check if dataset is too large for memory-efficient processing
            if len(unique_smiles) > batch_size:
                logger.info(f"Processing large dataset in batches of {batch_size}")
                # Process in batches to avoid memory issues
                all_features: List[NDArray[np.float32]] = []

                for i in range(0, len(unique_smiles), batch_size):
                    batch = unique_smiles[i : i + batch_size]

                    try:
                        # Preprocess batch with the featurizer's method
                        processed_batch, _ = featurizer.preprocess(batch)

                        # Transform preprocessed batch (assume scikit-learn API)
                        if hasattr(featurizer, "transform"):
                            batch_features = featurizer.transform(processed_batch)
                        else:
                            # Fallback if transform not available
                            batch_features = featurizer(processed_batch)

                        # Validate batch features
                        if batch_features is None:
                            raise RuntimeError(f"Featurizer returned None for batch {i // batch_size + 1}")

                        if not isinstance(batch_features, np.ndarray):
                            # Convert to numpy array if needed
                            batch_features = np.asarray(batch_features)

                        all_features.append(batch_features)

                    except Exception as e:
                        raise RuntimeError(f"Failed to process batch {i // batch_size + 1}: {e}") from e

                # Combine all batches with error handling
                try:
                    unique_features = np.vstack(all_features)
                except Exception as e:
                    raise RuntimeError(f"Failed to combine batch features: {e}") from e
            else:
                # Process all unique SMILES at once
                try:
                    processed_smiles, _ = featurizer.preprocess(unique_smiles)

                    # Transform preprocessed SMILES
                    if hasattr(featurizer, "transform"):
                        unique_features = featurizer.transform(processed_smiles)
                    else:
                        unique_features = featurizer(processed_smiles)

                    # Validate features
                    if unique_features is None:
                        raise RuntimeError("Featurizer returned None")

                    if not isinstance(unique_features, np.ndarray):
                        # Convert to numpy array if needed
                        unique_features = np.asarray(unique_features)  # type: ignore[unreachable]

                except Exception as e:
                    raise RuntimeError(f"Failed to compute features: {e}") from e

            # Validate feature dimensions
            if len(unique_features) != len(unique_smiles):
                raise ValueError(
                    f"Feature count ({len(unique_features)}) does not match unique SMILES count ({len(unique_smiles)})"
                )

            # Create a mapping from unique SMILES to their features
            smiles_to_features: Dict[str, NDArray[np.float32]] = {
                smiles: unique_features[i] for i, smiles in enumerate(unique_smiles)
            }

            # Create the full feature array, maintaining the original order
            try:
                features = np.array([smiles_to_features[smiles] for smiles in smiles_list])
            except Exception as e:
                raise RuntimeError(f"Failed to create feature array: {e}") from e

            if len(features) != len(self.data):
                raise ValueError(
                    f"Feature length ({len(features)}) does not match data length ({len(self.data)})"
                )

            # Update individual molecule features for efficient access later
            for i, molecule in enumerate(self.data):
                # Use a view instead of a copy when possible to save memory
                molecule._features = features[i]

            # Cache at dataset level
            self._features = features

            elapsed_time = time.time() - start_time
            logger.info(
                f"Successfully generated embeddings for dataset {self.task_id} in {elapsed_time:.2f} seconds"
            )

            return features

        except Exception as e:
            # Log the error for debugging
            logger.error(f"Error computing features for dataset {self.task_id}: {e}")
            raise

        finally:
            # Restore original n_jobs setting if we changed it
            if original_n_jobs is not None and hasattr(featurizer, "n_jobs"):
                featurizer.n_jobs = original_n_jobs
                logger.info(f"Restored featurizer to original setting of {original_n_jobs} jobs")

    def validate_dataset_integrity(self) -> bool:
        """Validate the integrity of the dataset.

        Returns:
            bool: True if dataset is valid, False otherwise

        Raises:
            ValueError: If critical integrity issues are found
        """
        if not self.data:
            raise ValueError("Dataset is empty")

        # Check for required attributes
        for i, datapoint in enumerate(self.data):
            if not hasattr(datapoint, "smiles") or not datapoint.smiles:
                raise ValueError(f"Datapoint at index {i} has invalid SMILES")
            if not isinstance(datapoint.smiles, str):
                raise ValueError(f"SMILES at index {i} must be string, got {type(datapoint.smiles)}")
            if not hasattr(datapoint, "bool_label"):
                raise ValueError(f"Datapoint at index {i} missing bool_label")

        logger.info(f"Dataset {self.task_id} passed integrity validation")
        return True

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics for the dataset.

        Returns:
            Dictionary with memory usage in MB for different components
        """
        import sys

        def get_size_mb(obj: Any) -> float:
            return sys.getsizeof(obj) / (1024 * 1024)

        memory_stats = {
            "dataset_object": get_size_mb(self),
            "data_list": get_size_mb(self.data),
            "datapoints": sum(get_size_mb(dp) for dp in self.data),
        }

        if self._features is not None:
            memory_stats["cached_features"] = get_size_mb(self._features)
        else:
            memory_stats["cached_features"] = 0.0

        memory_stats["total"] = sum(memory_stats.values())

        return memory_stats

    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up unnecessary data.

        Returns:
            Dictionary with optimization results
        """
        initial_memory = self.get_memory_usage()["total"]

        # Clear individual molecule feature caches if dataset features are available
        if self._features is not None:
            cleared_count = 0
            for molecule in self.data:
                if hasattr(molecule, "_features") and molecule._features is not None:
                    molecule._features = None
                    cleared_count += 1

            logger.info(f"Cleared {cleared_count} individual molecule feature caches")

        # Clear cache info that might be outdated
        self._cache_info.clear()

        final_memory = self.get_memory_usage()["total"]
        memory_saved = initial_memory - final_memory

        optimization_results = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_saved_mb": memory_saved,
            "memory_saved_percent": (memory_saved / initial_memory * 100) if initial_memory > 0 else 0,
        }

        logger.info(
            f"Memory optimization for {self.task_id}: saved {memory_saved:.2f} MB ({optimization_results['memory_saved_percent']:.1f}%)"
        )

        return optimization_results

    # Helper method to clear cache for benchmarking
    def clear_cache(self) -> None:
        """Clear all cached features."""
        self._features = None
        for molecule in self.data:
            molecule._features = None
        logger.info(f"Cleared cache for dataset {self.task_id}")

    def enable_persistent_cache(self, cache_dir: Union[str, Path]) -> None:
        """Enable persistent caching for this dataset.

        Args:
            cache_dir: Directory for storing cached features
        """
        self._persistent_cache = PersistentFeatureCache(cache_dir)
        logger.info(f"Enabled persistent cache for dataset {self.task_id} at {cache_dir}")

    def get_dataset_embedding_with_persistent_cache(
        self,
        featurizer_name: str,
        cache_dir: Optional[Union[str, Path]] = None,
        n_jobs: Optional[int] = None,
        force_recompute: bool = False,
        batch_size: int = 1000,
    ) -> NDArray[np.float32]:
        """Get dataset features with persistent caching enabled.

        This method provides the same functionality as get_dataset_embedding but with
        persistent disk caching to avoid recomputation across sessions.

        Args:
            featurizer_name: Name of the featurizer to use
            cache_dir: Directory for persistent cache (if None, uses existing cache)
            n_jobs: Number of parallel jobs
            force_recompute: Whether to force recomputation even if cached
            batch_size: Batch size for processing

        Returns:
            Features for the entire dataset
        """
        # Initialize persistent cache if needed
        if cache_dir is not None and self._persistent_cache is None:
            self.enable_persistent_cache(cache_dir)

        # Check persistent cache first if available
        if self._persistent_cache is not None and not force_recompute:
            smiles_list = [dp.smiles for dp in self.data]
            cache_key = self._persistent_cache.generate_cache_key(smiles_list, featurizer_name)

            cached_features = self._persistent_cache.get(cache_key)
            if cached_features is not None:
                logger.info(f"Using persistent cache for dataset {self.task_id}")
                self._features = cached_features

                # Update individual molecule features
                for i, molecule in enumerate(self.data):
                    molecule._features = cached_features[i]

                return cached_features

        # Compute features using the standard method
        features = self.get_dataset_embedding(
            featurizer_name=featurizer_name,
            n_jobs=n_jobs,
            force_recompute=force_recompute,
            batch_size=batch_size,
        )

        # Store in persistent cache if available
        if self._persistent_cache is not None:
            smiles_list = [dp.smiles for dp in self.data]
            cache_key = self._persistent_cache.generate_cache_key(smiles_list, featurizer_name)
            self._persistent_cache.store(cache_key, features)
            logger.info(f"Stored features in persistent cache for dataset {self.task_id}")

        return features

    def get_persistent_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the persistent cache.

        Returns:
            Cache statistics if persistent cache is enabled, None otherwise
        """
        if self._persistent_cache is None:
            return None

        return {
            "cache_stats": self._persistent_cache.get_stats(),
            "cache_size": self._persistent_cache.get_cache_size_info(),
        }

    # Method to measure cache efficiency
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        molecules_cached = sum(1 for dp in self.data if dp._features is not None)
        dataset_cached = self._features is not None

        cache_info = {
            "dataset_cached": dataset_cached,
            "molecules_cached": molecules_cached,
            "total_molecules": len(self.data),
            "cache_ratio": molecules_cached / len(self.data) if len(self.data) > 0 else 0,
        }
        logger.info(f"Cache info for dataset {self.task_id}: {cache_info}")
        self._cache_info.update(cache_info)
        return self._cache_info

    def get_prototype(self, featurizer_name: str) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get the prototype of the dataset.

        This method calculates the mean feature vector of positive and negative examples
        in the dataset using the specified featurizer.

        Args:
            featurizer_name (str): Name of the featurizer to use.

        Returns:
            Tuple[NDArray[np.float32], NDArray[np.float32]]: Tuple containing:
                - positive_prototype: Mean feature vector of positive examples
                - negative_prototype: Mean feature vector of negative examples

        Raises:
            ValueError: If there are no positive or negative examples in the dataset
            TypeError: If featurizer_name is not a string
            RuntimeError: If feature computation fails
        """
        # Input validation
        if not isinstance(featurizer_name, str):
            raise TypeError(f"featurizer_name must be a string, got {type(featurizer_name)}")

        if not featurizer_name.strip():
            raise ValueError("featurizer_name cannot be empty")

        logger.info(f"Calculating prototypes for dataset {self.task_id}")

        try:
            # Get the features for the entire dataset
            data_features = self.get_dataset_embedding(featurizer_name)
        except Exception as e:
            raise RuntimeError(f"Failed to compute features for prototyping: {e}") from e

        # Separate positive and negative examples with validation
        positive_indices = []
        negative_indices = []

        for i, datapoint in enumerate(self.data):
            if not hasattr(datapoint, "bool_label"):
                raise ValueError(f"Datapoint at index {i} missing bool_label attribute")

            if datapoint.bool_label:
                positive_indices.append(i)
            else:
                negative_indices.append(i)

        # Check if we have both positive and negative examples
        if not positive_indices:
            raise ValueError(f"Dataset {self.task_id} contains no positive examples")
        if not negative_indices:
            raise ValueError(f"Dataset {self.task_id} contains no negative examples")

        logger.info(f"Found {len(positive_indices)} positive and {len(negative_indices)} negative examples")

        try:
            # Calculate prototypes using numpy indexing for efficiency
            positive_features = data_features[positive_indices]
            negative_features = data_features[negative_indices]

            positive_prototype = np.mean(positive_features, axis=0)
            negative_prototype = np.mean(negative_features, axis=0)

            # Validate prototypes
            if np.any(np.isnan(positive_prototype)):
                raise ValueError("Positive prototype contains NaN values")
            if np.any(np.isnan(negative_prototype)):
                raise ValueError("Negative prototype contains NaN values")

        except Exception as e:
            raise RuntimeError(f"Failed to calculate prototypes: {e}") from e

        logger.info(f"Successfully calculated prototypes for dataset {self.task_id}")
        return positive_prototype, negative_prototype

    @property
    def get_features(self) -> Optional[NDArray[np.float32]]:
        """Get the cached features for the dataset.

        Returns:
            Optional[NDArray[np.float32]]: Cached features for the dataset if available, None otherwise.
        """
        if self._features is None:
            return None
        return np.array(self._features)

    @property
    def get_labels(self) -> NDArray[np.int32]:
        return np.array([data.bool_label for data in self.data])

    @property
    def get_smiles(self) -> List[str]:
        return [data.smiles for data in self.data]

    @property
    def get_ratio(self) -> float:
        """Get the ratio of positive to negative examples in the dataset.

        Returns:
            float: Ratio of positive to negative examples in the dataset.
        """
        return round(sum([data.bool_label for data in self.data]) / len(self.data), 2)

    @staticmethod
    def load_from_file(path: Union[str, RichPath]) -> "MoleculeDataset":
        """Load dataset from a JSONL.GZ file.

        Args:
            path (Union[str, RichPath]): Path to the JSONL.GZ file.

        Returns:
            MoleculeDataset: Loaded dataset.
        """
        if isinstance(path, str):
            path = RichPath.create(path)
        else:
            path = path

        logger.info(f"Loading dataset from {path}")
        samples = []
        for raw_sample in path.read_by_file_suffix():
            fingerprint_raw = raw_sample.get("fingerprints")
            if fingerprint_raw is not None:
                fingerprint: Optional[np.ndarray] = np.array(fingerprint_raw, dtype=np.int32)
            else:
                fingerprint = None

            descriptors_raw = raw_sample.get("descriptors")
            if descriptors_raw is not None:
                descriptors: Optional[np.ndarray] = np.array(descriptors_raw, dtype=np.float32)
            else:
                descriptors = None

            samples.append(
                MoleculeDatapoint(
                    task_id=get_task_name_from_path(path),
                    smiles=raw_sample["SMILES"],
                    bool_label=bool(float(raw_sample["Property"])),
                    numeric_label=float(raw_sample.get("RegressionProperty") or "nan"),
                    _fingerprint=fingerprint,
                    _features=descriptors,
                )
            )

        logger.info(f"Successfully loaded dataset from {path} with {len(samples)} samples")
        return MoleculeDataset(get_task_name_from_path(path), samples)

    def filter(self, condition: Callable[[MoleculeDatapoint], bool]) -> "MoleculeDataset":
        """Filter dataset based on a condition.

        Args:
            condition (Callable[[MoleculeDatapoint], bool]): Function that returns True/False
                for each datapoint.

        Returns:
            MoleculeDataset: New dataset containing only the filtered datapoints.
        """
        filtered_data = [dp for dp in self.data if condition(dp)]
        return MoleculeDataset(self.task_id, filtered_data)

    def get_statistics(self) -> DatasetStats:
        """Get statistics about the dataset.

        Returns:
            DatasetStats: Dictionary containing:
                - size: Total number of datapoints
                - positive_ratio: Ratio of positive to negative examples
                - avg_molecular_weight: Average molecular weight
                - avg_atoms: Average number of atoms
                - avg_bonds: Average number of bonds

        Raises:
            ValueError: If the dataset is empty
        """
        if not self.data:
            raise ValueError("Cannot compute statistics for empty dataset")

        return {
            "size": len(self),
            "positive_ratio": self.get_ratio,
            "avg_molecular_weight": float(np.mean([dp.molecular_weight for dp in self.data])),
            "avg_atoms": float(np.mean([dp.number_of_atoms for dp in self.data])),
            "avg_bonds": float(np.mean([dp.number_of_bonds for dp in self.data])),
        }
