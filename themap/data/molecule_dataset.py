from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import logging
import numpy as np
import time
from dpu_utils.utils import RichPath

from themap.data.molecule_datapoint import MoleculeDatapoint
from themap.utils.featurizer_utils import get_featurizer
from themap.utils.logging import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

# Type definitions for better type hints
FeatureArray = np.ndarray  # Type alias for numpy feature arrays
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
        _features (Optional[FeatureArray]): Cached features for the dataset.
        _cache_info (Dict[str, Any]): Information about the feature caching.
    """

    task_id: str
    data: List[MoleculeDatapoint] = field(default_factory=list)
    _features: Optional[FeatureArray] = None
    _cache_info: Dict[str, Any] = field(default_factory=dict)


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

    def get_dataset_embedding(self, featurizer_name: str, n_jobs: Optional[int] = None, 
                            force_recompute: bool = False, batch_size: int = 1000) -> FeatureArray:
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
            FeatureArray: Features for the entire dataset, shape (n_samples, n_features)
            
        Raises:
            ValueError: If the generated features length doesn't match the dataset length
        """
        # Start timing for performance measurement
        start_time = time.time()
        
        # Return cached features if available and not forcing recompute
        if self._features is not None and not force_recompute:
            logger.info(f"Using cached dataset features for {self.task_id}")
            return self._features
            
        logger.info(f"Generating embeddings for dataset {self.task_id} with {len(self.data)} molecules")
        
        # Get the featurizer
        featurizer = get_featurizer(featurizer_name)
        
        # Check if we're dealing with a BaseFeaturizer
        if not hasattr(featurizer, 'n_jobs'):
            logger.warning(f"Featurizer {featurizer_name} does not appear to be a BaseFeaturizer. "
                        f"Parallelization settings may not apply.")
        
        # Configure parallelization if needed and possible
        original_n_jobs = None
        if n_jobs is not None and hasattr(featurizer, 'n_jobs'):
            original_n_jobs = featurizer.n_jobs
            featurizer.n_jobs = n_jobs
            logger.info(f"Temporarily set featurizer to use {featurizer.n_jobs} jobs")
        
        try:
            # Get all SMILES strings
            smiles_list = [dp.smiles for dp in self.data]
            
            # Find unique SMILES for optimization (avoid computing duplicates)
            unique_smiles = []
            smiles_indices = {}
            for i, smiles in enumerate(smiles_list):
                if smiles not in smiles_indices:
                    unique_smiles.append(smiles)
                    smiles_indices[smiles] = []
                smiles_indices[smiles].append(i)
            
            has_duplicates = len(unique_smiles) < len(smiles_list)
            if has_duplicates:
                logger.info(f"Found {len(smiles_list) - len(unique_smiles)} duplicate SMILES - optimizing computation")
            
            # Check if dataset is too large for memory-efficient processing
            if len(unique_smiles) > batch_size:
                logger.info(f"Processing large dataset in batches of {batch_size}")
                # Process in batches to avoid memory issues
                all_features = []
                
                for i in range(0, len(unique_smiles), batch_size):
                    batch = unique_smiles[i:i+batch_size]
                    
                    # Preprocess batch with the featurizer's method
                    processed_batch, _ = featurizer.preprocess(batch)
                    
                    # Transform preprocessed batch (assume scikit-learn API)
                    if hasattr(featurizer, 'transform'):
                        batch_features = featurizer.transform(processed_batch)
                    else:
                        # Fallback if transform not available
                        batch_features = featurizer(processed_batch)
                    
                    all_features.append(batch_features)
                
                # Combine all batches
                unique_features = np.vstack(all_features)
            else:
                # Process all unique SMILES at once
                processed_smiles, _ = featurizer.preprocess(unique_smiles)
                
                # Transform preprocessed SMILES
                if hasattr(featurizer, 'transform'):
                    unique_features = featurizer.transform(processed_smiles)
                else:
                    unique_features = featurizer(processed_smiles)
            
            # Create a mapping from unique SMILES to their features
            smiles_to_features = {smiles: unique_features[i] for i, smiles in enumerate(unique_smiles)}
            
            # Create the full feature array, maintaining the original order
            features = np.array([smiles_to_features[smiles] for smiles in smiles_list])
            
            # Ensure we have the right shape and format
            if not isinstance(features, np.ndarray):
                features = np.array(features)
                
            if len(features) != len(self.data):
                raise ValueError(f"Feature length ({len(features)}) does not match data length ({len(self.data)})")
            
            # Update individual molecule features for efficient access later
            for i, molecule in enumerate(self.data):
                # Use a view instead of a copy when possible to save memory
                molecule._features = features[i]
                
            # Cache at dataset level
            self._features = features
            
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully generated embeddings for dataset {self.task_id} in {elapsed_time:.2f} seconds")
            
            return features
            
        finally:
            # Restore original n_jobs setting if we changed it
            if original_n_jobs is not None and hasattr(featurizer, 'n_jobs'):
                featurizer.n_jobs = original_n_jobs
                logger.info(f"Restored featurizer to original setting of {original_n_jobs} jobs")

    # Helper method to clear cache for benchmarking
    def clear_cache(self):
        """Clear all cached features."""
        self._features = None
        for molecule in self.data:
            molecule._features = None
        logger.info(f"Cleared cache for dataset {self.task_id}")

    # Method to measure cache efficiency
    def get_cache_info(self):
        """Get information about the current cache state."""
        molecules_cached = sum(1 for dp in self.data if dp._features is not None)
        dataset_cached = self._features is not None
        
        cache_info = {
            'dataset_cached': dataset_cached,
            'molecules_cached': molecules_cached,
            'total_molecules': len(self.data),
            'cache_ratio': molecules_cached / len(self.data) if len(self.data) > 0 else 0
        }
        logger.info(f"Cache info for dataset {self.task_id}: {cache_info}")
        self._cache_info.update(cache_info)
        return self._cache_info

    def get_prototype(self, featurizer_name: str) -> tuple[FeatureArray, FeatureArray]:
        """Get the prototype of the dataset.

        This method calculates the mean feature vector of positive and negative examples
        in the dataset using the specified featurizer.

        Args:
            featurizer_name (str): Name of the featurizer to use.

        Returns:
            tuple[FeatureArray, FeatureArray]: Tuple containing:
                - positive_prototype: Mean feature vector of positive examples
                - negative_prototype: Mean feature vector of negative examples
                
        Raises:
            ValueError: If there are no positive or negative examples in the dataset
        """
        logger.info(f"Calculating prototypes for dataset {self.task_id}")

        # Get the features for the entire dataset
        data_features = self.get_dataset_embedding(featurizer_name)

        # Calculate the mean feature vector of positive and negative examples
        positives = [data_features[i] for i in range(len(data_features)) if self.data[i].bool_label]
        negatives = [data_features[i] for i in range(len(data_features)) if not self.data[i].bool_label]

        if not positives or not negatives:
            raise ValueError("Dataset must contain both positive and negative examples")

        positive_prototype = np.array(positives).mean(axis=0)
        negative_prototype = np.array(negatives).mean(axis=0)
        logger.info(f"Successfully calculated prototypes for dataset {self.task_id}")
        return positive_prototype, negative_prototype

    @property
    def get_features(self) -> Optional[FeatureArray]:
        """Get the cached features for the dataset.

        Returns:
            Optional[FeatureArray]: Cached features for the dataset if available, None otherwise.
        """
        if self._features is None:
            return None
        return np.array(self._features)

    @property
    def get_labels(self) -> np.ndarray:
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
            "avg_molecular_weight": np.mean([dp.molecular_weight for dp in self.data]),
            "avg_atoms": np.mean([dp.number_of_atoms for dp in self.data]),
            "avg_bonds": np.mean([dp.number_of_bonds for dp in self.data]),
        }
