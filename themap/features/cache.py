"""
Feature caching module for THEMAP.

This module provides disk-based caching for computed features,
allowing reuse across pipeline runs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..utils.logging import get_logger

logger = get_logger(__name__)


class FeatureCache:
    """Disk-based feature caching for molecules and proteins.

    Caches computed features to NPZ/NPY files for efficient reuse across runs.
    Supports both molecule features (per-dataset with labels) and protein
    features (single vector per task).

    Attributes:
        cache_dir: Root directory for cached features.
        molecule_dir: Directory for molecule features.
        protein_dir: Directory for protein features.

    Directory Structure:
        cache_dir/
        ├── molecule/
        │   └── {featurizer}/
        │       └── {task_id}.npz  # features + labels
        └── protein/
            └── {featurizer}/
                └── {task_id}.npy  # single vector

    Examples:
        >>> cache = FeatureCache("./feature_cache")
        >>> # Save molecule features
        >>> cache.save_molecule_features("CHEMBL123", "ecfp", features, labels)
        >>> # Load if exists
        >>> features, labels = cache.load_molecule_features("CHEMBL123", "ecfp")
        >>> if features is None:
        ...     # Compute and save
        ...     cache.save_molecule_features("CHEMBL123", "ecfp", features, labels)
    """

    def __init__(self, cache_dir: Union[str, Path]):
        """Initialize the feature cache.

        Args:
            cache_dir: Root directory for cached features.
        """
        self.cache_dir = Path(cache_dir)
        self.molecule_dir = self.cache_dir / "molecule"
        self.protein_dir = self.cache_dir / "protein"

        # Create directories
        self.molecule_dir.mkdir(parents=True, exist_ok=True)
        self.protein_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Feature cache initialized at {self.cache_dir}")

    def _get_molecule_path(self, task_id: str, featurizer: str) -> Path:
        """Get path for molecule features."""
        featurizer_dir = self.molecule_dir / featurizer
        featurizer_dir.mkdir(parents=True, exist_ok=True)
        return featurizer_dir / f"{task_id}.npz"

    def _get_protein_path(self, task_id: str, featurizer: str) -> Path:
        """Get path for protein features."""
        featurizer_dir = self.protein_dir / featurizer
        featurizer_dir.mkdir(parents=True, exist_ok=True)
        return featurizer_dir / f"{task_id}.npy"

    def has_molecule_features(self, task_id: str, featurizer: str) -> bool:
        """Check if molecule features are cached.

        Args:
            task_id: Task ID to check.
            featurizer: Name of the featurizer.

        Returns:
            True if features are cached.
        """
        path = self._get_molecule_path(task_id, featurizer)
        return path.exists()

    def has_protein_features(self, task_id: str, featurizer: str) -> bool:
        """Check if protein features are cached.

        Args:
            task_id: Task ID to check.
            featurizer: Name of the featurizer.

        Returns:
            True if features are cached.
        """
        path = self._get_protein_path(task_id, featurizer)
        return path.exists()

    def save_molecule_features(
        self,
        task_id: str,
        featurizer: str,
        features: NDArray[np.float32],
        labels: NDArray[np.int32],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save molecule features to cache.

        Args:
            task_id: Task ID for the dataset.
            featurizer: Name of the featurizer used.
            features: Feature matrix of shape (n_molecules, feature_dim).
            labels: Binary labels of shape (n_molecules,).
            metadata: Optional metadata dictionary.

        Returns:
            Path to the saved file.
        """
        path = self._get_molecule_path(task_id, featurizer)

        save_dict = {
            "features": features.astype(np.float32),
            "labels": labels.astype(np.int32),
        }

        if metadata:
            # Save metadata as JSON string in a special key
            save_dict["metadata"] = np.array([json.dumps(metadata)])

        np.savez_compressed(path, **save_dict)
        logger.debug(f"Saved molecule features for {task_id} to {path}")
        return path

    def load_molecule_features(
        self,
        task_id: str,
        featurizer: str,
    ) -> Tuple[Optional[NDArray[np.float32]], Optional[NDArray[np.int32]]]:
        """Load molecule features from cache.

        Args:
            task_id: Task ID for the dataset.
            featurizer: Name of the featurizer.

        Returns:
            Tuple of (features, labels), or (None, None) if not cached.
        """
        path = self._get_molecule_path(task_id, featurizer)

        if not path.exists():
            return None, None

        try:
            data = np.load(path)
            features = data["features"].astype(np.float32)
            labels = data["labels"].astype(np.int32)
            logger.debug(f"Loaded molecule features for {task_id} from {path}")
            return features, labels
        except Exception as e:
            logger.warning(f"Failed to load cached features for {task_id}: {e}")
            return None, None

    def save_protein_features(
        self,
        task_id: str,
        featurizer: str,
        features: NDArray[np.float32],
    ) -> Path:
        """Save protein features to cache.

        Args:
            task_id: Task ID for the protein.
            featurizer: Name of the featurizer used.
            features: Feature vector of shape (feature_dim,).

        Returns:
            Path to the saved file.
        """
        path = self._get_protein_path(task_id, featurizer)
        np.save(path, features.astype(np.float32))
        logger.debug(f"Saved protein features for {task_id} to {path}")
        return path

    def load_protein_features(
        self,
        task_id: str,
        featurizer: str,
    ) -> Optional[NDArray[np.float32]]:
        """Load protein features from cache.

        Args:
            task_id: Task ID for the protein.
            featurizer: Name of the featurizer.

        Returns:
            Feature vector, or None if not cached.
        """
        path = self._get_protein_path(task_id, featurizer)

        if not path.exists():
            return None

        try:
            features = np.load(path).astype(np.float32)
            logger.debug(f"Loaded protein features for {task_id} from {path}")
            return features
        except Exception as e:
            logger.warning(f"Failed to load cached protein features for {task_id}: {e}")
            return None

    def save_all_molecule_features(
        self,
        features_dict: Dict[str, NDArray[np.float32]],
        labels_dict: Dict[str, NDArray[np.int32]],
        featurizer: str,
    ) -> List[Path]:
        """Save molecule features for multiple datasets.

        Args:
            features_dict: Dictionary mapping task IDs to feature matrices.
            labels_dict: Dictionary mapping task IDs to label arrays.
            featurizer: Name of the featurizer used.

        Returns:
            List of paths to saved files.
        """
        paths = []
        for task_id in features_dict:
            if task_id in labels_dict:
                path = self.save_molecule_features(
                    task_id, featurizer, features_dict[task_id], labels_dict[task_id]
                )
                paths.append(path)
        return paths

    def save_all_protein_features(
        self,
        features_dict: Dict[str, NDArray[np.float32]],
        featurizer: str,
    ) -> List[Path]:
        """Save protein features for multiple tasks.

        Args:
            features_dict: Dictionary mapping task IDs to feature vectors.
            featurizer: Name of the featurizer used.

        Returns:
            List of paths to saved files.
        """
        paths = []
        for task_id, features in features_dict.items():
            path = self.save_protein_features(task_id, featurizer, features)
            paths.append(path)
        return paths

    def load_all_molecule_features(
        self,
        task_ids: List[str],
        featurizer: str,
    ) -> Tuple[Dict[str, NDArray[np.float32]], Dict[str, NDArray[np.int32]], List[str]]:
        """Load molecule features for multiple datasets.

        Args:
            task_ids: List of task IDs to load.
            featurizer: Name of the featurizer.

        Returns:
            Tuple of (features_dict, labels_dict, missing_ids).
            missing_ids contains task IDs that were not found in cache.
        """
        features_dict = {}
        labels_dict = {}
        missing_ids = []

        for task_id in task_ids:
            features, labels = self.load_molecule_features(task_id, featurizer)
            if features is not None and labels is not None:
                features_dict[task_id] = features
                labels_dict[task_id] = labels
            else:
                missing_ids.append(task_id)

        return features_dict, labels_dict, missing_ids

    def load_all_protein_features(
        self,
        task_ids: List[str],
        featurizer: str,
    ) -> Tuple[Dict[str, NDArray[np.float32]], List[str]]:
        """Load protein features for multiple tasks.

        Args:
            task_ids: List of task IDs to load.
            featurizer: Name of the featurizer.

        Returns:
            Tuple of (features_dict, missing_ids).
            missing_ids contains task IDs that were not found in cache.
        """
        features_dict = {}
        missing_ids = []

        for task_id in task_ids:
            features = self.load_protein_features(task_id, featurizer)
            if features is not None:
                features_dict[task_id] = features
            else:
                missing_ids.append(task_id)

        return features_dict, missing_ids

    def clear(self, featurizer: Optional[str] = None) -> int:
        """Clear cached features.

        Args:
            featurizer: If specified, only clear features for this featurizer.
                       If None, clear all cached features.

        Returns:
            Number of files deleted.
        """
        count = 0

        if featurizer:
            # Clear specific featurizer
            for base_dir in [self.molecule_dir, self.protein_dir]:
                feat_dir = base_dir / featurizer
                if feat_dir.exists():
                    for f in feat_dir.glob("*"):
                        f.unlink()
                        count += 1
        else:
            # Clear all
            for base_dir in [self.molecule_dir, self.protein_dir]:
                for feat_dir in base_dir.iterdir():
                    if feat_dir.is_dir():
                        for f in feat_dir.glob("*"):
                            f.unlink()
                            count += 1

        logger.info(f"Cleared {count} cached feature files")
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about cached features.

        Returns:
            Dictionary with cache statistics.
        """
        stats = {
            "cache_dir": str(self.cache_dir),
            "molecule_featurizers": {},
            "protein_featurizers": {},
            "total_size_mb": 0,
        }

        total_size = 0

        # Molecule features
        if self.molecule_dir.exists():
            for feat_dir in self.molecule_dir.iterdir():
                if feat_dir.is_dir():
                    files = list(feat_dir.glob("*.npz"))
                    size = sum(f.stat().st_size for f in files)
                    stats["molecule_featurizers"][feat_dir.name] = {
                        "count": len(files),
                        "size_mb": size / (1024 * 1024),
                    }
                    total_size += size

        # Protein features
        if self.protein_dir.exists():
            for feat_dir in self.protein_dir.iterdir():
                if feat_dir.is_dir():
                    files = list(feat_dir.glob("*.npy"))
                    size = sum(f.stat().st_size for f in files)
                    stats["protein_featurizers"][feat_dir.name] = {
                        "count": len(files),
                        "size_mb": size / (1024 * 1024),
                    }
                    total_size += size

        stats["total_size_mb"] = total_size / (1024 * 1024)
        return stats


def create_cache(cache_dir: Union[str, Path] = "./feature_cache") -> FeatureCache:
    """Create a feature cache.

    Args:
        cache_dir: Directory for the cache.

    Returns:
        FeatureCache instance.
    """
    return FeatureCache(cache_dir)
