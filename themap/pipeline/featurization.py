"""
Explicit featurization pipeline with disk caching.

This module provides the core pipeline for computing and caching molecular
and metadata features for efficient N×M distance matrix computation.

Usage:
    # Initialize pipeline with cache directory
    pipeline = FeaturizationPipeline(
        cache_dir="./feature_cache",
        molecule_featurizer="ecfp",
        protein_featurizer="esm2_t33_650M_UR50D"
    )

    # Step 1: Featurize all tasks (saves to disk)
    pipeline.featurize_all_datasets(datasets, n_jobs=8)

    # Step 2: Load features for distance computation
    source_features, source_labels, source_ids = pipeline.load_dataset_features(
        datasets=[...],
        dataset_names=[...]
    )
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..data.molecule_dataset import MoleculeDataset
from ..utils.featurizer_utils import get_featurizer
from ..utils.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


@dataclass
class FeatureStore:
    """Disk-based feature storage for datasets.

    File structure:
        cache_dir/
            molecules/
                {featurizer_name}/
                    {task_id}.npz  # Contains 'features' and 'labels'
            protein/
                {featurizer_name}/
                    {task_id}.npy  # Single vector per task
            metadata/
                {metadata_type}/
                    {featurizer_name}/
                        {task_id}.npy  # Single vector per task

    Attributes:
        cache_dir: Root directory for feature storage
    """

    cache_dir: Path

    def __post_init__(self) -> None:
        """Ensure cache directory exists."""
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_molecule_path(self, task_id: str, featurizer_name: str) -> Path:
        """Get path for molecule features."""
        return self.cache_dir / "molecules" / featurizer_name / f"{task_id}.npz"

    def _get_metadata_path(self, task_id: str, metadata_type: str, featurizer_name: str) -> Path:
        """Get path for metadata features."""
        return self.cache_dir / metadata_type / featurizer_name / f"{task_id}.npy"

    def save_molecule_features(
        self,
        task_id: str,
        features: NDArray[np.float32],
        labels: NDArray[np.int32],
        featurizer_name: str,
    ) -> None:
        """Save molecule dataset features.

        Args:
            task_id: Task identifier
            features: Feature matrix of shape (n_molecules, feature_dim)
            labels: Label array of shape (n_molecules,)
            featurizer_name: Name of the featurizer used
        """
        path = self._get_molecule_path(task_id, featurizer_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=features.astype(np.float32), labels=labels.astype(np.int32))
        logger.debug(f"Saved molecule features for {task_id} at {path}")

    def load_molecule_features(self, task_id: str, featurizer_name: str) -> Optional[Dict[str, NDArray]]:
        """Load molecule dataset features.

        Args:
            task_id: Task identifier
            featurizer_name: Name of the featurizer used

        Returns:
            Dictionary with 'features' and 'labels' arrays, or None if not found
        """
        path = self._get_molecule_path(task_id, featurizer_name)
        if path.exists():
            data = np.load(path)
            return {"features": data["features"], "labels": data["labels"]}
        return None

    def save_metadata_features(
        self,
        task_id: str,
        features: NDArray[np.float32],
        metadata_type: str,
        featurizer_name: str,
    ) -> None:
        """Save metadata features (single vector per task).

        Args:
            task_id: Task identifier
            features: Feature vector of shape (feature_dim,)
            metadata_type: Type of metadata (e.g., 'protein', 'description')
            featurizer_name: Name of the featurizer used
        """
        path = self._get_metadata_path(task_id, metadata_type, featurizer_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features.astype(np.float32))
        logger.debug(f"Saved {metadata_type} features for {task_id} at {path}")

    def load_metadata_features(
        self, task_id: str, metadata_type: str, featurizer_name: str
    ) -> Optional[NDArray[np.float32]]:
        """Load metadata features.

        Args:
            task_id: Task identifier
            metadata_type: Type of metadata
            featurizer_name: Name of the featurizer used

        Returns:
            Feature vector or None if not found
        """
        path = self._get_metadata_path(task_id, metadata_type, featurizer_name)
        if path.exists():
            return np.load(path)
        return None

    def has_molecule_features(self, task_id: str, featurizer_name: str) -> bool:
        """Check if molecule features exist for a task."""
        return self._get_molecule_path(task_id, featurizer_name).exists()

    def has_metadata_features(self, task_id: str, metadata_type: str, featurizer_name: str) -> bool:
        """Check if metadata features exist for a task."""
        return self._get_metadata_path(task_id, metadata_type, featurizer_name).exists()

    def get_cached_task_ids(self, featurizer_name: str, feature_type: str = "molecules") -> List[str]:
        """Get list of task IDs that have cached features.

        Args:
            featurizer_name: Name of the featurizer
            feature_type: 'molecules' or metadata type name

        Returns:
            List of task IDs with cached features
        """
        if feature_type == "molecules":
            path = self.cache_dir / "molecules" / featurizer_name
        else:
            path = self.cache_dir / feature_type / featurizer_name

        if not path.exists():
            return []

        task_ids = []
        for f in path.glob("*.np*"):
            task_id = f.stem
            task_ids.append(task_id)

        return task_ids

    def clear_cache(self, featurizer_name: Optional[str] = None) -> None:
        """Clear cached features.

        Args:
            featurizer_name: If provided, only clear features for this featurizer.
                           If None, clear all cached features.
        """
        import shutil

        if featurizer_name:
            for subdir in ["molecules", "protein", "metadata"]:
                path = self.cache_dir / subdir / featurizer_name
                if path.exists():
                    shutil.rmtree(path)
                    logger.info(f"Cleared cache for {featurizer_name} in {subdir}")
        else:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    shutil.rmtree(subdir)
            logger.info(f"Cleared all feature cache at {self.cache_dir}")


class FeaturizationPipeline:
    """Pipeline for batch featurization of molecule datasets.

    This pipeline provides efficient batch featurization with:
    - Global SMILES deduplication across all datasets
    - Disk-based caching for reuse across sessions
    - Parallel computation support

    Attributes:
        store: FeatureStore for disk caching
        molecule_featurizer: Name of molecular featurizer to use
        protein_featurizer: Name of protein featurizer to use

    Example:
        >>> pipeline = FeaturizationPipeline(
        ...     cache_dir="./feature_cache",
        ...     molecule_featurizer="ecfp"
        ... )
        >>> # Featurize all datasets
        >>> pipeline.featurize_all_datasets(datasets)
        >>> # Load features for distance computation
        >>> features, labels, ids = pipeline.load_dataset_features(datasets, names)
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        molecule_featurizer: str = "ecfp",
        protein_featurizer: str = "esm2_t33_650M_UR50D",
    ):
        """Initialize the featurization pipeline.

        Args:
            cache_dir: Directory for storing cached features
            molecule_featurizer: Name of the molecular featurizer
            protein_featurizer: Name of the protein featurizer
        """
        self.store = FeatureStore(Path(cache_dir))
        self.molecule_featurizer = molecule_featurizer
        self.protein_featurizer = protein_featurizer

        logger.info(
            f"FeaturizationPipeline initialized with cache at {cache_dir}, "
            f"molecule_featurizer={molecule_featurizer}"
        )

    def featurize_all_datasets(
        self,
        datasets: List[MoleculeDataset],
        n_jobs: int = 8,
        batch_size: int = 1000,
        force_recompute: bool = False,
    ) -> Dict[str, bool]:
        """Featurize all datasets and save features to disk.

        This method:
        1. Collects all unique SMILES across datasets
        2. Batch computes features for unique SMILES
        3. Distributes and saves features per dataset

        Args:
            datasets: List of MoleculeDataset objects to featurize
            n_jobs: Number of parallel jobs for featurization
            batch_size: Batch size for featurizer
            force_recompute: If True, recompute even if cached

        Returns:
            Dictionary mapping task_id to success status
        """
        logger.info(f"Featurizing {len(datasets)} datasets with {self.molecule_featurizer}")

        # Check which datasets need featurization
        datasets_to_process = []
        for dataset in datasets:
            if force_recompute or not self.store.has_molecule_features(
                dataset.task_id, self.molecule_featurizer
            ):
                datasets_to_process.append(dataset)
            else:
                logger.debug(f"Skipping {dataset.task_id} - already cached")

        if not datasets_to_process:
            logger.info("All datasets already cached, skipping featurization")
            return {d.task_id: True for d in datasets}

        logger.info(
            f"Processing {len(datasets_to_process)} datasets (skipped {len(datasets) - len(datasets_to_process)} cached)"
        )

        # Collect all unique SMILES
        all_smiles = set()
        for dataset in datasets_to_process:
            all_smiles.update(dataset.smiles_list)

        unique_smiles = list(all_smiles)
        logger.info(f"Found {len(unique_smiles)} unique SMILES across all datasets")

        # Batch compute features for unique SMILES
        smiles_to_features = self._batch_featurize_smiles(unique_smiles, n_jobs, batch_size)

        # Distribute and save features for each dataset
        results = {}
        for dataset in datasets_to_process:
            try:
                features = np.array(
                    [smiles_to_features[s] for s in dataset.smiles_list],
                    dtype=np.float32,
                )
                self.store.save_molecule_features(
                    dataset.task_id,
                    features,
                    dataset.labels,
                    self.molecule_featurizer,
                )
                # Also set features on the dataset object
                dataset.set_features(features, self.molecule_featurizer)
                results[dataset.task_id] = True
            except Exception as e:
                logger.error(f"Failed to process dataset {dataset.task_id}: {e}")
                results[dataset.task_id] = False

        # Mark skipped datasets as successful
        for dataset in datasets:
            if dataset.task_id not in results:
                results[dataset.task_id] = True

        success_count = sum(results.values())
        logger.info(f"Successfully featurized {success_count}/{len(datasets)} datasets")

        return results

    def _batch_featurize_smiles(
        self,
        smiles_list: List[str],
        n_jobs: int,
        batch_size: int,
    ) -> Dict[str, NDArray[np.float32]]:
        """Batch featurize SMILES strings using molfeat.

        Args:
            smiles_list: List of SMILES strings to featurize
            n_jobs: Number of parallel jobs
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping SMILES to feature vectors
        """
        logger.info(f"Batch featurizing {len(smiles_list)} unique SMILES")

        featurizer = get_featurizer(self.molecule_featurizer, n_jobs=n_jobs)

        # Process in batches
        all_features = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(smiles_list) + batch_size - 1) // batch_size

            logger.debug(f"Processing batch {batch_num}/{total_batches}")

            try:
                processed, _ = featurizer.preprocess(batch)
                if hasattr(featurizer, "transform"):
                    batch_features = featurizer.transform(processed)
                else:
                    batch_features = featurizer(processed)

                all_features.append(batch_features)
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                raise

        # Combine all batches
        features = np.vstack(all_features)

        # Create SMILES -> features mapping
        return dict(zip(smiles_list, features))

    def load_dataset_features(
        self,
        datasets: List[MoleculeDataset],
        dataset_names: Optional[List[str]] = None,
    ) -> Tuple[List[NDArray[np.float32]], List[NDArray[np.int32]], List[str]]:
        """Load features for a list of datasets.

        Args:
            datasets: List of MoleculeDataset objects
            dataset_names: Optional list of names (uses task_id if not provided)

        Returns:
            Tuple of (features_list, labels_list, valid_names)
            - features_list: List of feature matrices, one per dataset
            - labels_list: List of label arrays, one per dataset
            - valid_names: List of dataset names that were successfully loaded
        """
        features_list = []
        labels_list = []
        valid_names = []

        names = dataset_names or [d.task_id for d in datasets]

        for dataset, name in zip(datasets, names):
            # Try to get from dataset object first
            if dataset.has_features() and dataset.featurizer_name == self.molecule_featurizer:
                features_list.append(dataset.features)
                labels_list.append(dataset.labels)
                valid_names.append(name)
                continue

            # Try to load from disk
            cached = self.store.load_molecule_features(dataset.task_id, self.molecule_featurizer)
            if cached is not None:
                features_list.append(cached["features"])
                labels_list.append(cached["labels"])
                valid_names.append(name)
                # Also set on dataset object
                dataset.set_features(cached["features"], self.molecule_featurizer)
            else:
                logger.warning(f"No features found for {dataset.task_id}. Run featurize_all_datasets first.")

        logger.info(f"Loaded features for {len(valid_names)}/{len(datasets)} datasets")
        return features_list, labels_list, valid_names

    def load_features_for_distance(
        self,
        source_datasets: List[MoleculeDataset],
        target_datasets: List[MoleculeDataset],
        source_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
    ) -> Tuple[
        List[NDArray[np.float32]],
        List[NDArray[np.int32]],
        List[str],
        List[NDArray[np.float32]],
        List[NDArray[np.int32]],
        List[str],
    ]:
        """Load features organized for N×M distance computation.

        Args:
            source_datasets: List of source MoleculeDataset objects (N)
            target_datasets: List of target MoleculeDataset objects (M)
            source_names: Optional names for source datasets
            target_names: Optional names for target datasets

        Returns:
            Tuple containing:
            - source_features: List of N feature matrices
            - source_labels: List of N label arrays
            - source_ids: List of N dataset names
            - target_features: List of M feature matrices
            - target_labels: List of M label arrays
            - target_ids: List of M dataset names
        """
        source_features, source_labels, source_ids = self.load_dataset_features(source_datasets, source_names)
        target_features, target_labels, target_ids = self.load_dataset_features(target_datasets, target_names)

        return (
            source_features,
            source_labels,
            source_ids,
            target_features,
            target_labels,
            target_ids,
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached features.

        Returns:
            Dictionary with cache statistics
        """
        cached_molecules = self.store.get_cached_task_ids(self.molecule_featurizer, "molecules")

        stats = {
            "cache_dir": str(self.store.cache_dir),
            "molecule_featurizer": self.molecule_featurizer,
            "cached_molecule_datasets": len(cached_molecules),
            "cached_task_ids": cached_molecules[:10],  # First 10 for preview
        }

        return stats
