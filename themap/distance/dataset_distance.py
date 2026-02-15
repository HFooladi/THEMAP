"""
Dataset distance computation for N×M task distance matrices.

This module provides efficient distance computation between molecule datasets,
supporting OTDD, Euclidean, and Cosine distance methods.

The key distinction from metadata distance:
- Dataset distance: Computes distance between SETS of molecules (using prototypes or OTDD)
- Metadata distance: Computes distance between SINGLE vectors (simple pairwise)
"""

import logging
import math
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Type aliases
DatasetDistanceMethod = Literal["otdd", "euclidean", "cosine"]
DistanceMatrix = Dict[str, Dict[str, float]]


def _matrix_to_dict(
    matrix: NDArray[np.float64],
    source_ids: List[str],
    target_ids: List[str],
) -> DistanceMatrix:
    """Convert distance matrix to nested dictionary format.

    Args:
        matrix: Distance matrix of shape (M, N) where M=targets, N=sources
        source_ids: List of source task identifiers
        target_ids: List of target task identifiers

    Returns:
        Nested dict mapping ``target_id -> source_id -> distance``.
    """
    return {
        target_ids[i]: {source_ids[j]: float(matrix[i, j]) for j in range(len(source_ids))}
        for i in range(len(target_ids))
    }


def _compute_prototype(
    features: NDArray[np.float32],
    labels: NDArray[np.int32],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute positive and negative prototypes from features.

    Args:
        features: Feature matrix of shape (n_samples, feature_dim)
        labels: Label array of shape (n_samples,)

    Returns:
        Tuple of (positive_prototype, negative_prototype)
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    # Handle cases where one class might be missing
    if pos_mask.any():
        pos_proto = features[pos_mask].mean(axis=0)
    else:
        pos_proto = np.zeros(features.shape[1], dtype=np.float32)

    if neg_mask.any():
        neg_proto = features[neg_mask].mean(axis=0)
    else:
        neg_proto = np.zeros(features.shape[1], dtype=np.float32)

    return pos_proto.astype(np.float32), neg_proto.astype(np.float32)


class DatasetDistance:
    """Compute distances between molecule datasets (N×M matrix).

    This class computes distances between sets of molecules, where each dataset
    contains multiple molecules with labels. Supports:
    - OTDD: Optimal Transport Dataset Distance (considers feature + label distributions)
    - Euclidean: L2 distance between positive/negative prototypes
    - Cosine: Cosine distance between positive/negative prototypes

    For prototype-based methods (Euclidean/Cosine), the distance is computed as:
    1. Compute prototype (mean feature) for each class in each dataset
    2. Concatenate [pos_prototype, neg_prototype] into a single vector
    3. Use scipy.cdist for efficient pairwise distance computation

    Args:
        method: Distance computation method ('otdd', 'euclidean', 'cosine')

    Examples:
        >>> distance_calc = DatasetDistance(method="euclidean")
        >>> matrix = distance_calc.compute_matrix(
        ...     source_features=[src_feat_1, src_feat_2],
        ...     source_labels=[src_labels_1, src_labels_2],
        ...     target_features=[tgt_feat_1, tgt_feat_2],
        ...     target_labels=[tgt_labels_1, tgt_labels_2],
        ...     source_ids=["CHEMBL001", "CHEMBL002"],
        ...     target_ids=["CHEMBL100", "CHEMBL101"],
        ...     n_jobs=8
        ... )
    """

    SUPPORTED_METHODS = ["otdd", "euclidean", "cosine"]

    def __init__(self, method: DatasetDistanceMethod = "euclidean"):
        """Initialize dataset distance calculator.

        Args:
            method: Distance method to use

        Raises:
            ValueError: If method is not supported
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method '{method}' not supported. Use one of: {self.SUPPORTED_METHODS}")
        self.method = method

    def compute_matrix(
        self,
        source_features: List[NDArray[np.float32]],
        source_labels: List[NDArray[np.int32]],
        target_features: List[NDArray[np.float32]],
        target_labels: List[NDArray[np.int32]],
        source_ids: List[str],
        target_ids: List[str],
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> DistanceMatrix:
        """Compute N×M distance matrix between source and target datasets.

        Args:
            source_features: List of N feature matrices, each (n_i, d)
            source_labels: List of N label arrays, each (n_i,)
            target_features: List of M feature matrices, each (m_j, d)
            target_labels: List of M label arrays, each (m_j,)
            source_ids: List of N source task identifiers
            target_ids: List of M target task identifiers
            n_jobs: Number of parallel jobs (for OTDD)
            **kwargs: Additional arguments for specific methods

        Returns:
            Nested dict mapping ``target_id -> source_id -> distance``.
        """
        if len(source_features) != len(source_labels) != len(source_ids):
            raise ValueError("Source features, labels, and ids must have same length")
        if len(target_features) != len(target_labels) != len(target_ids):
            raise ValueError("Target features, labels, and ids must have same length")

        if self.method == "otdd":
            return self._compute_otdd_matrix(
                source_features,
                source_labels,
                target_features,
                target_labels,
                source_ids,
                target_ids,
                n_jobs,
                **kwargs,
            )
        else:
            return self._compute_prototype_matrix(
                source_features, source_labels, target_features, target_labels, source_ids, target_ids, n_jobs
            )

    def _compute_prototype_matrix(
        self,
        source_features: List[NDArray[np.float32]],
        source_labels: List[NDArray[np.int32]],
        target_features: List[NDArray[np.float32]],
        target_labels: List[NDArray[np.int32]],
        source_ids: List[str],
        target_ids: List[str],
        n_jobs: int,
    ) -> DistanceMatrix:
        """Compute distance matrix using prototypes - highly efficient.

        Uses scipy.cdist for vectorized pairwise distance computation.
        """
        logger.info(f"Computing {len(target_ids)}×{len(source_ids)} {self.method} distance matrix")

        # Compute prototypes for all datasets
        source_prototypes = [
            _compute_prototype(feat, lab) for feat, lab in zip(source_features, source_labels)
        ]
        target_prototypes = [
            _compute_prototype(feat, lab) for feat, lab in zip(target_features, target_labels)
        ]

        # Concatenate prototypes into single vectors [pos, neg]
        source_vectors = np.vstack([np.concatenate([p[0], p[1]]) for p in source_prototypes])
        target_vectors = np.vstack([np.concatenate([p[0], p[1]]) for p in target_prototypes])

        # Compute pairwise distances using scipy.cdist
        metric = "euclidean" if self.method == "euclidean" else "cosine"
        distances = cdist(target_vectors, source_vectors, metric=metric)

        logger.info(f"Computed distance matrix of shape {distances.shape}")

        return _matrix_to_dict(distances, source_ids, target_ids)

    def _compute_otdd_matrix(
        self,
        source_features: List[NDArray[np.float32]],
        source_labels: List[NDArray[np.int32]],
        target_features: List[NDArray[np.float32]],
        target_labels: List[NDArray[np.int32]],
        source_ids: List[str],
        target_ids: List[str],
        n_jobs: int,
        maxsamples: int = 1000,
        **kwargs: Any,
    ) -> DistanceMatrix:
        """Compute distance matrix using OTDD.

        OTDD is more expensive but considers both feature and label distributions.
        """
        logger.info(
            f"Computing {len(target_ids)}×{len(source_ids)} OTDD distance matrix (maxsamples={maxsamples})"
        )

        try:
            from ..models.otdd.src.distance import DatasetDistance as OTDDDistance
        except ImportError:
            logger.error("OTDD not available. Install with: pip install pot geomloss")
            raise ImportError("OTDD requires additional dependencies. Install with: pip install pot geomloss")

        # Create data loaders for OTDD
        from ..data.molecule_dataset import MoleculeDataset
        from ..data.torch_dataset import MoleculeDataloader

        # Create temporary datasets for OTDD computation
        source_loaders = []
        for features, labels, task_id in zip(source_features, source_labels, source_ids):
            nan_count = np.isnan(features).sum()
            if nan_count > 0:
                nan_pct = nan_count / features.size * 100
                logger.warning(
                    f"Dataset {task_id}: {nan_count} NaN values ({nan_pct:.1f}%) in features. "
                    "Replacing NaN with 0 for OTDD computation."
                )
                features = np.nan_to_num(features, nan=0.0)

            dataset = MoleculeDataset(
                task_id=task_id,
                smiles_list=["C"] * len(labels),  # Placeholder SMILES
                labels=labels,
            )
            dataset.set_features(features, "precomputed")
            source_loaders.append(MoleculeDataloader(dataset))

        target_loaders = []
        for features, labels, task_id in zip(target_features, target_labels, target_ids):
            nan_count = np.isnan(features).sum()
            if nan_count > 0:
                nan_pct = nan_count / features.size * 100
                logger.warning(
                    f"Dataset {task_id}: {nan_count} NaN values ({nan_pct:.1f}%) in features. "
                    "Replacing NaN with 0 for OTDD computation."
                )
                features = np.nan_to_num(features, nan=0.0)

            dataset = MoleculeDataset(
                task_id=task_id,
                smiles_list=["C"] * len(labels),
                labels=labels,
            )
            dataset.set_features(features, "precomputed")
            target_loaders.append(MoleculeDataloader(dataset))

        # Compute N×M OTDD distances
        distances: DistanceMatrix = {}
        hopts = {
            "maxsamples": maxsamples,
            "device": "cpu",
            "verbose": 1 if logger.isEnabledFor(logging.DEBUG) else 0,
            **kwargs,
        }

        for i, (tgt_loader, tgt_id) in enumerate(zip(target_loaders, target_ids)):
            distances[tgt_id] = {}
            for j, (src_loader, src_id) in enumerate(zip(source_loaders, source_ids)):
                try:
                    dist_calc = OTDDDistance(src_loader, tgt_loader, **hopts)
                    dist_value = dist_calc.distance(maxsamples=maxsamples)
                    dist_float = float(dist_value)
                    if not math.isfinite(dist_float):
                        logger.warning(
                            f"OTDD returned non-finite value ({dist_float}) for {src_id}->{tgt_id}"
                        )
                    distances[tgt_id][src_id] = dist_float
                except Exception as e:
                    logger.error(f"OTDD computation failed for {src_id}->{tgt_id}: {type(e).__name__}: {e}")
                    distances[tgt_id][src_id] = float("inf")

            logger.debug(f"Completed OTDD row {i + 1}/{len(target_ids)}")

        # Summarize results
        total = sum(len(row) for row in distances.values())
        inf_count = sum(1 for row in distances.values() for v in row.values() if not math.isfinite(v))
        if inf_count > 0:
            logger.error(
                f"OTDD distance matrix: {inf_count}/{total} pairs failed (returned inf). "
                "Run with -v flag for detailed error messages."
            )

        return distances

    def compute_single_distance(
        self,
        source_features: NDArray[np.float32],
        source_labels: NDArray[np.int32],
        target_features: NDArray[np.float32],
        target_labels: NDArray[np.int32],
        **kwargs: Any,
    ) -> float:
        """Compute distance between two datasets.

        Args:
            source_features: Source feature matrix (n, d)
            source_labels: Source labels (n,)
            target_features: Target feature matrix (m, d)
            target_labels: Target labels (m,)

        Returns:
            Distance value
        """
        result = self.compute_matrix(
            [source_features],
            [source_labels],
            [target_features],
            [target_labels],
            ["source"],
            ["target"],
            **kwargs,
        )
        return result["target"]["source"]


def compute_dataset_distance_matrix(
    source_features: List[NDArray[np.float32]],
    source_labels: List[NDArray[np.int32]],
    target_features: List[NDArray[np.float32]],
    target_labels: List[NDArray[np.int32]],
    source_ids: List[str],
    target_ids: List[str],
    method: DatasetDistanceMethod = "euclidean",
    n_jobs: int = 1,
    **kwargs: Any,
) -> DistanceMatrix:
    """Convenience function to compute dataset distance matrix.

    This is the main entry point for computing N×M distance matrices
    between molecule datasets.

    Args:
        source_features: List of N source feature matrices
        source_labels: List of N source label arrays
        target_features: List of M target feature matrices
        target_labels: List of M target label arrays
        source_ids: List of N source task identifiers
        target_ids: List of M target task identifiers
        method: Distance method ('otdd', 'euclidean', 'cosine')
        n_jobs: Number of parallel jobs
        **kwargs: Additional method-specific arguments

    Returns:
        Nested dict mapping ``target_id -> source_id -> distance``.

    Examples:
        >>> distances = compute_dataset_distance_matrix(
        ...     source_features, source_labels,
        ...     target_features, target_labels,
        ...     source_ids, target_ids,
        ...     method="euclidean"
        ... )
    """
    calculator = DatasetDistance(method=method)
    return calculator.compute_matrix(
        source_features,
        source_labels,
        target_features,
        target_labels,
        source_ids,
        target_ids,
        n_jobs=n_jobs,
        **kwargs,
    )
