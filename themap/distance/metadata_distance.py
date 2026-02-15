"""
Metadata distance computation for N×M task distance matrices.

This module provides efficient distance computation between task metadata,
where each task has a SINGLE feature vector (e.g., protein embeddings,
assay description embeddings).

The key distinction from dataset distance:
- Metadata distance: Each task has ONE vector (e.g., protein embedding)
- Dataset distance: Each task has MULTIPLE vectors (molecule features + labels)

This makes metadata distance much simpler and faster - just scipy.cdist.
"""

from typing import Dict, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Type aliases
MetadataDistanceMethod = Literal["euclidean", "cosine", "manhattan"]
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


class MetadataDistance:
    """Compute distances between task metadata vectors (N×M matrix).

    This class computes distances between single feature vectors per task,
    such as protein embeddings or assay description embeddings.

    Supports:
    - Euclidean: L2 distance
    - Cosine: Cosine distance (1 - cosine_similarity)
    - Manhattan: L1 distance

    Since each task has exactly one vector, this reduces to simple
    pairwise distance computation using scipy.cdist - highly efficient.

    Args:
        method: Distance computation method ('euclidean', 'cosine', 'manhattan')

    Examples:
        >>> distance_calc = MetadataDistance(method="cosine")
        >>> matrix = distance_calc.compute_matrix(
        ...     source_vectors=protein_embeddings_train,
        ...     target_vectors=protein_embeddings_test,
        ...     source_ids=["CHEMBL001", "CHEMBL002"],
        ...     target_ids=["CHEMBL100", "CHEMBL101"]
        ... )
    """

    SUPPORTED_METHODS = ["euclidean", "cosine", "manhattan"]

    def __init__(self, method: MetadataDistanceMethod = "euclidean"):
        """Initialize metadata distance calculator.

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
        source_vectors: NDArray[np.float32],
        target_vectors: NDArray[np.float32],
        source_ids: List[str],
        target_ids: List[str],
    ) -> DistanceMatrix:
        """Compute N×M distance matrix between source and target metadata.

        This uses scipy.cdist for efficient vectorized computation -
        all N×M distances are computed in a single optimized call.

        Args:
            source_vectors: Source feature matrix of shape (N, d)
            target_vectors: Target feature matrix of shape (M, d)
            source_ids: List of N source task identifiers
            target_ids: List of M target task identifiers

        Returns:
            Nested dict mapping ``target_id -> source_id -> distance``.
        """
        if len(source_vectors) != len(source_ids):
            raise ValueError(
                f"Source vectors ({len(source_vectors)}) and ids ({len(source_ids)}) must have same length"
            )
        if len(target_vectors) != len(target_ids):
            raise ValueError(
                f"Target vectors ({len(target_vectors)}) and ids ({len(target_ids)}) must have same length"
            )

        logger.info(f"Computing {len(target_ids)}×{len(source_ids)} {self.method} metadata distance")

        # Map method names to scipy metric names
        metric_map = {
            "euclidean": "euclidean",
            "cosine": "cosine",
            "manhattan": "cityblock",
        }
        metric = metric_map[self.method]

        # Single optimized cdist call computes all N×M distances
        distances = cdist(target_vectors, source_vectors, metric=metric)

        logger.info(f"Computed metadata distance matrix of shape {distances.shape}")

        return _matrix_to_dict(distances, source_ids, target_ids)

    def compute_from_lists(
        self,
        source_vectors: List[NDArray[np.float32]],
        target_vectors: List[NDArray[np.float32]],
        source_ids: List[str],
        target_ids: List[str],
    ) -> DistanceMatrix:
        """Compute distance matrix from lists of vectors.

        Convenience method that stacks lists into arrays.

        Args:
            source_vectors: List of N source feature vectors
            target_vectors: List of M target feature vectors
            source_ids: List of N source task identifiers
            target_ids: List of M target task identifiers

        Returns:
            Nested dict mapping ``target_id -> source_id -> distance``.
        """
        source_array = np.vstack(source_vectors).astype(np.float32)
        target_array = np.vstack(target_vectors).astype(np.float32)

        return self.compute_matrix(source_array, target_array, source_ids, target_ids)

    def compute_single_distance(
        self,
        source_vector: NDArray[np.float32],
        target_vector: NDArray[np.float32],
    ) -> float:
        """Compute distance between two vectors.

        Args:
            source_vector: Source feature vector (d,)
            target_vector: Target feature vector (d,)

        Returns:
            Distance value
        """
        result = self.compute_matrix(
            source_vector.reshape(1, -1), target_vector.reshape(1, -1), ["source"], ["target"]
        )
        return result["target"]["source"]


def compute_metadata_distance_matrix(
    source_vectors: NDArray[np.float32],
    target_vectors: NDArray[np.float32],
    source_ids: List[str],
    target_ids: List[str],
    method: MetadataDistanceMethod = "euclidean",
) -> DistanceMatrix:
    """Convenience function to compute metadata distance matrix.

    This is the main entry point for computing N×M distance matrices
    between task metadata (protein embeddings, descriptions, etc.).

    Args:
        source_vectors: Source feature matrix of shape (N, d)
        target_vectors: Target feature matrix of shape (M, d)
        source_ids: List of N source task identifiers
        target_ids: List of M target task identifiers
        method: Distance method ('euclidean', 'cosine', 'manhattan')

    Returns:
        Nested dict mapping ``target_id -> source_id -> distance``.

    Examples:
        >>> # Compute protein distance matrix
        >>> protein_distances = compute_metadata_distance_matrix(
        ...     train_protein_embeddings,  # (N, 1280) for ESM
        ...     test_protein_embeddings,   # (M, 1280)
        ...     train_task_ids,
        ...     test_task_ids,
        ...     method="cosine"
        ... )
    """
    calculator = MetadataDistance(method=method)
    return calculator.compute_matrix(source_vectors, target_vectors, source_ids, target_ids)


def combine_distance_matrices(
    matrices: Dict[str, DistanceMatrix],
    weights: Optional[Dict[str, float]] = None,
    combination: str = "weighted_average",
) -> DistanceMatrix:
    """Combine multiple distance matrices into one.

    Args:
        matrices: Dict mapping aspect name to distance matrix
        weights: Optional weights for each aspect (default: equal weights)
        combination: Combination strategy ('weighted_average', 'min', 'max', 'sum')

    Returns:
        Combined distance matrix

    Examples:
        >>> combined = combine_distance_matrices(
        ...     {"molecules": mol_distances, "protein": prot_distances},
        ...     weights={"molecules": 0.7, "protein": 0.3},
        ...     combination="weighted_average"
        ... )
    """
    if not matrices:
        return {}

    aspect_names = list(matrices.keys())
    first_matrix = matrices[aspect_names[0]]

    if not first_matrix:
        return {}

    # Get all target and source IDs from first matrix
    target_ids = list(first_matrix.keys())
    source_ids = list(first_matrix[target_ids[0]].keys())

    # Default to equal weights
    if weights is None:
        weights = {name: 1.0 / len(aspect_names) for name in aspect_names}

    # Normalize weights for weighted_average
    if combination == "weighted_average":
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

    # Combine matrices
    combined: DistanceMatrix = {}
    for tgt_id in target_ids:
        combined[tgt_id] = {}
        for src_id in source_ids:
            values = []
            weighted_values = []

            for aspect in aspect_names:
                if tgt_id in matrices[aspect] and src_id in matrices[aspect][tgt_id]:
                    val = matrices[aspect][tgt_id][src_id]
                    values.append(val)
                    weighted_values.append(val * weights.get(aspect, 1.0))

            if not values:
                combined[tgt_id][src_id] = float("inf")
            elif combination == "weighted_average":
                combined[tgt_id][src_id] = sum(weighted_values)
            elif combination == "sum":
                combined[tgt_id][src_id] = sum(values)
            elif combination == "min":
                combined[tgt_id][src_id] = min(values)
            elif combination == "max":
                combined[tgt_id][src_id] = max(values)
            else:
                combined[tgt_id][src_id] = sum(weighted_values)

    return combined
