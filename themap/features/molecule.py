"""
Molecule featurization module for THEMAP.

This module provides efficient molecular featurization using molfeat,
with support for batch processing and SMILES deduplication.
"""

from typing import Any, Dict, List, Union

import numpy as np
from numpy.typing import NDArray

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Available molecule featurizers
MOLECULE_FEATURIZERS = [
    # Fingerprints (fast)
    "ecfp",
    "fcfp",
    "maccs",
    "avalon",
    "topological",
    "atompair",
    # Descriptors (medium)
    "desc2D",
    "desc3D",
    "mordred",
    # Neural embeddings (slow, requires GPU)
    "ChemBERTa-77M-MLM",
    "ChemBERTa-77M-MTR",
    "MolT5",
    "Roberta-Zinc480M-102M",
    "gin_supervised_infomax",
    "gin_supervised_contextpred",
    "gin_supervised_edgepred",
    "gin_supervised_masking",
]

# Fingerprint featurizers (fast, binary vectors)
FINGERPRINT_FEATURIZERS = ["ecfp", "fcfp", "maccs", "avalon", "topological", "atompair"]

# Descriptor featurizers (medium speed, continuous values)
DESCRIPTOR_FEATURIZERS = ["desc2D", "desc3D", "mordred"]

# Neural embedding featurizers (slow, requires ML libraries)
NEURAL_FEATURIZERS = [
    "ChemBERTa-77M-MLM",
    "ChemBERTa-77M-MTR",
    "MolT5",
    "Roberta-Zinc480M-102M",
    "gin_supervised_infomax",
    "gin_supervised_contextpred",
    "gin_supervised_edgepred",
    "gin_supervised_masking",
]


class MoleculeFeaturizer:
    """Efficient molecule featurization using molfeat.

    Provides batch featurization with SMILES deduplication for efficiency.
    Supports fingerprints, descriptors, and neural embeddings.

    Attributes:
        featurizer_name: Name of the featurizer to use.
        n_jobs: Number of parallel workers for featurization.
        _transformer: Cached molfeat transformer instance.

    Examples:
        >>> featurizer = MoleculeFeaturizer("ecfp")
        >>> features = featurizer.featurize(["CCO", "CCCO", "CCCCO"])
        >>> print(features.shape)  # (3, 2048)

        >>> # With SMILES deduplication
        >>> smiles = ["CCO", "CCCO", "CCO", "CCCCO", "CCCO"]
        >>> features = featurizer.featurize_deduplicated(smiles)
        >>> print(features.shape)  # (5, 2048) - returns features for all input SMILES
    """

    def __init__(
        self,
        featurizer_name: str = "ecfp",
        n_jobs: int = 8,
        device: str = "auto",
    ):
        """Initialize the molecule featurizer.

        Args:
            featurizer_name: Name of the molfeat featurizer to use.
            n_jobs: Number of parallel workers for featurization.
            device: Device for neural featurizers ('auto', 'cpu', 'cuda').
        """
        self.featurizer_name = featurizer_name
        self.n_jobs = n_jobs
        self.device = device
        self._transformer = None

        if featurizer_name not in MOLECULE_FEATURIZERS:
            logger.warning(
                f"Featurizer '{featurizer_name}' not in known list. Available: {MOLECULE_FEATURIZERS}"
            )

    @property
    def transformer(self):
        """Get or create the molfeat transformer (lazy initialization)."""
        if self._transformer is None:
            self._transformer = self._create_transformer()
        return self._transformer

    def _create_transformer(self):
        """Create the appropriate molfeat transformer."""
        try:
            from molfeat.trans import MoleculeTransformer
            from molfeat.trans.pretrained import (
                GraphormerTransformer,
                PretrainedDGLTransformer,
            )
            from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
        except ImportError as e:
            raise ImportError(
                "molfeat is required for molecule featurization. Install with: pip install molfeat"
            ) from e

        name = self.featurizer_name

        # Fingerprints and basic descriptors
        if name in FINGERPRINT_FEATURIZERS + DESCRIPTOR_FEATURIZERS:
            return MoleculeTransformer(name, n_jobs=self.n_jobs)

        # Graphormer
        elif name == "pcqm4mv2_graphormer_base":
            return GraphormerTransformer(kind=name, dtype=float, n_jobs=self.n_jobs)

        # HuggingFace transformers (ChemBERTa, MolT5, etc.)
        elif name in [
            "ChemBERTa-77M-MLM",
            "ChemBERTa-77M-MTR",
            "Roberta-Zinc480M-102M",
            "MolT5",
        ]:
            return PretrainedHFTransformer(kind=name, notation="smiles", dtype=float, n_jobs=self.n_jobs)

        # DGL pretrained models (GIN)
        elif name in [
            "gin_supervised_infomax",
            "gin_supervised_contextpred",
            "gin_supervised_edgepred",
            "gin_supervised_masking",
        ]:
            return PretrainedDGLTransformer(kind=name, dtype=float, n_jobs=self.n_jobs)

        else:
            # Try as generic MoleculeTransformer
            logger.warning(f"Unknown featurizer '{name}', trying as MoleculeTransformer")
            return MoleculeTransformer(name, n_jobs=self.n_jobs)

    def featurize(
        self,
        smiles: Union[str, List[str]],
        ignore_errors: bool = True,
    ) -> NDArray[np.float32]:
        """Featurize one or more SMILES strings.

        Args:
            smiles: Single SMILES string or list of SMILES.
            ignore_errors: If True, return NaN for invalid SMILES.

        Returns:
            Feature array of shape (n_molecules, feature_dim).
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        n_molecules = len(smiles)
        logger.debug(f"Featurizing {n_molecules} molecules with {self.featurizer_name}")

        try:
            result = self.transformer(smiles, ignore_errors=ignore_errors)

            # Handle molfeat's return format with ignore_errors=True
            # It returns a tuple: (valid_features, valid_indices)
            if isinstance(result, tuple) and len(result) == 2:
                valid_features, valid_indices = result

                if len(valid_features) == 0:
                    raise ValueError("All molecules failed featurization")

                # Get feature dimension
                feature_dim = (
                    valid_features[0].shape[0]
                    if hasattr(valid_features[0], "shape")
                    else len(valid_features[0])
                )

                # Create full array with NaN for failed molecules
                full_features = np.full((n_molecules, feature_dim), np.nan, dtype=np.float32)

                # Fill in valid features
                for feat, idx in zip(valid_features, valid_indices):
                    full_features[idx] = np.array(feat, dtype=np.float32)

                return full_features

            # Standard numpy array return
            if result is None:
                raise ValueError("Featurization returned None")

            return np.array(result, dtype=np.float32)
        except Exception as e:
            logger.error(f"Featurization failed: {e}")
            raise

    def featurize_deduplicated(
        self,
        smiles: List[str],
        ignore_errors: bool = True,
    ) -> NDArray[np.float32]:
        """Featurize SMILES with deduplication for efficiency.

        Unique SMILES are featurized once, then results are mapped back
        to the original list. This is efficient when there are many
        duplicate SMILES across datasets.

        Args:
            smiles: List of SMILES strings (may contain duplicates).
            ignore_errors: If True, return NaN for invalid SMILES.

        Returns:
            Feature array of shape (n_molecules, feature_dim) in original order.
        """
        if len(smiles) == 0:
            raise ValueError("Cannot featurize empty SMILES list")

        # Get unique SMILES and their mapping
        unique_smiles = list(set(smiles))
        smiles_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        logger.info(f"Featurizing {len(unique_smiles)} unique SMILES (deduplicated from {len(smiles)})")

        # Featurize unique SMILES
        unique_features = self.featurize(unique_smiles, ignore_errors=ignore_errors)

        # Map back to original order
        indices = [smiles_to_idx[s] for s in smiles]
        features = unique_features[indices]

        return features

    def featurize_datasets(
        self,
        datasets: Dict[str, "MoleculeDataset"],
        deduplicate: bool = True,
    ) -> Dict[str, NDArray[np.float32]]:
        """Featurize multiple datasets efficiently.

        When deduplicate=True, collects all unique SMILES across all datasets,
        featurizes them once, then maps back to each dataset.

        Args:
            datasets: Dictionary mapping task IDs to MoleculeDataset instances.
            deduplicate: If True, deduplicate SMILES across all datasets.

        Returns:
            Dictionary mapping task IDs to feature arrays.
        """
        if not datasets:
            return {}

        if deduplicate:
            return self._featurize_datasets_deduplicated(datasets)
        else:
            return self._featurize_datasets_individual(datasets)

    def _featurize_datasets_deduplicated(
        self,
        datasets: Dict[str, "MoleculeDataset"],
    ) -> Dict[str, NDArray[np.float32]]:
        """Featurize datasets with global SMILES deduplication."""
        # Collect all unique SMILES
        all_smiles: Dict[str, int] = {}  # SMILES -> unique index
        dataset_smiles_indices: Dict[str, List[int]] = {}  # task_id -> list of indices

        for task_id, dataset in datasets.items():
            indices = []
            for smiles in dataset.smiles_list:
                if smiles not in all_smiles:
                    all_smiles[smiles] = len(all_smiles)
                indices.append(all_smiles[smiles])
            dataset_smiles_indices[task_id] = indices

        unique_smiles_list = list(all_smiles.keys())
        logger.info(f"Featurizing {len(unique_smiles_list)} unique SMILES across {len(datasets)} datasets")

        # Featurize all unique SMILES
        all_features = self.featurize(unique_smiles_list, ignore_errors=True)

        # Map back to datasets
        results = {}
        for task_id, indices in dataset_smiles_indices.items():
            results[task_id] = all_features[indices].astype(np.float32)
            # Also set features on the dataset object
            datasets[task_id].set_features(results[task_id], self.featurizer_name)

        return results

    def _featurize_datasets_individual(
        self,
        datasets: Dict[str, "MoleculeDataset"],
    ) -> Dict[str, NDArray[np.float32]]:
        """Featurize datasets individually without deduplication."""
        results = {}
        for task_id, dataset in datasets.items():
            logger.info(f"Featurizing dataset {task_id} ({len(dataset)} molecules)")
            features = self.featurize(dataset.smiles_list)
            results[task_id] = features
            dataset.set_features(features, self.featurizer_name)
        return results

    def get_feature_dim(self) -> int:
        """Get the feature dimension for this featurizer.

        Returns:
            Number of features produced by this featurizer.
        """
        # Featurize a simple molecule to get dimensions
        test_smiles = "C"  # Methane
        features = self.featurize([test_smiles])
        return features.shape[1]

    @property
    def is_fingerprint(self) -> bool:
        """Check if this is a fingerprint-based featurizer."""
        return self.featurizer_name in FINGERPRINT_FEATURIZERS

    @property
    def is_neural(self) -> bool:
        """Check if this is a neural embedding featurizer."""
        return self.featurizer_name in NEURAL_FEATURIZERS


def get_featurizer(
    featurizer_name: str,
    n_jobs: int = 8,
    **kwargs: Any,
) -> MoleculeFeaturizer:
    """Create a molecule featurizer by name.

    Args:
        featurizer_name: Name of the featurizer.
        n_jobs: Number of parallel workers.
        **kwargs: Additional arguments passed to MoleculeFeaturizer.

    Returns:
        MoleculeFeaturizer instance.
    """
    return MoleculeFeaturizer(featurizer_name, n_jobs=n_jobs, **kwargs)


def list_featurizers() -> List[str]:
    """List all available molecule featurizers.

    Returns:
        List of featurizer names.
    """
    return MOLECULE_FEATURIZERS.copy()
