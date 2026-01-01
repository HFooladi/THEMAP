"""
Protein featurization module for THEMAP.

This module provides efficient protein featurization using ESM2/ESM3 models,
with support for batching and model caching.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Available protein featurizers
PROTEIN_FEATURIZERS = [
    # ESM2 models
    "esm2_t12_35M_UR50D",  # 35M parameters, 480 dim
    "esm2_t33_650M_UR50D",  # 650M parameters, 1280 dim
    "esm2_t36_3B_UR50D",  # 3B parameters, 2560 dim (large)
    # ESM3 models
    "esm3_sm_open_v1",
    "esm3_open_small",
]

# ESM2 models (well-tested, recommended)
ESM2_MODELS = [
    "esm2_t12_35M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
]

# ESM3 models (newer, experimental)
ESM3_MODELS = [
    "esm3_sm_open_v1",
    "esm3_open_small",
]

# Default embedding layers for ESM2 models
ESM2_DEFAULT_LAYERS = {
    "esm2_t12_35M_UR50D": 12,
    "esm2_t33_650M_UR50D": 33,
    "esm2_t36_3B_UR50D": 36,
}


class ProteinFeaturizer:
    """Efficient protein featurization using ESM models.

    Provides batch featurization of protein sequences using ESM2 or ESM3 models.
    Models are cached globally to avoid reloading.

    Attributes:
        featurizer_name: Name of the ESM model to use.
        layer: Which transformer layer to extract embeddings from.
        device: Device for computation ('auto', 'cpu', 'cuda').

    Examples:
        >>> featurizer = ProteinFeaturizer("esm2_t33_650M_UR50D")
        >>> sequences = {"P1": "MKTVRQ...", "P2": "MENLNM..."}
        >>> features = featurizer.featurize(sequences)
        >>> print(features.shape)  # (2, 1280)
    """

    def __init__(
        self,
        featurizer_name: str = "esm2_t33_650M_UR50D",
        layer: Optional[int] = None,
        device: str = "auto",
    ):
        """Initialize the protein featurizer.

        Args:
            featurizer_name: Name of the ESM model to use.
            layer: Which transformer layer to extract embeddings from.
                   If None, uses the default for the model.
            device: Device for computation ('auto', 'cpu', 'cuda').
        """
        self.featurizer_name = featurizer_name
        self.layer = layer
        self.device = device

        if featurizer_name not in PROTEIN_FEATURIZERS:
            logger.warning(
                f"Featurizer '{featurizer_name}' not in known list. Available: {PROTEIN_FEATURIZERS}"
            )

        # Set default layer if not specified
        if self.layer is None and featurizer_name in ESM2_DEFAULT_LAYERS:
            self.layer = ESM2_DEFAULT_LAYERS[featurizer_name]

    @property
    def is_esm2(self) -> bool:
        """Check if this is an ESM2 model."""
        return self.featurizer_name in ESM2_MODELS

    @property
    def is_esm3(self) -> bool:
        """Check if this is an ESM3 model."""
        return self.featurizer_name in ESM3_MODELS

    def featurize(
        self,
        sequences: Union[Dict[str, str], List[str]],
    ) -> NDArray[np.float32]:
        """Featurize protein sequences.

        Args:
            sequences: Either a dictionary mapping protein IDs to sequences,
                      or a list of sequences.

        Returns:
            Feature array of shape (n_proteins, embedding_dim).
        """
        # Convert list to dictionary if needed
        if isinstance(sequences, list):
            sequences = {f"protein_{i}": seq for i, seq in enumerate(sequences)}

        if not sequences:
            raise ValueError("Cannot featurize empty sequence dictionary")

        logger.info(f"Featurizing {len(sequences)} protein sequences with {self.featurizer_name}")

        try:
            from ..utils.protein_utils import get_protein_features

            features = get_protein_features(sequences, featurizer=self.featurizer_name, layer=self.layer)
            return features.astype(np.float32)
        except Exception as e:
            logger.error(f"Protein featurization failed: {e}")
            raise

    def featurize_from_fasta(
        self,
        fasta_path: Union[str, Path],
    ) -> Dict[str, NDArray[np.float32]]:
        """Featurize proteins from a FASTA file.

        Args:
            fasta_path: Path to the FASTA file.

        Returns:
            Dictionary mapping protein IDs to feature vectors.
        """
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        sequences = self._read_fasta(fasta_path)
        if not sequences:
            raise ValueError(f"No sequences found in FASTA file: {fasta_path}")

        features = self.featurize(sequences)

        # Map features back to protein IDs
        protein_ids = list(sequences.keys())
        return {pid: features[i] for i, pid in enumerate(protein_ids)}

    def featurize_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.fasta",
    ) -> Dict[str, NDArray[np.float32]]:
        """Featurize all proteins from FASTA files in a directory.

        Each FASTA file is expected to contain one protein sequence.
        The filename (without extension) is used as the task/protein ID.

        Args:
            directory: Path to directory containing FASTA files.
            pattern: Glob pattern for finding FASTA files.

        Returns:
            Dictionary mapping task IDs to feature vectors.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        fasta_files = list(directory.glob(pattern))
        if not fasta_files:
            logger.warning(f"No FASTA files found in {directory}")
            return {}

        logger.info(f"Found {len(fasta_files)} FASTA files in {directory}")

        # Collect all sequences
        all_sequences: Dict[str, str] = {}
        for fasta_path in fasta_files:
            task_id = fasta_path.stem
            sequences = self._read_fasta(fasta_path)
            if sequences:
                # Use first sequence in file
                first_seq = list(sequences.values())[0]
                all_sequences[task_id] = first_seq
            else:
                logger.warning(f"No sequence found in {fasta_path}")

        if not all_sequences:
            return {}

        # Featurize all sequences in one batch
        features = self.featurize(all_sequences)

        # Map back to task IDs
        task_ids = list(all_sequences.keys())
        return {tid: features[i] for i, tid in enumerate(task_ids)}

    def _read_fasta(self, path: Path) -> Dict[str, str]:
        """Read sequences from a FASTA file.

        Args:
            path: Path to the FASTA file.

        Returns:
            Dictionary mapping sequence IDs to sequences.
        """
        try:
            from Bio import SeqIO

            sequences = {}
            for record in SeqIO.parse(str(path), "fasta"):
                sequences[record.id] = str(record.seq)
            return sequences
        except ImportError:
            # Fallback parser if BioPython not available
            return self._read_fasta_simple(path)

    def _read_fasta_simple(self, path: Path) -> Dict[str, str]:
        """Simple FASTA parser without BioPython."""
        sequences = {}
        current_id = None
        current_seq = []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id is not None:
                        sequences[current_id] = "".join(current_seq)
                    current_id = line[1:].split()[0]  # Take first word after >
                    current_seq = []
                elif line:
                    current_seq.append(line)

            # Don't forget the last sequence
            if current_id is not None:
                sequences[current_id] = "".join(current_seq)

        return sequences

    def get_feature_dim(self) -> int:
        """Get the feature dimension for this featurizer.

        Returns:
            Number of features produced by this featurizer.
        """
        # Known dimensions for ESM2 models
        dims = {
            "esm2_t12_35M_UR50D": 480,
            "esm2_t33_650M_UR50D": 1280,
            "esm2_t36_3B_UR50D": 2560,
        }

        if self.featurizer_name in dims:
            return dims[self.featurizer_name]

        # For unknown models, compute by featurizing a test sequence
        test_seq = {"test": "MKTVRQERLKSIVRILERSKEPVSGAQ"}
        features = self.featurize(test_seq)
        return features.shape[1]


def get_featurizer(
    featurizer_name: str = "esm2_t33_650M_UR50D",
    **kwargs: Any,
) -> ProteinFeaturizer:
    """Create a protein featurizer by name.

    Args:
        featurizer_name: Name of the ESM model.
        **kwargs: Additional arguments passed to ProteinFeaturizer.

    Returns:
        ProteinFeaturizer instance.
    """
    return ProteinFeaturizer(featurizer_name, **kwargs)


def list_featurizers() -> List[str]:
    """List all available protein featurizers.

    Returns:
        List of featurizer names.
    """
    return PROTEIN_FEATURIZERS.copy()


def clear_model_cache() -> None:
    """Clear the global protein model cache to free memory."""
    try:
        from ..utils.protein_utils import clear_protein_model_cache

        clear_protein_model_cache()
    except ImportError:
        pass
