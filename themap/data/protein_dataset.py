from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dpu_utils.utils import RichPath

from themap.utils.protein_utils import (
    convert_fasta_to_dict,
    get_protein_features,
    get_task_name_from_uniprot,
)
from themap.utils.logging import get_logger

logger = get_logger(__name__)

# Type definitions for better type hints
ProteinDict = Dict[str, str]  # Maps protein ID to sequence
FeatureArray = np.ndarray  # Type alias for numpy feature arrays
ModelType = Any  # Type for model objects

@dataclass
class ProteinDataset:
    """Data structure holding information for proteins (list of protein).

    Args:
        task_id (List[str]): list of string describing the tasks these protein are taken from.
        protein (ProteinDict): dictionary mapping the protein id to the protein sequence.
        features (Optional[FeatureArray]): Optional pre-computed protein features.
    """

    task_id: List[str]
    protein: ProteinDict
    features: Optional[FeatureArray] = None

    def __post_init__(self) -> None:
        """Validate initialization data."""
        if not isinstance(self.task_id, list):
            raise TypeError("task_id must be a list")
        if not isinstance(self.protein, dict):
            raise TypeError("protein must be a dictionary")
        if not all(isinstance(key, str) for key in self.protein.keys()):
            raise TypeError("protein keys must be strings")

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return list(self.protein.keys())[idx], list(self.protein.values())[idx]

    def __len__(self) -> int:
        return len(self.protein)

    def __repr__(self) -> str:
        return f"ProteinDataset(task_id={self.task_id}, protein={self.protein})"

    def get_features(self, model: ModelType) -> FeatureArray:
        """Get protein features using the specified model.

        Args:
            model (ModelType): Model to use for feature extraction.

        Returns:
            FeatureArray: Computed protein features.
        """
        self.features = get_protein_features(self.protein, model)
        return self.features

    @staticmethod
    def load_from_file(path: Union[str, RichPath]) -> "ProteinDataset":
        """Load protein dataset from a FASTA file.

        Args:
            path (Union[str, RichPath]): Path to the FASTA file.

        Returns:
            ProteinDataset: Loaded protein dataset.
        """
        protein_dict = convert_fasta_to_dict(path)
        uniprot_ids = [key.split("|")[1] for key in protein_dict.keys()]
        return ProteinDataset(get_task_name_from_uniprot(uniprot_ids), protein_dict) 