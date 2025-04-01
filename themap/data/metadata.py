from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from themap.data.protein_dataset import ProteinDataset
from themap.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetaData:
    """Data structure holding metadata for a batch of tasks.

    Args:
        task_id (List[str]): list of string describing the tasks these metadata are taken from.
        protein (ProteinDataset): ProteinDataset object.
        text_desc (Optional[str]): Optional text description of the task.
    """

    task_id: List[str]
    protein: ProteinDataset
    text_desc: Optional[str]

    def __post_init__(self) -> None:
        """Validate initialization data."""
        if not isinstance(self.task_id, list):
            raise TypeError("task_id must be a list")
        if not isinstance(self.protein, ProteinDataset):
            raise TypeError("protein must be a ProteinDataset")

    def get_features(self, model: Any) -> np.ndarray:
        """Get features from the text description using the specified model.

        Args:
            model (Any): Model to use for feature extraction from text.

        Returns:
            np.ndarray: Computed features from text description.
        """
        return model.encode(self.text_desc)
