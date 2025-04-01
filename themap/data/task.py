from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

from themap.data.molecule_dataset import MoleculeDataset
from themap.data.metadata import MetaData
from themap.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Task:
    """A task represents a molecular property prediction problem.

    Args:
        task_id (str): Unique identifier for the task
        data (MoleculeDataset): Dataset containing molecular data
        metadata (MetaData): Metadata about the task
        hardness (Optional[float]): Optional measure of task difficulty
    """

    task_id: str
    data: MoleculeDataset
    metadata: MetaData
    hardness: Optional[float] = None

    def __post_init__(self):
        """Validate task initialization."""
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")
        if not isinstance(self.data, MoleculeDataset):
            raise TypeError("data must be a MoleculeDataset")
        if not isinstance(self.metadata, MetaData):
            raise TypeError("metadata must be a MetaData instance")
        if self.hardness is not None and not isinstance(self.hardness, (int, float)):
            raise TypeError("hardness must be a number or None")

    def __repr__(self) -> str:
        return f"Task(task_id={self.task_id}, data_size={len(self.data)}, hardness={self.hardness})"

    def get_task_embedding(self, data_model: Any, metadata_model: Any) -> np.ndarray:
        """Get combined embedding of data and metadata features.

        Args:
            data_model (Any): Model to use for data feature extraction
            metadata_model (Any): Model to use for metadata feature extraction

        Returns:
            np.ndarray: Combined feature vector
        """
        data_features = np.array([data.get_features(data_model) for data in self.data])
        metadata_features = self.metadata.get_features(metadata_model)
        return np.concatenate([data_features, metadata_features], axis=0) 