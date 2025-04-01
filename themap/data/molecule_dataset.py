from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
from dpu_utils.utils import RichPath

from themap.data.molecule_datapoint import MoleculeDatapoint
from themap.utils.featurizer_utils import get_featurizer
from themap.utils.logging import get_logger

logger = get_logger(__name__)

# Type definitions for better type hints
FeatureArray = np.ndarray  # Type alias for numpy feature arrays
ModelType = Any  # Type for model objects
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

    Args:
        task_id (str): String describing the task this dataset is taken from.
        data (List[MoleculeDatapoint]): List of MoleculeDatapoint objects.
    """

    task_id: str
    data: List[MoleculeDatapoint] = field(default_factory=list)
    _features: Optional[FeatureArray] = None

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

    def get_dataset_embedding(self, model: str) -> FeatureArray:
        """Get the features for the entire dataset.

        Args:
            model (str): Featurizer model to use.

        Returns:
            FeatureArray: Features for the entire dataset, shape (n_samples, n_features).
        """
        logger.info(f"Generating embeddings for dataset {self.task_id} with {len(self.data)} molecules")
        if self._features is not None:
            return self._features
        smiles = [data.smiles for data in self.data]
        features = get_featurizer(model)(smiles)
        for i, molecule in enumerate(self.data):
            molecule._features = features[i]
        assert len(features) == len(smiles), "Feature length does not match SMILES length"
        if isinstance(features, np.ndarray):
            self._features = features
        else:
            features = np.array(features)
            self._features = features
        logger.info(f"Successfully generated embeddings for dataset {self.task_id}")
        return features

    def get_prototype(self, model: ModelType) -> tuple[FeatureArray, FeatureArray]:
        """Get the prototype of the dataset.

        Args:
            model (ModelType): Featurizer model to use.

        Returns:
            tuple[FeatureArray, FeatureArray]: Tuple containing:
                - positive_prototype: Mean feature vector of positive examples
                - negative_prototype: Mean feature vector of negative examples
        """
        logger.info(f"Calculating prototypes for dataset {self.task_id}")
        data_features = self.get_dataset_embedding(model)
        positives = [data_features[i] for i in range(len(data_features)) if self.data[i].bool_label]
        negatives = [data_features[i] for i in range(len(data_features)) if not self.data[i].bool_label]

        positive_prototype = np.array(positives).mean(axis=0)
        negative_prototype = np.array(negatives).mean(axis=0)
        logger.info(f"Successfully calculated prototypes for dataset {self.task_id}")
        return positive_prototype, negative_prototype

    @property
    def get_features(self) -> Optional[FeatureArray]:
        if self._features is None:
            return None
        else:
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
        """
        return {
            "size": len(self),
            "positive_ratio": self.get_ratio,
            "avg_molecular_weight": np.mean([dp.molecular_weight for dp in self.data]),
            "avg_atoms": np.mean([dp.number_of_atoms for dp in self.data]),
            "avg_bonds": np.mean([dp.number_of_bonds for dp in self.data]),
        }
