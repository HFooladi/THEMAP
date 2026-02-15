"""
Simplified MoleculeDataset for efficient batch distance computation.

This module provides a streamlined data structure for molecule datasets,
optimized for N×M task distance matrix computation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dpu_utils.utils import RichPath
from numpy.typing import NDArray

from ..utils.logging import get_logger

logger = get_logger(__name__)


def get_task_name_from_path(path: Union[str, RichPath, Path]) -> str:
    """Extract task name from file path.

    Args:
        path: Path-like object

    Returns:
        str: Extracted task name.
    """
    try:
        if isinstance(path, (str, Path)):
            name = Path(path).name
        else:
            name = path.basename()
        return name[: -len(".jsonl.gz")] if name.endswith(".jsonl.gz") else name
    except Exception:
        return "unknown_task"


@dataclass
class MoleculeDataset:
    """Simplified dataset structure for molecules.

    Optimized for batch distance computation between tasks.
    Stores SMILES strings and labels directly without per-molecule object overhead.

    Attributes:
        task_id: String identifying the task this dataset belongs to.
        smiles_list: List of SMILES strings for all molecules.
        labels: Binary labels as numpy array (0/1).
        numeric_labels: Optional continuous labels (e.g., pIC50).
        _features: Precomputed feature matrix (set via set_features or pipeline).
        _featurizer_name: Name of featurizer used for current features.

    Examples:
        >>> dataset = MoleculeDataset.load_from_file("datasets/train/CHEMBL123.jsonl.gz")
        >>> print(len(dataset))  # Number of molecules
        >>> print(dataset.positive_ratio)  # Ratio of positive labels
        >>> # Features are set externally via FeaturizationPipeline
        >>> dataset.set_features(features_array, "ecfp")
        >>> pos_proto, neg_proto = dataset.get_prototype()
    """

    task_id: str
    smiles_list: List[str] = field(default_factory=list)
    labels: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))
    numeric_labels: Optional[NDArray[np.float32]] = None
    _features: Optional[NDArray[np.float32]] = field(default=None, repr=False)
    _featurizer_name: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate dataset initialization."""
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")
        if not isinstance(self.smiles_list, list):
            raise TypeError("smiles_list must be a list")

        # Convert labels to numpy array if not already
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels, dtype=np.int32)

        # Validate lengths match
        if len(self.smiles_list) > 0 and len(self.labels) > 0:
            if len(self.smiles_list) != len(self.labels):
                raise ValueError(
                    f"smiles_list length ({len(self.smiles_list)}) must match "
                    f"labels length ({len(self.labels)})"
                )

    def __len__(self) -> int:
        """Return number of molecules in the dataset."""
        return len(self.smiles_list)

    def __repr__(self) -> str:
        features_info = f", featurizer={self._featurizer_name}" if self._featurizer_name else ""
        return f"MoleculeDataset(task_id={self.task_id}, size={len(self)}{features_info})"

    @property
    def smiles(self) -> List[str]:
        """Get SMILES list (alias for backward compatibility)."""
        return self.smiles_list

    @property
    def positive_ratio(self) -> float:
        """Get ratio of positive to total examples."""
        if len(self.labels) == 0:
            return 0.0
        return round(float(self.labels.sum()) / len(self.labels), 4)

    @property
    def features(self) -> Optional[NDArray[np.float32]]:
        """Get precomputed features if available."""
        return self._features

    @property
    def featurizer_name(self) -> Optional[str]:
        """Get name of featurizer used for current features."""
        return self._featurizer_name

    def has_features(self) -> bool:
        """Check if features have been computed."""
        return self._features is not None

    def set_features(self, features: NDArray[np.float32], featurizer_name: str) -> None:
        """Set precomputed features for this dataset.

        Args:
            features: Feature matrix of shape (n_molecules, feature_dim)
            featurizer_name: Name of the featurizer used

        Raises:
            ValueError: If feature dimensions don't match dataset size
        """
        if len(features) != len(self.smiles_list):
            raise ValueError(
                f"Feature count ({len(features)}) must match dataset size ({len(self.smiles_list)})"
            )
        self._features = features.astype(np.float32)
        self._featurizer_name = featurizer_name
        logger.debug(f"Set features for {self.task_id}: shape={features.shape}")

    def clear_features(self) -> None:
        """Clear cached features to free memory."""
        self._features = None
        self._featurizer_name = None

    def get_features(self, featurizer_name: str = "ecfp", **kwargs: Any) -> NDArray[np.float32]:
        """Get molecular features, computing on demand if necessary.

        This method returns pre-computed features if available (set via set_features or FeaturizationPipeline),
        or computes features on demand using the specified featurizer.

        Args:
            featurizer_name: Name of molecular featurizer to use (e.g., "ecfp", "maccs", "desc2D")
            **kwargs: Additional featurizer arguments

        Returns:
            Feature matrix of shape (n_molecules, feature_dim)

        Raises:
            ValueError: If no molecules in dataset or featurization fails
        """
        # Check if features are already computed with matching featurizer
        if self._features is not None:
            if self._featurizer_name == featurizer_name:
                return self._features
            else:
                logger.debug(
                    f"Task {self.task_id}: requested featurizer '{featurizer_name}' differs from "
                    f"cached '{self._featurizer_name}', recomputing..."
                )

        if len(self.smiles_list) == 0:
            raise ValueError(f"Cannot compute features for empty dataset {self.task_id}")

        # Compute features on demand using the featurizer
        try:
            from ..utils.featurizer_utils import get_featurizer

            featurizer = get_featurizer(featurizer_name)
            features_list = []
            for smiles in self.smiles_list:
                feat = featurizer(smiles)
                features_list.append(feat)

            features = np.array(features_list, dtype=np.float32)
            self.set_features(features, featurizer_name)
            return features
        except Exception as e:
            logger.error(f"Failed to compute features for dataset {self.task_id}: {e}")
            raise ValueError(f"Feature computation failed: {e}") from e

    def get_prototype(
        self, featurizer_name: Optional[str] = None
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute positive and negative prototypes from features.

        Prototypes are the mean feature vectors for each class.

        Args:
            featurizer_name: Optional featurizer name. If provided and features
                aren't yet computed, they will be computed on demand.

        Returns:
            Tuple of (positive_prototype, negative_prototype)

        Raises:
            ValueError: If features haven't been set or no examples exist for a class
        """
        # If featurizer_name is provided and features aren't set, compute them
        if featurizer_name is not None and self._features is None:
            self.get_features(featurizer_name)

        if self._features is None:
            raise ValueError(
                f"Features must be set before computing prototype for {self.task_id}. "
                "Use set_features() or FeaturizationPipeline first."
            )

        pos_mask = self.labels == 1
        neg_mask = self.labels == 0

        if not pos_mask.any():
            raise ValueError(f"Dataset {self.task_id} contains no positive examples")
        if not neg_mask.any():
            raise ValueError(f"Dataset {self.task_id} contains no negative examples")

        positive_prototype = self._features[pos_mask].mean(axis=0).astype(np.float32)
        negative_prototype = self._features[neg_mask].mean(axis=0).astype(np.float32)

        return positive_prototype, negative_prototype

    def get_class_features(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get features separated by class.

        Returns:
            Tuple of (positive_features, negative_features)

        Raises:
            ValueError: If features haven't been set
        """
        if self._features is None:
            raise ValueError(f"Features must be set for {self.task_id}")

        pos_mask = self.labels == 1
        neg_mask = self.labels == 0

        return self._features[pos_mask], self._features[neg_mask]

    @staticmethod
    def load_from_file(path: Union[str, RichPath, Path]) -> "MoleculeDataset":
        """Load dataset from a JSONL.GZ file.

        Args:
            path: Path to the JSONL.GZ file.

        Returns:
            MoleculeDataset with loaded SMILES and labels.
        """
        if isinstance(path, (str, Path)):
            rich_path = RichPath.create(str(path))
        else:
            rich_path = path

        task_id = get_task_name_from_path(path)
        logger.info(f"Loading dataset {task_id} from {path}")

        smiles_list: List[str] = []
        labels: List[int] = []
        numeric_labels: List[Optional[float]] = []

        for raw_sample in rich_path.read_by_file_suffix():
            smiles_list.append(raw_sample["SMILES"])
            labels.append(int(float(raw_sample["Property"])))

            # Handle numeric label
            regression_property = raw_sample.get("RegressionProperty")
            if regression_property is not None and regression_property != "":
                try:
                    numeric_value = float(regression_property)
                    if -float("inf") < numeric_value < float("inf"):
                        numeric_labels.append(numeric_value)
                    else:
                        numeric_labels.append(None)
                except (ValueError, TypeError):
                    numeric_labels.append(None)
            else:
                numeric_labels.append(None)

        # Convert to numpy arrays
        labels_array = np.array(labels, dtype=np.int32)

        # Only create numeric_labels array if any values exist
        numeric_array = None
        if any(v is not None for v in numeric_labels):
            numeric_array = np.array(
                [v if v is not None else np.nan for v in numeric_labels], dtype=np.float32
            )

        logger.info(f"Loaded {len(smiles_list)} molecules for {task_id}")

        return MoleculeDataset(
            task_id=task_id,
            smiles_list=smiles_list,
            labels=labels_array,
            numeric_labels=numeric_array,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary representation."""
        return {
            "task_id": self.task_id,
            "smiles_list": self.smiles_list,
            "labels": self.labels.tolist(),
            "numeric_labels": self.numeric_labels.tolist() if self.numeric_labels is not None else None,
            "featurizer_name": self._featurizer_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MoleculeDataset":
        """Create dataset from dictionary representation."""
        return cls(
            task_id=data["task_id"],
            smiles_list=data["smiles_list"],
            labels=np.array(data["labels"], dtype=np.int32),
            numeric_labels=np.array(data["numeric_labels"], dtype=np.float32)
            if data.get("numeric_labels")
            else None,
        )

    def filter_by_indices(self, indices: List[int]) -> "MoleculeDataset":
        """Create a new dataset with only the specified indices.

        Args:
            indices: List of indices to keep

        Returns:
            New MoleculeDataset with filtered data
        """
        return MoleculeDataset(
            task_id=self.task_id,
            smiles_list=[self.smiles_list[i] for i in indices],
            labels=self.labels[indices],
            numeric_labels=self.numeric_labels[indices] if self.numeric_labels is not None else None,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        if len(self) == 0:
            return {"size": 0, "positive_ratio": 0.0}

        stats: Dict[str, Any] = {
            "size": len(self),
            "positive_count": int(self.labels.sum()),
            "negative_count": int((self.labels == 0).sum()),
            "positive_ratio": self.positive_ratio,
        }

        if self.numeric_labels is not None:
            valid_labels = self.numeric_labels[~np.isnan(self.numeric_labels)]
            if len(valid_labels) > 0:
                stats["numeric_mean"] = float(valid_labels.mean())
                stats["numeric_std"] = float(valid_labels.std())

        if self._features is not None:
            stats["feature_dim"] = self._features.shape[1]
            stats["featurizer"] = self._featurizer_name

        return stats

    # Backward compatibility: datapoints property for metalearning module
    @property
    def datapoints(self) -> List[Dict[str, Any]]:
        """Legacy property for backward compatibility with metalearning module.

        Returns list of dictionaries with molecule data.
        """
        return [
            {
                "smiles": self.smiles_list[i],
                "labels": self.labels[i],
                "bool_label": bool(self.labels[i]),
                "numeric_label": self.numeric_labels[i] if self.numeric_labels is not None else None,
            }
            for i in range(len(self))
        ]

    @property
    def data(self) -> List[Dict[str, Any]]:
        """Legacy property - alias for datapoints."""
        return self.datapoints
