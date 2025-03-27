from dataclasses import dataclass, field
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Callable, TypeVar, Generic, Sequence
from typing_extensions import TypedDict, NotRequired
import os
from enum import Enum

import numpy as np
import pandas as pd
from dpu_utils.utils import RichPath
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import torch
import torch.utils.data.dataloader as dataloader

from themap.utils.featurizer_utils import get_featurizer, make_mol
from themap.utils.protein_utils import (
    convert_fasta_to_dict,
    get_protein_features,
    get_task_name_from_uniprot,
)
from themap.utils.logging import get_logger

logger = get_logger(__name__)


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


# Type definitions for better type hints
ProteinDict = Dict[str, str]  # Maps protein ID to sequence
FeatureArray = np.ndarray  # Type alias for numpy feature arrays
ModelType = Any  # Type for model objects (could be made more specific based on actual model types)
DatasetStats = Dict[str, Union[int, float]]  # Type for dataset statistics

class MoleculeFeatures(TypedDict):
    fingerprint: NotRequired[np.ndarray]
    features: NotRequired[np.ndarray]

def get_task_name_from_path(path: RichPath) -> str:
    """
    Extract task name from file path.
    
    Args:
        path (RichPath): Path-like object
    
    Returns:
        str: Extracted task name
    
    Raises:
        ValueError: If the task name cannot be extracted from the path.
    """
    try:
        name = path.basename()
        return name[:-len(".jsonl.gz")] if name.endswith(".jsonl.gz") else name
    except Exception:
        return "unknown_task"


@dataclass
class MoleculeDatapoint:
    """Data structure holding information for a single molecule and associated features.

    This class represents a single molecule datapoint with its associated features and labels.
    It provides methods to compute molecular fingerprints and features, and includes various
    molecular properties as properties.

    Args:
        task_id (str): String describing the task this datapoint is taken from.
        smiles (str): SMILES string describing the molecule this datapoint corresponds to.
        bool_label (bool): bool classification label, usually derived from the numeric label using a threshold.
        numeric_label (Optional[float]): numerical label (e.g., activity), usually measured in the lab
        _fingerprint (Optional[np.ndarray]): optional ECFP (Extended-Connectivity Fingerprint) for the molecule.
        _features (Optional[np.ndarray]): optional features for the molecule. features are how we represent the molecule in the model

    Examples:
        >>> # Create a molecule datapoint
        >>> datapoint = MoleculeDatapoint(
        ...     task_id="task1",
        ...     smiles="CCO",
        ...     bool_label=True,
        ...     numeric_label=0.8
        ... )
        >>> # Get molecular properties
        >>> print(f"Number of atoms: {datapoint.number_of_atoms}")
        >>> print(f"Molecular weight: {datapoint.molecular_weight}")
        >>> # Get features
        >>> features = datapoint.get_features(featurizer="ecfp")
        >>> fingerprint = datapoint.get_fingerprint()
    """

    task_id: str
    smiles: str
    bool_label: bool
    numeric_label: Optional[float] = None
    _fingerprint: Optional[np.ndarray] = field(default=None, repr=False)
    _features: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate initialization data.
        
        This method is called after initialization to validate the input data.
        It ensures that all fields have the correct types and values.

        Raises:
            TypeError: If any of the fields have incorrect types.
        """
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")
        if not isinstance(self.smiles, str):
            raise TypeError("smiles must be a string")
        if not isinstance(self.bool_label, bool):
            raise TypeError("bool_label must be a boolean")
        if self.numeric_label is not None and not isinstance(self.numeric_label, (int, float)):
            raise TypeError("numeric_label must be a number or None")

    def __repr__(self) -> str:
        return f"MoleculeDatapoint(task_id={self.task_id}, smiles={self.smiles}, bool_label={self.bool_label}, numeric_label={self.numeric_label})"

    def get_fingerprint(self) -> np.ndarray:
        """Get the Morgan fingerprint for a molecule.

        This method computes the Extended-Connectivity Fingerprint (ECFP) for the molecule
        using RDKit's Morgan fingerprint generator. The fingerprint is cached after first
        computation to avoid recomputing.

        Returns:
            np.ndarray: Morgan fingerprint for the molecule (r=2, nbits=2048).
                The fingerprint is a binary vector representing the molecular structure.

        Examples:
            >>> datapoint = MoleculeDatapoint(
            ...     task_id="task1",
            ...     smiles="CCO",
            ...     bool_label=True
            ... )
            >>> fingerprint = datapoint.get_fingerprint()
            >>> print(f"Fingerprint shape: {fingerprint.shape}")
            >>> print(f"Fingerprint sum: {np.sum(fingerprint)}")
        """
        if self._fingerprint is not None:
            return self._fingerprint
        
        logger.debug(f"Generating fingerprint for molecule {self.smiles}")
        self._fingerprint = get_featurizer("ecfp")(self.smiles)
        logger.debug(f"Successfully generated fingerprint for molecule {self.smiles}")
        return self._fingerprint

    def get_features(self, featurizer: Optional[str] = None) -> np.ndarray:
        """Get features for a molecule using a featurizer model.

        This method computes molecular features using the specified featurizer model.
        The features are cached after first computation to avoid recomputing.

        Args:
            featurizer (Optional[str]): Name of the featurizer model to use.
                If None, no featurization is performed.

        Returns:
            np.ndarray: Features for the molecule. The shape and content depend on
                the featurizer used.

        Examples:
            >>> datapoint = MoleculeDatapoint(
            ...     task_id="task1",
            ...     smiles="CCO",
            ...     bool_label=True
            ... )
            >>> # Get features using Morgan fingerprint
            >>> features = datapoint.get_features(featurizer="ecfp")
            >>> print(f"Feature shape: {features.shape}")
            >>> # Get features using a different featurizer
            >>> features = datapoint.get_features(featurizer="graph")
        """
        if self._features is not None:
            return self._features
            
        logger.debug(f"Generating features for molecule {self.smiles} using featurizer {featurizer}")
        model = get_featurizer(featurizer) if featurizer else None
        features = model(self.smiles) if model else None

        self._features = features
        logger.debug(f"Successfully generated features for molecule {self.smiles}")
        return features

    @property
    def number_of_atoms(self) -> int:
        """Get the number of atoms in the molecule.

        This property computes the number of atoms in the molecule using RDKit.

        Returns:
            int: Number of atoms in the molecule.

        Examples:
            >>> datapoint = MoleculeDatapoint(
            ...     task_id="task1",
            ...     smiles="CCO",
            ...     bool_label=True
            ... )
            >>> print(f"Number of atoms: {datapoint.number_of_atoms}")
        """
        mol = make_mol(self.smiles)
        return len(mol.GetAtoms())

    @property
    def number_of_bonds(self) -> int:
        """Get the number of bonds in the molecule.

        This property computes the number of bonds in the molecule using RDKit.

        Returns:
            int: Number of bonds in the molecule.

        Examples:
            >>> datapoint = MoleculeDatapoint(
            ...     task_id="task1",
            ...     smiles="CCO",
            ...     bool_label=True
            ... )
            >>> print(f"Number of bonds: {datapoint.number_of_bonds}")
        """
        mol = make_mol(self.smiles)
        return len(mol.GetBonds())

    @property
    def molecular_weight(self) -> float:
        """Get the molecular weight of the molecule.

        This property computes the exact molecular weight of the molecule using RDKit.

        Returns:
            float: Molecular weight of the molecule in atomic mass units.

        Examples:
            >>> datapoint = MoleculeDatapoint(
            ...     task_id="task1",
            ...     smiles="CCO",
            ...     bool_label=True
            ... )
            >>> print(f"Molecular weight: {datapoint.molecular_weight:.2f}")
        """
        mol = make_mol(self.smiles)
        return Chem.Descriptors.ExactMolWt(mol)


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
        self.features = get_protein_features(self.protein, model)
        return self.features

    @staticmethod
    def load_from_file(path: str) -> "ProteinDataset":
        protein_dict = convert_fasta_to_dict(path)
        uniprot_ids = [key.split("|")[1] for key in protein_dict.keys()]
        return ProteinDataset(get_task_name_from_uniprot(uniprot_ids), protein_dict)


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

    def get_features(self, model: ModelType) -> FeatureArray:
        return model.encode(self.text_desc)


@dataclass
class MoleculeDataset:
    """Data structure holding information for a dataset of molecules.

    This class represents a collection of molecule datapoints, providing methods for
    dataset manipulation, feature computation, and statistical analysis.

    Args:
        task_id (str): String describing the task this dataset is taken from.
        data (List[MoleculeDatapoint]): List of MoleculeDatapoint objects.

    Examples:
        >>> # Create a dataset
        >>> dataset = MoleculeDataset(
        ...     task_id="task1",
        ...     data=[
        ...         MoleculeDatapoint("task1", "CCO", True),
        ...         MoleculeDatapoint("task1", "CCCO", False)
        ...     ]
        ... )
        >>> # Get dataset statistics
        >>> stats = dataset.get_statistics()
        >>> print(f"Dataset size: {stats['size']}")
        >>> print(f"Positive ratio: {stats['positive_ratio']}")
        >>> # Filter dataset
        >>> filtered_dataset = dataset.filter(lambda x: x.number_of_atoms > 3)
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

        This method computes features for all molecules in the dataset using the specified
        featurizer model. The features are stored in each datapoint for future use.

        Args:
            model (ModelType): Featurizer model to use.

        Returns:
            FeatureArray: Features for the entire dataset, shape (n_samples, n_features).

        Examples:
            >>> dataset = MoleculeDataset(
            ...     task_id="task1",
            ...     data=[MoleculeDatapoint("task1", "CCO", True),
            ...           MoleculeDatapoint("task1", "CCCO", False)]
            ... )
            >>> features = dataset.get_dataset_embedding(model="ecfp")
            >>> print(f"Feature matrix shape: {features.shape}")
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

    def get_prototype(self, model: ModelType) -> Tuple[FeatureArray, FeatureArray]:
        """Get the prototype of the dataset.

        This method computes the mean feature vectors for positive and negative examples
        in the dataset, which can be used as prototypes for each class.

        Args:
            model (ModelType): Featurizer model to use.
        
        Returns:
            Tuple[FeatureArray, FeatureArray]: Tuple containing:
                - positive_prototype: Mean feature vector of positive examples
                - negative_prototype: Mean feature vector of negative examples

        Examples:
            >>> dataset = MoleculeDataset(
            ...     task_id="task1",
            ...     data=[
            ...         MoleculeDatapoint("task1", "CCO", True),
            ...         MoleculeDatapoint("task1", "CCCO", False)
            ...     ]
            ... )
            >>> pos_proto, neg_proto = dataset.get_prototype(model="ecfp")
            >>> print(f"Positive prototype shape: {pos_proto.shape}")
            >>> print(f"Negative prototype shape: {neg_proto.shape}")
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
        """
        Get the ratio of positive to negative examples in the dataset.

        Returns:
            float: Ratio of positive to negative examples in the dataset.
        """
        return round(sum([data.bool_label for data in self.data]) / len(self.data), 2)

    @staticmethod
    def load_from_file(path: Union[str, RichPath]) -> "MoleculeDataset":
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

    def filter(self, condition: Callable[[MoleculeDatapoint], bool]) -> 'MoleculeDataset':
        """Filter dataset based on a condition.
        
        This method creates a new dataset containing only the datapoints that satisfy
        the given condition.

        Args:
            condition (Callable[[MoleculeDatapoint], bool]): Function that returns True/False
                for each datapoint. Should take a MoleculeDatapoint as input and return a boolean.
        
        Returns:
            MoleculeDataset: New dataset containing only the filtered datapoints.

        Examples:
            >>> dataset = MoleculeDataset(
            ...     task_id="task1",
            ...     data=[
            ...         MoleculeDatapoint("task1", "CCO", True),
            ...         MoleculeDatapoint("task1", "CCCO", False)
            ...     ]
            ... )
            >>> # Filter for molecules with more than 3 atoms
            >>> filtered = dataset.filter(lambda x: x.number_of_atoms > 3)
            >>> print(f"Original size: {len(dataset)}")
            >>> print(f"Filtered size: {len(filtered)}")
        """
        filtered_data = [dp for dp in self.data if condition(dp)]
        return MoleculeDataset(self.task_id, filtered_data)

    def get_statistics(self) -> DatasetStats:
        """Get statistics about the dataset.

        This method computes various statistical measures about the dataset, including
        size, class balance, and molecular properties.

        Returns:
            DatasetStats: Dictionary containing:
                - size: Total number of datapoints
                - positive_ratio: Ratio of positive to negative examples
                - avg_molecular_weight: Average molecular weight
                - avg_atoms: Average number of atoms
                - avg_bonds: Average number of bonds

        Examples:
            >>> dataset = MoleculeDataset(
            ...     task_id="task1",
            ...     data=[
            ...         MoleculeDatapoint("task1", "CCO", True),
            ...         MoleculeDatapoint("task1", "CCCO", False)
            ...     ]
            ... )
            >>> stats = dataset.get_statistics()
            >>> print(f"Dataset size: {stats['size']}")
            >>> print(f"Average molecular weight: {stats['avg_molecular_weight']:.2f}")
        """
        return {
            "size": len(self),
            "positive_ratio": self.get_ratio,
            "avg_molecular_weight": np.mean([dp.molecular_weight for dp in self.data]),
            "avg_atoms": np.mean([dp.number_of_atoms for dp in self.data]),
            "avg_bonds": np.mean([dp.number_of_bonds for dp in self.data])
        }


class MoleculeDatasets:
    """Dataset of related tasks, provided as individual files split into meta-train, meta-valid and
    meta-test sets."""

    def __init__(
        self,
        train_data_paths: List[RichPath] = [],
        valid_data_paths: List[RichPath] = [],
        test_data_paths: List[RichPath] = [],
        num_workers: Optional[int] = None,
    ) -> None:
        logger.info("Initializing MoleculeDatasets")
        self._fold_to_data_paths: Dict[DataFold, List[RichPath]] = {
            DataFold.TRAIN: train_data_paths,
            DataFold.VALIDATION: valid_data_paths,
            DataFold.TEST: test_data_paths,
        }
        self._num_workers = num_workers
        logger.info(f"Initialized with {len(train_data_paths)} training, {len(valid_data_paths)} validation, and {len(test_data_paths)} test paths")

    def __repr__(self) -> str:
        return f"MoleculeDatasets(train={len(self._fold_to_data_paths[DataFold.TRAIN])}, valid={len(self._fold_to_data_paths[DataFold.VALIDATION])}, test={len(self._fold_to_data_paths[DataFold.TEST])})"

    def get_num_fold_tasks(self, fold: DataFold) -> int:
        return len(self._fold_to_data_paths[fold])

    @staticmethod
    def from_directory(
        directory: Union[str, RichPath],
        task_list_file: Optional[Union[str, RichPath]] = None,
        **kwargs,
    ) -> "MoleculeDatasets":
        logger.info(f"Loading datasets from directory {directory}")
        if isinstance(directory, str):
            directory = RichPath.create(directory)
        else:
            directory = directory

        if task_list_file is not None:
            if isinstance(task_list_file, str):
                task_list_file = RichPath.create(task_list_file)
            logger.info(f"Using task list file: {task_list_file}")
            with open(task_list_file, "r") as f:
                task_list = [line.strip() for line in f.readlines()]
        else:
            task_list = None

        def get_fold_file_names(data_fold_name: str):
            fold_dir = directory.join(data_fold_name)
            if not fold_dir.exists():
                logger.warning(f"Directory {fold_dir} does not exist")
                return []
            return [
                f for f in fold_dir.iterate_filtered(glob_pattern="*.jsonl.gz")
                if task_list is None or get_task_name_from_path(f) in task_list
            ]

        train_data_paths = get_fold_file_names("train")
        valid_data_paths = get_fold_file_names("valid")
        test_data_paths = get_fold_file_names("test")

        logger.info(f"Found {len(train_data_paths)} training, {len(valid_data_paths)} validation, and {len(test_data_paths)} test tasks")
        return MoleculeDatasets(
            train_data_paths=train_data_paths,
            valid_data_paths=valid_data_paths,
            test_data_paths=test_data_paths,
            **kwargs,
        )

    def get_task_names(self, data_fold: DataFold) -> List[str]:
        return [get_task_name_from_path(path) for path in self._fold_to_data_paths[data_fold]]


class TorchMoleculeDataset(torch.utils.data.Dataset):
    """PYTORCH Dataset for molecular data.

    Args:
        data (MoleculeDataset): MoleculeDataset object
        transform (callable): transform to apply to data
        target_transform (callable): transform to apply to targets

    """

    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [0, 1]

        if self.data.get_features is None:
            logger.warning("Dataset does not have features")
            X = torch.ones(len(self.data), 2)
        else:
            X = torch.from_numpy(self.data.get_features)
        
        if isinstance(self.data.get_labels, np.ndarray):
            y = torch.from_numpy(self.data.get_labels).type(torch.LongTensor)
        else:
            y = self.data.get_labels.type(torch.LongTensor)
        self.smiles = self.data.get_smiles
        self.tensors = [X, y]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

    def __repr__(self):
        return f"TorchMoleculeDataset(task_id={self.data.task_id}, task_size={len(self.data.data)})"

    @classmethod
    def create_dataloader(
        cls,
        data: MoleculeDataset,
        batch_size: int = 64,
        shuffle: bool = True,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader.
        
        Args:
            data: Input dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            DataLoader: PyTorch data loader
        """
        dataset = cls(data)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )


def MoleculeDataloader(data, batch_size=64, shuffle=True, transform=None, target_transform=None):
    """Load molecular data and create PYTORCH dataloader.
    Args:
        data (MoleculeDataset): MoleculeDataset object
        batch_size (int): batch size
        shuffle (bool): whether to shuffle data
        transform (callable): transform to apply to data
        target_transform (callable): transform to apply to targets

    Returns:
        dataset_loader (DataLoader): PYTORCH dataloader

    """
    dataset = TorchMoleculeDataset(data)
    dataset_loader = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset_loader


@dataclass
class Task:
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

    def __repr__(self):
        return f"Task(task_id={self.task_id}, smiles={self.smiles}, protein={self.protein}, label={self.label}, hardness={self.hardness})"

    def get_task_embedding(self, data_model, metadata_model) -> np.ndarray:
        data_features = np.array([data.get_features(data_model) for data in self.data])
        metadata_features = self.metadata.get_features(metadata_model)
        return np.concatenate([data_features, metadata_features], axis=0)
