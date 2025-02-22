from dataclasses import dataclass, field
import pickle
from typing import Dict, List, Optional, Tuple, Union
import os
import logging
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

logger = logging.getLogger(__name__)


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


def get_task_name_from_path(path: RichPath) -> str:
    """
    Extract task name from file path.
    
    Args:
        path (Any): Path-like object
    
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

    Args:
        task_id (str): String describing the task this datapoint is taken from.
        smiles (str): SMILES string describing the molecule this datapoint corresponds to.
        bool_label (bool): bool classification label, usually derived from the numeric label using a threshold.
        numeric_label (float): numerical label (e.g., activity), usually measured in the lab
        fingerprint (np.ndarray): optional ECFP (Extended-Connectivity Fingerprint) for the molecule.
        features (np.ndarray): optional features for the molecule. features are how we represent the molecule in the model
    """

    task_id: str
    smiles: str
    bool_label: bool
    numeric_label: Optional[float] = None
    _fingerprint: Optional[np.ndarray] = field(default=None, repr=False)
    _features: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate initialization data."""
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")
        if not isinstance(self.smiles, str):
            raise TypeError("smiles must be a string")
        if not isinstance(self.bool_label, bool):
            raise TypeError("bool_label must be a boolean")
        if self.numeric_label is not None and not isinstance(self.numeric_label, (int, float)):
            raise TypeError("numeric_label must be a number or None")
    

    def __repr__(self):
        return f"MoleculeDatapoint(task_id={self.task_id}, smiles={self.smiles}, bool_label={self.bool_label}, numeric_label={self.numeric_label})"

    def get_fingerprint(self) -> np.ndarray:
        """
        Get the fingerprint for a molecule.

        Returns:
            np.ndarray: Morgan fingerprint for the molecule (r=2, nbits=2048).
        """
        if self._fingerprint is not None:
            return self._fingerprint
        
        mol = make_mol(self.smiles)
        fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
            [mol], fpType=rdFingerprintGenerator.MorganFP
        )[0]
        fingerprint = np.zeros((0,), np.float32)  # Generate target pointer to fill
        DataStructs.ConvertToNumpyArray(fingerprints_vect, fingerprint)
        self._fingerprint = fingerprint
        return fingerprint

    def get_features(self, featurizer: Optional[str] = None) -> np.ndarray:
        """
        Get features for a molecule using a featurizer model.

        Args:
            featurizer (str): Name of the featurizer model to use.

        Returns:
            np.ndarray: Features for the molecule.
        """
        if self._features is not None:
            return self._features
        model = get_featurizer(featurizer) if featurizer else None
        features = model(self.smiles) if model else None

        self._features = features
        return features

    @property
    def number_of_atoms(self) -> int:
        """
        Gets the number of atoms in the :class:`MoleculeDatapoint`.

        Returns:
            int: Number of atoms in the molecule.

        """
        mol = make_mol(self.smiles)
        return len(mol.GetAtoms())

    @property
    def number_of_bonds(self) -> int:
        """
        Gets the number of bonds in the :class:`MoleculeDatapoint`.

        Returns:
            int: Number of bonds in the molecule.
        """
        mol = make_mol(self.smiles)
        return len(mol.GetBonds())

    @property
    def molecular_weight(self) -> float:
        """
        Gets the molecular weight of the :class:`MoleculeDatapoint`.

        Returns:
            float: Molecular weight of the molecule.
        """
        mol = make_mol(self.smiles)
        return Chem.Descriptors.ExactMolWt(mol)


@dataclass
class ProteinDataset:
    """Data structure holding information for proteins (list of protein).

    Args:
        task_id (list[str]): list of string describing the tasks these protein are taken from.
        protein (dict): dictionary mapping the protein id to the protein sequence.
    """

    task_id: list[str]
    protein: dict
    features: Optional[np.ndarray] = None

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return list(self.protein.keys())[idx], list(self.protein.values())[idx]

    def __len__(self) -> int:
        return len(self.protein)

    def __repr__(self) -> str:
        return f"ProteinDataset(task_id={self.task_id}, protein={self.protein})"

    def get_features(self, model) -> np.ndarray:
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
        task_id (list): list of string describing the tasks these metadata are taken from.
        protein (ProteinDataset): ProteinDataset object.
    """

    task_id: list[str]
    protein: ProteinDataset
    text_desc: Optional[str]

    def get_features(self, model) -> np.ndarray:
        return model.encode(self.text_desc)


@dataclass
class MoleculeDataset:
    """Data structure holding information for a dataset of molecules.

    Args:
        task_id (str): String describing the task this dataset is taken from.
        data (List[MoleculeDatapoint]): List of MoleculeDatapoint objects.
    """

    task_id: str
    data: List[MoleculeDatapoint] = field(default_factory=list)

    def __post_init__(self):
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

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"MoleculeDataset(task_id={self.task_id}, task_size={len(self.data)})"

    def get_dataset_embedding(self, model) -> np.ndarray:
        """
        Get the features for the entire dataset.

        Args:
            model: Featurizer model to use.

        Returns:
            np.ndarray: Features for the entire dataset.
        """
        smiles = [data.smiles for data in self.data]
        features = get_featurizer(model)(smiles)
        for i, molecule in enumerate(self.data):
            molecule.features = features[i]
        assert len(features) == len(smiles)
        return features

    def get_prototype(self, model) -> Tuple[np.ndarray]:
        """
        Get the prototype of the dataset.

        Args:
            model: Featurizer model to use.
        
        Returns:
            Tuple[np.ndarray]: Tuple containing the positive and negative prototype of the dataset.
        """
        data_features = self.get_dataset_embedding(model)
        # Calculate the prototype of the dataset
        # First find all positive and negative samples
        # Then just average over all positives and negatives
        positives = [data_features[i] for i in range(len(data_features)) if self.data[i].bool_label]
        negatives = [data_features[i] for i in range(len(data_features)) if not self.data[i].bool_label]

        positive_prototype = np.array(positives).mean(axis=0)
        negative_prototype = np.array(negatives).mean(axis=0)
        return positive_prototype, negative_prototype

    @property
    def get_features(self) -> np.ndarray:
        return np.array([data.features for data in self.data])

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

        return MoleculeDataset(get_task_name_from_path(path), samples)

    def filter(self, condition: callable) -> 'MoleculeDataset':
        """
        Filter dataset based on a condition.
        
        Args:
            condition: Callable that returns True/False for each datapoint
        
        Returns:
            Filtered MoleculeDataset
        """
        filtered_data = [dp for dp in self.data if condition(dp)]
        return MoleculeDataset(self.task_id, filtered_data)


class MoleculeDatasets:
    """Dataset of related tasks, provided as individual files split into meta-train, meta-valid and
    meta-test sets."""

    def __init__(
        self,
        train_data_paths: List[RichPath] = [],
        valid_data_paths: List[RichPath] = [],
        test_data_paths: List[RichPath] = [],
        num_workers: Optional[int] = None,
    ):
        self._fold_to_data_paths: Dict[DataFold, List[RichPath]] = {
            DataFold.TRAIN: train_data_paths,
            DataFold.VALIDATION: valid_data_paths,
            DataFold.TEST: test_data_paths,
        }
        self._num_workers = num_workers if num_workers is not None else os.cpu_count() or 1
        logger.info(f"Identified {len(self._fold_to_data_paths[DataFold.TRAIN])} training tasks.")
        logger.info(f"Identified {len(self._fold_to_data_paths[DataFold.VALIDATION])} validation tasks.")
        logger.info(f"Identified {len(self._fold_to_data_paths[DataFold.TEST])} test tasks.")

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
        """Create a new MoleculeDatasets object from a directory containing the pre-processed
        files (*.jsonl.gz) split in to train/valid/test subdirectories.

        Args:
            directory: Path containing .jsonl.gz files representing the pre-processed tasks.
            task_list_file: (Optional) path of the .json file that stores which assays are to be
            used in each fold. Used for subset selection.
            **kwargs: remaining arguments are forwarded to the MoleculeDatasets constructor.
        """
        if isinstance(directory, str):
            data_rp = RichPath.create(directory)
        else:
            data_rp = directory

        if task_list_file is not None:
            if isinstance(task_list_file, str):
                task_list_file = RichPath.create(task_list_file)
            else:
                task_list_file = task_list_file
            task_list = task_list_file.read_by_file_suffix()
        else:
            task_list = None

        def get_fold_file_names(data_fold_name: str):
            fold_dir = data_rp.join(data_fold_name)
            if task_list is None:
                return fold_dir.get_filtered_files_in_dir("*.jsonl.gz")
            else:
                return [
                    file_name
                    for file_name in fold_dir.get_filtered_files_in_dir("*.jsonl.gz")
                    if any(
                        file_name.basename() == f"{task_name}.jsonl.gz"
                        for task_name in task_list[data_fold_name]
                    )
                ]

        return MoleculeDatasets(
            train_data_paths=get_fold_file_names("train"),
            valid_data_paths=sorted(get_fold_file_names("valid")),
            test_data_paths=sorted(get_fold_file_names("test")),
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

        if isinstance(self.data.get_features, np.ndarray):
            X = torch.from_numpy(self.data.get_features)
        else:
            X = self.data.get_features
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
