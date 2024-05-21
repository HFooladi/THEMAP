from dataclasses import dataclass
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dpu_utils.utils import RichPath  # I should see whether I can remove this dependency or not
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from themap.utils.featurizer_utils import get_featurizer, make_mol
from themap.utils.protein_utils import (
    convert_fasta_to_dict,
    get_protein_features,
    get_task_name_from_uniprot,
)


def get_task_name_from_path(path: RichPath) -> str:
    # Use filename as task name:
    name = path.basename()
    if name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")]
    return name


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
    fingerprint: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None

    def __repr__(self):
        return f"MoleculeDatapoint(task_id={self.task_id}, smiles={self.smiles}, bool_label={self.bool_label}, numeric_label={self.numeric_label})"

    def get_fingerprint(self) -> np.ndarray:
        """
        Get the fingerprint for a molecule.

        Returns:
            np.ndarray: Morgan fingerprint for the molecule (r=2, nbits=2048).
        """
        if self.fingerprint is not None:
            return self.fingerprint
        else:
            mol = make_mol(self.smiles)
            fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
                [mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
            fingerprint = np.zeros((0,), np.float32)  # Generate target pointer to fill
            DataStructs.ConvertToNumpyArray(fingerprints_vect, fingerprint)
            self.fingerprint = fingerprint
            return fingerprint

    def get_features(self, featurizer: Optional[str] = None) -> np.ndarray:
        """
        Get features for a molecule using a featurizer model.

        Args:
            featurizer (str): Name of the featurizer model to use.

        Returns:
            np.ndarray: Features for the molecule.
        """
        if self.features is not None:
            return self.features
        model = get_featurizer(featurizer)
        features = model(self.smiles)
        self.features = features
        return features

    @property
    def number_of_atoms(self) -> int:
        """
        Gets the number of atoms in the :class:`MoleculeDatapoint`.

        Returns:
            int: Number of atoms in the molecule.

        TODO: maybe I can create make_mol function to create mol from smiles and then use it in the class
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
    """Data structure holding information for proteins.

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
    """Data structure holding metadata for a single task.

    Args:
        task_id: String describing the task this datapoint is taken from.
    """

    task_id: str
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
    data: List[MoleculeDatapoint]

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

    def get_prototype(self, model) -> MoleculeDatapoint:
        data_features = self.get_dataset_embedding(model)
        prototype = data_features.mean(axis=0)
        return prototype

    @property
    def get_features(self) -> np.ndarray:
        return np.array([data.features for data in self.data])

    @property
    def get_labels(self) -> np.ndarray:
        return np.array([data.bool_label for data in self.data])

    @property
    def get_smiles(self) -> List[str]:
        return [data.smiles for data in self.data]

    @staticmethod
    def load_from_file(path: RichPath) -> "MoleculeDataset":
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
                    fingerprint=fingerprint,
                    features=descriptors,
                )
            )

        return MoleculeDataset(get_task_name_from_path(path), samples)


@dataclass
class Task:
    task_id: str
    data: MoleculeDataset
    metadata: MetaData
    hardness: None

    def __repr__(self):
        return f"Task(task_id={self.task_id}, smiles={self.smiles}, protein={self.protein}, label={self.label}, hardness={self.hardness})"

    def get_task_embedding(self, data_model, metadata_model) -> np.ndarray:
        data_features = np.array([data.get_features(data_model) for data in self.data])
        metadata_features = self.metadata.get_features(metadata_model)
        return np.concatenate([data_features, metadata_features], axis=0)


@dataclass
class TaskDistance:
    source_task_ids: List[str] = None
    target_task_ids: List[str] = None
    external_chemical_space: np.ndarray = None
    external_protein_space: np.ndarray = None
    internal_chemical_space: np.ndarray = None

    def __repr__(self) -> str:
        return f"TaskDistance(source_task_ids={len(self.source_task_ids)}, target_task_ids={len(self.target_task_ids)})"

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.source_task_ids), len(self.target_task_ids)

    def compute_ext_chem_distance(self, method):
        pass

    def compute_ext_prot_distance(self, method):
        pass

    def compute_int_chem_distance(self, method):
        pass

    @staticmethod
    def load_ext_chem_distance(path):
        with open(path, "rb") as f:
            x = pickle.load(f)

        if "train_chembl_ids" in x.keys():
            source_task_ids = x["train_chembl_ids"]
        elif "train_pubchem_ids" in x.keys():
            source_task_ids = x["train_pubchem_ids"]
        elif "source_task_ids" in x.keys():
            source_task_ids = x["source_task_ids"]

        if "test_chembl_ids" in x.keys():
            target_task_ids = x["test_chembl_ids"]
        elif "test_pubchem_ids" in x.keys():
            target_task_ids = x["test_pubchem_ids"]
        elif "target_task_ids" in x.keys():
            target_task_ids = x["target_task_ids"]

        return TaskDistance(source_task_ids, target_task_ids, external_chemical_space=x["distance_matrices"])

    @staticmethod
    def load_ext_prot_distance(path):
        with open(path, "rb") as f:
            x = pickle.load(f)

            if "train_chembl_ids" in x.keys():
                source_task_ids = x["train_chembl_ids"]
            elif "train_pubchem_ids" in x.keys():
                source_task_ids = x["train_pubchem_ids"]
            elif "source_task_ids" in x.keys():
                source_task_ids = x["source_task_ids"]

            if "test_chembl_ids" in x.keys():
                target_task_ids = x["test_chembl_ids"]
            elif "test_pubchem_ids" in x.keys():
                target_task_ids = x["test_pubchem_ids"]
            elif "target_task_ids" in x.keys():
                target_task_ids = x["target_task_ids"]

        return TaskDistance(source_task_ids, target_task_ids, external_protein_space=x["distance_matrices"])

    def to_pandas(self):
        df = pd.DataFrame(
            self.external_chemical_space, index=self.source_task_ids, columns=self.target_task_ids
        )
        return df


@dataclass
class TaskHardness:
    task_id: List[str] = None
    external_chemical_space: np.ndarray = None
    external_protein_space: np.ndarray = None
    internal_chemical_space: np.ndarray = None
    hardness: np.ndarray = None

    def compute_hardness(self, w_exc=0.1, w_exp=1.0, w_inc=0.1):
        final_hardness = (
            w_exc * self.external_chemical_space
            + w_exp * self.external_protein_space
            + w_inc * self.internal_chemical_space
        )
        return final_hardness

    @staticmethod
    def compute_from_distance(task_distance: TaskDistance):
        if task_distance.external_chemical_space is not None:
            pass
        elif task_distance.external_protein_space is not None:
            pass
        elif task_distance.internal_chemical_space is not None:
            pass

        pass
