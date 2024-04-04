from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from dpu_utils.utils import RichPath  # I should see whether I can remove this dependency or not

from themap.utils.featurizer_utils import get_featurizer, make_mol
from themap.utils.protein_utils import get_protein_features


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

    def get_features(self, featurizer: str) -> np.ndarray:
        """
        Get features for a molecule using a featurizer model.

        Args:
            featurizer (str): Name of the featurizer model to use.

        Returns:
            np.ndarray: Features for the molecule.
        """

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


@dataclass(frozen=True)
class ProteinDatapoint:
    """Data structure holding information for a single protein.

    Args:
        task_id (str): String describing the task this datapoint is taken from.
        protein (str): protein sequence string
        numeric_label: numerical label (e.g., activity), usually measured in the lab
        bool_label: bool classification label, usually derived from the numeric label using a
            threshold.
    """

    task_id: str
    protein: str
    numeric_label: float
    bool_label: bool

    def get_features(self, model) -> np.ndarray:
        return get_protein_features(model, self.protein)


@dataclass
class MetaData:
    """Data structure holding metadata for a single task.

    Args:
        task_id: String describing the task this datapoint is taken from.
    """

    task_id: str
    protein: ProteinDatapoint
    text_desc: Optional[str]

    def get_features(self, model) -> np.ndarray:
        return model.encode(self.text_desc)


@dataclass
class MoleculeDataset:
    task_id: str
    data: List[MoleculeDatapoint]
    metadata: MetaData

    def get_dataset_embedding(self, model) -> np.ndarray:
        data_features = np.array([data.get_features(model) for data in self.data])
        return data_features

    def get_prototype(self, model) -> MoleculeDatapoint:
        data_features = self.get_dataset_embedding(model)
        prototype = data_features.mean(axis=0)
        return prototype
    
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
    external_chemical_space: float
    external_protein_space: float
    internal_chemical_space: float


@dataclass
class TaskHardness:
    external_chemical_space: float
    external_protein_space: float
    internal_chemical_space: float

    def compute_hardness(self, w_exc=0.1, w_exp=1.0, w_inc=0.1):
        final_hardness = (
            w_exc * self.external_chemical_space
            + w_exp * self.external_protein_space
            + w_inc * self.internal_chemical_space
        )
        return final_hardness
