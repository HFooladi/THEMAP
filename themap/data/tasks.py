from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


@dataclass(frozen=True)
class MoleculeDatapoint:
    """Data structure holding information for a single molecule.

    Args:
        task_id (str): String describing the task this datapoint is taken from.
        smiles (str): SMILES string describing the molecule this datapoint corresponds to.
        bool_label (bool): bool classification label, usually derived from the numeric label using a threshold.
        numeric_label (float): numerical label (e.g., activity), usually measured in the lab
        fingerprint (np.ndarray): optional ECFP (Extended-Connectivity Fingerprint) for the molecule.
    """

    task_id: str
    smiles: str
    bool_label: bool
    fingerprint: Optional[np.ndarray]
    numeric_label: Optional[float] = None

    def get_fingerprint(self) -> np.ndarray:
        if self.fingerprint is not None:
            return self.fingerprint
        else:
            mol = Chem.MolFromSmiles(self.smiles)
            fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
                [mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
            fingerprint = np.zeros((0,), np.float32)  # Generate target pointer to fill
            DataStructs.ConvertToNumpyArray(fingerprints_vect, fingerprint)
            return fingerprint


@dataclass(frozen=True)
class ProteinDatapoint:
    """Data structure holding information for a single protein.

    Args:
        task_id: String describing the task this datapoint is taken from.
        protein: protein sequence string
        numeric_label: numerical label (e.g., activity), usually measured in the lab
        bool_label: bool classification label, usually derived from the numeric label using a
            threshold.
    """

    task_id: str
    protein: str
    numeric_label: float
    bool_label: bool


@dataclass
class MetaData:
    """Data structure holding metadata for a single task.

    Args:
        task_id: String describing the task this datapoint is taken from.
    """

    task_id: str
    protein: ProteinDatapoint
    text_desc: Optional[str]


@dataclass
class Task:
    task_id: str
    data: List[MoleculeDatapoint]
    metadata: MetaData
    hardness: None

    def __repr__(self):
        return f"Task(task_id={self.task_id}, smiles={self.smiles}, protein={self.protein}, label={self.label}, hardness={self.hardness})"


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
