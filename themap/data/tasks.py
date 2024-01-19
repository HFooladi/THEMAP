from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

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
        final_hardness = w_exc*self.external_chemical_space + w_exp*self.external_protein_space + w_inc*self.internal_chemical_space
        return final_hardness
    


@dataclass(frozen=True)
class MoleculeDatapoint:
    """Data structure holding information for a single molecule.

    Args:
        task_id: String describing the task this datapoint is taken from.
        smiles: SMILES string describing the molecule this datapoint corresponds to.
        numeric_label: numerical label (e.g., activity), usually measured in the lab
        bool_label: bool classification label, usually derived from the numeric label using a
            threshold.
        fingerprint: optional ECFP (Extended-Connectivity Fingerprint) for the molecule.
        descriptors: optional phys-chem descriptors for the molecule.
    """

    task_id: str
    smiles: str
    numeric_label: float
    bool_label: bool
    fingerprint: Optional[np.ndarray]
    descriptors: Optional[np.ndarray]



@dataclass
class Task:
    task_id: str
    data: str
    metadata: str
    hardness: None

    def __repr__(self):
        return f"Task(task_id={self.task_id}, smiles={self.smiles}, protein={self.protein}, label={self.label}, hardness={self.hardness})"