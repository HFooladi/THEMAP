from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem

from themap.utils.featurizer_utils import get_featurizer, make_mol
from themap.utils.logging import get_logger

logger = get_logger(__name__)


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
    """

    task_id: str
    smiles: str
    bool_label: bool
    numeric_label: Optional[float] = None
    _fingerprint: Optional[np.ndarray] = field(default=None, repr=False)
    _features: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate initialization data."""
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

        Returns:
            int: Number of atoms in the molecule.
        """
        mol = make_mol(self.smiles)
        return len(mol.GetAtoms())

    @property
    def number_of_bonds(self) -> int:
        """Get the number of bonds in the molecule.

        Returns:
            int: Number of bonds in the molecule.
        """
        mol = make_mol(self.smiles)
        return len(mol.GetBonds())

    @property
    def molecular_weight(self) -> float:
        """Get the molecular weight of the molecule.

        Returns:
            float: Molecular weight of the molecule in atomic mass units.
        """
        mol = make_mol(self.smiles)
        return Chem.Descriptors.ExactMolWt(mol)
