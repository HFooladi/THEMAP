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

    Attributes:
        task_id (str): String describing the task this datapoint is taken from.
        smiles (str): SMILES string describing the molecule this datapoint corresponds to.
        bool_label (bool): bool classification label, usually derived from the numeric label using a threshold.
        numeric_label (Optional[float]): numerical label (e.g., activity), usually measured in the lab
        _fingerprint (Optional[np.ndarray]): optional ECFP (Extended-Connectivity Fingerprint) for the molecule.
        _features (Optional[np.ndarray]): optional features for the molecule. features are how we represent the molecule in the model
        _rdkit_mol (Optional[Chem.Mol]): cached RDKit molecule object

    Properties:
        number_of_atoms (int): Number of atoms in the molecule
        number_of_bonds (int): Number of bonds in the molecule
        molecular_weight (float): Molecular weight in atomic mass units
        rdkit_mol (Chem.Mol): RDKit molecule object (lazy loaded)

    Methods:
        get_fingerprint(): Computes and returns the Morgan fingerprint for the molecule
        get_features(): Computes and returns molecular features using specified featurizer

    Example:
        >>> # Create a molecule datapoint
        >>> datapoint = MoleculeDatapoint(
        ...     task_id="toxicity_prediction",
        ...     smiles="CCO",  # ethanol
        ...     bool_label=True,
        ...     numeric_label=0.8
        ... )
        >>> 
        >>> # Access molecular properties
        >>> print(f"Number of atoms: {datapoint.number_of_atoms}")
        Number of atoms: 3
        >>> print(f"Molecular weight: {datapoint.molecular_weight:.2f}")
        Molecular weight: 46.04
        >>> 
        >>> # Get molecular features
        >>> fingerprint = datapoint.get_fingerprint()
        >>> features = datapoint.get_features(featurizer="ecfp")
    """

    task_id: str
    smiles: str
    bool_label: bool
    numeric_label: Optional[float] = None
    _fingerprint: Optional[np.ndarray] = field(default=None, repr=False)
    _features: Optional[np.ndarray] = field(default=None, repr=False)
    _rdkit_mol: Optional[Chem.Mol] = field(default=None, repr=False)

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
    
    def to_dict(self) -> dict:
        """Convert datapoint to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "smiles": self.smiles,
            "bool_label": self.bool_label,
            "numeric_label": self.numeric_label
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MoleculeDatapoint":
        """Create datapoint from dictionary."""
        return cls(
            task_id=data["task_id"],
            smiles=data["smiles"],
            bool_label=data["bool_label"],
            numeric_label=data.get("numeric_label")
        )

    def get_fingerprint(self, force_recompute: bool = False) -> Optional[np.ndarray]:
        """Get the Morgan fingerprint for a molecule.

        This method computes the Extended-Connectivity Fingerprint (ECFP) for the molecule
        using RDKit's Morgan fingerprint generator. The fingerprint is cached after first
        computation to avoid recomputing.

        Args:
            force_recompute (bool): If True, the fingerprint is recomputed even if it
                has already been computed.

        Returns:
            Optional[np.ndarray]: Morgan fingerprint for the molecule (r=2, nbits=2048).
                The fingerprint is a binary vector representing the molecular structure.
                Returns None if fingerprint generation fails.
        """
        if self._fingerprint is not None and not force_recompute:
            return self._fingerprint

        logger.debug(f"Generating fingerprint for molecule {self.smiles}")
        features = get_featurizer("ecfp")(self.smiles)
        if features is None:
            logger.error(f"Failed to generate fingerprint for molecule {self.smiles}")
            return None
        self._fingerprint = features[0]
        logger.debug(f"Successfully generated fingerprint for molecule {self.smiles}")
        return self._fingerprint

    def get_features(self, featurizer_name: Optional[str] = None, force_recompute: bool = False) -> Optional[np.ndarray]:
        """Get features for a molecule using a featurizer model.

        This method computes molecular features using the specified featurizer model.
        The features are cached after first computation to avoid recomputing.

        Args:
            featurizer_name (Optional[str]): Name of the featurizer model to use.
                If None, no featurization is performed.
            force_recompute (bool): If True, the features are recomputed even if they
                have already been computed.

        Returns:
            Optional[np.ndarray]: Features for the molecule. The shape and content depend on
                the featurizer used. Returns None if feature generation fails.
        """
        if self._features is not None and not force_recompute:
            return self._features

        logger.debug(f"Generating features for molecule {self.smiles} using featurizer {featurizer_name}")
        model = get_featurizer(featurizer_name) if featurizer_name else None
        if model is None:
            return None
        features = model(self.smiles)
        if features is None:
            logger.error(f"Failed to generate features for molecule {self.smiles}")
            return None

        self._features = features[0]
        logger.debug(f"Successfully generated features for molecule {self.smiles}")
        return self._features

    @property
    def rdkit_mol(self) -> Optional[Chem.Mol]:
        """Get the RDKit molecule object.

        This property lazily initializes the RDKit molecule if it hasn't been created yet.
        The molecule is cached to avoid recreating it multiple times.

        Returns:
            Optional[Chem.Mol]: RDKit molecule object. Returns None if molecule creation fails.
        """
        if self._rdkit_mol is None:
            self._rdkit_mol = make_mol(self.smiles)
        return self._rdkit_mol

    @property
    def number_of_atoms(self) -> int:
        """Get the number of heavy atoms in the molecule.

        Returns:
            int: Number of heavy atoms in the molecule.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return len(mol.GetAtoms())

    @property
    def number_of_bonds(self) -> int:
        """Get the number of bonds in the molecule.

        Returns:
            int: Number of bonds in the molecule.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return len(mol.GetBonds())

    @property
    def molecular_weight(self) -> float:
        """Get the molecular weight of the molecule.

        Returns:
            float: Molecular weight of the molecule in atomic mass units.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return Chem.Descriptors.ExactMolWt(mol)  # type: ignore[attr-defined]

    @property
    def logp(self) -> float:
        """Calculate octanol-water partition coefficient.
        
        Returns:
            float: LogP value of the molecule.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return Chem.Descriptors.MolLogP(mol)  # type: ignore[attr-defined]

    @property
    def num_rotatable_bonds(self) -> int:
        """Get number of rotatable bonds.
        
        Returns:
            int: Number of rotatable bonds in the molecule.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return Chem.Descriptors.NumRotatableBonds(mol)  # type: ignore[attr-defined]

    @property
    def smiles_canonical(self) -> str:
        """Get canonical SMILES representation.
        
        Returns:
            str: Canonical SMILES string for the molecule.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return Chem.MolToSmiles(mol, isomericSmiles=True)