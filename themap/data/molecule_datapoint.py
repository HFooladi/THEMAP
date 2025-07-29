from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from ..utils.cache_utils import CacheKey, get_global_feature_cache
from ..utils.featurizer_utils import get_featurizer, make_mol
from ..utils.logging import get_logger, setup_logging
from .exceptions import FeaturizationError, InvalidSMILESError

# Setup logging
setup_logging()
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
        _rdkit_mol (Optional[Chem.Mol]): cached RDKit molecule object

    Properties:
        number_of_atoms (int): Number of heavy atoms in the molecule
        number_of_bonds (int): Number of bonds in the molecule
        molecular_weight (float): Molecular weight in atomic mass units
        logp (float): Octanol-water partition coefficient (LogP)
        num_rotatable_bonds (int): Number of rotatable bonds in the molecule
        smiles_canonical (str): Canonical SMILES representation
        rdkit_mol (Chem.Mol): RDKit molecule object (lazy loaded)

    Methods:
        get_fingerprint(): Computes and returns the Morgan fingerprint for the molecule
        get_features(): Computes and returns molecular features using specified featurizer

    Example:
        >>> # Create a molecule datapoint
        >>> datapoint = MoleculeDatapoint(
        ...     task_id="toxicity_prediction",
        ...     smiles="CCCO",  # propanol
        ...     bool_label=True,
        ...     numeric_label=0.8
        ... )
        >>>
        >>> # Access molecular properties
        >>> print(f"Number of heavy atoms: {datapoint.number_of_atoms}")
        # Number of heavy atoms: 4
        >>> print(f"Molecular weight: {datapoint.molecular_weight:.2f}")
        # Molecular weight: 60.06
        >>> print(f"LogP: {datapoint.logp:.2f}")
        # LogP: 0.39
        >>> print(f"Number of rotatable bonds: {datapoint.num_rotatable_bonds}")
        # Number of rotatable bonds: 1
        >>> print(f"SMILES canonical: {datapoint.smiles_canonical}")
        # SMILES canonical: CCCO
        >>>
        >>> # Get molecular features
        >>> fingerprint = datapoint.get_fingerprint()
        >>> print(f"Fingerprint shape: {fingerprint.shape if fingerprint is not None else None}")
        # Fingerprint shape: (2048,)
        >>> features = datapoint.get_features(featurizer_name="ecfp")
        >>> print(f"Features shape: {features.shape if features is not None else None}")
        # Features shape: (2048,)
    """

    task_id: str
    smiles: str
    bool_label: bool
    numeric_label: Optional[float] = None
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

        # Validate SMILES string
        if not self.smiles.strip():
            raise InvalidSMILESError(self.smiles, "SMILES string cannot be empty")
        test_mol = make_mol(self.smiles)
        if test_mol is None:
            raise InvalidSMILESError(self.smiles, "RDKit cannot parse this SMILES")

    def __repr__(self) -> str:
        return f"MoleculeDatapoint(task_id={self.task_id}, smiles={self.smiles}, bool_label={self.bool_label}, numeric_label={self.numeric_label})"

    def to_dict(self) -> dict:
        """Convert datapoint to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "smiles": self.smiles,
            "bool_label": self.bool_label,
            "numeric_label": self.numeric_label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MoleculeDatapoint":
        """Create datapoint from dictionary."""
        return cls(
            task_id=data["task_id"],
            smiles=data["smiles"],
            bool_label=data["bool_label"],
            numeric_label=data.get("numeric_label"),
        )

    def get_fingerprint(self, force_recompute: bool = False) -> Optional[np.ndarray]:
        """Get the Morgan fingerprint for a molecule.

        This method computes the Extended-Connectivity Fingerprint (ECFP) for the molecule
        using RDKit's Morgan fingerprint generator. Features are cached globally to avoid
        recomputation across different instances.

        Args:
            force_recompute (bool): If True, the fingerprint is recomputed even if cached.

        Returns:
            Optional[np.ndarray]: Morgan fingerprint for the molecule (r=2, nbits=2048).
                The fingerprint is a binary vector representing the molecular structure.
                Returns None if fingerprint generation fails.

        Note:
            dtype of the fingerprint is np.uint8.
        """
        cache_key = CacheKey(smiles=self.smiles, featurizer_name="ecfp")
        cache = get_global_feature_cache()

        # Check cache first unless forcing recomputation
        if not force_recompute:
            cached_features = cache.get(cache_key)
            if cached_features is not None:
                logger.debug(f"Cache hit for fingerprint of molecule {self.smiles}")
                return cached_features

        logger.debug(f"Computing fingerprint for molecule {self.smiles}")
        try:
            featurizer = get_featurizer("ecfp")
            features = featurizer(self.smiles)
            if features is None:
                logger.error(f"Failed to generate fingerprint for molecule {self.smiles}")
                return None

            fingerprint = features[0]
            cache.store(cache_key, fingerprint)
            logger.debug(f"Successfully computed and cached fingerprint for molecule {self.smiles}")
            return fingerprint

        except Exception as e:
            logger.error(f"Error computing fingerprint for molecule {self.smiles}: {e}")
            raise FeaturizationError(self.smiles, "ecfp", str(e))

    def get_features(
        self, featurizer_name: Optional[str] = None, force_recompute: bool = False
    ) -> Optional[np.ndarray]:
        """Get features for a molecule using a featurizer model.

        This method computes molecular features using the specified featurizer model.
        Features are cached globally to avoid recomputation across different instances.

        Args:
            featurizer_name (Optional[str]): Name of the featurizer model to use.
                If None, returns None.
            force_recompute (bool): If True, features are recomputed even if cached.

        Returns:
            Optional[np.ndarray]: Features for the molecule. The shape and content depend on
                the featurizer used. Returns None if feature generation fails or featurizer_name is None.

        Note:
            dtype of the features is different for different featurizers.
            For example, ecfp and fcfp dtype is np.uint8, while mordred dtype is np.float64.
        """
        if featurizer_name is None:
            return None

        cache_key = CacheKey(smiles=self.smiles, featurizer_name=featurizer_name)
        cache = get_global_feature_cache()

        # Check cache first unless forcing recomputation
        if not force_recompute:
            cached_features = cache.get(cache_key)
            if cached_features is not None:
                logger.debug(f"Cache hit for features of molecule {self.smiles} with {featurizer_name}")
                return cached_features

        logger.debug(f"Computing features for molecule {self.smiles} using featurizer {featurizer_name}")
        try:
            featurizer = get_featurizer(featurizer_name)
            if featurizer is None:
                logger.warning(f"Featurizer {featurizer_name} not found")
                return None

            features = featurizer(self.smiles)
            if features is None:
                logger.error(f"Failed to generate features for molecule {self.smiles}")
                return None

            feature_vector = features[0]
            cache.store(cache_key, feature_vector)
            logger.debug(f"Successfully computed and cached features for molecule {self.smiles}")
            return feature_vector

        except Exception as e:
            logger.error(f"Error computing features for molecule {self.smiles} with {featurizer_name}: {e}")
            raise FeaturizationError(self.smiles, featurizer_name, str(e))

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
        return mol.GetNumAtoms()

    @property
    def number_of_bonds(self) -> int:
        """Get the number of bonds in the molecule.

        Returns:
            int: Number of bonds in the molecule.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return mol.GetNumBonds()

    @property
    def molecular_weight(self) -> float:
        """Get the molecular weight of the molecule.

        Returns:
            float: Molecular weight of the molecule in atomic mass units.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return float(Descriptors.ExactMolWt(mol))  # type: ignore[attr-defined]

    @property
    def logp(self) -> float:
        """Calculate octanol-water partition coefficient.

        Returns:
            float: LogP value of the molecule.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return float(Chem.Descriptors.MolLogP(mol))  # type: ignore[attr-defined]

    @property
    def num_rotatable_bonds(self) -> int:
        """Get number of rotatable bonds.

        Returns:
            int: Number of rotatable bonds in the molecule.
        Raises:
            ValueError: If the molecule cannot be created.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return int(Chem.Descriptors.NumRotatableBonds(mol))  # type: ignore[attr-defined]

    @property
    def smiles_canonical(self) -> str:
        """Get canonical SMILES representation.

        Returns:
            str: Canonical SMILES string for the molecule.
        Raises:
            ValueError: If the molecule cannot be created.
        """
        mol = self.rdkit_mol
        if mol is None:
            raise ValueError("Failed to create RDKit molecule")
        return Chem.MolToSmiles(mol, isomericSmiles=True)
