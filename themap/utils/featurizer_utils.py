from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from molfeat.trans import MoleculeTransformer
from molfeat.trans.pretrained import GraphormerTransformer, PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from rdkit import Chem

from .logging import get_logger

# Setup logging
logger = get_logger(__name__)


class BaseFeaturizer(ABC):
    """Abstract base class for molecular featurizers."""

    @abstractmethod
    def __call__(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """Convert SMILES to feature vector(s).

        Args:
            smiles: SMILES string or list of SMILES strings

        Returns:
            np.ndarray: Feature vector(s)
        """
        pass


def make_mol(
    smiles: str, keep_h: bool = True, add_h: bool = False, keep_atom_map: bool = False
) -> Optional[Chem.Mol]:
    """
    Builds an RDKit molecule from a SMILES string.

    Args:
        smiles: SMILES string.
        keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
        add_h: Boolean whether to add hydrogens to the input smiles.
        keep_atom_map: Boolean whether to keep the original atom mapping.

    Returns:
        RDKit molecule or None if the molecule is invalid.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h  # type: ignore
    mol = Chem.MolFromSmiles(smiles, params)

    if add_h:
        mol = Chem.AddHs(mol)

    if keep_atom_map and mol is not None:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        for idx, map_num in enumerate(atom_map_numbers):
            if idx + 1 != map_num:
                new_order = np.argsort(atom_map_numbers).tolist()
                return Chem.rdmolops.RenumberAtoms(mol, new_order)
    elif not keep_atom_map and mol is not None:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    return mol


def get_featurizer(
    featurizer: str, n_jobs: int = -1
) -> Union[MoleculeTransformer, GraphormerTransformer, PretrainedHFTransformer, PretrainedDGLTransformer]:
    """
    Returns a featurizer object based on the input string.

    Args:
        featurizer (str): String specifying the featurizer to use.
        n_jobs (int): Number of jobs to use for parallel processing.

    Returns:
        Featurizer object.

    Raises:
        ValueError: If the specified featurizer is not found.
    """
    assert isinstance(featurizer, str), "Featurizer must be a string"
    assert isinstance(n_jobs, int), "Number of jobs must be an integer"

    if featurizer in ["ecfp", "fcfp", "mordred", "desc2D", "desc3D", "maccs", "usrcat"]:
        transformer = MoleculeTransformer(featurizer, n_jobs=n_jobs)

    elif featurizer in ["pcqm4mv2_graphormer_base"]:
        transformer = GraphormerTransformer(kind=featurizer, dtype=float, n_jobs=n_jobs)

    elif featurizer in [
        "ChemBERTa-77M-MLM",
        "ChemBERTa-77M-MTR",
        "Roberta-Zinc480M-102M",
        "MolT5",
    ]:
        transformer = PretrainedHFTransformer(kind=featurizer, notation="smiles", dtype=float, n_jobs=n_jobs)

    elif featurizer in [
        "gin_supervised_infomax",
        "gin_supervised_contextpred",
        "gin_supervised_edgepred",
        "gin_supervised_masking",
    ]:
        transformer = PretrainedDGLTransformer(kind=featurizer, dtype=float, n_jobs=n_jobs)

    else:
        raise ValueError(f"Featurizer {featurizer} not found.")

    return transformer
