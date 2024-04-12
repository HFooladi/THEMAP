import datamol as dm
import numpy as np
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.pretrained import GraphormerTransformer, PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from rdkit import Chem


def make_mol(smiles: str, keep_h: bool = True, add_h: bool = False, keep_atom_map: bool = False):
    """
    Builds an RDKit molecule from a SMILES string.

    Args:
        smiles: SMILES string.
        keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
        add_h: Boolean whether to add hydrogens to the input smiles.
        keep_atom_map: Boolean whether to keep the original atom mapping.

    Returns:
        RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h
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


def get_featurizer(featurizer: str, n_jobs: int = -1):
    """
    Returns a featurizer object based on the input string.

    Args:
        featurizer: String specifying the featurizer to use.
        n_jobs: Number of jobs to use for parallel processing.

    Returns:
        Featurizer object.
    """
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
        transformer = PretrainedHFTransformer(
            kind=featurizer, notation="smiles", dtype=float, n_jobs=n_jobs
        )

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
