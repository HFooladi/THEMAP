import heapq
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import distance


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def compute_fp_similarity(first_mol, second_mol) -> np.ndarray:
    """Compute similarity between molecules. It receives two MoleculeDatapoint objects,
    extracts their fingerprints and computes the similarity between them.

    Args:
        first_mol: MoleculeDatapoint object
        second_mol: MoleculeDatapoint object
    Returns:
        similarity: similarity between the two molecules
    """
    fp1 = first_mol.get_fingerprint()
    fp2 = second_mol.get_fingerprint()
    sim = 1 - distance.cdist(fp1, fp2, metric="jaccard")
    return sim.astype(np.float32)


def compute_fp_similarities(first_mol_list: List, second_mol_list: List) -> np.ndarray:
    """Compute similarities between two lists of molecules. It receives two lists of
    MoleculeDatapoint objects, extracts their fingerprints and computes the similarities
    between them.

    Args:
        first_mol_list: list of MoleculeDatapoint objects
        second_mol_list: list of MoleculeDatapoint objects
    Returns:
        sims: matrix of similarities between the two lists of molecules
    """
    fps1 = [mol.get_fingerprint() for mol in first_mol_list]    # assumed train set
    fps2 = [mol.get_fingerprint() for mol in second_mol_list]  # assumed test set
    sims = 1 - distance.cdist(fps1, fps2, metric="jaccard")
    return sims.astype(np.float32)