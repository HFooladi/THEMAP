import heapq
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance
import torch



def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def compute_fp_similarity(first_mol, second_mol) -> np.ndarray:
    """Compute similarity between molecules. It receives two MoleculeDatapoint objects,
    extracts their fingerprints and computes the similarity between them.

    Args:
        first_mol: MoleculeDatapoint object
        second_mol: MoleculeDatapoint object
    Returns:
        np.ndarray: similarity between the two molecules
    """
    fp1 = first_mol.get_fingerprint()
    fp2 = second_mol.get_fingerprint()
    sim = 1 - distance.cdist(fp1, fp2, metric="jaccard")
    return sim.astype(np.float32)


def compute_fps_similarity(first_mol_list: List, second_mol_list: List) -> np.ndarray:
    """Compute similarity between two lists of molecules. It receives two lists of
    MoleculeDatapoint objects, extracts their fingerprints and computes the similarities
    between them.

    Args:
        first_mol_list: list of MoleculeDatapoint objects
        second_mol_list: list of MoleculeDatapoint objects
    Returns:
        np.ndarray: matrix of similarities between the two lists of molecules
    """
    fps1 = [mol.get_fingerprint() for mol in first_mol_list]    # assumed train set
    fps2 = [mol.get_fingerprint() for mol in second_mol_list]   # assumed test set
    sims = 1 - distance.cdist(fps1, fps2, metric="jaccard")
    return sims.astype(np.float32)


def calculate_task_hardness_weight(chembl_ids: List, evaluated_resuts: Dict, method: str = "rf") -> torch.Tensor:
    """Computes the internal hardness of a tasks

    Actually  the output indicate how easy is the task based on different methods.
    The higher the value the easier the task.
    """
    assert method in ["rf", "knn" "scaffold"], "Method should be within valid methods"
    weights=[]
    for chembl_id in chembl_ids:
        if method in ["rf", "knn"]:
            weights.append(evaluated_resuts[chembl_id][0].roc_auc)
        elif method == "scaffold":
            weights.append(1 - evaluated_resuts[chembl_id]['neg'])

    weights = torch.tensor(weights)
    return weights


def otdd_hardness(path_to_otdd, path_to_intra_hardness, k=10, train_tasks_weighted=False, weighting_method="rf") -> pd.DataFrame:
    """This function computes the hardness of the test tasks from distance matirix
    computed with OTDD method. The hardeness can be weighted based on the internal hardness of the
    training tasks.

    hardness = sum_{j=1}^k distance_matrix[i,j] * weight[j]

    Args:
        path_to_otdd: Path to the otdd file (pickle file)
        path_to_intra_hardness: Path to the file containing hardness of the training tasks (pickle file)
        k: Number of nearest neighbors to consider for hardness calculation
        train_tasks_weighted: If True, the hardness of the test tasks will be weighted by the hardness of the closest train tasks.
    
    Returns:
        A dataframe containing the hardness of the test tasks
    """    
    PATH_TO_OTDD = path_to_otdd
    with open(PATH_TO_OTDD, 'rb') as f:
        data = pickle.load(f)

    ## data is a dictionary with following keys:
    print(data.keys())

    ## data['distnce_matrices] is a list. We will convert it to the pytorch tensor
    distance_matrix = torch.stack(data['distance_matrices']) # shape: #TRAIN_TASK * TEST_TASKS
    distance_matrix_sorted,  distance_matrix_indices = torch.sort(distance_matrix, dim=0)

    if train_tasks_weighted:
        ## We will weight the distance matrix by the hardness of the closest train tasks
        with open(path_to_intra_hardness, 'rb') as f:
            train_tasks_hardness = pickle.load(f)
        weights = calculate_task_hardness_weight(data['train_chembl_ids'], train_tasks_hardness, method=weighting_method)
        weighted_distance_matrix = (1 - weights[distance_matrix_indices]) * distance_matrix_sorted
        results = torch.mean(weighted_distance_matrix[:k, :], dim=0)
    else:
        results = torch.mean(distance_matrix_sorted[:k, :], dim=0)
    ## Some entries can contain nan values. We will replace them with mean of the results.
    print("Numebr of NaN values in the hardness matrix: ", torch.isnan(results).sum().item())
    results = torch.nan_to_num(results, nan=torch.nanmean(results).item())

    hardness_df = pd.DataFrame({'hardness':results, 'assay': data['test_chembl_ids']})
    return hardness_df