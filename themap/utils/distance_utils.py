import heapq
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance
import torch
from joblib import Parallel, delayed
from tqdm import tqdm


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
    fps1 = [mol.get_fingerprint() for mol in first_mol_list]  # assumed train set
    fps2 = [mol.get_fingerprint() for mol in second_mol_list]  # assumed test set
    sims = 1 - distance.cdist(fps1, fps2, metric="jaccard")
    return sims.astype(np.float32)


def compute_similarities_mean_nearest(mol_list1: List, mol_list2: List) -> float:
    """Compute similarities between two lists of molecules. It receives two lists of
    MoleculeDatapoint objects, extracts their fingerprints and computes the similarities
    between them.

    Args:
        mol_list1: list of MoleculeDatapoint objects
        mol_list2: list of MoleculeDatapoint objects
    """
    result = compute_fps_similarity(mol_list1, mol_list2).max(axis=1).mean()
    return result


def similar_dissimilar_indices(similarity_matrix, threshold) -> Tuple[np.ndarray, np.ndarray]:
    """Compute indices of similar and dissimilar pairs of molecules.

    Args:
        similarity_matrix: matrix of similarities between molecules
        threshold: threshold for similarity
    """
    similar_indices = [sim_col.max(axis=0) >= threshold for sim_col in similarity_matrix.T]
    dissimilar_indices = [np.logical_not(ind) for ind in similar_indices]

    similar_indices = np.where(similar_indices)[0]
    dissimilar_indices = np.where(dissimilar_indices)[0]

    return similar_indices, dissimilar_indices


def inter_distance(test_tasks, train_tasks):
    inter_dist = Parallel(n_jobs=32)(
        delayed(compute_similarities_mean_nearest)(test_tasks[i].samples, train_tasks[j].samples)
        for i in tqdm(range(len(test_tasks)))
        for j in range(len(train_tasks))
    )

    return inter_dist


def intra_distance(tasks_pos, tasks_neg):
    intra_dist = Parallel(n_jobs=16)(
        delayed(compute_fps_similarity)(tasks_pos[i].samples, tasks_neg[i].samples)
        for i in tqdm(range(len(tasks_pos)))
    )
    return intra_dist


def compute_task_hardness_from_distance_matrix(
    distance_matrix: torch.Tensor, proportion: float = 0.01, aggr="mean"
) -> List[torch.Tensor]:
    """Computes the task hardness for each tasks in the dataset.
    We first sort the distance matrix along the test dimension (from min distance to max for each test task) and
    then take the mean and median of the first k elements.

    Args:
        distance_matrix: [N_train * N_test] tensor with the pairwise distances between train and test samples.
        proportion: proportion (percent) of training tasks that should be condidered for calculating hardness
        aggr: aggregation method to use. Can be 'mean', 'median' or 'both'
    """
    assert (
        distance_matrix.shape[0] > distance_matrix.shape[1]
    ), "training set tasks should be larger than test set tasks"
    # Sort the distance matrix along the test dimension
    sorted_distance_matrix = torch.sort(distance_matrix, dim=0)[0]
    # Take the mean of the first k elements
    results = []
    if proportion < 1:
        k: int = int(proportion * distance_matrix.shape[0])
    else:
        k: int = int(proportion)
    if aggr == "mean":
        results.append(torch.mean(sorted_distance_matrix[:k, :], dim=0))
        return results
    elif aggr == "median":
        results.append(torch.median(sorted_distance_matrix[:k, :], dim=0).values)
        return results
    else:
        results.append(torch.mean(sorted_distance_matrix[:k, :], dim=0))
        results.append(torch.median(sorted_distance_matrix[:k, :], dim=0).values)
        return results


def compute_task_hardness_molecule_intra(distance_list: List[np.ndarray]) -> List:
    """Computes the task hardness for each task in the dataset.
    We have a list of arrays, where each element of the list is a N_pos*N_neg array where N_pos is the number of positives
    and N_neg is the number of negatives. Each element of the array is the tanimoto similarity between a positive and a negative.

    Higher Tanimoto similarity means harder thet task.
    Args:
        distance_list: List of N_pos*N_neg array where each array is tanimoto similarity between positives and negatives.

    """

    task_hardness = [item.max(axis=1).mean() for item in distance_list]
    return task_hardness


def compute_task_hardness_molecule_inter(
    distance_list: List[np.ndarray], test_size=157, train_size=4938, topk=100
) -> List:
    """Computes the task hardness for each task in the dataset.
    We have a list of arrays, where each element of the list is a N_pos*N_neg array where N_pos is the number of positives
    and N_neg is the number of negatives. Each element of the array is the tanimoto similarity between a positive and a negative.

    Higher Tanimoto similarity means harder thet task.
    Args:
        distance_list: List of N_pos*N_neg array where each array is tanimoto similarity between positives and negatives.

    """
    distance = []
    for i in range(test_size):
        d = heapq.nlargest(topk, distance_list[i * train_size : i * train_size + train_size])
        distance.append(1 - np.array(d).mean())

    return distance


def compute_correlation(task_df_with_perf, col1, col2, method="pearson"):
    """Computes the correlation between two columns of a dataframe.

    Args:
        task_df_with_perf: Dataframe with the performance of the tasks. It should also
        contain the hardness of the tasks.
        col1: First column name (usually a measure of task hardness).
        col2: Second column name (usually a performance measure of a task).
        method: Correlation method to use.
    """
    corr = task_df_with_perf[col1].corr(task_df_with_perf[col2], method=method)
    return corr


def corr_protein_hardness_metric(
    df,
    chembl_ids: List,
    distance_matrix: torch.Tensor,
    proportions: List = [0.01, 0.1, 0.5, 0.9],
    metric: str = "delta_auprc",
):
    # Correlation between protein hardness and delta_auprc for different k (nearest neighbors)
    protein_hardness_diff_k = {}
    corr_list = []
    k = [int(item * distance_matrix.shape[0]) for item in proportions]
    for item in k:
        hardness_protien = compute_task_hardness_from_distance_matrix(distance_matrix, k=item)
        hardness_protein_norm = (
            hardness_protien[0].numpy() - np.min(hardness_protien[0].numpy())
        ) / (np.max(hardness_protien[0].numpy()) - np.min(hardness_protien[0].numpy()))
        protein_hardness_diff_k["k" + str(item)] = hardness_protein_norm
        protein_hardness_diff_k["assay"] = chembl_ids

    protein_hardness_diff_k_df = pd.DataFrame(protein_hardness_diff_k)
    z = pd.merge(df, protein_hardness_diff_k_df, on="assay")
    for item in k:
        corr_list.append(compute_correlation(z, "k" + str(item), metric))

    return corr_list


def extract_class_indices(labels: torch.Tensor, which_class: torch.Tensor) -> torch.Tensor:
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def compute_class_prototypes(
    support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
    """Compute the prototype for each class in the support set."""
    means = []
    for c in torch.unique(support_labels):
        # filter out feature vectors which have class c
        class_features = torch.index_select(
            support_features, 0, extract_class_indices(support_labels, c)
        )
        means.append(torch.mean(class_features, dim=0))
    return torch.stack(means)


def compute_prototype_datamol(task, transformer) -> torch.Tensor:
    support_smiles = [item.smiles for item in task.samples]
    support_features = torch.Tensor(np.array(transformer(support_smiles)))
    support_labels = torch.Tensor(np.array([item.bool_label for item in task.samples]))
    prototypes = compute_class_prototypes(support_features, support_labels)
    return prototypes


def compute_features(task, transformer) -> torch.Tensor:
    support_smiles = [item.smiles for item in task.samples]
    support_features = torch.Tensor(np.array(transformer(support_smiles)))
    return support_features


def compute_features_smiles_labels(
    task, transformer
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    support_smiles = np.array([item.smiles for item in task.samples])
    support_labels = torch.Tensor(np.array([item.bool_label for item in task.samples]))
    support_features = torch.Tensor(np.array(transformer(support_smiles)))
    return support_features, support_labels, support_smiles


def calculate_task_hardness_weight(
    chembl_ids: List, evaluated_resuts: Dict, method: str = "rf"
) -> torch.Tensor:
    """Computes the internal hardness of a tasks

    Actually  the output indicate how easy is the task based on different methods.
    The higher the value the easier the task.
    """
    assert method in ["rf", "knn" "scaffold"], "Method should be within valid methods"
    weights = []
    for chembl_id in chembl_ids:
        if method in ["rf", "knn"]:
            weights.append(evaluated_resuts[chembl_id][0].roc_auc)
        elif method == "scaffold":
            weights.append(1 - evaluated_resuts[chembl_id]["neg"])

    weights = torch.tensor(weights)
    return weights


def otdd_hardness(
    path_to_otdd, path_to_intra_hardness, k=10, train_tasks_weighted=False, weighting_method="rf"
) -> pd.DataFrame:
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
    with open(PATH_TO_OTDD, "rb") as f:
        data = pickle.load(f)

    ## data is a dictionary with following keys:
    print(data.keys())

    ## data['distnce_matrices] is a list. We will convert it to the pytorch tensor
    distance_matrix = torch.stack(data["distance_matrices"])  # shape: #TRAIN_TASK * TEST_TASKS
    distance_matrix_sorted, distance_matrix_indices = torch.sort(distance_matrix, dim=0)

    if train_tasks_weighted:
        ## We will weight the distance matrix by the hardness of the closest train tasks
        with open(path_to_intra_hardness, "rb") as f:
            train_tasks_hardness = pickle.load(f)
        weights = calculate_task_hardness_weight(
            data["train_chembl_ids"], train_tasks_hardness, method=weighting_method
        )
        weighted_distance_matrix = (1 - weights[distance_matrix_indices]) * distance_matrix_sorted
        results = torch.mean(weighted_distance_matrix[:k, :], dim=0)
    else:
        results = torch.mean(distance_matrix_sorted[:k, :], dim=0)
    ## Some entries can contain nan values. We will replace them with mean of the results.
    print("Numebr of NaN values in the hardness matrix: ", torch.isnan(results).sum().item())
    results = torch.nan_to_num(results, nan=torch.nanmean(results).item())

    hardness_df = pd.DataFrame({"hardness": results, "assay": data["test_chembl_ids"]})
    return hardness_df


def prototype_hardness(
    path_to_prototypes_distance,
    path_to_intra_hardness,
    k=10,
    train_tasks_weighted=False,
    weighting_method="rf",
) -> pd.DataFrame:
    """This function computes the hardness of the test tasks from distance matirix
    computed based on prototype distance. The hardeness can be weighted based on the internal hardness of the
    training tasks

    Args:
        path_to_prototypes_distance: Path to the prototype distance file (pickle file)
        path_to_intra_hardness: Path to the file containing hardness of the training tasks (pickle file)
        k: Number of nearest neighbors to consider for hardness calculation
        train_tasks_weighted: If True, the hardness of the test tasks will be weighted by the hardness of the closest train tasks.

    Returns:
        A dataframe containing the hardness of the test tasks
    """
    PATH_TO_PROTOTYPE_DISTANCE = path_to_prototypes_distance
    with open(PATH_TO_PROTOTYPE_DISTANCE, "rb") as f:
        data = pickle.load(f)

    ## data is a dictionary with following keys:
    print(data.keys())

    distance_matrix = data["distance_matrix"]  # shape: #TRAIN_TASK * TEST_TASKS
    distance_matrix_sorted, distance_matrix_indices = torch.sort(distance_matrix, dim=0)

    if train_tasks_weighted:
        ## We will weight the distance matrix by the hardness of the closest train tasks
        with open(path_to_intra_hardness, "rb") as f:
            train_tasks_hardness = pickle.load(f)
        weights = calculate_task_hardness_weight(
            data["train_task_name"], train_tasks_hardness, method=weighting_method
        )
        weighted_distance_matrix = (1 - weights[distance_matrix_indices]) * distance_matrix_sorted
        results = torch.mean(weighted_distance_matrix[:k, :], dim=0)
    else:
        results = torch.mean(distance_matrix_sorted[:k, :], dim=0)
    ## Some entries can contain nan values. We will replace them with mean of the results.
    print("Numebr of NaN values in the hardness matrix: ", torch.isnan(results).sum().item())
    results = torch.nan_to_num(results, nan=torch.nanmean(results).item())

    hardness_df = pd.DataFrame({"hardness": results, "assay": data["test_task_name"]})
    return hardness_df
