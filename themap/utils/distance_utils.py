"""Utility functions for computing various types of distances in the THEMAP framework.

This module provides functions for computing:
- Molecular fingerprint similarities
- Task hardness metrics
- Protein sequence distances
- Dataset distances
- Correlation analysis between different distance metrics

The module includes both low-level distance computation functions and high-level
analysis tools for understanding dataset relationships and task difficulty.
"""

import heapq
import json
import logging
import multiprocessing
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.spatial import distance
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_N_JOBS = min(8, multiprocessing.cpu_count())
MAX_FILE_SIZE_MB = 100


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize an array to the range [0, 1].

    Args:
        x: Input array to normalize

    Returns:
        Normalized array with values between 0 and 1
    """
    normalized: np.ndarray = (x - x.min()) / (x.max() - x.min())
    return normalized


def compute_fp_similarity(first_mol: Any, second_mol: Any) -> np.ndarray:
    """Compute similarity between molecules using their fingerprints.

    This function receives two molecule objects with get_fingerprint methods,
    extracts their fingerprints and computes the similarity using Jaccard distance.

    Args:
        first_mol: First molecule object with get_fingerprint() method
        second_mol: Second molecule object with get_fingerprint() method

    Returns:
        Array containing the similarity between the two molecules' fingerprints

    Raises:
        TypeError: If first_mol or second_mol don't have get_fingerprint method
        ValueError: If fingerprints are empty or incompatible shapes
    """
    # Input validation
    if not hasattr(first_mol, "get_fingerprint") or not callable(getattr(first_mol, "get_fingerprint")):
        raise TypeError("First molecule object doesn't have a get_fingerprint method")
    if not hasattr(second_mol, "get_fingerprint") or not callable(getattr(second_mol, "get_fingerprint")):
        raise TypeError("Second molecule object doesn't have a get_fingerprint method")

    # Get fingerprints
    fp1 = first_mol.get_fingerprint()
    fp2 = second_mol.get_fingerprint()

    # Validate fingerprints
    if fp1 is None or fp2 is None:
        raise ValueError("One or both fingerprints are None")

    if len(fp1) == 0 or len(fp2) == 0:
        raise ValueError("One or both fingerprints are empty")

    # Calculate similarity
    sim = 1 - distance.cdist(fp1, fp2, metric="jaccard")
    sim_array: np.ndarray = sim.astype(np.float32)
    return sim_array


def compute_fps_similarity(first_mol_list: List[Any], second_mol_list: List[Any]) -> np.ndarray:
    """Compute similarity between two lists of molecules using their fingerprints.

    This function receives two lists of molecule objects with get_fingerprint methods,
    extracts their fingerprints and computes similarities using Jaccard distance.

    Args:
        first_mol_list: List of molecule objects with get_fingerprint() method
        second_mol_list: List of molecule objects with get_fingerprint() method

    Returns:
        Matrix of similarities between the two lists of molecules
    """
    fps1 = [mol.get_fingerprint() for mol in first_mol_list]  # assumed train set
    fps2 = [mol.get_fingerprint() for mol in second_mol_list]  # assumed test set
    sims = 1 - distance.cdist(fps1, fps2, metric="jaccard")
    sims_array: np.ndarray = sims.astype(np.float32)
    return sims_array


def compute_similarities_mean_nearest(mol_list1: List[Any], mol_list2: List[Any]) -> float:
    """Compute mean similarity between nearest neighbors of two molecule lists.

    This function computes the similarities between two lists of molecules and
    returns the mean of the maximum similarities for each molecule in the first list.

    Args:
        mol_list1: First list of molecule objects with get_fingerprint() method
        mol_list2: Second list of molecule objects with get_fingerprint() method

    Returns:
        Mean similarity between nearest neighbors
    """
    result = compute_fps_similarity(mol_list1, mol_list2).max(axis=1).mean()
    return float(result)


def similar_dissimilar_indices(
    similarity_matrix: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute indices of similar and dissimilar pairs of molecules.

    Args:
        similarity_matrix: Matrix of similarities between molecules
        threshold: Threshold value for determining similarity

    Returns:
        Tuple containing:
        - Array of indices for similar pairs
        - Array of indices for dissimilar pairs
    """
    similar_indices = [sim_col.max(axis=0) >= threshold for sim_col in similarity_matrix.T]
    dissimilar_indices = [np.logical_not(ind) for ind in similar_indices]

    similar_indices_arr = np.where(similar_indices)[0]
    dissimilar_indices_arr = np.where(dissimilar_indices)[0]

    return similar_indices_arr, dissimilar_indices_arr


def inter_distance(test_tasks: List[Any], train_tasks: List[Any]) -> List[float]:
    """Compute inter-task distances between test and train tasks.

    This function computes the mean similarity between nearest neighbors for each
    pair of test and train tasks in parallel.

    Args:
        test_tasks: List of test tasks
        train_tasks: List of train tasks

    Returns:
        List of inter-task distances
    """
    n_jobs = min(DEFAULT_N_JOBS, multiprocessing.cpu_count())
    inter_dist = Parallel(n_jobs=n_jobs)(
        delayed(compute_similarities_mean_nearest)(test_tasks[i].samples, train_tasks[j].samples)
        for i in tqdm(range(len(test_tasks)))
        for j in range(len(train_tasks))
    )
    return cast(List[float], inter_dist)


def intra_distance(tasks_pos: List[Any], tasks_neg: List[Any]) -> List[np.ndarray]:
    """Compute intra-task distances between positive and negative samples.

    This function computes the similarities between positive and negative samples
    within each task in parallel.

    Args:
        tasks_pos: List of tasks containing positive samples
        tasks_neg: List of tasks containing negative samples

    Returns:
        List of similarity matrices between positive and negative samples
    """
    n_jobs = min(DEFAULT_N_JOBS // 2, multiprocessing.cpu_count())
    intra_dist = Parallel(n_jobs=n_jobs)(
        delayed(compute_fps_similarity)(tasks_pos[i].samples, tasks_neg[i].samples)
        for i in tqdm(range(len(tasks_pos)))
    )
    return cast(List[np.ndarray], intra_dist)


def compute_task_hardness_from_distance_matrix(
    distance_matrix: torch.Tensor, proportion: float = 0.01, aggr: str = "mean"
) -> List[torch.Tensor]:
    """Compute task hardness from a distance matrix.

    This function computes the hardness of each task by considering the distances
    to the nearest neighbors. The hardness is computed by taking the mean or median
    of the k nearest neighbors, where k is determined by the proportion parameter.

    Args:
        distance_matrix: [N_train * N_test] tensor with pairwise distances
        proportion: Proportion of training tasks to consider for hardness calculation
        aggr: Aggregation method ('mean', 'median', or 'both')

    Returns:
        List of tensors containing task hardness values. If aggr is 'mean' or 'median',
        returns a single tensor. If 'both', returns [mean_tensor, median_tensor]

    Raises:
        ValueError: If training set is not larger than test set
    """
    if distance_matrix.shape[0] <= distance_matrix.shape[1]:
        raise ValueError(
            f"Training set tasks ({distance_matrix.shape[0]}) should be larger than "
            f"test set tasks ({distance_matrix.shape[1]})"
        )

    # Sort the distance matrix along the test dimension
    sorted_distance_matrix = torch.sort(distance_matrix, dim=0)[0]

    # Determine k based on proportion
    k: int = int(proportion * distance_matrix.shape[0]) if proportion < 1 else int(proportion)

    results: List[torch.Tensor] = []
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


def compute_task_hardness_molecule_intra(distance_list: List[np.ndarray]) -> List[float]:
    """Compute task hardness from intra-task distances.

    This function computes the hardness of each task based on the similarities
    between positive and negative samples within the task.

    Args:
        distance_list: List of N_pos*N_neg arrays containing Tanimoto similarities
                      between positive and negative samples

    Returns:
        List of task hardness values (higher values indicate harder tasks)
    """
    task_hardness = [float(item.max(axis=1).mean()) for item in distance_list]
    return task_hardness


def compute_task_hardness_molecule_inter(
    distance_list: List[float], test_size: int = 157, train_size: int = 4938, topk: int = 100
) -> List[float]:
    """Compute task hardness from inter-task distances.

    This function computes the hardness of each task based on the similarities
    between test and train tasks.

    Args:
        distance_list: List of inter-task distances
        test_size: Number of test tasks
        train_size: Number of train tasks
        topk: Number of nearest neighbors to consider

    Returns:
        List of task hardness values (higher values indicate harder tasks)
    """
    task_hardness: List[float] = []
    for i in range(test_size):
        start_idx = i * train_size
        end_idx = (i + 1) * train_size
        task_distances = distance_list[start_idx:end_idx]
        task_hardness.append(float(np.mean(heapq.nlargest(topk, task_distances))))
    return task_hardness


def compute_correlation(
    task_df_with_perf: pd.DataFrame, col1: str, col2: str, method: str = "pearson"
) -> float:
    """Compute correlation between two columns in a DataFrame.

    Args:
        task_df_with_perf: DataFrame containing task performance metrics
        col1: Name of first column
        col2: Name of second column
        method: Correlation method ('pearson', 'spearman', or 'kendall')

    Returns:
        Correlation coefficient between the two columns
    """
    return float(task_df_with_perf[col1].corr(task_df_with_perf[col2], method=method))


def corr_protein_hardness_metric(
    df: pd.DataFrame,
    chembl_ids: List[str],
    distance_matrix: torch.Tensor,
    proportions: List[float] = [0.01, 0.1, 0.5, 0.9],
    metric: str = "delta_auprc",
) -> List[float]:
    """Compute correlation between protein hardness and performance metric.

    This function computes the correlation between protein hardness (computed using
    different proportions of nearest neighbors) and a performance metric.

    Args:
        df: DataFrame containing task performance metrics
        chembl_ids: List of ChEMBL IDs for tasks
        distance_matrix: Matrix of protein distances
        proportions: List of proportions to use for hardness calculation
        metric: Performance metric to correlate with hardness

    Returns:
        List of correlation coefficients for each proportion
    """
    protein_hardness_diff_k: Dict[str, Union[np.ndarray, List[str]]] = {}
    corr_list: List[float] = []
    k = [int(item * distance_matrix.shape[0]) for item in proportions]

    for item in k:
        hardness_protein = compute_task_hardness_from_distance_matrix(distance_matrix, proportion=item)
        hardness_protein_norm = normalize(hardness_protein[0].numpy())
        protein_hardness_diff_k[f"k{item}"] = hardness_protein_norm
        protein_hardness_diff_k["assay"] = chembl_ids

    protein_hardness_diff_k_df = pd.DataFrame(protein_hardness_diff_k)
    z = pd.merge(df, protein_hardness_diff_k_df, on="assay")

    for item in k:
        corr_list.append(compute_correlation(z, f"k{item}", metric))

    return corr_list


def extract_class_indices(labels: torch.Tensor, which_class: torch.Tensor) -> torch.Tensor:
    """Extract indices of samples belonging to a specific class.

    Args:
        labels: Tensor of class labels
        which_class: Class label to find indices for

    Returns:
        Tensor containing indices of samples with the specified class label
    """
    class_mask = torch.eq(labels, which_class)
    class_mask_indices = torch.nonzero(class_mask)
    return torch.reshape(class_mask_indices, (-1,))


def compute_class_prototypes(support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
    """Compute prototype vectors for each class in the support set.

    Args:
        support_features: Feature vectors of support set samples
        support_labels: Labels of support set samples

    Returns:
        Tensor containing prototype vectors for each class
    """
    means: List[torch.Tensor] = []
    for c in torch.unique(support_labels):
        class_features = torch.index_select(support_features, 0, extract_class_indices(support_labels, c))
        means.append(torch.mean(class_features, dim=0))
    return torch.stack(means)


def compute_prototype_datamol(task: Any, transformer: Any) -> torch.Tensor:
    """Compute prototype vectors for a molecule dataset.

    Args:
        task: Molecule dataset task
        transformer: Feature transformer for SMILES strings

    Returns:
        Tensor containing prototype vectors for each class
    """
    support_smiles = [item.smiles for item in task.samples]
    support_features = torch.Tensor(np.array(transformer(support_smiles)))
    support_labels = torch.Tensor(np.array([item.bool_label for item in task.samples]))
    prototypes = compute_class_prototypes(support_features, support_labels)
    return prototypes


def compute_features(task: Any, transformer: Any) -> torch.Tensor:
    """Compute feature vectors for a molecule dataset.

    Args:
        task: Molecule dataset task
        transformer: Feature transformer for SMILES strings

    Returns:
        Tensor containing feature vectors for all samples
    """
    support_smiles = [item.smiles for item in task.samples]
    support_features = torch.Tensor(np.array(transformer(support_smiles)))
    return support_features


def compute_features_smiles_labels(
    task: Any, transformer: Any
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Compute features, labels, and SMILES strings for a molecule dataset.

    Args:
        task: Molecule dataset task
        transformer: Feature transformer for SMILES strings

    Returns:
        Tuple containing:
        - Feature vectors tensor
        - Labels tensor
        - SMILES strings array
    """
    support_smiles = np.array([item.smiles for item in task.samples])
    support_labels = torch.Tensor(np.array([item.bool_label for item in task.samples]))
    support_features = torch.Tensor(np.array(transformer(support_smiles)))
    return support_features, support_labels, support_smiles


def calculate_task_hardness_weight(
    chembl_ids: List[str], evaluated_results: Dict[str, Dict[str, float]], method: str = "rf"
) -> torch.Tensor:
    """Calculate task hardness weights based on different methods.

    Args:
        chembl_ids: List of ChEMBL IDs for the tasks
        evaluated_results: Dictionary containing task performance metrics
        method: Method to use for calculating hardness ('rf', 'knn', or 'scaffold')

    Returns:
        Tensor containing task hardness weights (higher values indicate easier tasks)

    Raises:
        ValueError: If method is not one of the supported methods
    """
    if method not in ["rf", "knn", "scaffold"]:
        raise ValueError(f"Method must be one of ['rf', 'knn', 'scaffold'], got '{method}'")

    weights: List[float] = []
    for chembl_id in chembl_ids:
        if method in ["rf", "knn"]:
            weights.append(evaluated_results[chembl_id]["roc_auc"])
        elif method == "scaffold":
            weights.append(1 - evaluated_results[chembl_id]["neg"])

    return torch.tensor(weights)


def otdd_hardness(
    path_to_otdd: str,
    path_to_intra_hardness: str,
    k: int = 10,
    train_tasks_weighted: bool = False,
    weighting_method: str = "rf",
) -> pd.DataFrame:
    """Compute task hardness using OTDD distances.

    This function computes the hardness of test tasks based on their distances
    to training tasks, optionally weighted by the internal hardness of training tasks.

    Args:
        path_to_otdd: Path to OTDD distance matrix file
        path_to_intra_hardness: Path to training task hardness file
        k: Number of nearest neighbors to consider
        train_tasks_weighted: Whether to weight distances by training task hardness
        weighting_method: Method to use for weighting ('rf', 'knn', or 'scaffold')

    Returns:
        DataFrame containing task hardness values and assay IDs
    """
    # Validate file size before loading
    file_size_mb = os.path.getsize(path_to_otdd) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"Large file detected: {file_size_mb:.1f}MB. This may consume significant memory.")

    try:
        with open(path_to_otdd, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)
        logger.info(f"Loaded OTDD data with keys: {list(data.keys())}")
    except Exception as e:
        logger.error(f"Failed to load OTDD data from {path_to_otdd}: {e}")
        raise

    if isinstance(data["distance_matrices"], list):
        distance_matrix = torch.stack(data["distance_matrices"])
    else:
        distance_matrix = data["distance_matrices"]

    distance_matrix_sorted, distance_matrix_indices = torch.sort(distance_matrix, dim=0)

    if train_tasks_weighted:
        with open(path_to_intra_hardness, "rb") as f:
            train_tasks_hardness: Dict[str, Dict[str, float]] = pickle.load(f)
        weights = calculate_task_hardness_weight(
            data["train_chembl_ids"], train_tasks_hardness, method=weighting_method
        )
        weighted_distance_matrix = (1 - weights[distance_matrix_indices]) * distance_matrix_sorted
        results = torch.mean(weighted_distance_matrix[:k, :], dim=0)
    else:
        results = torch.mean(distance_matrix_sorted[:k, :], dim=0)

    nan_count = torch.isnan(results).sum().item()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in hardness matrix, replacing with mean")
    results = torch.nan_to_num(results, nan=torch.nanmean(results).item())

    hardness_df = pd.DataFrame({"hardness": results, "assay": data["test_chembl_ids"]})
    return hardness_df


def prototype_hardness(
    path_to_prototypes_distance: str,
    path_to_intra_hardness: str,
    k: int = 10,
    train_tasks_weighted: bool = False,
    weighting_method: str = "rf",
) -> pd.DataFrame:
    """Compute task hardness using prototype distances.

    This function computes the hardness of test tasks based on their distances
    to training task prototypes, optionally weighted by training task hardness.

    Args:
        path_to_prototypes_distance: Path to prototype distance matrix file
        path_to_intra_hardness: Path to training task hardness file
        k: Number of nearest neighbors to consider
        train_tasks_weighted: Whether to weight distances by training task hardness
        weighting_method: Method to use for weighting ('rf', 'knn', or 'scaffold')

    Returns:
        DataFrame containing task hardness values and assay IDs
    """
    # Validate file size before loading
    file_size_mb = os.path.getsize(path_to_prototypes_distance) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"Large file detected: {file_size_mb:.1f}MB. This may consume significant memory.")

    try:
        with open(path_to_prototypes_distance, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)
        logger.info(f"Loaded prototype data with keys: {list(data.keys())}")
    except Exception as e:
        logger.error(f"Failed to load prototype data from {path_to_prototypes_distance}: {e}")
        raise

    distance_matrix = data["distance_matrix"]
    distance_matrix_sorted, distance_matrix_indices = torch.sort(distance_matrix, dim=0)

    if train_tasks_weighted:
        with open(path_to_intra_hardness, "rb") as f:
            train_tasks_hardness: Dict[str, Dict[str, float]] = pickle.load(f)
        weights = calculate_task_hardness_weight(
            data["train_task_name"], train_tasks_hardness, method=weighting_method
        )
        weighted_distance_matrix = (1 - weights[distance_matrix_indices]) * distance_matrix_sorted
        results = torch.mean(weighted_distance_matrix[:k, :], dim=0)
    else:
        results = torch.mean(distance_matrix_sorted[:k, :], dim=0)

    nan_count = torch.isnan(results).sum().item()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in hardness matrix, replacing with mean")
    results = torch.nan_to_num(results, nan=torch.nanmean(results).item())

    hardness_df = pd.DataFrame({"hardness": results, "assay": data["test_task_name"]})
    return hardness_df


def internal_hardness(hardness_df: pd.DataFrame, internal_hardness_path: str) -> pd.DataFrame:
    """Add internal hardness to a hardness DataFrame.

    Args:
        hardness_df: DataFrame containing task hardness values
        internal_hardness_path: Path to save internal hardness values

    Returns:
        DataFrame containing combined hardness values
    """
    with open(internal_hardness_path, "rb") as f:
        test_tasks_hardness: Dict[str, Dict[str, float]] = pickle.load(f)

    weights: List[float] = []
    for chembl_id in hardness_df["assay"]:
        weights.append(test_tasks_hardness[chembl_id]["roc_auc"])

    weights_tensor = torch.tensor(weights)
    hardness_df["internal_hardness"] = 1 - weights_tensor

    return hardness_df


def protein_hardness_from_distance_matrix(path: str, k: int) -> pd.DataFrame:
    """Compute protein hardness from a distance matrix.

    Args:
        path: Path to protein distance matrix file
        k: Number of nearest neighbors to consider

    Returns:
        DataFrame containing protein hardness values and metrics

    Raises:
        KeyError: If required keys are missing from the distance matrix file
    """
    with open(path, "rb") as f:
        protein_distance_matrix: Dict[str, Any] = pickle.load(f)

    if "distance_matrices" not in protein_distance_matrix.keys():
        raise KeyError("'distance_matrices' key should be present in the dictionary")
    if "test_chembl_ids" not in protein_distance_matrix.keys():
        raise KeyError("'test_chembl_ids' key should be present in the dictionary")

    hardness_protein = compute_task_hardness_from_distance_matrix(
        protein_distance_matrix["distance_matrices"], aggr="mean_median", proportion=k
    )

    hardness_protein_mean_norm = normalize(hardness_protein[0].numpy())
    hardness_protein_median_norm = normalize(hardness_protein[1].numpy())

    protein_hardness_df = pd.DataFrame(
        {
            "protein_hardness_mean": hardness_protein[0],
            "protein_hardness_median": hardness_protein[1],
            "protein_hardness_mean_norm": hardness_protein_mean_norm,
            "protein_hardness_median_norm": hardness_protein_median_norm,
            "assay": protein_distance_matrix["test_chembl_ids"],
        }
    )

    return protein_hardness_df


def get_configure(distance: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a distance metric.

    Args:
        distance: Name of the distance metric

    Returns:
        Dictionary containing distance metric configuration, or None if not found

    Raises:
        ValueError: If distance name contains invalid characters
    """
    # Validate distance name to prevent path traversal
    if not distance.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid distance name: {distance}")

    # Use pathlib for safer path construction
    source_path = Path(__file__).parent.parent
    config_path = source_path / "models" / "distance_configures" / f"{distance}.json"

    # Ensure the path is within the expected directory
    try:
        config_path.resolve().relative_to(source_path.resolve())
    except ValueError:
        logger.error(f"Attempted path traversal with distance name: {distance}")
        return None

    if not config_path.exists():
        logger.warning(f"Configuration file not found for distance: {distance}")
        return None

    try:
        with open(config_path, "r") as f:
            config: Dict[str, Any] = json.load(f)
        logger.debug(f"Loaded configuration for distance: {distance}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration for {distance}: {e}")
        return None
