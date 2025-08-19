"""
Combined task distance computation for the THEMAP framework.

This module provides functionality to compute and manage distances between tasks,
supporting both chemical and protein space distances with various combination strategies.
"""

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.tasks import Tasks
from ..utils.distance_utils import get_configure
from .base import (
    DATASET_DISTANCE_METHODS,
    METADATA_DISTANCE_METHODS,
    AbstractTasksDistance,
)
from .molecule_distance import MoleculeDatasetDistance
from .protein_distance import ProteinDatasetDistance

# Configure logging
logger = logging.getLogger(__name__)


class TaskDistance(AbstractTasksDistance):
    """Class for computing and managing distances between tasks.

    This class handles the computation and storage of distances between tasks,
    supporting both dataset distances (molecules) and metadata distances (protein, etc.).
    It can compute distances directly from Tasks collections or work with pre-computed matrices.

    Args:
        tasks: Tasks collection for distance computation (optional)
        dataset_method: Distance method for datasets (molecules) (default: "euclidean")
        metadata_method: Distance method for metadata including protein (default: "euclidean")
        method: Default distance computation method (legacy, optional)
        source_task_ids: List of task IDs for source tasks (legacy, optional)
        target_task_ids: List of task IDs for target tasks (legacy, optional)
        external_dataset_matrix: Pre-computed dataset distance matrix (optional)
        external_metadata_matrices: Dict of pre-computed metadata distance matrices (optional)

        # Deprecated parameters for backward compatibility:
        molecule_method: Deprecated alias for dataset_method
        protein_method: Deprecated - protein is metadata, use metadata_method
        external_chemical_space: Deprecated alias for external_dataset_matrix
        external_protein_space: Deprecated - protein is metadata
    """

    def __init__(
        self,
        tasks: Optional[Tasks] = None,
        molecule_method: str = "euclidean",
        protein_method: str = "euclidean",
        metadata_method: str = "euclidean",
        method: Optional[str] = None,
        source_task_ids: Optional[List[str]] = None,
        target_task_ids: Optional[List[str]] = None,
        external_chemical_space: Optional[np.ndarray] = None,
        external_protein_space: Optional[np.ndarray] = None,
    ):
        # Initialize parent class
        super().__init__(
            tasks=tasks,
            molecule_method=molecule_method,
            protein_method=protein_method,
            metadata_method=metadata_method,
            method=method,
        )

        # Handle legacy mode - override parent setup if using legacy parameters
        if tasks is None and (source_task_ids is not None or target_task_ids is not None):
            self.source_task_ids = source_task_ids or []
            self.target_task_ids = target_task_ids or []
            self.source = None
            self.target = None

        self.external_chemical_space = external_chemical_space
        self.external_protein_space = external_protein_space

        # Storage for computed distances
        self.molecule_distances: Optional[Dict[str, Dict[str, float]]] = None
        self.protein_distances: Optional[Dict[str, Dict[str, float]]] = None
        self.combined_distances: Optional[Dict[str, Dict[str, float]]] = None

    def get_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute and return the default distance between tasks.

        Uses the combined distance if both molecule and protein data are available,
        otherwise uses molecule distance, then protein distance as fallback.

        Returns:
            Dictionary containing distance matrix between source and target tasks.
        """
        # Try to compute combined distance first
        if self.tasks is not None:
            try:
                if self.combined_distances is None:
                    self.compute_combined_distance()
                if self.combined_distances:
                    return self.combined_distances
            except Exception:
                pass

            # Fall back to molecule distance
            try:
                if self.molecule_distances is None:
                    self.compute_molecule_distance()
                if self.molecule_distances:
                    return self.molecule_distances
            except Exception:
                pass

            # Fall back to protein distance
            try:
                if self.protein_distances is None:
                    self.compute_protein_distance()
                if self.protein_distances:
                    return self.protein_distances
            except Exception:
                pass

        # If nothing worked, return empty dict
        return {}

    def get_hopts(self, data_type: str = "dataset") -> Optional[Dict[str, Any]]:
        """Get hyperparameters for distance computation.

        Args:
            data_type: Type of data ("dataset", "metadata")
                      Legacy: "molecule" (alias for "dataset"), "protein" (alias for "metadata")

        Returns:
            Dictionary of hyperparameters for the specified data type distance computation method.
        """
        if data_type in ["dataset", "molecule"]:
            return get_configure(self.dataset_method)
        elif data_type in ["metadata", "protein"]:
            return get_configure(self.metadata_method)
        else:
            raise ValueError(f"Unknown data type: {data_type}. Use 'dataset' or 'metadata'")

    def get_supported_methods(self, data_type: str) -> List[str]:
        """Get list of supported methods for a specific data type.

        Args:
            data_type: Type of data ("dataset", "metadata")
                      Legacy: "molecule" (alias for "dataset"), "protein" (alias for "metadata")

        Returns:
            List of supported method names for the data type
        """
        if data_type in ["dataset", "molecule"]:
            return DATASET_DISTANCE_METHODS.copy()
        elif data_type in ["metadata", "protein"]:
            return METADATA_DISTANCE_METHODS.copy()
        else:
            raise ValueError(f"Unknown data type: {data_type}. Use 'dataset' or 'metadata'")

    def __repr__(self) -> str:
        """Return a string representation of the TaskDistance instance.

        Returns:
            String containing the number of source and target tasks and the mode.
        """
        mode = "tasks" if self.tasks is not None else "legacy"
        num_computed = 0
        if self.molecule_distances:
            num_computed += 1
        if self.protein_distances:
            num_computed += 1
        if self.combined_distances:
            num_computed += 1

        return (
            f"TaskDistance(mode={mode}, source_tasks={len(self.source_task_ids)}, "
            f"target_tasks={len(self.target_task_ids)}, computed_distances={num_computed}, "
            f"dataset_method={self.dataset_method}, metadata_method={self.metadata_method})"
        )

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the distance matrix.

        Returns:
            Tuple containing (number of source tasks, number of target tasks).
        """
        return len(self.source_task_ids), len(self.target_task_ids)

    def compute_molecule_distance(
        self, method: Optional[str] = None, molecule_featurizer: str = "ecfp"
    ) -> Dict[str, Dict[str, float]]:
        """Compute distances between tasks using molecule data.

        Args:
            method: Distance computation method ('euclidean', 'cosine', or 'otdd').
                   If None, uses the molecule_method from initialization.
            molecule_featurizer: Molecular featurizer to use

        Returns:
            Dictionary containing molecule-based distances between tasks.
        """
        if self.tasks is None:
            raise ValueError("Tasks collection required for computing molecule distances")

        # Use provided method or fall back to instance method
        actual_method = method if method is not None else self.molecule_method

        # Use MoleculeDatasetDistance to compute distances
        mol_distance = MoleculeDatasetDistance(
            tasks=self.tasks,
            molecule_method=actual_method,
        )
        self.molecule_distances = mol_distance.get_distance()
        return self.molecule_distances

    def compute_protein_distance(
        self, method: Optional[str] = None, protein_featurizer: str = "esm2_t33_650M_UR50D"
    ) -> Dict[str, Dict[str, float]]:
        """Compute distances between tasks using protein data.

        Args:
            method: Distance computation method ('euclidean' or 'cosine').
                   If None, uses the protein_method from initialization.
            protein_featurizer: Protein featurizer to use

        Returns:
            Dictionary containing protein-based distances between tasks.
        """
        if self.tasks is None:
            raise ValueError("Tasks collection required for computing protein distances")

        # Use provided method or fall back to instance method
        actual_method = method if method is not None else self.protein_method

        # Use ProteinDatasetDistance to compute distances
        prot_distance = ProteinDatasetDistance(
            tasks=self.tasks,
            protein_method=actual_method,
        )
        self.protein_distances = prot_distance.get_distance()
        return self.protein_distances

    def compute_combined_distance(
        self,
        molecule_method: Optional[str] = None,
        protein_method: Optional[str] = None,
        combination_strategy: str = "average",
        molecule_weight: float = 0.5,
        protein_weight: float = 0.5,
        molecule_featurizer: str = "ecfp",
        protein_featurizer: str = "esm2_t33_650M_UR50D",
    ) -> Dict[str, Dict[str, float]]:
        """Compute combined distances using both molecule and protein data.

        Args:
            molecule_method: Method for molecule distance computation
            protein_method: Method for protein distance computation
            combination_strategy: How to combine distances ('average', 'weighted_average', 'min', 'max')
            molecule_weight: Weight for molecule distances (used with 'weighted_average')
            protein_weight: Weight for protein distances (used with 'weighted_average')
            molecule_featurizer: Molecular featurizer to use
            protein_featurizer: Protein featurizer to use

        Returns:
            Dictionary containing combined distances between tasks.
        """
        # Use provided methods or fall back to instance methods
        actual_molecule_method = molecule_method if molecule_method is not None else self.molecule_method
        actual_protein_method = protein_method if protein_method is not None else self.protein_method

        # Compute individual distances if not already computed
        if self.molecule_distances is None:
            self.compute_molecule_distance(
                method=actual_molecule_method, molecule_featurizer=molecule_featurizer
            )
        if self.protein_distances is None:
            self.compute_protein_distance(method=actual_protein_method, protein_featurizer=protein_featurizer)

        if self.molecule_distances is None or self.protein_distances is None:
            raise ValueError("Could not compute both molecule and protein distances")

        # Combine distances
        self.combined_distances = {}

        # Get all target task IDs present in both distance matrices
        mol_target_ids = set(self.molecule_distances.keys())
        prot_target_ids = set(self.protein_distances.keys())
        common_target_ids = mol_target_ids.intersection(prot_target_ids)

        for target_id in common_target_ids:
            self.combined_distances[target_id] = {}

            # Get all source task IDs present in both distance matrices for this target
            mol_source_ids = set(self.molecule_distances[target_id].keys())
            prot_source_ids = set(self.protein_distances[target_id].keys())
            common_source_ids = mol_source_ids.intersection(prot_source_ids)

            for source_id in common_source_ids:
                mol_dist = self.molecule_distances[target_id][source_id]
                prot_dist = self.protein_distances[target_id][source_id]

                if combination_strategy == "average":
                    combined_dist = (mol_dist + prot_dist) / 2.0
                elif combination_strategy == "weighted_average":
                    # Normalize weights
                    total_weight = molecule_weight + protein_weight
                    if total_weight == 0:
                        combined_dist = (mol_dist + prot_dist) / 2.0
                    else:
                        combined_dist = (
                            mol_dist * molecule_weight + prot_dist * protein_weight
                        ) / total_weight
                elif combination_strategy == "min":
                    combined_dist = min(mol_dist, prot_dist)
                elif combination_strategy == "max":
                    combined_dist = max(mol_dist, prot_dist)
                else:
                    raise ValueError(f"Unknown combination strategy: {combination_strategy}")

                self.combined_distances[target_id][source_id] = combined_dist

        return self.combined_distances

    def compute_all_distances(
        self,
        molecule_method: Optional[str] = None,
        protein_method: Optional[str] = None,
        combination_strategy: str = "average",
        molecule_weight: float = 0.5,
        protein_weight: float = 0.5,
        molecule_featurizer: str = "ecfp",
        protein_featurizer: str = "esm2_t33_650M_UR50D",
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute all distance types (molecule, protein, and combined).

        Args:
            molecule_method: Method for molecule distance computation
            protein_method: Method for protein distance computation
            combination_strategy: How to combine distances
            molecule_weight: Weight for molecule distances
            protein_weight: Weight for protein distances
            molecule_featurizer: Molecular featurizer to use
            protein_featurizer: Protein featurizer to use

        Returns:
            Dictionary with keys 'molecule', 'protein', 'combined' containing respective distance matrices.
        """
        # Use provided methods or fall back to instance methods
        actual_molecule_method = molecule_method if molecule_method is not None else self.molecule_method
        actual_protein_method = protein_method if protein_method is not None else self.protein_method

        results = {}

        # Compute molecule distances
        try:
            results["molecule"] = self.compute_molecule_distance(
                method=actual_molecule_method, molecule_featurizer=molecule_featurizer
            )
        except Exception as e:
            print(f"Warning: Could not compute molecule distances: {e}")
            results["molecule"] = {}

        # Compute protein distances
        try:
            results["protein"] = self.compute_protein_distance(
                method=actual_protein_method, protein_featurizer=protein_featurizer
            )
        except Exception as e:
            print(f"Warning: Could not compute protein distances: {e}")
            results["protein"] = {}

        # Compute combined distances if both are available
        if self.molecule_distances and self.protein_distances:
            try:
                results["combined"] = self.compute_combined_distance(
                    molecule_method=actual_molecule_method,
                    protein_method=actual_protein_method,
                    combination_strategy=combination_strategy,
                    molecule_weight=molecule_weight,
                    protein_weight=protein_weight,
                    molecule_featurizer=molecule_featurizer,
                    protein_featurizer=protein_featurizer,
                )
            except Exception as e:
                print(f"Warning: Could not compute combined distances: {e}")
                results["combined"] = {}
        else:
            results["combined"] = {}

        return results

    def compute_ext_chem_distance(self, method: str) -> Dict[str, Dict[str, float]]:
        """Compute chemical space distances between tasks using external matrices.

        Args:
            method: Distance computation method to use

        Returns:
            Dictionary containing chemical space distances between tasks.

        Raises:
            NotImplementedError: If external chemical space is not provided
        """
        if self.external_chemical_space is None:
            raise NotImplementedError(
                "External chemical space matrix not provided. "
                "Use compute_molecule_distance() for direct computation."
            )

        # Convert external matrix to expected format
        result: Dict[str, Dict[str, float]] = {}
        for i, target_id in enumerate(self.target_task_ids):
            result[target_id] = {}
            for j, source_id in enumerate(self.source_task_ids):
                result[target_id][source_id] = float(self.external_chemical_space[i, j])

        return result

    def compute_ext_prot_distance(self, method: str) -> Dict[str, Dict[str, float]]:
        """Compute protein space distances between tasks using external matrices.

        Args:
            method: Distance computation method to use

        Returns:
            Dictionary containing protein space distances between tasks.

        Raises:
            NotImplementedError: If external protein space is not provided
        """
        if self.external_protein_space is None:
            raise NotImplementedError(
                "External protein space matrix not provided. "
                "Use compute_protein_distance() for direct computation."
            )

        # Convert external matrix to expected format
        result: Dict[str, Dict[str, float]] = {}
        for i, target_id in enumerate(self.target_task_ids):
            result[target_id] = {}
            for j, source_id in enumerate(self.source_task_ids):
                result[target_id][source_id] = float(self.external_protein_space[i, j])

        return result

    @staticmethod
    def load_ext_chem_distance(path: str) -> "TaskDistance":
        """Load pre-computed chemical space distances from a file.

        Args:
            path: Path to the file containing pre-computed chemical space distances

        Returns:
            TaskDistance instance initialized with the loaded distances.

        Note:
            The file should contain a dictionary with keys:
            - 'train_chembl_ids' or 'train_pubchem_ids' or 'source_task_ids'
            - 'test_chembl_ids' or 'test_pubchem_ids' or 'target_task_ids'
            - 'distance_matrices'
        """
        with open(path, "rb") as f:
            x = pickle.load(f)

        if "train_chembl_ids" in x.keys():
            source_task_ids = x["train_chembl_ids"]
        elif "train_pubchem_ids" in x.keys():
            source_task_ids = x["train_pubchem_ids"]
        elif "source_task_ids" in x.keys():
            source_task_ids = x["source_task_ids"]
        else:
            raise ValueError("No source task IDs found in the loaded file")

        if "test_chembl_ids" in x.keys():
            target_task_ids = x["test_chembl_ids"]
        elif "test_pubchem_ids" in x.keys():
            target_task_ids = x["test_pubchem_ids"]
        elif "target_task_ids" in x.keys():
            target_task_ids = x["target_task_ids"]
        else:
            raise ValueError("No target task IDs found in the loaded file")

        return TaskDistance(
            tasks=None,
            source_task_ids=source_task_ids,
            target_task_ids=target_task_ids,
            external_chemical_space=x["distance_matrices"],
        )

    @staticmethod
    def load_ext_prot_distance(path: str) -> "TaskDistance":
        """Load pre-computed protein space distances from a file.

        Args:
            path: Path to the file containing pre-computed protein space distances

        Returns:
            TaskDistance instance initialized with the loaded distances.

        Note:
            The file should contain a dictionary with keys:
            - 'train_chembl_ids' or 'train_pubchem_ids' or 'source_task_ids'
            - 'test_chembl_ids' or 'test_pubchem_ids' or 'target_task_ids'
            - 'distance_matrices'
        """
        with open(path, "rb") as f:
            x = pickle.load(f)

        if "train_chembl_ids" in x.keys():
            source_task_ids = x["train_chembl_ids"]
        elif "train_pubchem_ids" in x.keys():
            source_task_ids = x["train_pubchem_ids"]
        elif "source_task_ids" in x.keys():
            source_task_ids = x["source_task_ids"]
        else:
            raise ValueError("No source task IDs found in the loaded file")

        if "test_chembl_ids" in x.keys():
            target_task_ids = x["test_chembl_ids"]
        elif "test_pubchem_ids" in x.keys():
            target_task_ids = x["test_pubchem_ids"]
        elif "target_task_ids" in x.keys():
            target_task_ids = x["target_task_ids"]
        else:
            raise ValueError("No target task IDs found in the loaded file")

        return TaskDistance(
            tasks=None,
            source_task_ids=source_task_ids,
            target_task_ids=target_task_ids,
            external_protein_space=x["distance_matrices"],
        )

    def get_computed_distance(self, distance_type: str = "combined") -> Optional[Dict[str, Dict[str, float]]]:
        """Get computed distances of the specified type.

        Args:
            distance_type: Type of distance to return ('molecule', 'protein', 'combined')

        Returns:
            Dictionary containing the requested distances, or None if not computed.
        """
        if distance_type == "molecule":
            return self.molecule_distances
        elif distance_type == "protein":
            return self.protein_distances
        elif distance_type == "combined":
            return self.combined_distances
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")

    def to_pandas(self, distance_type: str = "combined") -> pd.DataFrame:
        """Convert distance matrix to a pandas DataFrame.

        Args:
            distance_type: Type of distance to convert ('molecule', 'protein', 'combined', 'external_chemical')

        Returns:
            DataFrame with source task IDs as index and target task IDs as columns,
            containing the distance values.

        Raises:
            ValueError: If no distances of the specified type are available
        """
        if distance_type == "external_chemical":
            if self.external_chemical_space is None:
                raise ValueError("No external chemical space distances available")
            df = pd.DataFrame(
                self.external_chemical_space, index=self.source_task_ids, columns=self.target_task_ids
            )
            return df
        elif distance_type == "external_protein":
            if self.external_protein_space is None:
                raise ValueError("No external protein space distances available")
            df = pd.DataFrame(
                self.external_protein_space, index=self.source_task_ids, columns=self.target_task_ids
            )
            return df
        else:
            distances = self.get_computed_distance(distance_type)
            if distances is None:
                raise ValueError(f"No {distance_type} distances available")
            return pd.DataFrame(distances)
