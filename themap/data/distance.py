"""
Module for calculating distances between datasets in the THEMAP framework.

This module provides functionality to compute various types of distances between:
- Molecule datasets (using OTDD, Euclidean, or cosine distances)
- Protein datasets (using Euclidean or cosine distances)
- Task distances (using external chemical or protein space)

The module supports both single dataset comparisons and batch comparisons
across multiple datasets.
"""

from typing import Any, Optional, Union
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pickle

from themap.utils.distance_utils import get_configure
from themap.data.tasks import MoleculeDataset, ProteinDataset, MoleculeDataloader
from otdd.pytorch.distance import DatasetDistance


MOLECULE_DISTANCE_METHODS = ["otdd", "euclidean", "cosine"]
PROTEIN_DISTANCE_METHODS = ["euclidean", "cosine"]


class AbstractDatasetDistance:
    """Base class for computing distances between datasets.
    
    This abstract class defines the interface for dataset distance computation.
    It provides a common structure for both molecule and protein dataset distances.
    
    Args:
        D1: First dataset for distance computation
        D2: Second dataset for distance computation (optional)
        method: Distance computation method to use
    """
    
    def __init__(
        self, 
        D1: Optional[Union[MoleculeDataset, ProteinDataset]] = None,
        D2: Optional[Union[MoleculeDataset, ProteinDataset]] = None,
        method: str = "euclidean"
    ):
        self.source = D1
        if D2 is None:
            self.target = self.source
            self.symmetric_tasks = True
        else:
            self.target = D2
        self.method = method

    def get_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute the distance between datasets.
        
        Returns:
            Dictionary containing distance matrix between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def get_hopts(self) -> Dict[str, Any]:
        """Get hyperparameters for distance computation.
        
        Returns:
            Dictionary of hyperparameters for the distance computation method.
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Dict[str, Dict[str, float]]:
        """Allow the class to be called as a function.
        
        Returns:
            The computed distance matrix.
        """
        return self.get_distance()


class MoleculeDatasetDistance(AbstractDatasetDistance):
    """Calculate distances between molecule datasets using various methods.
    
    This class implements distance computation between molecule datasets using:
    - Optimal Transport Dataset Distance (OTDD)
    - Euclidean distance
    - Cosine distance
    
    The class supports both single dataset comparisons and batch comparisons
    across multiple datasets.
    
    Args:
        D1: Single MoleculeDataset or list of MoleculeDatasets for source data
        D2: Single MoleculeDataset or list of MoleculeDatasets for target data
        method: Distance computation method ('otdd', 'euclidean', or 'cosine')
        **kwargs: Additional arguments passed to the distance computation method
        
    Raises:
        ValueError: If the specified method is not supported for molecule datasets
    """
    
    def __init__(
        self, 
        D1: Optional[Union[MoleculeDataset, List[MoleculeDataset]]] = None,
        D2: Optional[Union[MoleculeDataset, List[MoleculeDataset]]] = None,
        method: str = "euclidean",
        **kwargs: Any
    ):
        super().__init__(D1, D2, method)
        self.source = D1
        if D2 is None:
            self.target = self.source
            self.symmetric_tasks = True
        else:
            self.target = D2

        if not isinstance(self.source, list):
            self.source = [self.source]
        if not isinstance(self.target, list):
            self.target = [self.target]

        if method not in MOLECULE_DISTANCE_METHODS:
            raise ValueError(f"Method {method} not supported for molecule datasets. "
                           f"Supported methods are: {MOLECULE_DISTANCE_METHODS}")
        self.method = method

        self.source_task_ids = [d.task_id for d in self.source]
        self.target_task_ids = [d.task_id for d in self.target]
        self.distance: Optional[Dict[str, Dict[str, float]]] = None

    def get_hopts(self) -> Dict[str, Any]:
        """Get hyperparameters for the distance computation method.
        
        Returns:
            Dictionary of hyperparameters specific to the chosen distance method.
        """
        return get_configure(self.method)

    def otdd_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute Optimal Transport Dataset Distance between molecule datasets.
        
        This method uses the OTDD implementation to compute distances between
        molecule datasets, which takes into account both the feature space
        and label space of the datasets.
        
        Returns:
            Dictionary containing OTDD distances between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        chem_distances = {}
        hopts = self.get_hopts()
        loaders_src = [MoleculeDataloader(d) for d in self.source]
        loaders_tgt = [MoleculeDataloader(d) for d in self.target]
        for i, tgt in enumerate(loaders_tgt):
            chem_distance = {}
            for j, src in enumerate(loaders_src):
                dist = DatasetDistance(src, tgt, **hopts)
                d = dist.distance(maxsamples=1000)
                chem_distance[self.source_task_ids[j]] = d.cpu().item()
            chem_distances[self.target_task_ids[i]] = chem_distance
        return chem_distances

    def get_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute the distance between molecule datasets using the specified method.
        
        Returns:
            Dictionary containing distance matrix between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        if self.method == "otdd":
            self.distance = self.otdd_distance()
        else:
            self.distance = self.euclidean_distance()
        return self.distance

    def load_distance(self, path: str) -> None:
        """Load pre-computed distances from a file.
        
        Args:
            path: Path to the file containing pre-computed distances
            
        Note:
            This method is currently a placeholder and needs to be implemented.
        """
        pass

    def to_pandas(self) -> pd.DataFrame:
        """Convert the distance matrix to a pandas DataFrame.
        
        Returns:
            DataFrame with source task IDs as index and target task IDs as columns,
            containing the distance values.
        """
        return pd.DataFrame(self.distance)

    def __repr__(self) -> str:
        """Return a string representation of the MoleculeDatasetDistance instance.
        
        Returns:
            String containing the class name and initialization parameters.
        """
        return f"MoleculeDatasetDistance(D1={self.source}, D2={self.target}, method={self.method})"


class ProteinDatasetDistance(AbstractDatasetDistance):
    """Calculate distances between protein datasets using various methods.
    
    This class implements distance computation between protein datasets using:
    - Euclidean distance
    - Cosine distance
    
    The class supports both single dataset comparisons and batch comparisons
    across multiple datasets.
    
    Args:
        D1: ProteinDataset for source data
        D2: ProteinDataset for target data
        method: Distance computation method ('euclidean' or 'cosine')
        
    Raises:
        ValueError: If the specified method is not supported for protein datasets
    """
    
    def __init__(
        self, 
        D1: Optional[ProteinDataset] = None,
        D2: Optional[ProteinDataset] = None,
        method: str = "euclidean"
    ):
        super().__init__(D1, D2, method)
        self.source = D1
        if D2 is None:
            self.target = self.source
            self.symmetric_tasks = True
        else:
            self.target = D2
        self.method = method

        if method not in PROTEIN_DISTANCE_METHODS:
            raise ValueError(f"Method {method} not supported for protein datasets. "
                           f"Supported methods are: {PROTEIN_DISTANCE_METHODS}")

        self.source_task_ids = self.source.task_id
        self.target_task_ids = self.target.task_id
        self.distance: Optional[Dict[str, Dict[str, float]]] = None

    def get_hopts(self) -> Dict[str, Any]:
        """Get hyperparameters for the distance computation method.
        
        Returns:
            Dictionary of hyperparameters specific to the chosen distance method.
        """
        return get_configure(self.method)

    def euclidean_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute Euclidean distance between protein datasets.
        
        This method calculates the pairwise Euclidean distances between protein
        feature vectors in the datasets.
        
        Returns:
            Dictionary containing Euclidean distances between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        dist = cdist(self.target.features, self.source.features)
        prot_distances = {}
        for i, tgt in enumerate(self.target_task_ids):
            prot_distance = {}
            for j, src in enumerate(self.target.task_id):
                prot_distance[src] = dist[j, i]
            prot_distances[tgt] = prot_distance
        return prot_distances
    
    def cosine_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute cosine distance between protein datasets.
        
        This method calculates the pairwise cosine distances between protein
        feature vectors in the datasets.
        
        Returns:
            Dictionary containing cosine distances between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        dist = cdist(self.target.features, self.source.features, metric="cosine")
        prot_distances = {}
        for i, tgt in enumerate(self.target_task_ids):
            prot_distance = {}
            for j, src in enumerate(self.target.task_id):
                prot_distance[src] = dist[j, i]
            prot_distances[tgt] = prot_distance
        return prot_distances
    
    def sequence_identity_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute sequence identity-based distance between protein datasets.
        
        This method calculates distances based on protein sequence identity.
        Currently a placeholder for future implementation.
        
        Returns:
            Dictionary containing sequence identity-based distances between datasets.
            
        Note:
            This method is currently a placeholder and needs to be implemented.
        """
        pass

    def get_distance(self) -> Dict[str, Dict[str, float]]:
        """Compute the distance between protein datasets using the specified method.
        
        Returns:
            Dictionary containing distance matrix between source and target datasets.
            The outer dictionary is keyed by target task IDs, and the inner dictionary
            is keyed by source task IDs with distance values.
        """
        if self.method == "euclidean":
            self.distance = self.euclidean_distance()
        else:
            self.distance = self.cosine_distance()
        return self.distance

    def load_distance(self, path: str) -> None:
        """Load pre-computed distances from a file.
        
        Args:
            path: Path to the file containing pre-computed distances
            
        Note:
            This method is currently a placeholder and needs to be implemented.
        """
        pass

    def to_pandas(self) -> pd.DataFrame:
        """Convert the distance matrix to a pandas DataFrame.
        
        Returns:
            DataFrame with source task IDs as index and target task IDs as columns,
            containing the distance values.
        """
        return pd.DataFrame(self.distance)

    def __repr__(self) -> str:
        """Return a string representation of the ProteinDatasetDistance instance.
        
        Returns:
            String containing the class name and initialization parameters.
        """
        return f"ProteinDatasetDistance(D1={self.source}, D2={self.target}, method={self.method})"


class TaskDistance:
    """Class for computing and managing distances between tasks.
    
    This class handles the computation and storage of distances between tasks,
    supporting both chemical and protein space distances. It can work with
    pre-computed distance matrices or compute them on demand.
    
    Args:
        source_task_ids: List of task IDs for source tasks
        target_task_ids: List of task IDs for target tasks
        external_chemical_space: Pre-computed chemical space distance matrix (optional)
        external_protein_space: Pre-computed protein space distance matrix (optional)
    """
    
    def __init__(
        self,
        source_task_ids: List[str],
        target_task_ids: List[str],
        external_chemical_space: Optional[np.ndarray] = None,
        external_protein_space: Optional[np.ndarray] = None,
    ):
        self.source_task_ids = source_task_ids
        self.target_task_ids = target_task_ids
        self.external_chemical_space = external_chemical_space
        self.external_protein_space = external_protein_space

    def __repr__(self) -> str:
        """Return a string representation of the TaskDistance instance.
        
        Returns:
            String containing the number of source and target tasks.
        """
        return f"TaskDistance(source_task_ids={len(self.source_task_ids)}, target_task_ids={len(self.target_task_ids)})"

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the distance matrix.
        
        Returns:
            Tuple containing (number of source tasks, number of target tasks).
        """
        return len(self.source_task_ids), len(self.target_task_ids)

    def compute_ext_chem_distance(self, method: str) -> Dict[str, Dict[str, float]]:
        """Compute chemical space distances between tasks.
        
        Args:
            method: Distance computation method to use
            
        Returns:
            Dictionary containing chemical space distances between tasks.
            
        Note:
            This method is currently a placeholder and needs to be implemented.
        """
        pass

    def compute_ext_prot_distance(self, method: str) -> Dict[str, Dict[str, float]]:
        """Compute protein space distances between tasks.
        
        Args:
            method: Distance computation method to use
            
        Returns:
            Dictionary containing protein space distances between tasks.
            
        Note:
            This method is currently a placeholder and needs to be implemented.
        """
        pass

    @staticmethod
    def load_ext_chem_distance(path: str) -> 'TaskDistance':
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

        return TaskDistance(source_task_ids, target_task_ids, external_chemical_space=x["distance_matrices"])

    @staticmethod
    def load_ext_prot_distance(path: str) -> 'TaskDistance':
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

        return TaskDistance(source_task_ids, target_task_ids, external_protein_space=x["distance_matrices"])

    def to_pandas(self) -> pd.DataFrame:
        """Convert the chemical space distance matrix to a pandas DataFrame.
        
        Returns:
            DataFrame with source task IDs as index and target task IDs as columns,
            containing the chemical space distance values.
            
        Raises:
            ValueError: If no chemical space distances are available
        """
        if self.external_chemical_space is None:
            raise ValueError("No chemical space distances available")
        df = pd.DataFrame(
            self.external_chemical_space, index=self.source_task_ids, columns=self.target_task_ids
        )
        return df
