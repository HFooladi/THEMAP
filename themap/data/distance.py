from typing import Any
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
    def __init__(self, D1=None, D2=None, method="euclidean"):
        self.source = D1
        if D2 is None:
            self.target = self.source
            self.symmetric_tasks = True
        else:
            self.target = D2
        self.method = method

    def get_distance(self):
        raise NotImplementedError

    def get_hopts(self) -> Dict:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.get_distance()


class MoleculeDatasetDistance(AbstractDatasetDistance):
    def __init__(self, D1=None, D2=None, method="euclidean", **kwargs):
        """
        Calculate the distance between two molecule datasets or two lists of molecule datasets
        Args:
            D1 (MoleculeDataset or list of Dataloader): Torch dataloader for the first dataset
            D2 (MoleculeDataset or list of Dataloader): Torch dataloader for the second dataset
            method (str): The distance method to use
        """
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

        assert method in MOLECULE_DISTANCE_METHODS, f"Method {method} not supported for molecule datasets"
        self.method = method

        self.source_task_ids = [d.task_id for d in self.source]
        self.target_task_ids = [d.task_id for d in self.target]

        self.distance = None

    def get_hopts(self) -> Dict:
        return get_configure(self.method)

    def otdd_distance(self) -> Dict:
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

    def get_distance(self):
        if self.method == "otdd":
            self.distance = self.otdd_distance()
            return self.distance
        else:
            self.distance = self.euclidean_distance()
            return self.distance

    def load_distance(self, path):
        # Load the distance from a file
        pass

    def to_pandas(self) -> pd.DataFrame:
        # Convert the distance to a pandas dataframe
        return pd.DataFrame(self.distance)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.get_distance()

    def __repr__(self):
        return f"MoleculeDatasetDistance(D1={self.source}, D2={self.target}, method={self.method})"


class ProteinDatasetDistance(AbstractDatasetDistance):
    """
    Calculate the distance between two protein datasets
    Args:
        D1 (ProteinDataset): The first protein dataset
        D2 (ProteinDataset): The second protein dataset
        method (str): The distance method to use
    """

    def __init__(self, D1=None, D2=None, method="euclidean"):
        super().__init__(D1, D2, method)
        self.source = D1
        if D2 is None:
            self.target = self.source
            self.symmetric_tasks = True
        else:
            self.target = D2
        self.method = method

        assert method in PROTEIN_DISTANCE_METHODS, f"Method {method} not supported for protein datasets"

        self.source_task_ids = self.source.task_id
        self.target_task_ids = self.target.task_id

        self.distance = None

    def get_hopts(self) -> Dict:
        return get_configure(self.method)

    def euclidean_distance(self) -> Dict:
        # Calculate the Euclidean distance between two protein datasets
        dist = cdist(self.target.features, self.source.features)
        prot_distances = {}
        for i, tgt in enumerate(self.target_task_ids):
            prot_distance = {}
            for j, src in enumerate(self.target.task_id):
                prot_distance[src] = dist[j, i]
            prot_distances[tgt] = prot_distance
        return prot_distances

    def get_distance(self):
        if self.method == "euclidean":
            self.distance = self.euclidean_distance()
            return self.distance
        else:
            self.distance = self.cosine_distance()
            return self.distance

    def load_distance(self, path):
        # Load the distance from a file
        pass

    def to_pandas(self) -> pd.DataFrame:
        # Convert the distance to a pandas dataframe
        return pd.DataFrame(self.distance)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.get_distance()

    def __repr__(self):
        return f"ProteinDatasetDistance(D1={self.source}, D2={self.target}, method={self.method})"


class TaskDistance:
    def __init__(
        self,
        source_task_ids: List[str],
        target_task_ids: List[str],
        external_chemical_space: np.ndarray = None,
        external_protein_space: np.ndarray = None,
    ):
        self.source_task_ids = source_task_ids
        self.target_task_ids = target_task_ids
        self.external_chemical_space = external_chemical_space
        self.external_protein_space = external_protein_space

    def __repr__(self) -> str:
        return f"TaskDistance(source_task_ids={len(self.source_task_ids)}, target_task_ids={len(self.target_task_ids)})"

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.source_task_ids), len(self.target_task_ids)

    def compute_ext_chem_distance(self, method):
        pass

    def compute_ext_prot_distance(self, method):
        pass

    @staticmethod
    def load_ext_chem_distance(path):
        with open(path, "rb") as f:
            x = pickle.load(f)

        if "train_chembl_ids" in x.keys():
            source_task_ids = x["train_chembl_ids"]
        elif "train_pubchem_ids" in x.keys():
            source_task_ids = x["train_pubchem_ids"]
        elif "source_task_ids" in x.keys():
            source_task_ids = x["source_task_ids"]

        if "test_chembl_ids" in x.keys():
            target_task_ids = x["test_chembl_ids"]
        elif "test_pubchem_ids" in x.keys():
            target_task_ids = x["test_pubchem_ids"]
        elif "target_task_ids" in x.keys():
            target_task_ids = x["target_task_ids"]

        return TaskDistance(source_task_ids, target_task_ids, external_chemical_space=x["distance_matrices"])

    @staticmethod
    def load_ext_prot_distance(path):
        with open(path, "rb") as f:
            x = pickle.load(f)

            if "train_chembl_ids" in x.keys():
                source_task_ids = x["train_chembl_ids"]
            elif "train_pubchem_ids" in x.keys():
                source_task_ids = x["train_pubchem_ids"]
            elif "source_task_ids" in x.keys():
                source_task_ids = x["source_task_ids"]

            if "test_chembl_ids" in x.keys():
                target_task_ids = x["test_chembl_ids"]
            elif "test_pubchem_ids" in x.keys():
                target_task_ids = x["test_pubchem_ids"]
            elif "target_task_ids" in x.keys():
                target_task_ids = x["target_task_ids"]

        return TaskDistance(source_task_ids, target_task_ids, external_protein_space=x["distance_matrices"])

    def to_pandas(self):
        df = pd.DataFrame(
            self.external_chemical_space, index=self.source_task_ids, columns=self.target_task_ids
        )
        return df
