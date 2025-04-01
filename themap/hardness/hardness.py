from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TaskHardness:
    task_id: List[str] = None
    external_chemical_space: np.ndarray = None
    external_protein_space: np.ndarray = None
    internal_chemical_space: np.ndarray = None
    hardness: np.ndarray = None

    def compute_hardness(self, w_exc=0.1, w_exp=1.0, w_inc=0.1):
        final_hardness = (
            w_exc * self.external_chemical_space
            + w_exp * self.external_protein_space
            + w_inc * self.internal_chemical_space
        )
        return final_hardness

    @staticmethod
    def compute_from_distance(task_distance):
        if task_distance.external_chemical_space is not None:
            pass
        elif task_distance.external_protein_space is not None:
            pass
        elif task_distance.internal_chemical_space is not None:
            pass

        pass 