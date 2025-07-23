from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TaskHardness:
    task_id: Optional[List[str]] = None
    external_chemical_space: Optional[np.ndarray] = None
    external_protein_space: Optional[np.ndarray] = None
    internal_chemical_space: Optional[np.ndarray] = None
    hardness: Optional[np.ndarray] = None

    def compute_hardness(self, w_exc: float = 0.1, w_exp: float = 1.0, w_inc: float = 0.1) -> np.ndarray:
        final_hardness = (
            w_exc * self.external_chemical_space
            + w_exp * self.external_protein_space
            + w_inc * self.internal_chemical_space
        )
        return final_hardness

    @staticmethod
    def compute_from_distance(task_distance: "TaskHardness") -> None:
        if task_distance.external_chemical_space is not None:
            pass
        elif task_distance.external_protein_space is not None:
            pass
        elif task_distance.internal_chemical_space is not None:
            pass

        pass
