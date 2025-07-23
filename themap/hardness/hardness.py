from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class TaskHardness:
    task_id: Optional[List[str]] = None
    external_chemical_space: Optional[NDArray[np.float64]] = None
    external_protein_space: Optional[NDArray[np.float64]] = None
    internal_chemical_space: Optional[NDArray[np.float64]] = None
    hardness: Optional[NDArray[np.float64]] = None

    def compute_hardness(
        self, w_exc: float = 0.1, w_exp: float = 1.0, w_inc: float = 0.1
    ) -> NDArray[np.float64]:
        components: List[NDArray[np.float64]] = []
        if self.external_chemical_space is not None:
            components.append(w_exc * self.external_chemical_space)
        if self.external_protein_space is not None:
            components.append(w_exp * self.external_protein_space)
        if self.internal_chemical_space is not None:
            components.append(w_inc * self.internal_chemical_space)

        if not components:
            return np.array([0.0], dtype=np.float64)

        final_hardness = sum(components)
        return np.asarray(final_hardness, dtype=np.float64)

    @staticmethod
    def compute_from_distance(task_distance: "TaskHardness") -> None:
        if task_distance.external_chemical_space is not None:
            pass
        elif task_distance.external_protein_space is not None:
            pass
        elif task_distance.internal_chemical_space is not None:
            pass

        pass
