from .tasks_distance import (
    MOLECULE_DISTANCE_METHODS,
    PROTEIN_DISTANCE_METHODS,
    AbstractTasksDistance,
    MoleculeDatasetDistance,
    ProteinDatasetDistance,
    TaskDistance,
)

__all__ = [
    # Legacy distance classes
    "AbstractTasksDistance",
    "MoleculeDatasetDistance",
    "ProteinDatasetDistance",
    "TaskDistance",
    "MOLECULE_DISTANCE_METHODS",
    "PROTEIN_DISTANCE_METHODS",
]
