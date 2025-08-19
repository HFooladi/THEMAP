# Import from new modular structure
from .base import MOLECULE_DISTANCE_METHODS, PROTEIN_DISTANCE_METHODS, AbstractTasksDistance
from .exceptions import DataValidationError, DistanceComputationError
from .molecule_distance import MoleculeDatasetDistance
from .protein_distance import ProteinDatasetDistance
from .task_distance import TaskDistance

# Maintain backward compatibility by importing everything that was in tasks_distance
try:
    # Legacy import for backward compatibility
    from .tasks_distance import *
except ImportError:
    # If tasks_distance.py is removed, this will be silent
    pass

__all__ = [
    # Core classes
    "AbstractTasksDistance",
    "MoleculeDatasetDistance",
    "ProteinDatasetDistance",
    "TaskDistance",
    # Exceptions
    "DistanceComputationError",
    "DataValidationError",
    # Constants
    "MOLECULE_DISTANCE_METHODS",
    "PROTEIN_DISTANCE_METHODS",
]
