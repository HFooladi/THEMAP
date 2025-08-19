"""THEMAP: A library for calculating distances between chemical datasets."""

from ._version import __version__

# Only import version by default for fast loading
__all__ = [
    "__version__",
    "AbstractTasksDistance",
    "MoleculeDatasetDistance",
    "ProteinDatasetDistance",
    "TaskDistance",
    "TaskHardness",
    "PipelineConfig",
    "PipelineRunner",
    "OutputManager",
]


def __getattr__(name):
    """Lazy loading of heavy modules to keep imports fast."""
    if name == "AbstractTasksDistance":
        from .distance import AbstractTasksDistance

        return AbstractTasksDistance
    elif name == "MoleculeDatasetDistance":
        from .distance import MoleculeDatasetDistance

        return MoleculeDatasetDistance
    elif name == "ProteinDatasetDistance":
        from .distance import ProteinDatasetDistance

        return ProteinDatasetDistance
    elif name == "TaskDistance":
        from .distance import TaskDistance

        return TaskDistance
    elif name == "TaskHardness":
        from .hardness import TaskHardness

        return TaskHardness
    elif name == "PipelineConfig":
        from .pipeline import PipelineConfig

        return PipelineConfig
    elif name == "PipelineRunner":
        from .pipeline import PipelineRunner

        return PipelineRunner
    elif name == "OutputManager":
        from .pipeline import OutputManager

        return OutputManager
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
