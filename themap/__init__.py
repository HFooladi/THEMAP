"""THEMAP: A library for calculating distances between chemical datasets."""

from ._version import __version__

# Only import version by default for fast loading
__all__ = [
    "__version__",
    # New pipeline API
    "Pipeline",
    "PipelineConfig",
    "run_pipeline",
    "quick_distance",
    # Distance computation
    "DatasetDistance",
    "MetadataDistance",
    "compute_dataset_distance_matrix",
    "compute_metadata_distance_matrix",
    "combine_distance_matrices",
    # Data loading
    "DatasetLoader",
    "MoleculeDataset",
    # Features
    "MoleculeFeaturizer",
    "ProteinFeaturizer",
    "FeatureCache",
    # Legacy API (backward compatibility)
    "AbstractTasksDistance",
    "MoleculeDatasetDistance",
    "ProteinDatasetDistance",
    "TaskDistance",
    "TaskHardness",
]


def __getattr__(name):
    """Lazy loading of heavy modules to keep imports fast."""
    # New pipeline API
    if name == "Pipeline":
        from .pipeline.orchestrator import Pipeline

        return Pipeline
    elif name == "PipelineConfig":
        from .config import PipelineConfig

        return PipelineConfig
    elif name == "run_pipeline":
        from .pipeline.orchestrator import run_pipeline

        return run_pipeline
    elif name == "quick_distance":
        from .pipeline.orchestrator import quick_distance

        return quick_distance

    # Distance computation
    elif name == "DatasetDistance":
        from .distance import DatasetDistance

        return DatasetDistance
    elif name == "MetadataDistance":
        from .distance import MetadataDistance

        return MetadataDistance
    elif name == "compute_dataset_distance_matrix":
        from .distance import compute_dataset_distance_matrix

        return compute_dataset_distance_matrix
    elif name == "compute_metadata_distance_matrix":
        from .distance import compute_metadata_distance_matrix

        return compute_metadata_distance_matrix
    elif name == "combine_distance_matrices":
        from .distance import combine_distance_matrices

        return combine_distance_matrices

    # Data loading
    elif name == "DatasetLoader":
        from .data import DatasetLoader

        return DatasetLoader
    elif name == "MoleculeDataset":
        from .data import MoleculeDataset

        return MoleculeDataset

    # Features
    elif name == "MoleculeFeaturizer":
        from .features import MoleculeFeaturizer

        return MoleculeFeaturizer
    elif name == "ProteinFeaturizer":
        from .features import ProteinFeaturizer

        return ProteinFeaturizer
    elif name == "FeatureCache":
        from .features import FeatureCache

        return FeatureCache

    # Legacy API (backward compatibility)
    elif name == "AbstractTasksDistance":
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
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
