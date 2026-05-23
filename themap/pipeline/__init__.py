"""
Pipeline module for configuration-driven distance computation workflows.

This module provides tools for defining, configuring, and executing
benchmarking pipelines that compute distances between molecular and protein datasets.
"""

from .config import PipelineConfig
from .featurization import FeatureStore, FeaturizationPipeline
from .orchestrator import Pipeline, quick_distance, run_pipeline
from .output import OutputManager

__all__ = [
    # New simplified API
    "Pipeline",
    "run_pipeline",
    "quick_distance",
    # Existing API
    "PipelineConfig",
    "PipelineRunner",  # lazy
    "OutputManager",
    "FeatureStore",
    "FeaturizationPipeline",
]


def __getattr__(name):
    # PipelineRunner is lazy-loaded: it imports ProteinDatasetDistance which
    # transitively requires biopython + torch.
    if name == "PipelineRunner":
        from .runner import PipelineRunner

        globals()[name] = PipelineRunner
        return PipelineRunner
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
