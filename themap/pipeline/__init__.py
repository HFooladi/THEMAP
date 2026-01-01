"""
Pipeline module for configuration-driven distance computation workflows.

This module provides tools for defining, configuring, and executing
benchmarking pipelines that compute distances between molecular and protein datasets.
"""

from .config import PipelineConfig
from .featurization import FeatureStore, FeaturizationPipeline
from .orchestrator import Pipeline, quick_distance, run_pipeline
from .output import OutputManager
from .runner import PipelineRunner

__all__ = [
    # New simplified API
    "Pipeline",
    "run_pipeline",
    "quick_distance",
    # Existing API
    "PipelineConfig",
    "PipelineRunner",
    "OutputManager",
    "FeatureStore",
    "FeaturizationPipeline",
]
