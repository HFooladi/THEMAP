"""
Pipeline module for configuration-driven distance computation workflows.

This module provides tools for defining, configuring, and executing
benchmarking pipelines that compute distances between molecular and protein datasets.
"""

from .config import PipelineConfig
from .output import OutputManager
from .runner import PipelineRunner

__all__ = ["PipelineConfig", "PipelineRunner", "OutputManager"]
