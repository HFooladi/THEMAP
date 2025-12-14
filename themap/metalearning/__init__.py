"""
Meta-learning module for THEMAP.

This module provides prototypical networks implementation for few-shot learning
on molecular property prediction tasks.
"""

from .data import EpisodeSampler, MetaLearningDataset, TaskSplits, create_meta_splits
from .eval import EvaluationConfig, MetaLearningEvaluator
from .models import PrototypicalNetwork
from .train import MetaLearningTrainer, TrainingConfig

__all__ = [
    # Data handling
    "EpisodeSampler",
    "MetaLearningDataset",
    "TaskSplits",
    "create_meta_splits",
    # Models
    "PrototypicalNetwork",
    # Training
    "MetaLearningTrainer",
    "TrainingConfig",
    # Evaluation
    "MetaLearningEvaluator",
    "EvaluationConfig",
]
