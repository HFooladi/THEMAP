"""Meta-learning for molecular activity prediction.

This subpackage implements distance-guided meta-learning: for a target dataset,
select its k-nearest source datasets from a saved distance file, meta-train a
:class:`ProtoNet` or :class:`MAMLLearner` on those sources, then evaluate how
much meta-learning improves the target in a low-data regime versus a from-scratch
baseline.

Torch-free utilities (:mod:`config`, :mod:`selection`) are imported eagerly; the
torch-dependent models/trainers are loaded lazily via :pep:`562` ``__getattr__``
so ``import themap`` stays light when torch is not installed.
"""

from __future__ import annotations

from typing import Any

from .config import (
    EncoderConfig,
    ExperimentConfig,
    MAMLConfig,
    ProtoConfig,
    TrainConfig,
)
from .selection import load_distance_matrix, select_k_nearest_sources

__all__ = [
    "EncoderConfig",
    "ExperimentConfig",
    "MAMLConfig",
    "ProtoConfig",
    "TrainConfig",
    "load_distance_matrix",
    "select_k_nearest_sources",
    # Lazily loaded (torch-dependent):
    "MLPEncoder",
    "ProtoNet",
    "MAMLLearner",
    "EpisodeSampler",
    "FeatureBank",
    "MetaTrainer",
    "LowDataEvaluator",
    "MetaLearnExperiment",
]

_LAZY = {
    "MLPEncoder": ("models.encoder", "MLPEncoder"),
    "ProtoNet": ("models.protonet", "ProtoNet"),
    "MAMLLearner": ("models.maml", "MAMLLearner"),
    "EpisodeSampler": ("episodes", "EpisodeSampler"),
    "FeatureBank": ("episodes", "FeatureBank"),
    "MetaTrainer": ("trainer", "MetaTrainer"),
    "LowDataEvaluator": ("evaluation", "LowDataEvaluator"),
    "MetaLearnExperiment": ("runner", "MetaLearnExperiment"),
}


def __getattr__(name: str) -> Any:
    """Lazily import torch-dependent symbols on first access (PEP 562)."""
    if name in _LAZY:
        import importlib

        module_path, attr = _LAZY[name]
        module = importlib.import_module(f"{__name__}.{module_path}")
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
