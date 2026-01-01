"""
Feature computation module for THEMAP.

This module provides efficient featurization for molecules and proteins,
with support for batching, caching, and SMILES deduplication.
"""

from .cache import FeatureCache
from .molecule import MOLECULE_FEATURIZERS, MoleculeFeaturizer
from .protein import PROTEIN_FEATURIZERS, ProteinFeaturizer

__all__ = [
    "MoleculeFeaturizer",
    "MOLECULE_FEATURIZERS",
    "ProteinFeaturizer",
    "PROTEIN_FEATURIZERS",
    "FeatureCache",
]
