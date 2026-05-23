"""Utility subpackage for THEMAP.

Only utilities with light dependencies are imported eagerly. Heavier helpers
are exposed via ``__getattr__`` so that importing this package on a core-only
install does not transitively require ``molfeat``, ``torch``, ``biopython``,
or ``requests``. Deferred groups:

- ``get_featurizer``, ``make_mol``                              -> molfeat, rdkit
- Distance / hardness helpers (``inter_distance``, ...)          -> torch
- ``get_protein_*``, ``read_esm_embedding``                     -> biopython, requests, torch
"""

from .cache_utils import GlobalMoleculeCache, PersistentFeatureCache
from .memory_utils import MemoryEfficientFeatureStorage

__all__ = [
    # Core (eager)
    "GlobalMoleculeCache",
    "PersistentFeatureCache",
    "MemoryEfficientFeatureStorage",
    # Deferred (molfeat / rdkit)
    "get_featurizer",
    "make_mol",
    # Deferred (torch) -- distance / hardness helpers
    "calculate_task_hardness_weight",
    "compute_class_prototypes",
    "compute_correlation",
    "compute_features",
    "compute_features_smiles_labels",
    "compute_fp_similarity",
    "compute_fps_similarity",
    "compute_prototype_datamol",
    "compute_similarities_mean_nearest",
    "compute_task_hardness_from_distance_matrix",
    "compute_task_hardness_molecule_inter",
    "compute_task_hardness_molecule_intra",
    "corr_protein_hardness_metric",
    "extract_class_indices",
    "inter_distance",
    "internal_hardness",
    "intra_distance",
    "normalize",
    "otdd_hardness",
    "protein_hardness_from_distance_matrix",
    "prototype_hardness",
    "similar_dissimilar_indices",
    # Deferred (biopython / requests / torch) -- protein helpers
    "get_protein_accession",
    "get_protein_sequence",
    "get_target_chembl_id",
    "read_esm_embedding",
]

_FEATURIZER_NAMES = {"get_featurizer", "make_mol"}
_DISTANCE_NAMES = {
    "calculate_task_hardness_weight",
    "compute_class_prototypes",
    "compute_correlation",
    "compute_features",
    "compute_features_smiles_labels",
    "compute_fp_similarity",
    "compute_fps_similarity",
    "compute_prototype_datamol",
    "compute_similarities_mean_nearest",
    "compute_task_hardness_from_distance_matrix",
    "compute_task_hardness_molecule_inter",
    "compute_task_hardness_molecule_intra",
    "corr_protein_hardness_metric",
    "extract_class_indices",
    "inter_distance",
    "internal_hardness",
    "intra_distance",
    "normalize",
    "otdd_hardness",
    "protein_hardness_from_distance_matrix",
    "prototype_hardness",
    "similar_dissimilar_indices",
}
_PROTEIN_NAMES = {
    "get_protein_accession",
    "get_protein_sequence",
    "get_target_chembl_id",
    "read_esm_embedding",
}


def __getattr__(name):
    if name in _FEATURIZER_NAMES:
        from . import featurizer_utils as _mod
    elif name in _DISTANCE_NAMES:
        from . import distance_utils as _mod
    elif name in _PROTEIN_NAMES:
        from . import protein_utils as _mod
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    attr = getattr(_mod, name)
    globals()[name] = attr  # cache so subsequent accesses skip __getattr__
    return attr
