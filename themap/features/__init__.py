"""Feature computation subpackage for THEMAP.

Molecule and protein featurizers depend on disjoint heavy stacks (molfeat /
torch+transformers vs biopython / esm). Both are exposed via ``__getattr__``
so that

    from themap.features import MoleculeFeaturizer

does not transitively load biopython, and likewise

    from themap.features import ProteinFeaturizer

does not load molfeat. The peer subpackages ``themap.data`` and
``themap.distance`` follow the same pattern.

Featurizer name constants (lists of strings) ARE loaded eagerly because they
are cheap and the rest of the codebase relies on them at module load time.
"""

from ..utils.featurizer_utils import (
    AVAILABLE_FEATURIZERS as MOLECULE_FEATURIZERS,
)
from ..utils.featurizer_utils import (
    COUNT_FINGERPRINT_FEATURIZERS,
    DESCRIPTOR_FEATURIZERS,
    FINGERPRINT_FEATURIZERS,
    NEURAL_FEATURIZERS,
)

__all__ = [
    # Eager (cheap constant lists)
    "MOLECULE_FEATURIZERS",
    "FINGERPRINT_FEATURIZERS",
    "COUNT_FINGERPRINT_FEATURIZERS",
    "DESCRIPTOR_FEATURIZERS",
    "NEURAL_FEATURIZERS",
    "PROTEIN_FEATURIZERS",
    # Deferred (avoid pulling molfeat / biopython unless needed)
    "MoleculeFeaturizer",
    "ProteinFeaturizer",
    "FeatureCache",
]

_LAZY_IMPORTS = {
    "MoleculeFeaturizer": ".molecule",
    "ProteinFeaturizer": ".protein",
    "PROTEIN_FEATURIZERS": ".protein",
    "FeatureCache": ".cache",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(_LAZY_IMPORTS[name], package=__name__)
        attr = getattr(module, name)
        globals()[name] = attr  # cache so subsequent accesses skip __getattr__
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
