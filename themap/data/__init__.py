"""Data subpackage for THEMAP.

Light-weight items (MoleculeDataset, MoleculeDatasets, DatasetLoader, DataFold,
CSVConverter, etc.) are imported eagerly. Heavier items that transitively
require optional dependencies are exposed via ``__getattr__`` so that

    from themap.data import MoleculeDataset

succeeds on a core-only install. Deferred imports:

- ``ProteinMetadataDataset``, ``ProteinMetadataDatasets`` -> biopython, requests, torch
- ``Task``, ``Tasks``                                     -> biopython, torch (via protein_datasets)
- ``TorchMoleculeDataset``, ``MoleculeDataloader``, ...   -> torch
"""

from .converter import ConversionStats, CSVConverter, convert_csv_to_jsonl
from .enums import DataFold
from .loader import DatasetLoader, load_datasets_for_distance
from .molecule_dataset import MoleculeDataset
from .molecule_datasets import MoleculeDatasets

__all__ = [
    # Core (eager)
    "MoleculeDataset",
    "MoleculeDatasets",
    "DataFold",
    "DatasetLoader",
    "load_datasets_for_distance",
    "CSVConverter",
    "ConversionStats",
    "convert_csv_to_jsonl",
    # Deferred (require biopython / torch / requests)
    "ProteinMetadataDataset",
    "ProteinMetadataDatasets",
    "Task",
    "Tasks",
    "TorchMoleculeDataset",
    "TorchProteinMetadataDataset",
    "MoleculeDataloader",
    "ProteinDataloader",
]

_LAZY_IMPORTS = {
    "ProteinMetadataDataset": ".protein_datasets",
    "ProteinMetadataDatasets": ".protein_datasets",
    "Task": ".tasks",
    "Tasks": ".tasks",
    "TorchMoleculeDataset": ".torch_dataset",
    "TorchProteinMetadataDataset": ".torch_dataset",
    "MoleculeDataloader": ".torch_dataset",
    "ProteinDataloader": ".torch_dataset",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(_LAZY_IMPORTS[name], package=__name__)
        attr = getattr(module, name)
        globals()[name] = attr  # cache so subsequent accesses skip __getattr__
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
