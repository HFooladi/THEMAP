"""Custom exceptions for molecular data handling."""


class MolecularDataError(Exception):
    """Base exception for molecular data operations."""

    pass


class InvalidSMILESError(MolecularDataError):
    """Raised when SMILES string is invalid."""

    def __init__(self, smiles: str, reason: str = ""):
        self.smiles = smiles
        self.reason = reason
        super().__init__(f"Invalid SMILES '{smiles}': {reason}")


class FeaturizationError(MolecularDataError):
    """Raised when feature computation fails."""

    def __init__(self, smiles: str, featurizer_name: str, reason: str = ""):
        self.smiles = smiles
        self.featurizer_name = featurizer_name
        self.reason = reason
        super().__init__(f"Failed to compute features for '{smiles}' using {featurizer_name}: {reason}")


class DatasetValidationError(MolecularDataError):
    """Raised when dataset validation fails."""

    def __init__(self, dataset_id: str, issue: str):
        self.dataset_id = dataset_id
        self.issue = issue
        super().__init__(f"Dataset validation failed for '{dataset_id}': {issue}")


class CacheError(MolecularDataError):
    """Raised when cache operations fail."""

    pass


class MemoryError(MolecularDataError):
    """Raised when memory operations fail."""

    pass
