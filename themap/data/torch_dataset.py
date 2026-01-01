from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.logging import get_logger, setup_logging
from .molecule_dataset import MoleculeDataset
from .protein_datasets import ProteinMetadataDataset

# Setup logging
setup_logging()
logger = get_logger(__name__)


class TorchMoleculeDataset(Dataset):
    """Enhanced PyTorch Dataset wrapper for molecular data.

    This class wraps a MoleculeDataset to provide PyTorch Dataset functionality
    while maintaining access to all original MoleculeDataset methods through delegation.

    Args:
        data (MoleculeDataset): MoleculeDataset object
        transform (callable, optional): Transform to apply to features
        target_transform (callable, optional): Transform to apply to labels
        lazy_loading (bool, optional): Whether to load data lazily. Defaults to False.

    Example:
        >>> from themap.data import MoleculeDataset
        >>> from themap.data.torch_dataset import TorchMoleculeDataset
        >>>
        >>> # Load molecular dataset
        >>> mol_dataset = MoleculeDataset.load_from_file("data.jsonl.gz")
        >>>
        >>> # Create PyTorch wrapper
        >>> torch_dataset = TorchMoleculeDataset(mol_dataset)
        >>>
        >>> # Use as PyTorch Dataset
        >>> dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=32)
        >>>
        >>> # Access original methods through delegation
        >>> stats = torch_dataset.get_statistics()
        >>> features = torch_dataset.get_features("ecfp")
    """

    def __init__(
        self,
        data: MoleculeDataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        lazy_loading: bool = False,
    ) -> None:
        """Initialize TorchMoleculeDataset.

        Args:
            data (MoleculeDataset): Input molecular dataset
            transform (callable, optional): Transform to apply to features
            target_transform (callable, optional): Transform to apply to labels
            lazy_loading (bool, optional): Whether to load tensors lazily

        Raises:
            ValueError: If the dataset is empty or features/labels are invalid
            TypeError: If data is not a MoleculeDataset instance
        """
        if not isinstance(data, MoleculeDataset):
            raise TypeError(f"Expected MoleculeDataset, got {type(data)}")

        if len(data) == 0:
            raise ValueError("Cannot create TorchMoleculeDataset from empty MoleculeDataset")

        self._dataset = data
        self.transform = transform
        self.target_transform = target_transform
        self.lazy_loading = lazy_loading
        self.classes = [0, 1]  # Binary classification by default

        # Cache tensors if not using lazy loading
        if not lazy_loading:
            self._prepare_tensors()
        else:
            self.tensors = None

    def _prepare_tensors(self) -> None:
        """Prepare PyTorch tensors from the underlying dataset.

        Raises:
            RuntimeError: If features or labels cannot be converted to tensors
        """
        try:
            # Get features - use the features property from MoleculeDataset
            features = self._dataset.features
            if features is None:
                logger.warning("Dataset does not have features, creating dummy features")
                X = torch.ones(len(self._dataset), 2, dtype=torch.float32)
            else:
                if not isinstance(features, np.ndarray):
                    raise RuntimeError(f"Features must be numpy array, got {type(features)}")
                X = torch.from_numpy(features).float()

            # Get labels - use the labels attribute from MoleculeDataset
            labels = self._dataset.labels
            if isinstance(labels, np.ndarray):
                y = torch.from_numpy(labels).long()
            elif isinstance(labels, torch.Tensor):
                y = labels.long()
            else:
                raise RuntimeError(f"Labels must be numpy array or torch tensor, got {type(labels)}")

            # Validate tensor shapes
            if X.size(0) != y.size(0):
                raise RuntimeError(f"Feature and label count mismatch: {X.size(0)} vs {y.size(0)}")

            self.tensors = [X, y]
            logger.debug(f"Prepared tensors: features {X.shape}, labels {y.shape}")

        except Exception as e:
            logger.error(f"Failed to prepare tensors: {e}")
            raise RuntimeError(f"Tensor preparation failed: {e}") from e

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample.

        Args:
            index (int): Index of the sample to get

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (features, label)

        Raises:
            IndexError: If index is out of bounds
            RuntimeError: If lazy loading fails
        """
        if not (0 <= index < len(self)):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")

        try:
            # Handle lazy loading
            if self.lazy_loading or self.tensors is None:
                self._prepare_tensors()

            x = self.tensors[0][index]
            if self.transform:
                x = self.transform(x)

            y = self.tensors[1][index]
            if self.target_transform:
                y = self.target_transform(y)

            return x, y

        except Exception as e:
            logger.error(f"Failed to get item at index {index}: {e}")
            raise RuntimeError(f"Failed to retrieve sample {index}") from e

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self._dataset)

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"TorchMoleculeDataset(task_id={self._dataset.task_id}, size={len(self._dataset)}, lazy={self.lazy_loading})"

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying MoleculeDataset.

        Args:
            name: Attribute name

        Returns:
            The attribute from the underlying dataset

        Raises:
            AttributeError: If attribute doesn't exist in underlying dataset
        """
        try:
            return getattr(self._dataset, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def dataset(self) -> MoleculeDataset:
        """Access to the underlying MoleculeDataset.

        Returns:
            MoleculeDataset: The wrapped dataset
        """
        return self._dataset

    def get_smiles(self) -> list[str]:
        """Get SMILES strings for all molecules.

        Returns:
            list[str]: List of SMILES strings
        """
        return self._dataset.smiles_list

    def refresh_tensors(self) -> None:
        """Refresh cached tensors from the underlying dataset.

        Useful when the underlying dataset has been modified.
        """
        logger.debug("Refreshing cached tensors")
        self._prepare_tensors()

    @classmethod
    def create_dataloader(
        cls,
        data: MoleculeDataset,
        batch_size: int = 64,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        lazy_loading: bool = False,
        **kwargs: Any,
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader with enhanced options.

        Args:
            data: Input molecular dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            transform: Transform to apply to features
            target_transform: Transform to apply to labels
            lazy_loading: Whether to use lazy loading
            **kwargs: Additional arguments for DataLoader

        Returns:
            DataLoader: PyTorch data loader

        Example:
            >>> loader = TorchMoleculeDataset.create_dataloader(
            ...     dataset,
            ...     batch_size=32,
            ...     shuffle=True,
            ...     num_workers=4
            ... )
        """
        dataset = cls(data, transform=transform, target_transform=target_transform, lazy_loading=lazy_loading)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


class TorchProteinMetadataDataset(Dataset):
    """Enhanced PyTorch Dataset wrapper for protein data.

    This class wraps a ProteinMetadataDataset to provide PyTorch Dataset functionality
    while maintaining access to all original ProteinMetadataDataset methods through delegation.

    Args:
        data (ProteinMetadataDataset): ProteinMetadataDataset object
        transform (callable, optional): Transform to apply to features
        target_transform (callable, optional): Transform to apply to labels
        lazy_loading (bool, optional): Whether to load data lazily. Defaults to False.
        sequence_length (int, optional): Fixed sequence length for padding/truncation

    Example:
        >>> from themap.data import ProteinMetadataDataset
        >>> from themap.data.torch_dataset import TorchProteinMetadataDataset
        >>>
        >>> # Create protein dataset
        >>> protein_dataset = ProteinMetadataDataset(
        ...     task_id="CHEMBL123",
        ...     uniprot_id="P12345",
        ...     sequence="MKLLVFSLCLLAFSSATAAF"
        ... )
        >>>
        >>> # Create PyTorch wrapper
        >>> torch_dataset = TorchProteinMetadataDataset(protein_dataset)
        >>>
        >>> # Use as PyTorch Dataset
        >>> dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=1)
    """

    def __init__(
        self,
        data: ProteinMetadataDataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        lazy_loading: bool = False,
        sequence_length: Optional[int] = None,
    ) -> None:
        """Initialize TorchProteinMetadataDataset.

        Args:
            data (ProteinMetadataDataset): Input protein dataset
            transform (callable, optional): Transform to apply to features
            target_transform (callable, optional): Transform to apply to targets
            lazy_loading (bool, optional): Whether to load tensors lazily
            sequence_length (int, optional): Fixed sequence length for padding/truncation

        Raises:
            TypeError: If data is not a ProteinMetadataDataset instance
            ValueError: If the dataset is invalid
        """
        if not isinstance(data, ProteinMetadataDataset):
            raise TypeError(f"Expected ProteinMetadataDataset, got {type(data)}")

        if not data.sequence:
            raise ValueError(
                "Cannot create TorchProteinMetadataDataset from ProteinMetadataDataset with empty sequence"
            )

        self._dataset = data
        self.transform = transform
        self.target_transform = target_transform
        self.lazy_loading = lazy_loading
        self.sequence_length = sequence_length

        # Cache tensors if not using lazy loading
        if not lazy_loading:
            self._prepare_tensors()
        else:
            self.tensors = None

    def _prepare_tensors(self) -> None:
        """Prepare PyTorch tensors from the underlying protein dataset.

        Raises:
            RuntimeError: If features cannot be converted to tensors
        """
        try:
            # Get protein features
            # For ProteinMetadataDataset, features might be pre-computed or need to be computed
            if hasattr(self._dataset, "features") and self._dataset.features is not None:
                features = self._dataset.features
            else:
                # Try to get features using the get_features method if available
                try:
                    features = (
                        self._dataset.get_features() if hasattr(self._dataset, "get_features") else None
                    )
                except Exception:
                    features = None

            if features is None:
                # If no precomputed features, create sequence-based features
                logger.info("No precomputed features found, creating sequence-based features")
                # Simple amino acid encoding (can be enhanced with ESM embeddings)
                sequence = self._dataset.sequence
                if self.sequence_length:
                    # Pad or truncate sequence
                    if len(sequence) < self.sequence_length:
                        sequence = sequence + "X" * (self.sequence_length - len(sequence))
                    else:
                        sequence = sequence[: self.sequence_length]

                # Convert to numeric encoding (simplified)
                aa_to_idx = {
                    "A": 0,
                    "C": 1,
                    "D": 2,
                    "E": 3,
                    "F": 4,
                    "G": 5,
                    "H": 6,
                    "I": 7,
                    "K": 8,
                    "L": 9,
                    "M": 10,
                    "N": 11,
                    "P": 12,
                    "Q": 13,
                    "R": 14,
                    "S": 15,
                    "T": 16,
                    "V": 17,
                    "W": 18,
                    "Y": 19,
                    "X": 20,
                }
                sequence_indices = [aa_to_idx.get(aa, 20) for aa in sequence]
                X = torch.tensor(sequence_indices, dtype=torch.long)

                # If features are needed as floats for embedding layers
                if len(X.shape) == 1:
                    X = X.unsqueeze(-1).float()  # Add feature dimension
            else:
                if not isinstance(features, np.ndarray):
                    raise RuntimeError(f"Features must be numpy array, got {type(features)}")
                X = torch.from_numpy(features).float()

            # For protein datasets, we typically don't have labels in the same way as molecules
            # Create a dummy label (this can be customized based on your use case)
            y = torch.tensor([0], dtype=torch.long)  # Single dummy label

            self.tensors = [X, y]
            logger.debug(f"Prepared protein tensors: features {X.shape}, labels {y.shape}")

        except Exception as e:
            logger.error(f"Failed to prepare protein tensors: {e}")
            raise RuntimeError(f"Protein tensor preparation failed: {e}") from e

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a protein data sample.

        Args:
            index (int): Index of the sample to get (should be 0 for single protein)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (features, label)

        Raises:
            IndexError: If index is out of bounds
            RuntimeError: If lazy loading fails
        """
        if index != 0:
            raise IndexError(f"ProteinMetadataDataset contains only one protein, got index {index}")

        try:
            # Handle lazy loading
            if self.lazy_loading or self.tensors is None:
                self._prepare_tensors()

            x = self.tensors[0]
            if self.transform:
                x = self.transform(x)

            y = self.tensors[1]
            if self.target_transform:
                y = self.target_transform(y)

            return x, y

        except Exception as e:
            logger.error(f"Failed to get protein item at index {index}: {e}")
            raise RuntimeError(f"Failed to retrieve protein sample {index}") from e

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Always 1 for a single protein
        """
        return 1

    def __repr__(self) -> str:
        """String representation of the protein dataset."""
        return f"TorchProteinDataset(task_id={self._dataset.task_id}, uniprot_id={self._dataset.uniprot_id}, seq_len={len(self._dataset.sequence)})"

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying ProteinMetadataDataset.

        Args:
            name: Attribute name

        Returns:
            The attribute from the underlying dataset

        Raises:
            AttributeError: If attribute doesn't exist in underlying dataset
        """
        try:
            return getattr(self._dataset, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def dataset(self) -> ProteinMetadataDataset:
        """Access to the underlying ProteinMetadataDataset.

        Returns:
            ProteinMetadataDataset: The wrapped dataset
        """
        return self._dataset

    def refresh_tensors(self) -> None:
        """Refresh cached tensors from the underlying dataset.

        Useful when the underlying dataset has been modified.
        """
        logger.debug("Refreshing cached protein tensors")
        self._prepare_tensors()

    @classmethod
    def create_dataloader(
        cls,
        data: ProteinMetadataDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        lazy_loading: bool = False,
        sequence_length: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader for protein data.

        Args:
            data: Input protein dataset
            batch_size: Batch size (typically 1 for single proteins)
            shuffle: Whether to shuffle data (typically False for single protein)
            transform: Transform to apply to features
            target_transform: Transform to apply to labels
            lazy_loading: Whether to use lazy loading
            sequence_length: Fixed sequence length for padding/truncation
            **kwargs: Additional arguments for DataLoader

        Returns:
            DataLoader: PyTorch data loader
        """
        dataset = cls(
            data,
            transform=transform,
            target_transform=target_transform,
            lazy_loading=lazy_loading,
            sequence_length=sequence_length,
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


# Legacy function for backward compatibility
def MoleculeDataloader(
    data: MoleculeDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> torch.utils.data.DataLoader:
    """Load molecular data and create PyTorch dataloader.

    This function is kept for backward compatibility. Consider using
    TorchMoleculeDataset.create_dataloader() for new code.

    Args:
        data (MoleculeDataset): MoleculeDataset object
        batch_size (int): batch size
        shuffle (bool): whether to shuffle data
        transform (callable): transform to apply to data
        target_transform (callable): transform to apply to targets

    Returns:
        dataset_loader (DataLoader): PyTorch dataloader

    Example:
        >>> from themap.data.torch_dataset import MoleculeDataloader
        >>> from themap.data.tasks import Tasks
        >>> tasks = Tasks.from_directory(
        >>>     directory="datasets/",
        >>>     task_list_file="datasets/sample_tasks_list.json",
        >>>     load_molecules=True,
        >>>     load_proteins=False,
        >>>     load_metadata=False,
        >>>     cache_dir="./cache"
        >>> )
        >>> dataset_loader = MoleculeDataloader(tasks.get_task("TASK_ID").molecule_dataset, batch_size=10, shuffle=True)
        >>> for batch in dataset_loader:
        >>>     print(batch)
        >>>     break
    """
    dataset = TorchMoleculeDataset(data, transform=transform, target_transform=target_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Convenience function for protein dataloaders
def ProteinDataloader(
    data: ProteinMetadataDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    sequence_length: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Load protein data and create PyTorch dataloader.

    Args:
        data (ProteinMetadataDataset): ProteinMetadataDataset object
        batch_size (int): batch size (typically 1 for proteins)
        shuffle (bool): whether to shuffle data
        transform (callable): transform to apply to data
        target_transform (callable): transform to apply to targets
        sequence_length (int, optional): Fixed sequence length for padding/truncation

    Returns:
        dataset_loader (DataLoader): PyTorch dataloader

    Example:
        >>> from themap.data.torch_dataset import ProteinDataloader
        >>> from themap.data.protein_datasets import ProteinMetadataDataset
        >>>
        >>> protein_dataset = ProteinMetadataDataset(
        ...     task_id="CHEMBL123",
        ...     uniprot_id="P12345",
        ...     sequence="MKLLVFSLCLLAFSSATAAF"
        ... )
        >>> dataset_loader = ProteinDataloader(protein_dataset)
        >>> for batch in dataset_loader:
        >>>     print(batch)
        >>>     break
    """
    dataset = TorchProteinMetadataDataset(
        data, transform=transform, target_transform=target_transform, sequence_length=sequence_length
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
