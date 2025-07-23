from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.utils.data.dataloader as dataloader

from themap.data.molecule_dataset import MoleculeDataset
from themap.utils.logging import get_logger

logger = get_logger(__name__)


class TorchMoleculeDataset(torch.utils.data.Dataset):
    """PYTORCH Dataset for molecular data.

    Args:
        data (MoleculeDataset): MoleculeDataset object
        transform (callable): transform to apply to data
        target_transform (callable): transform to apply to targets
    """

    def __init__(
        self,
        data: MoleculeDataset,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize TorchMoleculeDataset.

        Args:
            data (MoleculeDataset): Input dataset
            transform (callable, optional): Transform to apply to data
            target_transform (callable, optional): Transform to apply to targets
        """
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [0, 1]

        if self.data.get_features is None:
            logger.warning("Dataset does not have features")
            X = torch.ones(len(self.data), 2)
        else:
            X = torch.from_numpy(self.data.get_features)

        if isinstance(self.data.get_labels, np.ndarray):
            y = torch.from_numpy(self.data.get_labels).long()
        else:
            y = self.data.get_labels.long()  # type: ignore
        self.smiles = self.data.get_smiles
        self.tensors = [X, y]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample.

        Args:
            index (int): Index of the sample to get

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (features, label)
        """
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return self.tensors[0].size(0)

    def __repr__(self) -> str:
        return f"TorchMoleculeDataset(task_id={self.data.task_id}, task_size={len(self.data.data)})"

    @classmethod
    def create_dataloader(
        cls, data: MoleculeDataset, batch_size: int = 64, shuffle: bool = True, **kwargs: Any
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader.

        Args:
            data: Input dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            **kwargs (any): Additional arguments for DataLoader

        Returns:
            DataLoader: PyTorch data loader
        """
        dataset = cls(data)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def MoleculeDataloader(
    data: MoleculeDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> torch.utils.data.DataLoader:
    """Load molecular data and create PYTORCH dataloader.

    Args:
        data (MoleculeDataset): MoleculeDataset object
        batch_size (int): batch size
        shuffle (bool): whether to shuffle data
        transform (callable): transform to apply to data
        target_transform (callable): transform to apply to targets

    Returns:
        dataset_loader (DataLoader): PYTORCH dataloader

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
        >>> dataset_loader = MoleculeDataloader(tasks[0][0].molecule_dataset, batch_size=10, shuffle=True)
        >>> for batch in dataset_loader:
        >>>     print(batch)
        >>>     break
    """
    dataset = TorchMoleculeDataset(data)
    dataset_loader = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset_loader
