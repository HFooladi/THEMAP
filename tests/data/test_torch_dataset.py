import torch

from themap.data.molecule_dataset import MoleculeDataset
from themap.data.molecule_datapoint import MoleculeDatapoint
from themap.data.torch_dataset import TorchMoleculeDataset, MoleculeDataloader

def test_TorchMoleculeDataset(dataset_CHEMBL2219358):
    """Test the TorchMoleculeDataset class."""
    # Load the dataset from a file
    dataset = MoleculeDataset.load_from_file(dataset_CHEMBL2219358)

    # Create a TorchMoleculeDataset object
    torch_dataset = TorchMoleculeDataset(dataset)

    # Test the __len__ method
    assert len(torch_dataset) == 157

    # Test the __getitem__ method
    features, label = torch_dataset[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    # Test the __repr__ method
    assert str(torch_dataset) == f"TorchMoleculeDataset(task_id={dataset.task_id}, task_size=157)"

def test_TorchMoleculeDataset_without_features():
    """Test TorchMoleculeDataset when dataset has no features."""
    # Create a dataset without features
    datapoints = [
        MoleculeDatapoint("test_task", "c1ccccc1", True),
        MoleculeDatapoint("test_task", "c1ccccc1", False)
    ]
    dataset = MoleculeDataset("test_task", datapoints)

    # Create TorchMoleculeDataset
    torch_dataset = TorchMoleculeDataset(dataset)

    # Check that default features are created
    features, label = torch_dataset[0]
    assert features.shape == (2,)  # Default shape when no features
    assert isinstance(label, torch.Tensor)

def test_TorchMoleculeDataset_transforms():
    """Test TorchMoleculeDataset with transforms."""
    # Create a test dataset
    datapoints = [
        MoleculeDatapoint("test_task", "c1ccccc1", True),
        MoleculeDatapoint("test_task", "c1ccccc1", False)
    ]
    dataset = MoleculeDataset("test_task", datapoints)

    # Define test transforms
    def feature_transform(x):
        return x * 2

    def label_transform(y):
        return y + 1

    # Create dataset with transforms
    torch_dataset = TorchMoleculeDataset(
        dataset,
        transform=feature_transform,
        target_transform=label_transform
    )

    # Test transforms
    features, label = torch_dataset[0]
    assert features[0] == 2  # Original value was 1
    assert label.item() == 2  # Original value was 1

def test_MoleculeDataloader(dataset_CHEMBL2219358):
    """Test the MoleculeDataloader function."""
    # Load the dataset from a file
    dataset = MoleculeDataset.load_from_file(dataset_CHEMBL2219358)

    # Create a MoleculeDataloader object
    dataloader = MoleculeDataloader(dataset, batch_size=32)

    # Test the __len__ method
    assert len(dataloader) == 5  # 157/32 rounded up

    # Test the __iter__ method
    for batch in dataloader:
        features, labels = batch
        assert isinstance(features, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert features.shape[0] <= 32  # Batch size
        assert labels.shape[0] <= 32  # Batch size
        break

def test_TorchMoleculeDataset_create_dataloader():
    """Test the create_dataloader classmethod."""
    # Create a test dataset
    datapoints = [
        MoleculeDatapoint("test_task", "c1ccccc1", True),
        MoleculeDatapoint("test_task", "c1ccccc1", False)
    ]
    dataset = MoleculeDataset("test_task", datapoints)

    # Create dataloader using classmethod
    dataloader = TorchMoleculeDataset.create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    # Test dataloader
    assert len(dataloader) == 1
    features, labels = next(iter(dataloader))
    assert features.shape == (2, 2)  # (batch_size, feature_dim)
    assert labels.shape == (2,) 