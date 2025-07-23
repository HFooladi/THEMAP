import numpy as np
import pytest

from themap.data.molecule_datapoint import MoleculeDatapoint
from themap.data.molecule_dataset import MoleculeDataset


def test_MoleculeDataset_load_from_file(dataset_CHEMBL2219236):
    """Test loading MoleculeDataset from file."""
    # Load the dataset from a file
    dataset = MoleculeDataset.load_from_file(dataset_CHEMBL2219236)

    # Test the __len__ method
    assert len(dataset) == 157

    # Test the __getitem__ method
    assert isinstance(dataset[0], MoleculeDatapoint)

    # Test the __repr__ method
    assert str(dataset) == "MoleculeDataset(task_id=CHEMBL2219236, task_size=157)"


def test_MoleculeDataset_validation():
    """Test input validation in MoleculeDataset."""
    # Test valid initialization
    dataset = MoleculeDataset(
        task_id="test_task", data=[MoleculeDatapoint(task_id="test_task", smiles="c1ccccc1", bool_label=True)]
    )
    assert dataset.task_id == "test_task"
    assert len(dataset) == 1

    # Test invalid task_id
    with pytest.raises(TypeError):
        MoleculeDataset(
            task_id=123,  # Should be string
            data=[],
        )

    # Test invalid data
    with pytest.raises(TypeError):
        MoleculeDataset(
            task_id="test_task",
            data="not_a_list",  # Should be list
        )

    # Test invalid data items
    with pytest.raises(TypeError):
        MoleculeDataset(
            task_id="test_task",
            data=["not_a_MoleculeDatapoint"],  # Should be MoleculeDatapoint
        )


def test_MoleculeDataset_properties():
    """Test MoleculeDataset properties."""
    # Create a test dataset
    datapoints = [
        MoleculeDatapoint("test_task", "c1ccccc1", True),
        MoleculeDatapoint("test_task", "c1ccccc1", False),
        MoleculeDatapoint("test_task", "c1ccccc1", True),
    ]
    dataset = MoleculeDataset("test_task", datapoints)

    # Test get_features property
    assert dataset.get_features is None  # Initially None

    # Test get_labels property
    labels = dataset.get_labels
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (3,)
    assert labels.dtype == bool

    # Test get_smiles property
    smiles = dataset.get_smiles
    assert isinstance(smiles, list)
    assert len(smiles) == 3
    assert all(isinstance(s, str) for s in smiles)

    # Test get_ratio property
    assert dataset.get_ratio == 0.67  # 2/3 rounded to 2 decimal places


def test_MoleculeDataset_filter():
    """Test MoleculeDataset filtering."""
    # Create a test dataset
    datapoints = [
        MoleculeDatapoint("test_task", "c1ccccc1", True),
        MoleculeDatapoint("test_task", "c1ccccc1", False),
        MoleculeDatapoint("test_task", "c1ccccc1", True),
    ]
    dataset = MoleculeDataset("test_task", datapoints)

    # Filter for positive examples
    filtered_dataset = dataset.filter(lambda x: x.bool_label)
    assert len(filtered_dataset) == 2
    assert all(dp.bool_label for dp in filtered_dataset)


def test_MoleculeDataset_statistics():
    """Test MoleculeDataset statistics."""
    # Create a test dataset
    datapoints = [
        MoleculeDatapoint("test_task", "c1ccccc1", True),
        MoleculeDatapoint("test_task", "c1ccccc1", False),
        MoleculeDatapoint("test_task", "c1ccccc1", True),
    ]
    dataset = MoleculeDataset("test_task", datapoints)

    # Get statistics
    stats = dataset.get_statistics()

    # Check statistics
    assert stats["size"] == 3
    assert stats["positive_ratio"] == 0.67
    assert isinstance(stats["avg_molecular_weight"], float)
    assert isinstance(stats["avg_atoms"], float)
    assert isinstance(stats["avg_bonds"], float)
