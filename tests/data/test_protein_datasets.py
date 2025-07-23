import pytest

from themap.data.protein_datasets import ProteinDatasets


def test_ProteinDataset():
    """Test the ProteinDataset class functionality."""
    # Create a ProteinDataset object
    protein_dataset = ProteinDatasets(
        task_id=["task_id"],
        protein={"task_id": "LNMHMNVQNG"},
    )

    # Test the __repr__ method
    assert str(protein_dataset) == "ProteinDataset(task_id=['task_id'], protein={'task_id': 'LNMHMNVQNG'})"

    # Test the __len__ method
    assert len(protein_dataset) == 1

    # Test the __getitem__ method
    key, value = protein_dataset[0]
    assert key == "task_id"
    assert value == "LNMHMNVQNG"


def test_ProteinDataset_validation():
    """Test input validation in ProteinDataset."""
    # Test valid initialization
    dataset = ProteinDatasets(task_id=["test_task"], protein={"test_task": "LNMHMNVQNG"})
    assert dataset.task_id == ["test_task"]
    assert dataset.protein == {"test_task": "LNMHMNVQNG"}

    # Test invalid task_id
    with pytest.raises(TypeError):
        ProteinDatasets(
            task_id="not_a_list",  # Should be list
            protein={"test_task": "LNMHMNVQNG"},
        )

    # Test invalid protein
    with pytest.raises(TypeError):
        ProteinDatasets(
            task_id=["test_task"],
            protein="not_a_dict",  # Should be dict
        )

    # Test invalid protein keys
    with pytest.raises(TypeError):
        ProteinDatasets(
            task_id=["test_task"],
            protein={123: "LNMHMNVQNG"},  # Keys should be strings
        )


def test_ProteinDataset_load_from_file(protein_dataset_train):
    """Test loading ProteinDataset from file."""
    # Load the dataset from a file
    dataset = ProteinDatasets.load_from_file(protein_dataset_train)

    # Test the __len__ method
    assert len(dataset) == 10

    # Test the __getitem__ method
    key, value = dataset[0]
    assert isinstance(key, str)
    assert isinstance(value, str)
