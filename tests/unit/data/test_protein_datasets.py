import pytest

from themap.data.protein_datasets import ProteinMetadataDataset


def test_ProteinMetadataDataset():
    """Test the ProteinMetadataDataset class functionality."""
    # Create a ProteinMetadataDataset object
    protein_dataset = ProteinMetadataDataset(
        task_id="task_id",
        uniprot_id="P00000",
        sequence="LNMHMNVQNG",
    )

    # Test the __repr__ method
    assert str(protein_dataset) == "ProteinMetadataDataset(task_id=task_id, uniprot_id=P00000, seq_len=10)"


def test_ProteinMetadataDataset_validation():
    """Test input validation in ProteinMetadataDataset."""
    # Test valid initialization
    dataset = ProteinMetadataDataset(task_id="test_task", uniprot_id="P00000", sequence="LNMHMNVQNG")
    assert dataset.task_id == "test_task"
    assert dataset.uniprot_id == "P00000"
    assert dataset.sequence == "LNMHMNVQNG"

    # Test invalid task_id
    with pytest.raises(TypeError):
        ProteinMetadataDataset(
            task_id=123,  # Should be string
            uniprot_id="P00000",
            sequence="LNMHMNVQNG",
        )

    # Test invalid sequence
    with pytest.raises(TypeError):
        ProteinMetadataDataset(
            task_id="test_task",
            uniprot_id="P00000",
            sequence=123,  # Should be string, not int
        )


def test_ProteinMetadataDataset_load_from_file(protein_dataset_train):
    """Test loading ProteinMetadataDataset from file."""
    pass
