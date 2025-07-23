import pytest

from themap.data.molecule_datapoint import MoleculeDatapoint


def test_MoleculeDatapoint(datapoint_molecule):
    """Test the MoleculeDatapoint class functionality."""
    # Test the __repr__ method
    assert (
        str(datapoint_molecule)
        == "MoleculeDatapoint(task_id=task_id, smiles=c1ccccc1, bool_label=True, numeric_label=1.0)"
    )

    # Test the number_of_atoms method
    assert datapoint_molecule.number_of_atoms == 6

    # Test the number_of_bonds method
    assert datapoint_molecule.number_of_bonds == 6

    # Test the molecular_weight method
    assert round(datapoint_molecule.molecular_weight) == 78


def test_MoleculeDatapoint_validation():
    """Test input validation in MoleculeDatapoint."""
    # Test valid initialization
    datapoint = MoleculeDatapoint(task_id="test_task", smiles="c1ccccc1", bool_label=True, numeric_label=1.0)
    assert datapoint.task_id == "test_task"
    assert datapoint.smiles == "c1ccccc1"
    assert datapoint.bool_label is True
    assert datapoint.numeric_label == 1.0

    # Test invalid task_id
    with pytest.raises(TypeError):
        MoleculeDatapoint(
            task_id=123,  # Should be string
            smiles="c1ccccc1",
            bool_label=True,
        )

    # Test invalid smiles
    with pytest.raises(TypeError):
        MoleculeDatapoint(
            task_id="test_task",
            smiles=123,  # Should be string
            bool_label=True,
        )

    # Test invalid bool_label
    with pytest.raises(TypeError):
        MoleculeDatapoint(
            task_id="test_task",
            smiles="c1ccccc1",
            bool_label=1,  # Should be bool
        )

    # Test invalid numeric_label
    with pytest.raises(TypeError):
        MoleculeDatapoint(
            task_id="test_task",
            smiles="c1ccccc1",
            bool_label=True,
            numeric_label="invalid",  # Should be number or None
        )
