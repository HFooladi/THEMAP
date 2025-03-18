"""Tests for the distance module."""

import numpy as np
import pytest
from themap.data.distance import (
    MoleculeDatasetDistance,
    ProteinDatasetDistance,
    TaskDistance,
    MOLECULE_DISTANCE_METHODS,
    PROTEIN_DISTANCE_METHODS,
)
from themap.data.tasks import MoleculeDataset, ProteinDataset


@pytest.fixture
def mock_molecule_dataset():
    """Create a mock molecule dataset for testing."""
    return MoleculeDataset(
        task_id="test_task",
        features=np.random.rand(10, 5),
        labels=np.random.randint(0, 2, 10),
    )


@pytest.fixture
def mock_protein_dataset():
    """Create a mock protein dataset for testing."""
    return ProteinDataset(
        task_id="test_task",
        features=np.random.rand(10, 5),
        labels=np.random.randint(0, 2, 10),
    )


def test_molecule_dataset_distance_initialization(mock_molecule_dataset):
    """Test initialization of MoleculeDatasetDistance."""
    # Test single dataset initialization
    dist = MoleculeDatasetDistance(D1=mock_molecule_dataset)
    assert dist.source == [mock_molecule_dataset]
    assert dist.target == [mock_molecule_dataset]
    assert dist.method == "euclidean"
    assert dist.symmetric_tasks is True

    # Test two dataset initialization
    dist = MoleculeDatasetDistance(D1=mock_molecule_dataset, D2=mock_molecule_dataset)
    assert dist.source == [mock_molecule_dataset]
    assert dist.target == [mock_molecule_dataset]
    assert dist.symmetric_tasks is False

    # Test invalid method
    with pytest.raises(ValueError):
        MoleculeDatasetDistance(D1=mock_molecule_dataset, method="invalid_method")


def test_protein_dataset_distance_initialization(mock_protein_dataset):
    """Test initialization of ProteinDatasetDistance."""
    # Test single dataset initialization
    dist = ProteinDatasetDistance(D1=mock_protein_dataset)
    assert dist.source == mock_protein_dataset
    assert dist.target == mock_protein_dataset
    assert dist.method == "euclidean"
    assert dist.symmetric_tasks is True

    # Test two dataset initialization
    dist = ProteinDatasetDistance(D1=mock_protein_dataset, D2=mock_protein_dataset)
    assert dist.source == mock_protein_dataset
    assert dist.target == mock_protein_dataset
    assert dist.symmetric_tasks is False

    # Test invalid method
    with pytest.raises(ValueError):
        ProteinDatasetDistance(D1=mock_protein_dataset, method="invalid_method")


def test_task_distance_initialization():
    """Test initialization of TaskDistance."""
    source_ids = ["task1", "task2"]
    target_ids = ["task3", "task4"]
    chem_space = np.random.rand(2, 2)
    prot_space = np.random.rand(2, 2)

    # Test initialization with chemical space
    dist = TaskDistance(
        source_task_ids=source_ids,
        target_task_ids=target_ids,
        external_chemical_space=chem_space,
    )
    assert dist.source_task_ids == source_ids
    assert dist.target_task_ids == target_ids
    assert np.array_equal(dist.external_chemical_space, chem_space)
    assert dist.external_protein_space is None

    # Test initialization with protein space
    dist = TaskDistance(
        source_task_ids=source_ids,
        target_task_ids=target_ids,
        external_protein_space=prot_space,
    )
    assert dist.source_task_ids == source_ids
    assert dist.target_task_ids == target_ids
    assert dist.external_chemical_space is None
    assert np.array_equal(dist.external_protein_space, prot_space)

    # Test shape property
    assert dist.shape == (2, 2)


def test_task_distance_to_pandas():
    """Test conversion of TaskDistance to pandas DataFrame."""
    source_ids = ["task1", "task2"]
    target_ids = ["task3", "task4"]
    chem_space = np.random.rand(2, 2)

    dist = TaskDistance(
        source_task_ids=source_ids,
        target_task_ids=target_ids,
        external_chemical_space=chem_space,
    )

    df = dist.to_pandas()
    assert df.index.tolist() == source_ids
    assert df.columns.tolist() == target_ids
    assert df.shape == (2, 2)

    # Test error when no chemical space is available
    dist = TaskDistance(source_task_ids=source_ids, target_task_ids=target_ids)
    with pytest.raises(ValueError):
        dist.to_pandas()


def test_distance_methods():
    """Test available distance methods."""
    assert "otdd" in MOLECULE_DISTANCE_METHODS
    assert "euclidean" in MOLECULE_DISTANCE_METHODS
    assert "cosine" in MOLECULE_DISTANCE_METHODS

    assert "euclidean" in PROTEIN_DISTANCE_METHODS
    assert "cosine" in PROTEIN_DISTANCE_METHODS


def test_distance_computation(mock_molecule_dataset, mock_protein_dataset):
    """Test distance computation for both molecule and protein datasets."""
    # Test molecule dataset distance computation
    mol_dist = MoleculeDatasetDistance(D1=mock_molecule_dataset)
    distance = mol_dist.get_distance()
    assert isinstance(distance, dict)
    assert mock_molecule_dataset.task_id in distance

    # Test protein dataset distance computation
    prot_dist = ProteinDatasetDistance(D1=mock_protein_dataset)
    distance = prot_dist.get_distance()
    assert isinstance(distance, dict)
    assert mock_protein_dataset.task_id in distance


def test_sequence_identity_distance(mock_protein_dataset):
    """Test sequence identity distance computation (placeholder)."""
    prot_dist = ProteinDatasetDistance(D1=mock_protein_dataset)
    # Currently a placeholder, should return None
    assert prot_dist.sequence_identity_distance() is None 