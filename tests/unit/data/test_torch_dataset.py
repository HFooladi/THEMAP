from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from themap.data.molecule_dataset import MoleculeDataset
from themap.data.protein_datasets import ProteinMetadataDataset
from themap.data.torch_dataset import (
    MoleculeDataloader,
    ProteinDataloader,
    TorchMoleculeDataset,
    TorchProteinMetadataDataset,
)


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
    assert str(torch_dataset) == f"TorchMoleculeDataset(task_id={dataset.task_id}, size=157, lazy=False)"


def test_TorchMoleculeDataset_without_features():
    """Test TorchMoleculeDataset when dataset has no features."""
    # Create a dataset without features using new simplified structure
    dataset = MoleculeDataset(
        task_id="test_task",
        smiles_list=["c1ccccc1", "c1ccccc1"],
        labels=np.array([1, 0], dtype=np.int32),
    )

    # Create TorchMoleculeDataset
    torch_dataset = TorchMoleculeDataset(dataset)

    # Check that default features are created
    features, label = torch_dataset[0]
    assert features.shape == (2,)  # Default shape when no features
    assert isinstance(label, torch.Tensor)


def test_TorchMoleculeDataset_transforms():
    """Test TorchMoleculeDataset with transforms."""
    # Create a test dataset using new simplified structure
    dataset = MoleculeDataset(
        task_id="test_task",
        smiles_list=["c1ccccc1", "c1ccccc1"],
        labels=np.array([1, 0], dtype=np.int32),
    )

    # Define test transforms
    def feature_transform(x):
        return x * 2

    def label_transform(y):
        return y + 1

    # Create dataset with transforms
    torch_dataset = TorchMoleculeDataset(
        dataset, transform=feature_transform, target_transform=label_transform
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
    # Create a test dataset using new simplified structure
    dataset = MoleculeDataset(
        task_id="test_task",
        smiles_list=["c1ccccc1", "c1ccccc1"],
        labels=np.array([1, 0], dtype=np.int32),
    )

    # Create dataloader using classmethod
    dataloader = TorchMoleculeDataset.create_dataloader(dataset, batch_size=2, shuffle=False)

    # Test dataloader
    assert len(dataloader) == 1
    features, labels = next(iter(dataloader))
    assert features.shape == (2, 2)  # (batch_size, feature_dim)
    assert labels.shape == (2,)


# ============================================================================
# Enhanced Tests for Improved Wrappers
# ============================================================================


@pytest.fixture
def sample_molecule_dataset():
    """Create a mock molecule dataset for testing."""
    mock_dataset = Mock(spec=MoleculeDataset)
    mock_dataset.task_id = "CHEMBL123456"
    mock_dataset.__len__ = Mock(return_value=10)
    mock_dataset.features = np.random.rand(10, 128).astype(np.float32)
    mock_dataset.labels = np.random.randint(0, 2, 10).astype(np.int32)
    mock_dataset.smiles_list = [f"C{i}" for i in range(10)]
    mock_dataset.get_statistics = Mock(return_value={"size": 10, "classes": 2})
    return mock_dataset


@pytest.fixture
def sample_protein_dataset():
    """Create a sample protein dataset for testing."""
    return ProteinMetadataDataset(
        task_id="PROTEIN123",
        uniprot_id="P12345",
        sequence="MKLLVFSLCLLAFSSATAAF",
        features=np.random.rand(1280).astype(np.float32),
    )


class TestEnhancedTorchMoleculeDataset:
    """Test enhanced TorchMoleculeDataset functionality."""

    def test_delegation_functionality(self, sample_molecule_dataset):
        """Test that methods are properly delegated to underlying dataset."""
        dataset = TorchMoleculeDataset(sample_molecule_dataset)

        # Test delegation works
        assert dataset.task_id == "CHEMBL123456"
        stats = dataset.get_statistics()
        assert stats["size"] == 10

        # Test delegation fails for non-existent attributes
        with pytest.raises(AttributeError):
            dataset.non_existent_attribute

    def test_lazy_loading(self, sample_molecule_dataset):
        """Test lazy loading functionality."""
        dataset = TorchMoleculeDataset(sample_molecule_dataset, lazy_loading=True)

        # Initially no tensors should be loaded
        assert dataset.tensors is None

        # First access should trigger loading
        features, label = dataset[0]
        assert dataset.tensors is not None
        assert isinstance(features, torch.Tensor)

    def test_error_handling_invalid_input(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(TypeError) as exc_info:
            TorchMoleculeDataset("not_a_dataset")
        assert "Expected MoleculeDataset" in str(exc_info.value)

    def test_error_handling_empty_dataset(self):
        """Test error handling for empty dataset."""
        empty_dataset = Mock(spec=MoleculeDataset)
        empty_dataset.__len__ = Mock(return_value=0)

        with pytest.raises(ValueError) as exc_info:
            TorchMoleculeDataset(empty_dataset)
        assert "empty MoleculeDataset" in str(exc_info.value)

    def test_refresh_tensors(self, sample_molecule_dataset):
        """Test tensor refresh functionality."""
        dataset = TorchMoleculeDataset(sample_molecule_dataset)

        # Modify underlying dataset
        sample_molecule_dataset.features = np.random.rand(10, 64).astype(np.float32)

        # Refresh should update tensors
        dataset.refresh_tensors()
        assert dataset.tensors[0].shape[1] == 64  # Feature dimension changed

    def test_enhanced_create_dataloader(self, sample_molecule_dataset):
        """Test enhanced create_dataloader with additional parameters."""

        def transform(x):
            return x * 2

        dataloader = TorchMoleculeDataset.create_dataloader(
            sample_molecule_dataset,
            batch_size=4,
            shuffle=True,
            transform=transform,
            lazy_loading=True,
            num_workers=0,  # Avoid multiprocessing in tests
        )

        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 4
        # DataLoader doesn't expose shuffle as an attribute, check dataset parameters instead
        assert hasattr(dataloader, "dataset")

    def test_tensor_preparation_errors(self, sample_molecule_dataset):
        """Test error handling during tensor preparation."""
        # Simulate mismatched feature and label counts
        sample_molecule_dataset.features = np.random.rand(5, 128).astype(np.float32)
        sample_molecule_dataset.labels = np.random.randint(0, 2, 10).astype(np.int32)

        with pytest.raises(RuntimeError) as exc_info:
            TorchMoleculeDataset(sample_molecule_dataset)
        assert "mismatch" in str(exc_info.value) or "Feature and label count mismatch" in str(exc_info.value)


class TestTorchProteinDataset:
    """Test TorchProteinDataset wrapper."""

    def test_initialization_success(self, sample_protein_dataset):
        """Test successful initialization."""
        dataset = TorchProteinMetadataDataset(sample_protein_dataset)

        assert dataset._dataset == sample_protein_dataset
        assert dataset.transform is None
        assert dataset.target_transform is None
        assert not dataset.lazy_loading
        assert dataset.sequence_length is None
        assert dataset.tensors is not None

    def test_initialization_with_sequence_length(self, sample_protein_dataset):
        """Test initialization with fixed sequence length."""
        dataset = TorchProteinMetadataDataset(sample_protein_dataset, sequence_length=50)

        assert dataset.sequence_length == 50

    def test_initialization_invalid_input(self):
        """Test initialization with invalid input."""
        with pytest.raises(TypeError) as exc_info:
            TorchProteinMetadataDataset("not_a_protein_dataset")
        assert "Expected ProteinMetadataDataset" in str(exc_info.value)

    def test_initialization_empty_sequence(self):
        """Test that ProteinMetadataDataset validates empty sequences at construction."""
        # ProteinMetadataDataset now validates at construction, so creating with empty sequence raises error
        from themap.data.exceptions import DatasetValidationError

        with pytest.raises(DatasetValidationError) as exc_info:
            ProteinMetadataDataset(task_id="EMPTY", uniprot_id="P00000", sequence="")
        assert "non-empty string" in str(exc_info.value)

    def test_getitem_success(self, sample_protein_dataset):
        """Test successful item retrieval."""
        dataset = TorchProteinMetadataDataset(sample_protein_dataset)

        features, label = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert features.dtype == torch.float32

    def test_getitem_invalid_index(self, sample_protein_dataset):
        """Test item retrieval with invalid index."""
        dataset = TorchProteinMetadataDataset(sample_protein_dataset)

        with pytest.raises(IndexError) as exc_info:
            dataset[1]
        assert "only one protein" in str(exc_info.value)

    def test_len(self, sample_protein_dataset):
        """Test dataset length."""
        dataset = TorchProteinMetadataDataset(sample_protein_dataset)

        assert len(dataset) == 1

    def test_repr(self, sample_protein_dataset):
        """Test string representation."""
        dataset = TorchProteinMetadataDataset(sample_protein_dataset)

        repr_str = repr(dataset)
        assert "TorchProteinDataset" in repr_str
        assert "PROTEIN123" in repr_str
        assert "P12345" in repr_str

    def test_delegation(self, sample_protein_dataset):
        """Test attribute delegation to underlying dataset."""
        dataset = TorchProteinMetadataDataset(sample_protein_dataset)

        assert dataset.task_id == "PROTEIN123"
        assert dataset.uniprot_id == "P12345"
        assert dataset.sequence == "MKLLVFSLCLLAFSSATAAF"

    def test_prepare_tensors_no_features(self):
        """Test tensor preparation without precomputed features."""
        protein_dataset = ProteinMetadataDataset(
            task_id="PROTEIN456", uniprot_id="P67890", sequence="ACDEFGHIKLMNPQRSTVWY", features=None
        )

        dataset = TorchProteinMetadataDataset(protein_dataset)

        # Should create sequence-based features
        assert dataset.tensors[0].shape[-1] == 1  # Feature dimension
        assert len(dataset.tensors[0]) == len(protein_dataset.sequence)

    def test_prepare_tensors_with_sequence_length(self):
        """Test tensor preparation with fixed sequence length."""
        protein_dataset = ProteinMetadataDataset(
            task_id="PROTEIN456", uniprot_id="P67890", sequence="ACDEFGHIKLMNPQRSTVWY", features=None
        )

        dataset = TorchProteinMetadataDataset(protein_dataset, sequence_length=50)

        # Should pad or truncate to fixed length
        assert len(dataset.tensors[0]) == 50

    def test_create_dataloader(self, sample_protein_dataset):
        """Test dataloader creation."""
        dataloader = TorchProteinMetadataDataset.create_dataloader(
            sample_protein_dataset, batch_size=1, shuffle=False
        )

        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 1
        # DataLoader doesn't expose shuffle as an attribute, check that it's created properly
        assert hasattr(dataloader, "dataset")

    def test_refresh_tensors(self, sample_protein_dataset):
        """Test tensor refresh functionality."""
        dataset = TorchProteinMetadataDataset(sample_protein_dataset)

        # Should refresh without error
        dataset.refresh_tensors()
        assert dataset.tensors is not None


class TestProteinDataloader:
    """Test ProteinDataloader function."""

    def test_protein_dataloader(self, sample_protein_dataset):
        """Test ProteinDataloader function."""
        dataloader = ProteinDataloader(sample_protein_dataset, batch_size=1, shuffle=False)

        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 1

        # Test iteration
        for batch_features, batch_labels in dataloader:
            assert batch_features.shape[0] == 1  # Single protein
            assert batch_labels.shape[0] == 1
            break

    def test_protein_dataloader_with_sequence_length(self, sample_protein_dataset):
        """Test ProteinDataloader with sequence length parameter."""
        dataloader = ProteinDataloader(sample_protein_dataset, sequence_length=100)

        assert isinstance(dataloader, torch.utils.data.DataLoader)


class TestIntegrationScenarios:
    """Test integration scenarios with both wrappers."""

    def test_mixed_usage_with_delegation(self, sample_molecule_dataset):
        """Test using PyTorch wrapper while accessing original methods."""
        dataset = TorchMoleculeDataset(sample_molecule_dataset)

        # Use as PyTorch dataset
        features, label = dataset[0]
        assert isinstance(features, torch.Tensor)

        # Use delegated methods
        stats = dataset.get_statistics()
        assert stats["size"] == 10

        # Access smiles_list via the wrapper
        smiles = dataset.smiles_list
        assert len(smiles) == 10

    @patch("themap.data.torch_dataset.logger")
    def test_error_logging(self, mock_logger, sample_molecule_dataset):
        """Test that errors are properly logged."""
        # Simulate tensor preparation error by providing invalid features
        sample_molecule_dataset.features = "invalid_features"

        with pytest.raises(RuntimeError):
            TorchMoleculeDataset(sample_molecule_dataset)

        # Verify error was logged
        mock_logger.error.assert_called()

    def test_dataset_property_access(self, sample_molecule_dataset, sample_protein_dataset):
        """Test dataset property provides access to underlying dataset."""
        mol_dataset = TorchMoleculeDataset(sample_molecule_dataset)
        prot_dataset = TorchProteinMetadataDataset(sample_protein_dataset)

        assert mol_dataset.dataset == sample_molecule_dataset
        assert prot_dataset.dataset == sample_protein_dataset
