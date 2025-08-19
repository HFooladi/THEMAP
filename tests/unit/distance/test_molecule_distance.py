"""
Tests for the molecule distance computation module.

This test suite covers:
- MoleculeDatasetDistance class
- OTDD, Euclidean, and Cosine distance methods
- Error handling and edge cases
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from themap.data.molecule_dataset import MoleculeDataset
from themap.data.tasks import Task, Tasks
from themap.distance.base import MOLECULE_DISTANCE_METHODS
from themap.distance.exceptions import DistanceComputationError
from themap.distance.molecule_distance import MoleculeDatasetDistance

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_molecule_dataset():
    """Create a mock molecule dataset for testing."""
    mock_dataset = Mock(spec=MoleculeDataset)
    mock_dataset.task_id = "CHEMBL123456"
    mock_dataset.data = [Mock() for _ in range(10)]
    return mock_dataset


@pytest.fixture
def sample_task(sample_molecule_dataset):
    """Create a mock task with molecule data."""
    task = Mock(spec=Task)
    task.task_id = "CHEMBL123456"
    task.molecule_dataset = sample_molecule_dataset
    task.protein_dataset = None
    task.metadata_datasets = None
    return task


@pytest.fixture
def sample_tasks(sample_task):
    """Create a mock Tasks collection."""
    tasks = Mock(spec=Tasks)
    tasks.get_tasks.return_value = [sample_task]
    tasks.get_num_fold_tasks.return_value = 1
    tasks.get_distance_computation_ready_features.return_value = (
        [np.random.rand(128)],  # source_features
        [np.random.rand(128)],  # target_features
        ["train_CHEMBL123456"],  # source_names
        ["test_CHEMBL123456"],  # target_names
    )
    return tasks


@pytest.fixture
def sample_features():
    """Create sample feature arrays for testing."""
    source_features = [np.random.rand(128) for _ in range(3)]
    target_features = [np.random.rand(128) for _ in range(2)]
    source_names = ["train_CHEMBL001", "train_CHEMBL002", "train_CHEMBL003"]
    target_names = ["test_CHEMBL004", "test_CHEMBL005"]
    return source_features, target_features, source_names, target_names


# ============================================================================
# MoleculeDatasetDistance Tests
# ============================================================================


class TestMoleculeDatasetDistance:
    """Test MoleculeDatasetDistance class."""

    def test_initialization_valid_method(self, sample_tasks):
        """Test initialization with valid molecule method."""
        for method in MOLECULE_DISTANCE_METHODS:
            distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method=method)
            assert distance.molecule_method == method

    def test_initialization_invalid_method(self, sample_tasks):
        """Test initialization with invalid molecule method."""
        with pytest.raises(ValueError) as exc_info:
            MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="invalid")
        assert "not supported for molecule datasets" in str(exc_info.value)

    def test_extract_molecule_datasets_none_tasks(self):
        """Test extraction when no tasks are provided."""
        distance = MoleculeDatasetDistance(tasks=None)
        assert distance.source_molecule_datasets == []
        assert distance.target_molecule_datasets == []
        assert distance.source_task_ids == []
        assert distance.target_task_ids == []

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_euclidean_distance_success(self, mock_compute_features, sample_tasks, sample_features):
        """Test successful euclidean distance computation."""
        mock_compute_features.return_value = sample_features

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")
        result = distance.euclidean_distance()

        assert isinstance(result, dict)
        assert "CHEMBL004" in result
        assert "CHEMBL005" in result
        assert "CHEMBL001" in result["CHEMBL004"]

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_euclidean_distance_no_features(self, mock_compute_features, sample_tasks):
        """Test euclidean distance computation with no features."""
        mock_compute_features.return_value = ([], [], [], [])

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")
        result = distance.euclidean_distance()

        assert result == {}

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_euclidean_distance_mismatched_features(self, mock_compute_features, sample_tasks):
        """Test euclidean distance with mismatched feature dimensions."""
        # Features with different dimensions
        source_features = [np.random.rand(64)]  # 64 dimensions
        target_features = [np.random.rand(128)]  # 128 dimensions
        source_names = ["train_CHEMBL001"]
        target_names = ["test_CHEMBL002"]

        mock_compute_features.return_value = (source_features, target_features, source_names, target_names)

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")

        with pytest.raises(DistanceComputationError) as exc_info:
            distance.euclidean_distance()
        assert "Feature dimension mismatch" in str(exc_info.value)

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_euclidean_distance_invalid_values(self, mock_compute_features, sample_tasks):
        """Test euclidean distance computation with NaN/inf values."""
        # Create features that will produce NaN distances
        source_features = [np.array([np.nan, 1.0, 2.0])]
        target_features = [np.array([1.0, np.inf, 2.0])]
        source_names = ["train_CHEMBL001"]
        target_names = ["test_CHEMBL002"]

        mock_compute_features.return_value = (source_features, target_features, source_names, target_names)

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")

        with patch("scipy.spatial.distance.cdist", return_value=np.array([[np.nan]])):
            result = distance.euclidean_distance()
            # Should handle NaN by replacing with default value
            assert result["CHEMBL002"]["CHEMBL001"] == 1.0

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    @patch("themap.distance.molecule_distance._get_dataset_distance")
    @patch("themap.distance.molecule_distance.MoleculeDataloader")
    def test_otdd_distance_success(
        self, mock_dataloader, mock_get_dataset_distance, mock_compute_features, sample_tasks, sample_features
    ):
        """Test successful OTDD distance computation."""
        mock_compute_features.return_value = sample_features

        # Mock OTDD computation
        mock_distance_class = Mock()
        mock_distance_instance = Mock()
        mock_distance_tensor = Mock()
        mock_distance_tensor.cpu.return_value.item.return_value = 0.5
        mock_distance_instance.distance.return_value = mock_distance_tensor
        mock_distance_class.return_value = mock_distance_instance
        mock_get_dataset_distance.return_value = mock_distance_class

        # Mock MoleculeDataloader
        mock_dataloader.return_value = Mock()

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="otdd")
        distance.source_molecule_datasets = [
            Mock(spec=MoleculeDataset),
            Mock(spec=MoleculeDataset),
            Mock(spec=MoleculeDataset),
        ]
        distance.target_molecule_datasets = [Mock(spec=MoleculeDataset), Mock(spec=MoleculeDataset)]
        distance.source_task_ids = ["CHEMBL001", "CHEMBL002", "CHEMBL003"]
        distance.target_task_ids = ["CHEMBL004", "CHEMBL005"]

        result = distance.otdd_distance()

        assert isinstance(result, dict)
        assert len(result) == 2  # Two target tasks

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    @patch("themap.distance.molecule_distance._get_dataset_distance")
    @patch("themap.distance.molecule_distance.MoleculeDataloader")
    def test_otdd_distance_error_handling(
        self, mock_dataloader, mock_get_dataset_distance, mock_compute_features, sample_tasks, sample_features
    ):
        """Test OTDD distance computation error handling."""
        mock_compute_features.return_value = sample_features
        mock_get_dataset_distance.side_effect = ImportError("OTDD not available")

        # Mock MoleculeDataloader
        mock_dataloader.return_value = Mock()

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="otdd")
        distance.source_molecule_datasets = [Mock(spec=MoleculeDataset)]
        distance.target_molecule_datasets = [Mock(spec=MoleculeDataset)]
        distance.source_task_ids = ["CHEMBL001"]
        distance.target_task_ids = ["CHEMBL002"]

        result = distance.otdd_distance()

        # Should handle error gracefully and return fallback distance
        assert result["CHEMBL002"]["CHEMBL001"] == 1.0

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_cosine_distance_success(self, mock_compute_features, sample_tasks, sample_features):
        """Test successful cosine distance computation."""
        mock_compute_features.return_value = sample_features

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="cosine")
        result = distance.cosine_distance()

        assert isinstance(result, dict)
        assert "CHEMBL004" in result
        assert "CHEMBL005" in result

    def test_get_distance_euclidean(self, sample_tasks):
        """Test get_distance method with euclidean method."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")

        with patch.object(distance, "euclidean_distance", return_value={"test": "result"}) as mock_euclidean:
            result = distance.get_distance()
            mock_euclidean.assert_called_once()
            assert result == {"test": "result"}

    def test_get_distance_cosine(self, sample_tasks):
        """Test get_distance method with cosine method."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="cosine")

        with patch.object(distance, "cosine_distance", return_value={"test": "result"}) as mock_cosine:
            result = distance.get_distance()
            mock_cosine.assert_called_once()
            assert result == {"test": "result"}

    def test_get_distance_otdd(self, sample_tasks):
        """Test get_distance method with OTDD method."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="otdd")

        with patch.object(distance, "otdd_distance", return_value={"test": "result"}) as mock_otdd:
            result = distance.get_distance()
            mock_otdd.assert_called_once()
            assert result == {"test": "result"}

    def test_get_distance_unknown_method(self, sample_tasks):
        """Test get_distance method with unknown method."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")
        distance.molecule_method = "unknown"  # Manually set invalid method

        with pytest.raises(ValueError) as exc_info:
            distance.get_distance()
        assert "Unknown molecule method" in str(exc_info.value)

    def test_load_distance_success(self, sample_tasks):
        """Test successful loading of distance data."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        # Create temporary file with valid distance data
        distance_data = {"task1": {"task2": 0.5, "task3": 0.8}, "task2": {"task1": 0.5, "task3": 0.3}}

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            pickle.dump(distance_data, f)
            temp_path = f.name

        try:
            distance.load_distance(temp_path)
            assert distance.distance == distance_data
        finally:
            Path(temp_path).unlink()

    def test_load_distance_file_not_found(self, sample_tasks):
        """Test loading distance from non-existent file."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        with pytest.raises(FileNotFoundError):
            distance.load_distance("/non/existent/file.pkl")

    def test_load_distance_invalid_format(self, sample_tasks):
        """Test loading distance from invalid file format."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        # Create temporary file with invalid data
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            pickle.dump("invalid_data", f)  # Not a dictionary
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                distance.load_distance(temp_path)
            assert "Distance file must contain a dictionary" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_to_pandas(self, sample_tasks):
        """Test conversion to pandas DataFrame."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)
        distance.distance = {"task1": {"task2": 0.5}, "task2": {"task1": 0.5}}

        df = distance.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_repr(self, sample_tasks):
        """Test string representation."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="cosine")
        distance.source_molecule_datasets = [Mock(), Mock()]
        distance.target_molecule_datasets = [Mock()]

        repr_str = repr(distance)
        assert "MoleculeDatasetDistance" in repr_str
        assert "cosine" in repr_str
        assert "source_tasks=2" in repr_str
        assert "target_tasks=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
