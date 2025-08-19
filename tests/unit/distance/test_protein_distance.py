"""
Tests for the protein distance computation module.

This test suite covers:
- ProteinDatasetDistance class
- Euclidean and Cosine distance methods
- Error handling and edge cases
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from themap.data.protein_datasets import ProteinMetadataDataset
from themap.data.tasks import Task, Tasks
from themap.distance.base import PROTEIN_DISTANCE_METHODS
from themap.distance.protein_distance import ProteinDatasetDistance

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_protein_dataset():
    """Create a mock protein dataset for testing."""
    mock_dataset = Mock(spec=ProteinMetadataDataset)
    mock_dataset.task_id = "PROTEIN123"
    mock_dataset.proteins = {"protein1": "MLSDEDFKAV", "protein2": "QLKEKGLF"}
    return mock_dataset


@pytest.fixture
def sample_task(sample_protein_dataset):
    """Create a mock task with protein data."""
    task = Mock(spec=Task)
    task.task_id = "PROTEIN123"
    task.molecule_dataset = None
    task.protein_dataset = sample_protein_dataset
    task.metadata_datasets = None
    return task


@pytest.fixture
def sample_tasks(sample_task):
    """Create a mock Tasks collection."""
    tasks = Mock(spec=Tasks)
    tasks.get_tasks.return_value = [sample_task]
    tasks.get_num_fold_tasks.return_value = 1
    tasks.get_distance_computation_ready_features.return_value = (
        [np.random.rand(1280)],  # source_features (ESM embeddings are larger)
        [np.random.rand(1280)],  # target_features
        ["train_PROTEIN123"],  # source_names
        ["test_PROTEIN123"],  # target_names
    )
    return tasks


@pytest.fixture
def sample_features():
    """Create sample feature arrays for testing."""
    source_features = [np.random.rand(1280) for _ in range(3)]
    target_features = [np.random.rand(1280) for _ in range(2)]
    source_names = ["train_PROT001", "train_PROT002", "train_PROT003"]
    target_names = ["test_PROT004", "test_PROT005"]
    return source_features, target_features, source_names, target_names


# ============================================================================
# ProteinDatasetDistance Tests
# ============================================================================


class TestProteinDatasetDistance:
    """Test ProteinDatasetDistance class."""

    def test_initialization_valid_method(self, sample_tasks):
        """Test initialization with valid protein method."""
        for method in PROTEIN_DISTANCE_METHODS:
            distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method=method)
            assert distance.protein_method == method

    def test_initialization_invalid_method(self, sample_tasks):
        """Test initialization with invalid protein method."""
        with pytest.raises(ValueError) as exc_info:
            ProteinDatasetDistance(tasks=sample_tasks, protein_method="invalid")
        assert "not supported for protein datasets" in str(exc_info.value)

    def test_extract_protein_datasets_none_tasks(self):
        """Test extraction when no tasks are provided."""
        distance = ProteinDatasetDistance(tasks=None)
        assert distance.source_protein_datasets == []
        assert distance.target_protein_datasets == []
        assert distance.source_task_ids == []
        assert distance.target_task_ids == []

    def test_sequence_identity_distance_not_implemented(self, sample_tasks):
        """Test that sequence identity distance raises NotImplementedError."""
        distance = ProteinDatasetDistance(tasks=sample_tasks)

        with pytest.raises(NotImplementedError) as exc_info:
            distance.sequence_identity_distance()
        assert "not yet implemented" in str(exc_info.value)

    @patch.object(ProteinDatasetDistance, "_compute_features")
    def test_euclidean_distance_success(self, mock_compute_features, sample_tasks, sample_features):
        """Test successful euclidean distance computation for proteins."""
        mock_compute_features.return_value = sample_features

        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="euclidean")
        result = distance.euclidean_distance()

        assert isinstance(result, dict)
        assert "PROT004" in result
        assert "PROT005" in result

    @patch.object(ProteinDatasetDistance, "_compute_features")
    def test_euclidean_distance_no_features(self, mock_compute_features, sample_tasks):
        """Test euclidean distance computation with no features."""
        mock_compute_features.return_value = ([], [], [], [])

        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="euclidean")
        result = distance.euclidean_distance()

        assert result == {}

    @patch.object(ProteinDatasetDistance, "_compute_features")
    def test_cosine_distance_success(self, mock_compute_features, sample_tasks, sample_features):
        """Test successful cosine distance computation for proteins."""
        mock_compute_features.return_value = sample_features

        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="cosine")
        result = distance.cosine_distance()

        assert isinstance(result, dict)
        assert "PROT004" in result
        assert "PROT005" in result

    @patch.object(ProteinDatasetDistance, "_compute_features")
    def test_cosine_distance_no_features(self, mock_compute_features, sample_tasks):
        """Test cosine distance computation with no features."""
        mock_compute_features.return_value = ([], [], [], [])

        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="cosine")
        result = distance.cosine_distance()

        assert result == {}

    def test_get_distance_euclidean(self, sample_tasks):
        """Test get_distance method with euclidean method."""
        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="euclidean")

        with patch.object(distance, "euclidean_distance", return_value={"test": "result"}) as mock_euclidean:
            result = distance.get_distance()
            mock_euclidean.assert_called_once()
            assert result == {"test": "result"}

    def test_get_distance_cosine(self, sample_tasks):
        """Test get_distance method with cosine method."""
        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="cosine")

        with patch.object(distance, "cosine_distance", return_value={"test": "result"}) as mock_cosine:
            result = distance.get_distance()
            mock_cosine.assert_called_once()
            assert result == {"test": "result"}

    def test_get_distance_unknown_method(self, sample_tasks):
        """Test get_distance method with unknown method."""
        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="euclidean")
        distance.protein_method = "unknown"  # Manually set invalid method

        with pytest.raises(ValueError) as exc_info:
            distance.get_distance()
        assert "Unknown protein method" in str(exc_info.value)

    def test_get_supported_methods(self, sample_tasks):
        """Test getting supported methods for different data types."""
        distance = ProteinDatasetDistance(tasks=sample_tasks)

        prot_methods = distance.get_supported_methods("protein")
        assert set(prot_methods) == set(PROTEIN_DISTANCE_METHODS)

        meta_methods = distance.get_supported_methods("metadata")
        assert "euclidean" in meta_methods
        assert "cosine" in meta_methods

    def test_get_supported_methods_invalid_type(self, sample_tasks):
        """Test getting supported methods with invalid data type."""
        distance = ProteinDatasetDistance(tasks=sample_tasks)

        with pytest.raises(ValueError) as exc_info:
            distance.get_supported_methods("invalid")
        assert "Unknown data type" in str(exc_info.value)

    def test_load_distance_success(self, sample_tasks):
        """Test successful loading of distance data."""
        distance = ProteinDatasetDistance(tasks=sample_tasks)

        # Create temporary file with valid distance data
        distance_data = {"prot1": {"prot2": 0.5, "prot3": 0.8}, "prot2": {"prot1": 0.5, "prot3": 0.3}}

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
        distance = ProteinDatasetDistance(tasks=sample_tasks)

        with pytest.raises(FileNotFoundError):
            distance.load_distance("/non/existent/file.pkl")

    def test_load_distance_invalid_format(self, sample_tasks):
        """Test loading distance from invalid file format."""
        distance = ProteinDatasetDistance(tasks=sample_tasks)

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
        distance = ProteinDatasetDistance(tasks=sample_tasks)
        distance.distance = {"prot1": {"prot2": 0.5}, "prot2": {"prot1": 0.5}}

        df = distance.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_repr(self, sample_tasks):
        """Test string representation."""
        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="cosine")
        distance.source_protein_datasets = [Mock(), Mock()]
        distance.target_protein_datasets = [Mock()]

        repr_str = repr(distance)
        assert "ProteinDatasetDistance" in repr_str
        assert "cosine" in repr_str
        assert "source_tasks=2" in repr_str
        assert "target_tasks=1" in repr_str

    def test_compute_features_method(self, sample_tasks):
        """Test _compute_features method calls Tasks.get_distance_computation_ready_features."""
        distance = ProteinDatasetDistance(tasks=sample_tasks)

        result = distance._compute_features(protein_featurizer="custom_esm", combination_method="average")

        # Verify that the tasks method was called with correct parameters
        sample_tasks.get_distance_computation_ready_features.assert_called_once()
        call_args = sample_tasks.get_distance_computation_ready_features.call_args
        assert call_args.kwargs["protein_featurizer"] == "custom_esm"
        assert call_args.kwargs["combination_method"] == "average"

    def test_compute_features_no_tasks(self):
        """Test _compute_features method with no tasks."""
        distance = ProteinDatasetDistance(tasks=None)

        result = distance._compute_features()

        assert result == ([], [], [], [])


if __name__ == "__main__":
    pytest.main([__file__])
