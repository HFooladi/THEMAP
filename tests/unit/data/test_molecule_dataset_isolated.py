"""
Isolated unit tests for the MoleculeDataset class.

This module provides comprehensive isolated unit tests for MoleculeDataset,
focusing on mocking external dependencies and testing individual methods
in isolation to ensure robust behavior.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from themap.data.molecule_datapoint import MoleculeDatapoint
from themap.data.molecule_dataset import MoleculeDataset, get_task_name_from_path


class TestGetTaskNameFromPath:
    """Test the helper function get_task_name_from_path."""

    def test_get_task_name_from_path_jsonl_gz(self):
        """Test extracting task name from .jsonl.gz file."""
        mock_path = Mock()
        mock_path.basename.return_value = "CHEMBL123456.jsonl.gz"

        result = get_task_name_from_path(mock_path)
        assert result == "CHEMBL123456"

    def test_get_task_name_from_path_other_extension(self):
        """Test extracting task name from non-.jsonl.gz file."""
        mock_path = Mock()
        mock_path.basename.return_value = "CHEMBL123456.csv"

        result = get_task_name_from_path(mock_path)
        assert result == "CHEMBL123456.csv"

    def test_get_task_name_from_path_exception(self):
        """Test handling exception in task name extraction."""
        mock_path = Mock()
        mock_path.basename.side_effect = Exception("Path error")

        result = get_task_name_from_path(mock_path)
        assert result == "unknown_task"


class TestMoleculeDatasetInitialization:
    """Test MoleculeDataset initialization and validation."""

    def test_init_valid_minimal(self):
        """Test minimal valid initialization."""
        dataset = MoleculeDataset("test_task")

        assert dataset.task_id == "test_task"
        assert dataset.data == []
        assert dataset._current_featurizer is None
        assert isinstance(dataset._cache_info, dict)
        assert dataset._persistent_cache is None

    def test_init_with_data(self):
        """Test initialization with data."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        assert dataset.task_id == "test_task"
        assert len(dataset.data) == 2
        assert all(isinstance(dp, MoleculeDatapoint) for dp in dataset.data)

    def test_init_invalid_task_id_none(self):
        """Test initialization with None task_id."""
        with pytest.raises(TypeError, match="task_id must be a string"):
            MoleculeDataset(None)

    def test_init_invalid_task_id_number(self):
        """Test initialization with numeric task_id."""
        with pytest.raises(TypeError, match="task_id must be a string"):
            MoleculeDataset(12345)

    def test_init_invalid_data_dict(self):
        """Test initialization with dict as data."""
        with pytest.raises(TypeError, match="data must be a list"):
            MoleculeDataset("test_task", {"not": "a_list"})

    def test_init_invalid_data_items_mixed(self):
        """Test initialization with mixed data types."""
        valid_dp = MoleculeDatapoint("test_task", "c1ccccc1", True)
        with pytest.raises(TypeError, match="All items in data must be MoleculeDatapoint"):
            MoleculeDataset("test_task", [valid_dp, "invalid_item", 123])


class TestMoleculeDatasetDunderMethods:
    """Test MoleculeDataset dunder methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
            MoleculeDatapoint("test_task", "CC", True),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_len_empty(self):
        """Test __len__ with empty dataset."""
        dataset = MoleculeDataset("test_task", [])
        assert len(dataset) == 0

    def test_len_with_data(self, sample_dataset):
        """Test __len__ with data."""
        assert len(sample_dataset) == 3

    def test_getitem_valid_index(self, sample_dataset):
        """Test __getitem__ with valid index."""
        item = sample_dataset[1]
        assert isinstance(item, MoleculeDatapoint)
        assert item.smiles == "CCO"

    def test_getitem_negative_index(self, sample_dataset):
        """Test __getitem__ with negative index."""
        item = sample_dataset[-1]
        assert isinstance(item, MoleculeDatapoint)
        assert item.smiles == "CC"

    def test_getitem_out_of_bounds(self, sample_dataset):
        """Test __getitem__ with out of bounds index."""
        with pytest.raises(IndexError):
            sample_dataset[10]

    def test_iter(self, sample_dataset):
        """Test __iter__ method."""
        smiles_list = [dp.smiles for dp in sample_dataset]
        assert smiles_list == ["c1ccccc1", "CCO", "CC"]

    def test_repr_empty(self):
        """Test __repr__ with empty dataset."""
        dataset = MoleculeDataset("empty_task", [])
        repr_str = repr(dataset)
        assert "MoleculeDataset" in repr_str
        assert "task_id=empty_task" in repr_str
        assert "task_size=0" in repr_str

    def test_repr_with_data(self, sample_dataset):
        """Test __repr__ with data."""
        repr_str = repr(sample_dataset)
        assert "MoleculeDataset" in repr_str
        assert "task_id=test_task" in repr_str
        assert "task_size=3" in repr_str


class TestMoleculeDatasetProperties:
    """Test MoleculeDataset property methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True, 1.5),
            MoleculeDatapoint("test_task", "CCO", False, -0.5),
            MoleculeDatapoint("test_task", "CC", True, 2.0),
        ]
        return MoleculeDataset("test_task", datapoints)

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    def test_get_features_no_featurizer(self, mock_get_cache, sample_dataset):
        """Test get_features when no current featurizer."""
        result = sample_dataset.get_computed_features
        assert result is None
        mock_get_cache.assert_not_called()

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    def test_get_features_with_featurizer(self, mock_get_cache, sample_dataset):
        """Test get_features with current featurizer."""
        # Setup
        sample_dataset._current_featurizer = "test_featurizer"
        mock_cache = Mock()
        mock_features = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        mock_cache.batch_get.return_value = mock_features
        mock_get_cache.return_value = mock_cache

        result = sample_dataset.get_computed_features

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        mock_get_cache.assert_called_once()
        mock_cache.batch_get.assert_called_once()

    def test_get_labels(self, sample_dataset):
        """Test get_labels property."""
        labels = sample_dataset.get_labels

        assert isinstance(labels, np.ndarray)
        assert labels.dtype == np.int32
        np.testing.assert_array_equal(labels, [1, 0, 1])

    def test_get_labels_empty_dataset(self):
        """Test get_labels with empty dataset."""
        dataset = MoleculeDataset("test_task", [])
        labels = dataset.get_labels

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 0

    def test_get_smiles(self, sample_dataset):
        """Test get_smiles property."""
        smiles = sample_dataset.get_smiles

        assert isinstance(smiles, list)
        assert smiles == ["c1ccccc1", "CCO", "CC"]

    def test_get_smiles_empty_dataset(self):
        """Test get_smiles with empty dataset."""
        dataset = MoleculeDataset("test_task", [])
        smiles = dataset.get_smiles

        assert isinstance(smiles, list)
        assert len(smiles) == 0

    def test_get_ratio_balanced(self, sample_dataset):
        """Test get_ratio with balanced dataset."""
        ratio = sample_dataset.get_ratio
        assert ratio == 0.67  # 2/3 rounded to 2 decimal places

    def test_get_ratio_all_positive(self):
        """Test get_ratio with all positive labels."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", True),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        assert dataset.get_ratio == 1.0

    def test_get_ratio_all_negative(self):
        """Test get_ratio with all negative labels."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", False),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        assert dataset.get_ratio == 0.0

    def test_get_ratio_single_positive(self):
        """Test get_ratio with single positive example."""
        datapoints = [MoleculeDatapoint("test_task", "c1ccccc1", True)]
        dataset = MoleculeDataset("test_task", datapoints)

        assert dataset.get_ratio == 1.0

    def test_get_ratio_single_negative(self):
        """Test get_ratio with single negative example."""
        datapoints = [MoleculeDatapoint("test_task", "c1ccccc1", False)]
        dataset = MoleculeDataset("test_task", datapoints)

        assert dataset.get_ratio == 0.0


class TestMoleculeDatasetValidation:
    """Test MoleculeDataset validation methods."""

    def test_validate_dataset_integrity_valid(self):
        """Test validation with valid dataset."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        assert dataset.validate_dataset_integrity() is True

    def test_validate_dataset_integrity_empty_raises(self):
        """Test validation with empty dataset raises ValueError."""
        dataset = MoleculeDataset("test_task", [])

        with pytest.raises(ValueError, match="Dataset is empty"):
            dataset.validate_dataset_integrity()

    def test_validate_dataset_integrity_no_smiles_attribute(self):
        """Test validation when datapoint lacks smiles attribute."""
        datapoint = MoleculeDatapoint("test_task", "c1ccccc1", True)
        delattr(datapoint, "smiles")
        dataset = MoleculeDataset("test_task", [datapoint])

        with pytest.raises(ValueError, match="has invalid SMILES"):
            dataset.validate_dataset_integrity()

    def test_validate_dataset_integrity_empty_smiles(self):
        """Test validation with empty SMILES string - should fail at datapoint creation."""
        from themap.data.exceptions import InvalidSMILESError

        with pytest.raises(InvalidSMILESError, match="SMILES string cannot be empty"):
            MoleculeDatapoint("test_task", "", True)

    def test_validate_dataset_integrity_none_smiles(self):
        """Test validation with None SMILES."""
        datapoint = MoleculeDatapoint("test_task", "c1ccccc1", True)
        datapoint.smiles = None
        dataset = MoleculeDataset("test_task", [datapoint])

        with pytest.raises(ValueError, match="has invalid SMILES"):
            dataset.validate_dataset_integrity()

    def test_validate_dataset_integrity_non_string_smiles(self):
        """Test validation with non-string SMILES."""
        datapoint = MoleculeDatapoint("test_task", "c1ccccc1", True)
        datapoint.smiles = 12345
        dataset = MoleculeDataset("test_task", [datapoint])

        with pytest.raises(ValueError, match="SMILES.*must be string"):
            dataset.validate_dataset_integrity()

    def test_validate_dataset_integrity_missing_bool_label(self):
        """Test validation when bool_label is missing."""
        datapoint = MoleculeDatapoint("test_task", "c1ccccc1", True)
        delattr(datapoint, "bool_label")
        dataset = MoleculeDataset("test_task", [datapoint])

        with pytest.raises(ValueError, match="missing bool_label"):
            dataset.validate_dataset_integrity()


class TestMoleculeDatasetMemoryManagement:
    """Test MoleculeDataset memory management methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        return MoleculeDataset("test_task", datapoints)

    @patch("sys.getsizeof")
    def test_get_memory_usage_structure(self, mock_getsizeof, sample_dataset):
        """Test get_memory_usage returns correct structure."""
        mock_getsizeof.return_value = 1024 * 1024  # 1MB

        memory_stats = sample_dataset.get_memory_usage()

        expected_keys = ["dataset_object", "data_list", "datapoints", "cached_features", "total"]
        assert all(key in memory_stats for key in expected_keys)
        assert all(isinstance(value, float) for value in memory_stats.values())
        assert all(value >= 0 for value in memory_stats.values())

    @patch("sys.getsizeof")
    def test_get_memory_usage_calculation(self, mock_getsizeof, sample_dataset):
        """Test memory usage calculation logic."""
        mock_getsizeof.side_effect = [2048, 1024, 512, 512]  # bytes

        memory_stats = sample_dataset.get_memory_usage()

        expected_total = (2048 + 1024 + 512 + 512) / (1024 * 1024)  # Convert to MB
        assert abs(memory_stats["total"] - expected_total) < 0.001

    def test_optimize_memory_no_features(self, sample_dataset):
        """Test optimize_memory when no features are cached."""
        result = sample_dataset.optimize_memory()

        expected_keys = ["initial_memory_mb", "final_memory_mb", "memory_saved_mb", "memory_saved_percent"]
        assert all(key in result for key in expected_keys)
        assert all(isinstance(value, (int, float)) for value in result.values())
        assert result["memory_saved_percent"] >= 0

    def test_optimize_memory_with_current_featurizer(self, sample_dataset):
        """Test optimize_memory when current featurizer is set."""
        sample_dataset._current_featurizer = "test_featurizer"

        result = sample_dataset.optimize_memory()

        assert "memory_saved_mb" in result
        assert len(sample_dataset._cache_info) == 0  # Should be cleared

    def test_optimize_memory_zero_initial_memory_edge_case(self, sample_dataset):
        """Test optimize_memory when initial memory is zero."""
        with patch.object(sample_dataset, "get_memory_usage") as mock_memory:
            mock_memory.side_effect = [{"total": 0.0}, {"total": 0.0}]

            result = sample_dataset.optimize_memory()

            assert result["memory_saved_percent"] == 0


class TestMoleculeDatasetCacheManagement:
    """Test MoleculeDataset cache management methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        return MoleculeDataset("test_task", datapoints)

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    def test_clear_cache_no_featurizer(self, mock_get_cache, sample_dataset):
        """Test clear_cache when no current featurizer."""
        sample_dataset.clear_cache()

        mock_get_cache.assert_not_called()
        assert sample_dataset._current_featurizer is None

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    def test_clear_cache_with_featurizer(self, mock_get_cache, sample_dataset):
        """Test clear_cache with current featurizer."""
        sample_dataset._current_featurizer = "test_featurizer"
        mock_cache = Mock()
        mock_cache.evict.return_value = True
        mock_get_cache.return_value = mock_cache

        sample_dataset.clear_cache()

        mock_get_cache.assert_called_once()
        assert mock_cache.evict.call_count == 2  # Once per molecule
        assert sample_dataset._current_featurizer is None

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    def test_get_cache_info_no_featurizer(self, mock_get_cache, sample_dataset):
        """Test get_cache_info when no current featurizer."""
        result = sample_dataset.get_cache_info()

        expected_keys = [
            "dataset_cached",
            "molecules_cached",
            "total_molecules",
            "cache_ratio",
            "current_featurizer",
        ]
        assert all(key in result for key in expected_keys)
        assert result["dataset_cached"] is False
        assert result["molecules_cached"] == 0
        assert result["total_molecules"] == 2
        assert result["cache_ratio"] == 0
        assert result["current_featurizer"] is None

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    def test_get_cache_info_with_featurizer_partial_cache(self, mock_get_cache, sample_dataset):
        """Test get_cache_info with partial cache."""
        sample_dataset._current_featurizer = "test_featurizer"
        mock_cache = Mock()
        mock_cache.get.side_effect = [np.array([1, 2, 3]), None]  # First cached, second not
        mock_get_cache.return_value = mock_cache

        result = sample_dataset.get_cache_info()

        assert result["molecules_cached"] == 1
        assert result["cache_ratio"] == 0.5
        assert result["dataset_cached"] is False
        assert result["current_featurizer"] == "test_featurizer"

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    def test_get_cache_info_full_cache(self, mock_get_cache, sample_dataset):
        """Test get_cache_info with full cache."""
        sample_dataset._current_featurizer = "test_featurizer"
        mock_cache = Mock()
        mock_cache.get.return_value = np.array([1, 2, 3])  # All cached
        mock_get_cache.return_value = mock_cache

        result = sample_dataset.get_cache_info()

        assert result["molecules_cached"] == 2
        assert result["cache_ratio"] == 1.0
        assert result["dataset_cached"] is True

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    def test_get_cache_info_empty_dataset(self, mock_get_cache):
        """Test get_cache_info with empty dataset."""
        dataset = MoleculeDataset("test_task", [])

        result = dataset.get_cache_info()

        assert result["total_molecules"] == 0
        assert result["cache_ratio"] == 0
        assert result["dataset_cached"] is False


class TestMoleculeDatasetPersistentCache:
    """Test MoleculeDataset persistent cache functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        return MoleculeDataset("test_task", datapoints)

    @patch("themap.data.molecule_dataset.PersistentFeatureCache")
    def test_enable_persistent_cache_string_path(self, mock_cache_class, sample_dataset):
        """Test enable_persistent_cache with string path."""
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache

        sample_dataset.enable_persistent_cache("/tmp/cache")

        mock_cache_class.assert_called_once_with("/tmp/cache")
        assert sample_dataset._persistent_cache == mock_cache

    @patch("themap.data.molecule_dataset.PersistentFeatureCache")
    def test_enable_persistent_cache_path_object(self, mock_cache_class, sample_dataset):
        """Test enable_persistent_cache with Path object."""
        mock_cache = Mock()
        mock_cache_class.return_value = mock_cache
        cache_path = Path("/tmp/cache")

        sample_dataset.enable_persistent_cache(cache_path)

        mock_cache_class.assert_called_once_with(cache_path)
        assert sample_dataset._persistent_cache == mock_cache

    def test_get_persistent_cache_stats_no_cache(self, sample_dataset):
        """Test get_persistent_cache_stats when no cache is enabled."""
        result = sample_dataset.get_persistent_cache_stats()
        assert result is None

    def test_get_persistent_cache_stats_with_cache(self, sample_dataset):
        """Test get_persistent_cache_stats with cache enabled."""
        mock_cache = Mock()
        mock_cache.get_stats.return_value = {"hits": 10, "misses": 2}
        mock_cache.get_cache_size_info.return_value = {"size_mb": 5.2}
        sample_dataset._persistent_cache = mock_cache

        result = sample_dataset.get_persistent_cache_stats()

        assert isinstance(result, dict)
        assert "cache_stats" in result
        assert "cache_size" in result
        assert result["cache_stats"]["hits"] == 10
        assert result["cache_size"]["size_mb"] == 5.2


class TestMoleculeDatasetGetFeatures:
    """Test MoleculeDataset get_features method in isolation."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_get_features_input_validation_type(self, sample_dataset):
        """Test input validation for featurizer_name type."""
        with pytest.raises(TypeError, match="featurizer_name must be a string"):
            sample_dataset.get_features(123)

    def test_get_features_input_validation_empty(self, sample_dataset):
        """Test input validation for empty featurizer_name."""
        with pytest.raises(ValueError, match="featurizer_name cannot be empty"):
            sample_dataset.get_features("   ")

    def test_get_features_empty_dataset(self):
        """Test with empty dataset."""
        dataset = MoleculeDataset("test_task", [])

        with pytest.raises(IndexError, match="Cannot compute features for empty dataset"):
            dataset.get_features("test_featurizer")

    def test_get_features_invalid_batch_size(self, sample_dataset):
        """Test with invalid batch_size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            sample_dataset.get_features("test_featurizer", batch_size=-1)

    def test_get_features_invalid_n_jobs_zero(self, sample_dataset):
        """Test with n_jobs=0."""
        with pytest.raises(ValueError, match="n_jobs cannot be 0"):
            sample_dataset.get_features("test_featurizer", n_jobs=0)

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_features_featurizer_load_failure(self, mock_get_featurizer, sample_dataset):
        """Test when featurizer loading fails."""
        mock_get_featurizer.side_effect = Exception("Featurizer not found")

        with pytest.raises(RuntimeError, match="Failed to load featurizer"):
            sample_dataset.get_features("nonexistent_featurizer")

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_features_invalid_smiles_empty(self, mock_get_featurizer, sample_dataset):
        """Test with empty SMILES string."""
        sample_dataset.data[0].smiles = ""

        with pytest.raises(ValueError, match="Invalid SMILES at index 0"):
            sample_dataset.get_features("test_featurizer")

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_features_invalid_smiles_none(self, mock_get_featurizer, sample_dataset):
        """Test with None SMILES."""
        sample_dataset.data[0].smiles = None

        with pytest.raises(ValueError, match="Invalid SMILES at index 0"):
            sample_dataset.get_features("test_featurizer")

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_features_successful_computation(self, mock_get_featurizer, mock_get_cache, sample_dataset):
        """Test successful feature computation."""
        # Setup mocks
        mock_featurizer = Mock()
        mock_featurizer.preprocess.return_value = (["c1ccccc1", "CCO"], None)
        mock_featurizer.transform.return_value = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        mock_get_featurizer.return_value = mock_featurizer

        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache

        result = sample_dataset.get_features("test_featurizer")

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        assert sample_dataset._current_featurizer == "test_featurizer"

        # Verify cache storage
        assert mock_cache.store.call_count == 2

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_features_with_duplicates(self, mock_get_featurizer, mock_get_cache, sample_dataset):
        """Test feature computation with duplicate SMILES."""
        # Add duplicate SMILES
        sample_dataset.data.append(MoleculeDatapoint("test_task", "c1ccccc1", False))  # Duplicate of first

        mock_featurizer = Mock()
        mock_featurizer.preprocess.return_value = (["c1ccccc1", "CCO"], None)  # Only unique SMILES
        mock_featurizer.transform.return_value = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        mock_get_featurizer.return_value = mock_featurizer

        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache

        result = sample_dataset.get_features("test_featurizer")

        assert result.shape == (3, 3)  # 3 molecules, 3 features each
        # First and last should be identical (same SMILES)
        np.testing.assert_array_equal(result[0], result[2])

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_features_cached_return(self, mock_get_featurizer, mock_get_cache, sample_dataset):
        """Test returning cached features."""
        sample_dataset._current_featurizer = "test_featurizer"

        # Mock cached features
        cached_features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        mock_cache = Mock()
        mock_cache.batch_get.return_value = [cached_features[0], cached_features[1]]
        mock_get_cache.return_value = mock_cache

        result = sample_dataset.get_features("test_featurizer")

        np.testing.assert_array_equal(result, cached_features)
        mock_get_featurizer.assert_not_called()  # Should not call featurizer

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_features_force_recompute(self, mock_get_featurizer, mock_get_cache, sample_dataset):
        """Test force recompute ignores cache."""
        sample_dataset._current_featurizer = "test_featurizer"

        # Setup featurizer
        mock_featurizer = Mock()
        mock_featurizer.preprocess.return_value = (["c1ccccc1", "CCO"], None)
        mock_featurizer.transform.return_value = np.array(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32
        )
        mock_get_featurizer.return_value = mock_featurizer

        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache

        result = sample_dataset.get_features("test_featurizer", force_recompute=True)

        assert result.shape == (2, 3)
        mock_get_featurizer.assert_called_once()  # Should call featurizer despite cache

    @patch("themap.data.molecule_dataset.get_global_feature_cache")
    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_features_n_jobs_setting(self, mock_get_featurizer, mock_get_cache, sample_dataset):
        """Test n_jobs setting and restoration."""
        mock_featurizer = Mock()
        mock_featurizer.n_jobs = 1  # Original setting
        mock_featurizer.preprocess.return_value = (["c1ccccc1", "CCO"], None)
        mock_featurizer.transform.return_value = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        mock_get_featurizer.return_value = mock_featurizer

        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache

        sample_dataset.get_features("test_featurizer", n_jobs=4)

        # Should be restored to original value
        assert mock_featurizer.n_jobs == 1

    def test_get_features_basic_functionality(self, sample_dataset):
        """Test basic functionality of get_features."""
        # Simple test that just checks the method exists and validates inputs
        with pytest.raises(RuntimeError, match="Failed to load featurizer"):
            sample_dataset.get_features("nonexistent_featurizer")


class TestMoleculeDatasetGetPrototype:
    """Test MoleculeDataset get_prototype method in isolation."""

    @pytest.fixture
    def balanced_dataset(self):
        """Create balanced dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", True),
            MoleculeDatapoint("test_task", "CC", False),
            MoleculeDatapoint("test_task", "CCC", False),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_get_prototype_input_validation_type(self, balanced_dataset):
        """Test input validation for featurizer_name type."""
        with pytest.raises(TypeError, match="featurizer_name must be a string"):
            balanced_dataset.get_prototype(123)

    def test_get_prototype_input_validation_empty(self, balanced_dataset):
        """Test input validation for empty featurizer_name."""
        with pytest.raises(ValueError, match="featurizer_name cannot be empty"):
            balanced_dataset.get_prototype("   ")

    @patch("themap.data.molecule_dataset.MoleculeDataset.get_features")
    def test_get_prototype_feature_computation_failure(self, mock_get_embedding, balanced_dataset):
        """Test when feature computation fails."""
        mock_get_embedding.side_effect = RuntimeError("Feature computation failed")

        with pytest.raises(RuntimeError, match="Failed to compute features for prototyping"):
            balanced_dataset.get_prototype("test_featurizer")

    def test_get_prototype_no_positive_examples(self):
        """Test with no positive examples."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", False),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        with patch.object(dataset, "get_features") as mock_get_embedding:
            mock_get_embedding.return_value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

            with pytest.raises(ValueError, match="contains no positive examples"):
                dataset.get_prototype("test_featurizer")

    def test_get_prototype_no_negative_examples(self):
        """Test with no negative examples."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", True),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        with patch.object(dataset, "get_features") as mock_get_embedding:
            mock_get_embedding.return_value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

            with pytest.raises(ValueError, match="contains no negative examples"):
                dataset.get_prototype("test_featurizer")

    def test_get_prototype_missing_bool_label(self, balanced_dataset):
        """Test with missing bool_label attribute."""
        delattr(balanced_dataset.data[0], "bool_label")

        with patch.object(balanced_dataset, "get_features") as mock_get_embedding:
            mock_get_embedding.return_value = np.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32
            )

            with pytest.raises(ValueError, match="missing bool_label attribute"):
                balanced_dataset.get_prototype("test_featurizer")

    def test_get_prototype_successful_computation(self, balanced_dataset):
        """Test successful prototype computation."""
        mock_features = np.array(
            [
                [1.0, 2.0, 3.0],  # positive
                [2.0, 3.0, 4.0],  # positive
                [5.0, 6.0, 7.0],  # negative
                [6.0, 7.0, 8.0],  # negative
            ],
            dtype=np.float32,
        )

        with patch.object(balanced_dataset, "get_features") as mock_get_embedding:
            mock_get_embedding.return_value = mock_features

            pos_proto, neg_proto = balanced_dataset.get_prototype("test_featurizer")

            assert isinstance(pos_proto, np.ndarray)
            assert isinstance(neg_proto, np.ndarray)
            assert pos_proto.shape == (3,)
            assert neg_proto.shape == (3,)

            # Check values
            expected_pos = np.mean([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], axis=0)
            expected_neg = np.mean([[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]], axis=0)

            np.testing.assert_array_almost_equal(pos_proto, expected_pos)
            np.testing.assert_array_almost_equal(neg_proto, expected_neg)

    def test_get_prototype_nan_in_positive_features(self, balanced_dataset):
        """Test with NaN values in positive features."""
        mock_features = np.array(
            [
                [1.0, 2.0, np.nan],  # positive with NaN
                [2.0, 3.0, 4.0],  # positive
                [5.0, 6.0, 7.0],  # negative
                [6.0, 7.0, 8.0],  # negative
            ],
            dtype=np.float32,
        )

        with patch.object(balanced_dataset, "get_features") as mock_get_embedding:
            mock_get_embedding.return_value = mock_features

            with pytest.raises(RuntimeError, match="Positive prototype contains NaN"):
                balanced_dataset.get_prototype("test_featurizer")

    def test_get_prototype_nan_in_negative_features(self, balanced_dataset):
        """Test with NaN values in negative features."""
        mock_features = np.array(
            [
                [1.0, 2.0, 3.0],  # positive
                [2.0, 3.0, 4.0],  # positive
                [5.0, 6.0, np.nan],  # negative with NaN
                [6.0, 7.0, 8.0],  # negative
            ],
            dtype=np.float32,
        )

        with patch.object(balanced_dataset, "get_features") as mock_get_embedding:
            mock_get_embedding.return_value = mock_features

            with pytest.raises(RuntimeError, match="Negative prototype contains NaN"):
                balanced_dataset.get_prototype("test_featurizer")


class TestMoleculeDatasetUtilityMethods:
    """Test MoleculeDataset utility methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True, 1.5),
            MoleculeDatapoint("test_task", "CCO", False, -0.5),
            MoleculeDatapoint("test_task", "CC", True, 2.0),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_filter_basic_functionality(self, sample_dataset):
        """Test filter method basic functionality."""
        filtered = sample_dataset.filter(lambda x: x.bool_label)

        assert isinstance(filtered, MoleculeDataset)
        assert len(filtered) == 2
        assert filtered.task_id == sample_dataset.task_id
        assert all(dp.bool_label for dp in filtered.data)

    def test_filter_empty_result(self, sample_dataset):
        """Test filter that returns empty result."""
        filtered = sample_dataset.filter(lambda x: x.smiles == "nonexistent")

        assert len(filtered) == 0
        assert filtered.task_id == sample_dataset.task_id

    def test_filter_all_match(self, sample_dataset):
        """Test filter where all items match."""
        filtered = sample_dataset.filter(lambda x: isinstance(x, MoleculeDatapoint))

        assert len(filtered) == len(sample_dataset)
        assert filtered.task_id == sample_dataset.task_id

    def test_filter_complex_condition(self, sample_dataset):
        """Test filter with complex condition."""
        filtered = sample_dataset.filter(lambda x: x.bool_label and x.numeric_label > 1.0)

        assert len(filtered) == 2
        assert filtered.data[0].smiles == "c1ccccc1"
        assert filtered.data[1].smiles == "CC"

    def test_get_statistics_valid_dataset(self, sample_dataset):
        """Test get_statistics with valid dataset."""
        stats = sample_dataset.get_statistics()

        expected_keys = ["size", "positive_ratio", "avg_molecular_weight", "avg_atoms", "avg_bonds"]
        assert all(key in stats for key in expected_keys)

        assert stats["size"] == 3
        assert stats["positive_ratio"] == 0.67
        assert isinstance(stats["avg_molecular_weight"], float)
        assert isinstance(stats["avg_atoms"], float)
        assert isinstance(stats["avg_bonds"], float)

    def test_get_statistics_empty_dataset(self):
        """Test get_statistics with empty dataset."""
        dataset = MoleculeDataset("test_task", [])

        with pytest.raises(ValueError, match="Cannot compute statistics for empty dataset"):
            dataset.get_statistics()

    def test_get_statistics_calculations(self, sample_dataset):
        """Test that statistics calculations work with actual molecular data."""
        stats = sample_dataset.get_statistics()

        # Just verify that the calculated values are reasonable
        assert isinstance(stats["avg_molecular_weight"], float)
        assert stats["avg_molecular_weight"] > 0
        assert isinstance(stats["avg_atoms"], float)
        assert stats["avg_atoms"] > 0
        assert isinstance(stats["avg_bonds"], float)
        assert stats["avg_bonds"] > 0


class TestMoleculeDatasetLoadFromFile:
    """Test MoleculeDataset.load_from_file method."""

    @patch("themap.data.molecule_dataset.RichPath")
    @patch("themap.data.molecule_dataset.get_task_name_from_path")
    def test_load_from_file_string_path(self, mock_get_task_name, mock_rich_path):
        """Test load_from_file with string path."""
        # Setup mocks
        mock_path = Mock()
        mock_path.read_by_file_suffix.return_value = [
            {"SMILES": "c1ccccc1", "Property": "1", "RegressionProperty": "1.5"},
            {"SMILES": "CCO", "Property": "0", "RegressionProperty": "-0.5"},
        ]
        mock_rich_path.create.return_value = mock_path
        mock_get_task_name.return_value = "test_task"

        dataset = MoleculeDataset.load_from_file("test_path.jsonl.gz")

        mock_rich_path.create.assert_called_once_with("test_path.jsonl.gz")
        assert dataset.task_id == "test_task"
        assert len(dataset.data) == 2
        assert dataset.data[0].smiles == "c1ccccc1"
        assert dataset.data[0].bool_label is True
        assert dataset.data[0].numeric_label == 1.5
        assert dataset.data[1].smiles == "CCO"
        assert dataset.data[1].bool_label is False
        assert dataset.data[1].numeric_label == -0.5

    @patch("themap.data.molecule_dataset.RichPath")
    @patch("themap.data.molecule_dataset.get_task_name_from_path")
    def test_load_from_file_rich_path(self, mock_get_task_name, mock_rich_path):
        """Test load_from_file with RichPath object."""
        mock_path = Mock()
        mock_path.read_by_file_suffix.return_value = []
        mock_get_task_name.return_value = "test_task"

        dataset = MoleculeDataset.load_from_file(mock_path)

        mock_rich_path.create.assert_not_called()  # Should use provided path directly
        mock_get_task_name.assert_called_with(mock_path)
        assert dataset.task_id == "test_task"
        assert len(dataset.data) == 0

    @patch("themap.data.molecule_dataset.RichPath")
    @patch("themap.data.molecule_dataset.get_task_name_from_path")
    def test_load_from_file_missing_regression_property(self, mock_get_task_name, mock_rich_path):
        """Test load_from_file with missing RegressionProperty."""
        mock_path = Mock()
        mock_path.read_by_file_suffix.return_value = [
            {"SMILES": "c1ccccc1", "Property": "1"}  # Missing RegressionProperty
        ]
        mock_rich_path.create.return_value = mock_path
        mock_get_task_name.return_value = "test_task"

        dataset = MoleculeDataset.load_from_file("test_path.jsonl.gz")

        assert len(dataset.data) == 1
        assert np.isnan(dataset.data[0].numeric_label)

    @patch("themap.data.molecule_dataset.RichPath")
    @patch("themap.data.molecule_dataset.get_task_name_from_path")
    def test_load_from_file_empty_file(self, mock_get_task_name, mock_rich_path):
        """Test load_from_file with empty file."""
        mock_path = Mock()
        mock_path.read_by_file_suffix.return_value = []
        mock_rich_path.create.return_value = mock_path
        mock_get_task_name.return_value = "empty_task"

        dataset = MoleculeDataset.load_from_file("empty_file.jsonl.gz")

        assert dataset.task_id == "empty_task"
        assert len(dataset.data) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=themap.data.molecule_dataset", "--cov-report=term-missing"])
