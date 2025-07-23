"""
Comprehensive unit tests for MoleculeDataset class.

This test suite provides extensive coverage of the MoleculeDataset class,
including all methods, error handling, edge cases, and performance aspects.
"""

import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from themap.data.molecule_datapoint import MoleculeDatapoint
from themap.data.molecule_dataset import MoleculeDataset


class TestMoleculeDatasetInit:
    """Test initialization and validation of MoleculeDataset."""

    def test_valid_initialization(self):
        """Test valid initialization of MoleculeDataset."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        assert dataset.task_id == "test_task"
        assert len(dataset.data) == 2
        assert dataset._features is None
        assert isinstance(dataset._cache_info, dict)

    def test_invalid_task_id_type(self):
        """Test that non-string task_id raises TypeError."""
        with pytest.raises(TypeError, match="task_id must be a string"):
            MoleculeDataset(123, [])

    def test_invalid_data_type(self):
        """Test that non-list data raises TypeError."""
        with pytest.raises(TypeError, match="data must be a list"):
            MoleculeDataset("test_task", "not_a_list")

    def test_invalid_data_items(self):
        """Test that non-MoleculeDatapoint items raise TypeError."""
        with pytest.raises(TypeError, match="All items in data must be MoleculeDatapoint"):
            MoleculeDataset("test_task", ["not_a_datapoint"])

    def test_empty_dataset(self):
        """Test initialization with empty dataset."""
        dataset = MoleculeDataset("test_task", [])
        assert len(dataset) == 0
        assert dataset.task_id == "test_task"


class TestMoleculeDatasetBasicMethods:
    """Test basic methods of MoleculeDataset."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True, 1.5),
            MoleculeDatapoint("test_task", "CCO", False, 2.0),
            MoleculeDatapoint("test_task", "CC", True, 0.5),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_len(self, sample_dataset):
        """Test __len__ method."""
        assert len(sample_dataset) == 3

    def test_getitem(self, sample_dataset):
        """Test __getitem__ method."""
        first_item = sample_dataset[0]
        assert isinstance(first_item, MoleculeDatapoint)
        assert first_item.smiles == "c1ccccc1"

    def test_iter(self, sample_dataset):
        """Test __iter__ method."""
        smiles_list = [dp.smiles for dp in sample_dataset]
        assert smiles_list == ["c1ccccc1", "CCO", "CC"]

    def test_repr(self, sample_dataset):
        """Test __repr__ method."""
        repr_str = repr(sample_dataset)
        assert "MoleculeDataset" in repr_str
        assert "test_task" in repr_str
        assert "task_size=3" in repr_str


class TestMoleculeDatasetProperties:
    """Test property methods of MoleculeDataset."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
            MoleculeDatapoint("test_task", "CC", True),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_get_features_none(self, sample_dataset):
        """Test get_features when no features are cached."""
        assert sample_dataset.get_features is None

    def test_get_features_with_cache(self, sample_dataset):
        """Test get_features when features are cached."""
        mock_features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sample_dataset._features = mock_features

        features = sample_dataset.get_features
        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 3)
        np.testing.assert_array_equal(features, mock_features)

    def test_get_labels(self, sample_dataset):
        """Test get_labels property."""
        labels = sample_dataset.get_labels
        assert isinstance(labels, np.ndarray)
        assert labels.dtype == bool
        np.testing.assert_array_equal(labels, [True, False, True])

    def test_get_smiles(self, sample_dataset):
        """Test get_smiles property."""
        smiles = sample_dataset.get_smiles
        assert isinstance(smiles, list)
        assert smiles == ["c1ccccc1", "CCO", "CC"]

    def test_get_ratio(self, sample_dataset):
        """Test get_ratio property."""
        ratio = sample_dataset.get_ratio
        assert ratio == 0.67  # 2/3 rounded to 2 decimal places

    def test_get_ratio_all_positive(self):
        """Test get_ratio with all positive examples."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", True),
        ]
        dataset = MoleculeDataset("test_task", datapoints)
        assert dataset.get_ratio == 1.0

    def test_get_ratio_all_negative(self):
        """Test get_ratio with all negative examples."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", False),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        dataset = MoleculeDataset("test_task", datapoints)
        assert dataset.get_ratio == 0.0


class TestMoleculeDatasetValidation:
    """Test validation methods of MoleculeDataset."""

    def test_validate_dataset_integrity_valid(self):
        """Test validate_dataset_integrity with valid dataset."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        assert dataset.validate_dataset_integrity() is True

    def test_validate_dataset_integrity_empty(self):
        """Test validate_dataset_integrity with empty dataset."""
        dataset = MoleculeDataset("test_task", [])

        with pytest.raises(ValueError, match="Dataset is empty"):
            dataset.validate_dataset_integrity()

    def test_validate_dataset_integrity_invalid_smiles(self):
        """Test validate_dataset_integrity with invalid SMILES."""
        # Create a datapoint with invalid SMILES
        datapoint = MoleculeDatapoint("test_task", "", True)  # Empty SMILES
        dataset = MoleculeDataset("test_task", [datapoint])

        with pytest.raises(ValueError, match="has invalid SMILES"):
            dataset.validate_dataset_integrity()

    def test_validate_dataset_integrity_non_string_smiles(self):
        """Test validate_dataset_integrity with non-string SMILES."""
        # Mock a datapoint with non-string SMILES
        datapoint = MoleculeDatapoint("test_task", "c1ccccc1", True)
        datapoint.smiles = 123  # Invalid type
        dataset = MoleculeDataset("test_task", [datapoint])

        with pytest.raises(ValueError, match="SMILES.*must be string"):
            dataset.validate_dataset_integrity()

    def test_validate_dataset_integrity_missing_label(self):
        """Test validate_dataset_integrity with missing bool_label."""
        # Create a datapoint and remove bool_label
        datapoint = MoleculeDatapoint("test_task", "c1ccccc1", True)
        delattr(datapoint, "bool_label")
        dataset = MoleculeDataset("test_task", [datapoint])

        with pytest.raises(ValueError, match="missing bool_label"):
            dataset.validate_dataset_integrity()


class TestMoleculeDatasetMemoryManagement:
    """Test memory management methods of MoleculeDataset."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_get_memory_usage(self, sample_dataset):
        """Test get_memory_usage method."""
        memory_stats = sample_dataset.get_memory_usage()

        assert isinstance(memory_stats, dict)
        assert "dataset_object" in memory_stats
        assert "data_list" in memory_stats
        assert "datapoints" in memory_stats
        assert "cached_features" in memory_stats
        assert "total" in memory_stats

        # All values should be floats (MB)
        for key, value in memory_stats.items():
            assert isinstance(value, float)
            assert value >= 0

    def test_get_memory_usage_with_features(self, sample_dataset):
        """Test get_memory_usage with cached features."""
        # Add mock features
        sample_dataset._features = np.array([[1, 2, 3], [4, 5, 6]])

        memory_stats = sample_dataset.get_memory_usage()
        assert memory_stats["cached_features"] > 0

    def test_optimize_memory_no_features(self, sample_dataset):
        """Test optimize_memory when no features are cached."""
        results = sample_dataset.optimize_memory()

        assert isinstance(results, dict)
        assert "initial_memory_mb" in results
        assert "final_memory_mb" in results
        assert "memory_saved_mb" in results
        assert "memory_saved_percent" in results

    def test_optimize_memory_with_features(self, sample_dataset):
        """Test optimize_memory with cached features."""
        # Set up dataset with features
        mock_features = np.array([[1, 2, 3], [4, 5, 6]])
        sample_dataset._features = mock_features

        # Set individual molecule features
        for i, molecule in enumerate(sample_dataset.data):
            molecule._features = mock_features[i]

        results = sample_dataset.optimize_memory()

        # Check that individual features were cleared
        for molecule in sample_dataset.data:
            assert molecule._features is None

        # Dataset features should remain
        assert sample_dataset._features is not None

        # Should report some memory savings
        assert results["memory_saved_mb"] >= 0


class TestMoleculeDatasetFeatureComputation:
    """Test feature computation methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_get_dataset_embedding_validation_invalid_type(self, sample_dataset):
        """Test get_dataset_embedding with invalid featurizer type."""
        with pytest.raises(TypeError, match="featurizer_name must be a string"):
            sample_dataset.get_dataset_embedding(123)

    def test_get_dataset_embedding_validation_empty_string(self, sample_dataset):
        """Test get_dataset_embedding with empty featurizer name."""
        with pytest.raises(ValueError, match="featurizer_name cannot be empty"):
            sample_dataset.get_dataset_embedding("")

    def test_get_dataset_embedding_validation_empty_dataset(self):
        """Test get_dataset_embedding with empty dataset."""
        dataset = MoleculeDataset("test_task", [])

        with pytest.raises(IndexError, match="Cannot compute features for empty dataset"):
            dataset.get_dataset_embedding("test_featurizer")

    def test_get_dataset_embedding_validation_invalid_batch_size(self, sample_dataset):
        """Test get_dataset_embedding with invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            sample_dataset.get_dataset_embedding("test_featurizer", batch_size=0)

    def test_get_dataset_embedding_validation_invalid_n_jobs(self, sample_dataset):
        """Test get_dataset_embedding with invalid n_jobs."""
        with pytest.raises(ValueError, match="n_jobs cannot be 0"):
            sample_dataset.get_dataset_embedding("test_featurizer", n_jobs=0)

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_dataset_embedding_featurizer_not_found(self, mock_get_featurizer, sample_dataset):
        """Test get_dataset_embedding when featurizer cannot be loaded."""
        mock_get_featurizer.side_effect = Exception("Featurizer not found")

        with pytest.raises(RuntimeError, match="Failed to load featurizer"):
            sample_dataset.get_dataset_embedding("nonexistent_featurizer")

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_dataset_embedding_success_small_dataset(self, mock_get_featurizer, sample_dataset):
        """Test successful feature computation for small dataset."""
        # Mock featurizer
        mock_featurizer = Mock()
        mock_featurizer.preprocess.return_value = (["c1ccccc1", "CCO"], None)
        mock_featurizer.transform.return_value = np.array([[1, 2, 3], [4, 5, 6]])
        mock_get_featurizer.return_value = mock_featurizer

        features = sample_dataset.get_dataset_embedding("test_featurizer")

        assert isinstance(features, np.ndarray)
        assert features.shape == (2, 3)
        assert sample_dataset._features is not None
        np.testing.assert_array_equal(features, sample_dataset._features)

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_dataset_embedding_cached_features(self, mock_get_featurizer, sample_dataset):
        """Test that cached features are returned when available."""
        # Set cached features
        cached_features = np.array([[1, 2, 3], [4, 5, 6]])
        sample_dataset._features = cached_features

        features = sample_dataset.get_dataset_embedding("test_featurizer")

        # Should return cached features without calling featurizer
        mock_get_featurizer.assert_not_called()
        np.testing.assert_array_equal(features, cached_features)

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_dataset_embedding_force_recompute(self, mock_get_featurizer, sample_dataset):
        """Test force recompute ignores cached features."""
        # Set cached features
        sample_dataset._features = np.array([[1, 2, 3], [4, 5, 6]])

        # Mock featurizer for recomputation
        mock_featurizer = Mock()
        mock_featurizer.preprocess.return_value = (["c1ccccc1", "CCO"], None)
        mock_featurizer.transform.return_value = np.array([[7, 8, 9], [10, 11, 12]])
        mock_get_featurizer.return_value = mock_featurizer

        features = sample_dataset.get_dataset_embedding("test_featurizer", force_recompute=True)

        # Should call featurizer despite cached features
        mock_get_featurizer.assert_called_once()
        assert not np.array_equal(features, [[1, 2, 3], [4, 5, 6]])

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_dataset_embedding_large_dataset_batching(self, mock_get_featurizer):
        """Test batching for large datasets."""
        # Create large dataset
        datapoints = [MoleculeDatapoint("test_task", f"smiles_{i}", True) for i in range(5)]
        dataset = MoleculeDataset("test_task", datapoints)

        # Mock featurizer
        mock_featurizer = Mock()
        mock_featurizer.preprocess.side_effect = lambda batch: (batch, None)
        mock_featurizer.transform.side_effect = lambda batch: np.random.rand(len(batch), 3)
        mock_get_featurizer.return_value = mock_featurizer

        features = dataset.get_dataset_embedding("test_featurizer", batch_size=2)

        assert isinstance(features, np.ndarray)
        assert features.shape == (5, 3)

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_dataset_embedding_invalid_smiles(self, mock_get_featurizer):
        """Test handling of invalid SMILES."""
        # Create dataset with invalid SMILES
        datapoint = MoleculeDatapoint("test_task", "c1ccccc1", True)
        datapoint.smiles = ""  # Invalid empty SMILES
        dataset = MoleculeDataset("test_task", [datapoint])

        with pytest.raises(ValueError, match="Invalid SMILES at index"):
            dataset.get_dataset_embedding("test_featurizer")

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_dataset_embedding_featurizer_returns_none(self, mock_get_featurizer, sample_dataset):
        """Test handling when featurizer returns None."""
        mock_featurizer = Mock()
        mock_featurizer.preprocess.return_value = (["c1ccccc1", "CCO"], None)
        mock_featurizer.transform.return_value = None
        mock_get_featurizer.return_value = mock_featurizer

        with pytest.raises(RuntimeError, match="Featurizer returned None"):
            sample_dataset.get_dataset_embedding("test_featurizer")


class TestMoleculeDatasetPrototype:
    """Test prototype computation methods."""

    @pytest.fixture
    def balanced_dataset(self):
        """Create a balanced dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", True),
            MoleculeDatapoint("test_task", "CC", False),
            MoleculeDatapoint("test_task", "CCC", False),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_get_prototype_validation_invalid_type(self, balanced_dataset):
        """Test get_prototype with invalid featurizer type."""
        with pytest.raises(TypeError, match="featurizer_name must be a string"):
            balanced_dataset.get_prototype(123)

    def test_get_prototype_validation_empty_string(self, balanced_dataset):
        """Test get_prototype with empty featurizer name."""
        with pytest.raises(ValueError, match="featurizer_name cannot be empty"):
            balanced_dataset.get_prototype("")

    @patch("themap.data.molecule_dataset.get_featurizer")
    def test_get_prototype_feature_computation_failure(self, mock_get_featurizer, balanced_dataset):
        """Test get_prototype when feature computation fails."""
        mock_get_featurizer.side_effect = Exception("Feature computation failed")

        with pytest.raises(RuntimeError, match="Failed to compute features for prototyping"):
            balanced_dataset.get_prototype("test_featurizer")

    def test_get_prototype_no_positive_examples(self):
        """Test get_prototype with no positive examples."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", False),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        # Mock features
        dataset._features = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError, match="contains no positive examples"):
            dataset.get_prototype("test_featurizer")

    def test_get_prototype_no_negative_examples(self):
        """Test get_prototype with no negative examples."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", True),
        ]
        dataset = MoleculeDataset("test_task", datapoints)

        # Mock features
        dataset._features = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError, match="contains no negative examples"):
            dataset.get_prototype("test_featurizer")

    def test_get_prototype_missing_bool_label(self):
        """Test get_prototype with missing bool_label."""
        datapoint = MoleculeDatapoint("test_task", "c1ccccc1", True)
        delattr(datapoint, "bool_label")
        dataset = MoleculeDataset("test_task", [datapoint])

        with pytest.raises(ValueError, match="missing bool_label attribute"):
            dataset.get_prototype("test_featurizer")

    def test_get_prototype_success(self, balanced_dataset):
        """Test successful prototype computation."""
        # Mock features
        mock_features = np.array(
            [
                [1, 2, 3],  # positive
                [2, 3, 4],  # positive
                [5, 6, 7],  # negative
                [6, 7, 8],  # negative
            ]
        )
        balanced_dataset._features = mock_features

        pos_proto, neg_proto = balanced_dataset.get_prototype("test_featurizer")

        assert isinstance(pos_proto, np.ndarray)
        assert isinstance(neg_proto, np.ndarray)
        assert pos_proto.shape == (3,)
        assert neg_proto.shape == (3,)

        # Check prototype values
        expected_pos = np.mean([[1, 2, 3], [2, 3, 4]], axis=0)
        expected_neg = np.mean([[5, 6, 7], [6, 7, 8]], axis=0)

        np.testing.assert_array_equal(pos_proto, expected_pos)
        np.testing.assert_array_equal(neg_proto, expected_neg)

    def test_get_prototype_nan_values(self, balanced_dataset):
        """Test get_prototype with NaN values in features."""
        # Mock features with NaN
        mock_features = np.array(
            [
                [1, 2, np.nan],  # positive with NaN
                [2, 3, 4],  # positive
                [5, 6, 7],  # negative
                [6, 7, 8],  # negative
            ]
        )
        balanced_dataset._features = mock_features

        with pytest.raises(ValueError, match="Positive prototype contains NaN"):
            balanced_dataset.get_prototype("test_featurizer")


class TestMoleculeDatasetUtilityMethods:
    """Test utility methods of MoleculeDataset."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
            MoleculeDatapoint("test_task", "CC", True),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_clear_cache(self, sample_dataset):
        """Test clear_cache method."""
        # Set up cached features
        mock_features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sample_dataset._features = mock_features

        for i, molecule in enumerate(sample_dataset.data):
            molecule._features = mock_features[i]

        # Clear cache
        sample_dataset.clear_cache()

        # Check that cache is cleared
        assert sample_dataset._features is None
        for molecule in sample_dataset.data:
            assert molecule._features is None

    def test_filter_method(self, sample_dataset):
        """Test filter method."""
        # Filter for positive examples
        filtered = sample_dataset.filter(lambda x: x.bool_label)

        assert isinstance(filtered, MoleculeDataset)
        assert len(filtered) == 2
        assert all(dp.bool_label for dp in filtered.data)
        assert filtered.task_id == sample_dataset.task_id

    def test_get_statistics_valid(self, sample_dataset):
        """Test get_statistics method."""
        stats = sample_dataset.get_statistics()

        assert isinstance(stats, dict)
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

    def test_get_cache_info(self, sample_dataset):
        """Test get_cache_info method."""
        cache_info = sample_dataset.get_cache_info()

        assert isinstance(cache_info, dict)
        assert "dataset_cached" in cache_info
        assert "molecules_cached" in cache_info
        assert "total_molecules" in cache_info
        assert "cache_ratio" in cache_info

        assert cache_info["dataset_cached"] is False
        assert cache_info["total_molecules"] == 3


class TestMoleculeDatasetPersistentCache:
    """Test persistent caching functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        datapoints = [
            MoleculeDatapoint("test_task", "c1ccccc1", True),
            MoleculeDatapoint("test_task", "CCO", False),
        ]
        return MoleculeDataset("test_task", datapoints)

    def test_enable_persistent_cache(self, sample_dataset):
        """Test enable_persistent_cache method."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dataset.enable_persistent_cache(tmp_dir)

            assert sample_dataset._persistent_cache is not None

    def test_get_persistent_cache_stats_no_cache(self, sample_dataset):
        """Test get_persistent_cache_stats without cache."""
        stats = sample_dataset.get_persistent_cache_stats()
        assert stats is None

    @patch("themap.data.molecule_dataset.PersistentFeatureCache")
    def test_get_persistent_cache_stats_with_cache(self, mock_cache_class, sample_dataset):
        """Test get_persistent_cache_stats with cache."""
        # Mock cache
        mock_cache = Mock()
        mock_cache.get_stats.return_value = {"hits": 10, "misses": 5}
        mock_cache.get_cache_size_info.return_value = {"size_mb": 1.5}
        sample_dataset._persistent_cache = mock_cache

        stats = sample_dataset.get_persistent_cache_stats()

        assert isinstance(stats, dict)
        assert "cache_stats" in stats
        assert "cache_size" in stats


class TestMoleculeDatasetLoadFromFile:
    """Test loading datasets from files."""

    def test_load_from_file_string_path(self):
        """Test load_from_file with string path."""
        # This would require actual test data files
        # For now, test the path conversion logic
        with patch("themap.data.molecule_dataset.RichPath") as mock_rich_path:
            mock_path = Mock()
            mock_path.read_by_file_suffix.return_value = []
            mock_rich_path.create.return_value = mock_path

            with patch("themap.data.molecule_dataset.get_task_name_from_path") as mock_get_name:
                mock_get_name.return_value = "test_task"

                dataset = MoleculeDataset.load_from_file("test_path")

                mock_rich_path.create.assert_called_once_with("test_path")
                assert dataset.task_id == "test_task"
                assert len(dataset.data) == 0


# Integration tests
class TestMoleculeDatasetIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test complete workflow from creation to feature computation."""
        # Create dataset
        datapoints = [
            MoleculeDatapoint("integration_test", "c1ccccc1", True),
            MoleculeDatapoint("integration_test", "CCO", False),
        ]
        dataset = MoleculeDataset("integration_test", datapoints)

        # Validate integrity
        assert dataset.validate_dataset_integrity() is True

        # Check initial memory usage
        initial_memory = dataset.get_memory_usage()
        assert initial_memory["cached_features"] == 0.0

        # Mock feature computation
        with patch("themap.data.molecule_dataset.get_featurizer") as mock_get_featurizer:
            mock_featurizer = Mock()
            mock_featurizer.preprocess.return_value = (["c1ccccc1", "CCO"], None)
            mock_featurizer.transform.return_value = np.array([[1, 2, 3], [4, 5, 6]])
            mock_get_featurizer.return_value = mock_featurizer

            # Compute features
            features = dataset.get_dataset_embedding("test_featurizer")
            assert features.shape == (2, 3)

            # Check memory after features
            memory_with_features = dataset.get_memory_usage()
            assert memory_with_features["cached_features"] > 0

            # Test prototypes
            pos_proto, neg_proto = dataset.get_prototype("test_featurizer")
            assert pos_proto.shape == (3,)
            assert neg_proto.shape == (3,)

            # Optimize memory
            optimization_results = dataset.optimize_memory()
            assert "memory_saved_mb" in optimization_results

            # Clear cache
            dataset.clear_cache()
            assert dataset._features is None


# Performance tests
class TestMoleculeDatasetPerformance:
    """Performance-related tests."""

    def test_large_dataset_memory_efficiency(self):
        """Test memory efficiency with large dataset."""
        # Create larger dataset
        datapoints = [MoleculeDatapoint("perf_test", f"smiles_{i}", i % 2 == 0) for i in range(100)]
        dataset = MoleculeDataset("perf_test", datapoints)

        # Check that memory usage is reasonable
        memory_stats = dataset.get_memory_usage()
        assert memory_stats["total"] < 10.0  # Should be less than 10MB

    def test_batch_processing_efficiency(self):
        """Test that batch processing works for different sizes."""
        datapoints = [MoleculeDatapoint("batch_test", f"smiles_{i}", True) for i in range(10)]
        dataset = MoleculeDataset("batch_test", datapoints)

        with patch("themap.data.molecule_dataset.get_featurizer") as mock_get_featurizer:
            mock_featurizer = Mock()
            mock_featurizer.preprocess.side_effect = lambda batch: (batch, None)
            mock_featurizer.transform.side_effect = lambda batch: np.random.rand(len(batch), 3)
            mock_get_featurizer.return_value = mock_featurizer

            # Test different batch sizes
            for batch_size in [3, 5, 15]:  # Including larger than dataset
                features = dataset.get_dataset_embedding(
                    "test_featurizer", batch_size=batch_size, force_recompute=True
                )
                assert features.shape == (10, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=themap.data.molecule_dataset", "--cov-report=term-missing"])
