"""
Isolated unit tests for the simplified MoleculeDataset class.

This module provides comprehensive isolated unit tests for MoleculeDataset,
focusing on the streamlined structure optimized for N×M distance computation.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from themap.data.molecule_dataset import MoleculeDataset, get_task_name_from_path


class TestGetTaskNameFromPath:
    """Test the helper function get_task_name_from_path."""

    def test_get_task_name_from_path_jsonl_gz(self):
        """Test extracting task name from .jsonl.gz file."""
        mock_path = Mock()
        mock_path.basename.return_value = "CHEMBL123456.jsonl.gz"

        result = get_task_name_from_path(mock_path)
        assert result == "CHEMBL123456"

    def test_get_task_name_from_path_string(self):
        """Test extracting task name from string path."""
        result = get_task_name_from_path("/path/to/CHEMBL123456.jsonl.gz")
        assert result == "CHEMBL123456"

    def test_get_task_name_from_path_pathlib(self):
        """Test extracting task name from pathlib.Path."""
        result = get_task_name_from_path(Path("/path/to/CHEMBL789.jsonl.gz"))
        assert result == "CHEMBL789"

    def test_get_task_name_from_path_other_extension(self):
        """Test extracting task name from non-.jsonl.gz file."""
        result = get_task_name_from_path("/path/to/CHEMBL123456.csv")
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
        dataset = MoleculeDataset(task_id="test_task")

        assert dataset.task_id == "test_task"
        assert dataset.smiles_list == []
        assert len(dataset.labels) == 0
        assert dataset._features is None
        assert dataset._featurizer_name is None

    def test_init_with_data(self):
        """Test initialization with data."""
        smiles = ["c1ccccc1", "CCO", "CC"]
        labels = np.array([1, 0, 1], dtype=np.int32)

        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=smiles,
            labels=labels,
        )

        assert dataset.task_id == "test_task"
        assert len(dataset) == 3
        assert dataset.smiles_list == smiles
        np.testing.assert_array_equal(dataset.labels, labels)

    def test_init_with_numeric_labels(self):
        """Test initialization with numeric labels."""
        smiles = ["CCO", "CCN"]
        labels = np.array([1, 0], dtype=np.int32)
        numeric_labels = np.array([1.5, -0.5], dtype=np.float32)

        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=smiles,
            labels=labels,
            numeric_labels=numeric_labels,
        )

        np.testing.assert_array_equal(dataset.numeric_labels, numeric_labels)

    def test_init_labels_list_converted_to_array(self):
        """Test that labels list is converted to numpy array."""
        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=["CCO", "CCN"],
            labels=[1, 0],  # Pass as list
        )

        assert isinstance(dataset.labels, np.ndarray)
        assert dataset.labels.dtype == np.int32

    def test_init_invalid_task_id_none(self):
        """Test initialization with None task_id."""
        with pytest.raises(TypeError, match="task_id must be a string"):
            MoleculeDataset(task_id=None)

    def test_init_invalid_task_id_number(self):
        """Test initialization with numeric task_id."""
        with pytest.raises(TypeError, match="task_id must be a string"):
            MoleculeDataset(task_id=12345)

    def test_init_invalid_smiles_list_type(self):
        """Test initialization with non-list smiles_list."""
        with pytest.raises(TypeError, match="smiles_list must be a list"):
            MoleculeDataset(task_id="test_task", smiles_list="not_a_list")

    def test_init_mismatched_lengths(self):
        """Test initialization with mismatched smiles and labels lengths."""
        with pytest.raises(ValueError, match="smiles_list length.*must match"):
            MoleculeDataset(
                task_id="test_task",
                smiles_list=["CCO", "CCN", "CCC"],
                labels=np.array([1, 0], dtype=np.int32),
            )


class TestMoleculeDatasetDunderMethods:
    """Test MoleculeDataset dunder methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return MoleculeDataset(
            task_id="test_task",
            smiles_list=["c1ccccc1", "CCO", "CC"],
            labels=np.array([1, 0, 1], dtype=np.int32),
        )

    def test_len_empty(self):
        """Test __len__ with empty dataset."""
        dataset = MoleculeDataset(task_id="test_task")
        assert len(dataset) == 0

    def test_len_with_data(self, sample_dataset):
        """Test __len__ with data."""
        assert len(sample_dataset) == 3

    def test_repr_empty(self):
        """Test __repr__ with empty dataset."""
        dataset = MoleculeDataset(task_id="empty_task")
        repr_str = repr(dataset)
        assert "MoleculeDataset" in repr_str
        assert "task_id=empty_task" in repr_str
        assert "size=0" in repr_str

    def test_repr_with_data(self, sample_dataset):
        """Test __repr__ with data."""
        repr_str = repr(sample_dataset)
        assert "MoleculeDataset" in repr_str
        assert "task_id=test_task" in repr_str
        assert "size=3" in repr_str

    def test_repr_with_featurizer(self, sample_dataset):
        """Test __repr__ shows featurizer when set."""
        features = np.random.rand(3, 10).astype(np.float32)
        sample_dataset.set_features(features, "ecfp")

        repr_str = repr(sample_dataset)
        assert "featurizer=ecfp" in repr_str


class TestMoleculeDatasetProperties:
    """Test MoleculeDataset property methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return MoleculeDataset(
            task_id="test_task",
            smiles_list=["c1ccccc1", "CCO", "CC"],
            labels=np.array([1, 0, 1], dtype=np.int32),
            numeric_labels=np.array([1.5, -0.5, 2.0], dtype=np.float32),
        )

    def test_smiles_property(self, sample_dataset):
        """Test smiles property (alias for smiles_list)."""
        assert sample_dataset.smiles == sample_dataset.smiles_list
        assert sample_dataset.smiles == ["c1ccccc1", "CCO", "CC"]

    def test_positive_ratio(self, sample_dataset):
        """Test positive_ratio property."""
        ratio = sample_dataset.positive_ratio
        assert ratio == 0.6667  # 2/3 rounded to 4 decimal places

    def test_positive_ratio_all_positive(self):
        """Test positive_ratio with all positive labels."""
        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=["CCO", "CCN"],
            labels=np.array([1, 1], dtype=np.int32),
        )
        assert dataset.positive_ratio == 1.0

    def test_positive_ratio_all_negative(self):
        """Test positive_ratio with all negative labels."""
        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=["CCO", "CCN"],
            labels=np.array([0, 0], dtype=np.int32),
        )
        assert dataset.positive_ratio == 0.0

    def test_positive_ratio_empty(self):
        """Test positive_ratio with empty dataset."""
        dataset = MoleculeDataset(task_id="test_task")
        assert dataset.positive_ratio == 0.0

    def test_features_property_none(self, sample_dataset):
        """Test features property when not set."""
        assert sample_dataset.features is None

    def test_features_property_after_set(self, sample_dataset):
        """Test features property after setting features."""
        features = np.random.rand(3, 10).astype(np.float32)
        sample_dataset.set_features(features, "ecfp")
        np.testing.assert_array_equal(sample_dataset.features, features)

    def test_featurizer_name_property(self, sample_dataset):
        """Test featurizer_name property."""
        assert sample_dataset.featurizer_name is None

        features = np.random.rand(3, 10).astype(np.float32)
        sample_dataset.set_features(features, "ecfp")
        assert sample_dataset.featurizer_name == "ecfp"

    def test_has_features(self, sample_dataset):
        """Test has_features method."""
        assert sample_dataset.has_features() is False

        features = np.random.rand(3, 10).astype(np.float32)
        sample_dataset.set_features(features, "ecfp")
        assert sample_dataset.has_features() is True


class TestMoleculeDatasetFeatures:
    """Test MoleculeDataset feature methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return MoleculeDataset(
            task_id="test_task",
            smiles_list=["c1ccccc1", "CCO", "CC", "CCC"],
            labels=np.array([1, 0, 1, 0], dtype=np.int32),
        )

    def test_set_features_valid(self, sample_dataset):
        """Test set_features with valid input."""
        features = np.random.rand(4, 10).astype(np.float32)
        sample_dataset.set_features(features, "ecfp")

        assert sample_dataset._features is not None
        assert sample_dataset._featurizer_name == "ecfp"
        assert sample_dataset._features.dtype == np.float32

    def test_set_features_wrong_count(self, sample_dataset):
        """Test set_features with wrong feature count."""
        features = np.random.rand(3, 10).astype(np.float32)  # Wrong count

        with pytest.raises(ValueError, match="Feature count.*must match"):
            sample_dataset.set_features(features, "ecfp")

    def test_clear_features(self, sample_dataset):
        """Test clear_features method."""
        features = np.random.rand(4, 10).astype(np.float32)
        sample_dataset.set_features(features, "ecfp")

        sample_dataset.clear_features()

        assert sample_dataset._features is None
        assert sample_dataset._featurizer_name is None


class TestMoleculeDatasetPrototype:
    """Test MoleculeDataset prototype computation."""

    @pytest.fixture
    def balanced_dataset(self):
        """Create balanced dataset for prototype testing."""
        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=["mol1", "mol2", "mol3", "mol4"],
            labels=np.array([1, 1, 0, 0], dtype=np.int32),
        )
        # Set features with known values
        features = np.array([
            [1.0, 2.0, 3.0],  # positive
            [2.0, 3.0, 4.0],  # positive
            [5.0, 6.0, 7.0],  # negative
            [6.0, 7.0, 8.0],  # negative
        ], dtype=np.float32)
        dataset.set_features(features, "test_featurizer")
        return dataset

    def test_get_prototype_success(self, balanced_dataset):
        """Test successful prototype computation."""
        pos_proto, neg_proto = balanced_dataset.get_prototype()

        assert isinstance(pos_proto, np.ndarray)
        assert isinstance(neg_proto, np.ndarray)
        assert pos_proto.shape == (3,)
        assert neg_proto.shape == (3,)

        # Check values
        expected_pos = np.mean([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], axis=0)
        expected_neg = np.mean([[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]], axis=0)

        np.testing.assert_array_almost_equal(pos_proto, expected_pos)
        np.testing.assert_array_almost_equal(neg_proto, expected_neg)

    def test_get_prototype_no_features(self):
        """Test get_prototype when features not set."""
        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=["CCO", "CCN"],
            labels=np.array([1, 0], dtype=np.int32),
        )

        with pytest.raises(ValueError, match="Features must be set"):
            dataset.get_prototype()

    def test_get_prototype_no_positive(self):
        """Test get_prototype with no positive examples."""
        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=["CCO", "CCN"],
            labels=np.array([0, 0], dtype=np.int32),
        )
        features = np.random.rand(2, 5).astype(np.float32)
        dataset.set_features(features, "test")

        with pytest.raises(ValueError, match="no positive examples"):
            dataset.get_prototype()

    def test_get_prototype_no_negative(self):
        """Test get_prototype with no negative examples."""
        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=["CCO", "CCN"],
            labels=np.array([1, 1], dtype=np.int32),
        )
        features = np.random.rand(2, 5).astype(np.float32)
        dataset.set_features(features, "test")

        with pytest.raises(ValueError, match="no negative examples"):
            dataset.get_prototype()

    def test_get_class_features(self, balanced_dataset):
        """Test get_class_features method."""
        pos_features, neg_features = balanced_dataset.get_class_features()

        assert pos_features.shape == (2, 3)
        assert neg_features.shape == (2, 3)

    def test_get_class_features_no_features(self):
        """Test get_class_features when features not set."""
        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=["CCO"],
            labels=np.array([1], dtype=np.int32),
        )

        with pytest.raises(ValueError, match="Features must be set"):
            dataset.get_class_features()


class TestMoleculeDatasetLoadFromFile:
    """Test MoleculeDataset.load_from_file method."""

    @patch("themap.data.molecule_dataset.RichPath")
    def test_load_from_file_string_path(self, mock_rich_path):
        """Test load_from_file with string path."""
        mock_path = Mock()
        mock_path.read_by_file_suffix.return_value = [
            {"SMILES": "c1ccccc1", "Property": "1", "RegressionProperty": "1.5"},
            {"SMILES": "CCO", "Property": "0", "RegressionProperty": "-0.5"},
        ]
        mock_rich_path.create.return_value = mock_path

        dataset = MoleculeDataset.load_from_file("datasets/train/test_task.jsonl.gz")

        mock_rich_path.create.assert_called_once_with("datasets/train/test_task.jsonl.gz")
        assert dataset.task_id == "test_task"
        assert len(dataset) == 2
        assert dataset.smiles_list == ["c1ccccc1", "CCO"]
        np.testing.assert_array_equal(dataset.labels, [1, 0])
        np.testing.assert_array_almost_equal(dataset.numeric_labels, [1.5, -0.5])

    @patch("themap.data.molecule_dataset.RichPath")
    def test_load_from_file_missing_regression_property(self, mock_rich_path):
        """Test load_from_file with missing RegressionProperty."""
        mock_path = Mock()
        mock_path.read_by_file_suffix.return_value = [
            {"SMILES": "c1ccccc1", "Property": "1"},  # Missing RegressionProperty
        ]
        mock_rich_path.create.return_value = mock_path

        dataset = MoleculeDataset.load_from_file("test_path.jsonl.gz")

        assert len(dataset) == 1
        assert dataset.numeric_labels is None

    @patch("themap.data.molecule_dataset.RichPath")
    def test_load_from_file_empty_file(self, mock_rich_path):
        """Test load_from_file with empty file."""
        mock_path = Mock()
        mock_path.read_by_file_suffix.return_value = []
        mock_rich_path.create.return_value = mock_path

        dataset = MoleculeDataset.load_from_file("empty_file.jsonl.gz")

        assert dataset.task_id == "empty_file"
        assert len(dataset) == 0


class TestMoleculeDatasetSerialization:
    """Test MoleculeDataset serialization methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return MoleculeDataset(
            task_id="test_task",
            smiles_list=["c1ccccc1", "CCO"],
            labels=np.array([1, 0], dtype=np.int32),
            numeric_labels=np.array([1.5, -0.5], dtype=np.float32),
        )

    def test_to_dict(self, sample_dataset):
        """Test to_dict method."""
        result = sample_dataset.to_dict()

        assert result["task_id"] == "test_task"
        assert result["smiles_list"] == ["c1ccccc1", "CCO"]
        assert result["labels"] == [1, 0]
        assert result["numeric_labels"] == [1.5, -0.5]
        assert result["featurizer_name"] is None

    def test_from_dict(self):
        """Test from_dict method."""
        data = {
            "task_id": "test_task",
            "smiles_list": ["c1ccccc1", "CCO"],
            "labels": [1, 0],
            "numeric_labels": [1.5, -0.5],
        }

        dataset = MoleculeDataset.from_dict(data)

        assert dataset.task_id == "test_task"
        assert dataset.smiles_list == ["c1ccccc1", "CCO"]
        np.testing.assert_array_equal(dataset.labels, [1, 0])

    def test_roundtrip_serialization(self, sample_dataset):
        """Test that serialization roundtrip preserves data."""
        data = sample_dataset.to_dict()
        restored = MoleculeDataset.from_dict(data)

        assert restored.task_id == sample_dataset.task_id
        assert restored.smiles_list == sample_dataset.smiles_list
        np.testing.assert_array_equal(restored.labels, sample_dataset.labels)


class TestMoleculeDatasetUtilityMethods:
    """Test MoleculeDataset utility methods."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return MoleculeDataset(
            task_id="test_task",
            smiles_list=["c1ccccc1", "CCO", "CC"],
            labels=np.array([1, 0, 1], dtype=np.int32),
            numeric_labels=np.array([1.5, -0.5, 2.0], dtype=np.float32),
        )

    def test_filter_by_indices(self, sample_dataset):
        """Test filter_by_indices method."""
        filtered = sample_dataset.filter_by_indices([0, 2])

        assert len(filtered) == 2
        assert filtered.smiles_list == ["c1ccccc1", "CC"]
        np.testing.assert_array_equal(filtered.labels, [1, 1])
        np.testing.assert_array_almost_equal(filtered.numeric_labels, [1.5, 2.0])

    def test_filter_by_indices_empty(self, sample_dataset):
        """Test filter_by_indices with empty indices."""
        filtered = sample_dataset.filter_by_indices([])

        assert len(filtered) == 0
        assert filtered.task_id == sample_dataset.task_id

    def test_get_statistics_valid(self, sample_dataset):
        """Test get_statistics method."""
        stats = sample_dataset.get_statistics()

        assert stats["size"] == 3
        assert stats["positive_count"] == 2
        assert stats["negative_count"] == 1
        assert stats["positive_ratio"] == 0.6667
        assert "numeric_mean" in stats
        assert "numeric_std" in stats

    def test_get_statistics_empty(self):
        """Test get_statistics with empty dataset."""
        dataset = MoleculeDataset(task_id="test_task")
        stats = dataset.get_statistics()

        assert stats["size"] == 0
        assert stats["positive_ratio"] == 0.0

    def test_get_statistics_with_features(self, sample_dataset):
        """Test get_statistics shows feature info when available."""
        features = np.random.rand(3, 10).astype(np.float32)
        sample_dataset.set_features(features, "ecfp")

        stats = sample_dataset.get_statistics()

        assert stats["feature_dim"] == 10
        assert stats["featurizer"] == "ecfp"


class TestMoleculeDatasetBackwardCompatibility:
    """Test backward compatibility with metalearning module."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return MoleculeDataset(
            task_id="test_task",
            smiles_list=["c1ccccc1", "CCO", "CC"],
            labels=np.array([1, 0, 1], dtype=np.int32),
            numeric_labels=np.array([1.5, -0.5, 2.0], dtype=np.float32),
        )

    def test_datapoints_property(self, sample_dataset):
        """Test datapoints property for backward compatibility."""
        datapoints = sample_dataset.datapoints

        assert len(datapoints) == 3
        assert all(isinstance(dp, dict) for dp in datapoints)

        # Check structure of each datapoint
        assert datapoints[0]["smiles"] == "c1ccccc1"
        assert datapoints[0]["labels"] == 1
        assert datapoints[0]["bool_label"] is True
        assert datapoints[0]["numeric_label"] == 1.5

        assert datapoints[1]["smiles"] == "CCO"
        assert datapoints[1]["labels"] == 0
        assert datapoints[1]["bool_label"] is False
        assert datapoints[1]["numeric_label"] == -0.5

    def test_data_property_alias(self, sample_dataset):
        """Test data property is alias for datapoints."""
        assert sample_dataset.data == sample_dataset.datapoints

    def test_datapoints_without_numeric_labels(self):
        """Test datapoints property without numeric labels."""
        dataset = MoleculeDataset(
            task_id="test_task",
            smiles_list=["CCO"],
            labels=np.array([1], dtype=np.int32),
        )

        datapoints = dataset.datapoints
        assert datapoints[0]["numeric_label"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=themap.data.molecule_dataset", "--cov-report=term-missing"])
