"""Tests for the MoleculeDatasets class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dpu_utils.utils import RichPath

from themap.data.molecule_datapoint import MoleculeDatapoint
from themap.data.molecule_dataset import MoleculeDataset
from themap.data.molecule_datasets import DataFold, MoleculeDatasets


@pytest.fixture
def mock_molecule_datapoints():
    """Create mock molecule datapoints for testing."""
    return [
        MoleculeDatapoint(
            task_id="test_task_1",
            smiles="CCO",  # ethanol
            bool_label=True,
            numeric_label=0.8,
        ),
        MoleculeDatapoint(
            task_id="test_task_1",
            smiles="CCN",  # ethylamine
            bool_label=False,
            numeric_label=0.2,
        ),
        MoleculeDatapoint(
            task_id="test_task_1",
            smiles="CCC",  # propane
            bool_label=True,
            numeric_label=0.9,
        ),
    ]


@pytest.fixture
def mock_molecule_dataset(mock_molecule_datapoints):
    """Create a mock MoleculeDataset for testing."""
    dataset = MoleculeDataset(task_id="test_task_1", data=mock_molecule_datapoints)
    # Add mock features
    dataset._features = np.random.rand(3, 10)
    return dataset


@pytest.fixture
def sample_train_paths():
    """Create sample training data paths."""
    return [
        RichPath.create("datasets/train/CHEMBL894522.jsonl.gz"),
        RichPath.create("datasets/train/CHEMBL1023359.jsonl.gz"),
    ]


@pytest.fixture
def sample_valid_paths():
    """Create sample validation data paths."""
    return [RichPath.create("datasets/valid/CHEMBL2219358.jsonl.gz")]


@pytest.fixture
def sample_test_paths():
    """Create sample test data paths."""
    return [
        RichPath.create("datasets/test/CHEMBL2219236.jsonl.gz"),
        RichPath.create("datasets/test/CHEMBL1963831.jsonl.gz"),
    ]


class TestDataFold:
    """Test the DataFold enum."""

    def test_data_fold_values(self):
        """Test DataFold enum values."""
        assert DataFold.TRAIN == 0
        assert DataFold.VALIDATION == 1
        assert DataFold.TEST == 2


class TestMoleculeDatasets:
    """Test the MoleculeDatasets class."""

    def test_init_default(self):
        """Test default initialization."""
        datasets = MoleculeDatasets()

        assert len(datasets._fold_to_data_paths[DataFold.TRAIN]) == 0
        assert len(datasets._fold_to_data_paths[DataFold.VALIDATION]) == 0
        assert len(datasets._fold_to_data_paths[DataFold.TEST]) == 0
        assert datasets._num_workers is None
        assert datasets.cache_dir is None
        assert datasets.global_cache is None
        assert len(datasets._loaded_datasets) == 0

    def test_init_with_paths(self, sample_train_paths, sample_valid_paths, sample_test_paths):
        """Test initialization with paths."""
        datasets = MoleculeDatasets(
            train_data_paths=sample_train_paths,
            valid_data_paths=sample_valid_paths,
            test_data_paths=sample_test_paths,
            num_workers=4,
        )

        assert len(datasets._fold_to_data_paths[DataFold.TRAIN]) == 2
        assert len(datasets._fold_to_data_paths[DataFold.VALIDATION]) == 1
        assert len(datasets._fold_to_data_paths[DataFold.TEST]) == 2
        assert datasets._num_workers == 4

    def test_init_with_cache_dir(self, sample_train_paths):
        """Test initialization with cache directory."""
        with tempfile.TemporaryDirectory() as cache_dir:
            datasets = MoleculeDatasets(train_data_paths=sample_train_paths, cache_dir=cache_dir)

            assert datasets.cache_dir == Path(cache_dir)
            assert datasets.global_cache is not None

    def test_repr(self, sample_train_paths, sample_valid_paths, sample_test_paths):
        """Test string representation."""
        datasets = MoleculeDatasets(
            train_data_paths=sample_train_paths,
            valid_data_paths=sample_valid_paths,
            test_data_paths=sample_test_paths,
        )

        expected = "MoleculeDatasets(train=2, valid=1, test=2)"
        assert repr(datasets) == expected

    def test_get_num_fold_tasks(self, sample_train_paths, sample_valid_paths, sample_test_paths):
        """Test getting number of tasks in each fold."""
        datasets = MoleculeDatasets(
            train_data_paths=sample_train_paths,
            valid_data_paths=sample_valid_paths,
            test_data_paths=sample_test_paths,
        )

        assert datasets.get_num_fold_tasks(DataFold.TRAIN) == 2
        assert datasets.get_num_fold_tasks(DataFold.VALIDATION) == 1
        assert datasets.get_num_fold_tasks(DataFold.TEST) == 2

    def test_get_task_names(self, sample_train_paths, sample_valid_paths, sample_test_paths):
        """Test getting task names for a fold."""
        datasets = MoleculeDatasets(
            train_data_paths=sample_train_paths,
            valid_data_paths=sample_valid_paths,
            test_data_paths=sample_test_paths,
        )

        train_names = datasets.get_task_names(DataFold.TRAIN)
        assert train_names == ["CHEMBL894522", "CHEMBL1023359"]

        valid_names = datasets.get_task_names(DataFold.VALIDATION)
        assert valid_names == ["CHEMBL2219358"]

        test_names = datasets.get_task_names(DataFold.TEST)
        assert test_names == ["CHEMBL2219236", "CHEMBL1963831"]


class TestMoleculeDatasetsFromDirectory:
    """Test the from_directory static method."""

    def test_from_directory_basic(self):
        """Test basic from_directory functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            train_dir = Path(temp_dir) / "train"
            valid_dir = Path(temp_dir) / "valid"
            test_dir = Path(temp_dir) / "test"

            train_dir.mkdir()
            valid_dir.mkdir()
            test_dir.mkdir()

            # Create mock files
            (train_dir / "CHEMBL894522.jsonl.gz").touch()
            (train_dir / "CHEMBL1023359.jsonl.gz").touch()
            (valid_dir / "CHEMBL1613776.jsonl.gz").touch()
            (test_dir / "CHEMBL1963831.jsonl.gz").touch()

            # Test loading
            datasets = MoleculeDatasets.from_directory(temp_dir)

            assert datasets.get_num_fold_tasks(DataFold.TRAIN) == 2
            assert datasets.get_num_fold_tasks(DataFold.VALIDATION) == 1
            assert datasets.get_num_fold_tasks(DataFold.TEST) == 1

    def test_from_directory_with_task_list_text(self):
        """Test from_directory with text task list file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            train_dir = Path(temp_dir) / "train"
            train_dir.mkdir()

            # Create mock files
            (train_dir / "CHEMBL894522.jsonl.gz").touch()
            (train_dir / "CHEMBL1023359.jsonl.gz").touch()
            (train_dir / "CHEMBL1613776.jsonl.gz").touch()

            # Create task list file
            task_list_file = Path(temp_dir) / "tasks.txt"
            task_list_file.write_text("CHEMBL894522\nCHEMBL1023359\n")

            # Test loading
            datasets = MoleculeDatasets.from_directory(temp_dir, task_list_file=str(task_list_file))

            # Should only load tasks in the list
            assert datasets.get_num_fold_tasks(DataFold.TRAIN) == 2
            train_names = datasets.get_task_names(DataFold.TRAIN)
            assert "CHEMBL894522" in train_names
            assert "CHEMBL1023359" in train_names
            assert "CHEMBL1613776" not in train_names

    def test_from_directory_with_task_list_json(self):
        """Test from_directory with JSON task list file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            train_dir = Path(temp_dir) / "train"
            valid_dir = Path(temp_dir) / "valid"
            train_dir.mkdir()
            valid_dir.mkdir()

            # Create mock files
            (train_dir / "CHEMBL894522.jsonl.gz").touch()
            (train_dir / "CHEMBL1023359.jsonl.gz").touch()
            (valid_dir / "CHEMBL1613776.jsonl.gz").touch()
            (valid_dir / "CHEMBL2219358.jsonl.gz").touch()

            # Create JSON task list file
            task_list_data = {"train": ["CHEMBL894522"], "valid": ["CHEMBL2219358"], "test": []}
            task_list_file = Path(temp_dir) / "tasks.json"
            task_list_file.write_text(json.dumps(task_list_data))

            # Test loading
            datasets = MoleculeDatasets.from_directory(temp_dir, task_list_file=str(task_list_file))

            # Should only load tasks specified in JSON
            assert datasets.get_num_fold_tasks(DataFold.TRAIN) == 1
            assert datasets.get_num_fold_tasks(DataFold.VALIDATION) == 1

            train_names = datasets.get_task_names(DataFold.TRAIN)
            assert train_names == ["CHEMBL894522"]

            valid_names = datasets.get_task_names(DataFold.VALIDATION)
            assert valid_names == ["CHEMBL2219358"]

    def test_from_directory_nonexistent_directory(self):
        """Test from_directory with nonexistent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only some directories
            train_dir = Path(temp_dir) / "train"
            train_dir.mkdir()
            (train_dir / "CHEMBL894522.jsonl.gz").touch()

            # Don't create valid/test directories

            datasets = MoleculeDatasets.from_directory(temp_dir)

            assert datasets.get_num_fold_tasks(DataFold.TRAIN) == 1
            assert datasets.get_num_fold_tasks(DataFold.VALIDATION) == 0
            assert datasets.get_num_fold_tasks(DataFold.TEST) == 0

    def test_from_directory_with_cache_dir(self):
        """Test from_directory with cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.TemporaryDirectory() as cache_dir:
                # Create directory structure
                train_dir = Path(temp_dir) / "train"
                train_dir.mkdir()
                (train_dir / "CHEMBL894522.jsonl.gz").touch()

                datasets = MoleculeDatasets.from_directory(temp_dir, cache_dir=cache_dir)

                assert datasets.cache_dir == Path(cache_dir)
                assert datasets.global_cache is not None


class TestMoleculeDatasetsLoadDatasets:
    """Test class for testing the load_datasets method of MoleculeDatasets."""

    @patch("themap.data.molecule_dataset.MoleculeDataset.load_from_file")  # Mock the load_from_file method
    def test_load_datasets_all_folds(
        self, mock_load, sample_train_paths, sample_valid_paths, sample_test_paths, mock_molecule_dataset
    ):
        """Test loading datasets from all folds (train, validation, test).

        Args:
            mock_load: Mocked load_from_file method
            sample_train_paths: Fixture containing training data paths
            sample_valid_paths: Fixture containing validation data paths
            sample_test_paths: Fixture containing test data paths
            mock_molecule_dataset: Fixture containing a mock dataset
        """
        # Configure the mock to return our mock dataset when load_from_file is called
        mock_load.return_value = mock_molecule_dataset

        # Create a MoleculeDatasets instance with paths for all three folds
        datasets = MoleculeDatasets(
            train_data_paths=sample_train_paths,
            valid_data_paths=sample_valid_paths,
            test_data_paths=sample_test_paths,
        )

        # Load all datasets from all folds
        loaded_datasets = datasets.load_datasets()

        # Verify total number of loaded datasets matches sum of datasets from all folds
        expected_count = len(sample_train_paths) + len(sample_valid_paths) + len(sample_test_paths)
        assert len(loaded_datasets) == expected_count

        # Verify load_from_file was called once for each dataset
        assert mock_load.call_count == expected_count

        # Verify specific dataset names are present in loaded results
        assert "train_CHEMBL894522" in loaded_datasets  # Check first training dataset
        assert "train_CHEMBL1023359" in loaded_datasets  # Check second training dataset
        assert "valid_CHEMBL2219358" in loaded_datasets  # Check validation dataset
        assert "test_CHEMBL2219236" in loaded_datasets  # Check first test dataset
        assert "test_CHEMBL1963831" in loaded_datasets  # Check second test dataset

    @patch("themap.data.molecule_dataset.MoleculeDataset.load_from_file")
    def test_load_datasets_specific_folds(
        self, mock_load, sample_train_paths, sample_valid_paths, mock_molecule_dataset
    ):
        """Test loading datasets from specific folds."""
        # Setup mock
        mock_load.return_value = mock_molecule_dataset

        datasets = MoleculeDatasets(train_data_paths=sample_train_paths, valid_data_paths=sample_valid_paths)

        # Load only training datasets
        loaded_datasets = datasets.load_datasets(folds=[DataFold.TRAIN])

        assert len(loaded_datasets) == len(sample_train_paths)
        assert mock_load.call_count == len(sample_train_paths)

        # Check that only training datasets are loaded
        assert "train_CHEMBL894522" in loaded_datasets
        assert "train_CHEMBL1023359" in loaded_datasets
        assert "valid_CHEMBL2219358" not in loaded_datasets

    @patch("themap.data.molecule_dataset.MoleculeDataset.load_from_file")
    def test_load_datasets_caching(self, mock_load, sample_train_paths, mock_molecule_dataset):
        """Test that datasets are cached and not reloaded."""
        # Setup mock
        mock_load.return_value = mock_molecule_dataset

        datasets = MoleculeDatasets(train_data_paths=sample_train_paths)

        # Load datasets twice
        datasets.load_datasets(folds=[DataFold.TRAIN])
        datasets.load_datasets(folds=[DataFold.TRAIN])

        # Should only call load_from_file once per dataset
        assert mock_load.call_count == len(sample_train_paths)


class TestMoleculeDatasetsFeatureComputation:
    """Test feature computation methods."""

    @patch("themap.data.molecule_datasets.MoleculeDatasets.load_datasets")
    def test_compute_all_features_without_cache(self, mock_load_datasets, mock_molecule_dataset):
        """Test feature computation without global cache."""
        # Setup mocks
        mock_datasets = {
            "train_CHEMBL123": mock_molecule_dataset,
            "valid_CHEMBL789": mock_molecule_dataset,
        }
        mock_load_datasets.return_value = mock_datasets

        # Mock the get_dataset_embedding method
        with patch.object(mock_molecule_dataset, "get_dataset_embedding") as mock_get_features:
            mock_features = np.random.rand(3, 10)
            mock_get_features.return_value = mock_features

            datasets = MoleculeDatasets()

            result = datasets.compute_all_features_with_deduplication(
                featurizer_name="ecfp", folds=[DataFold.TRAIN, DataFold.VALIDATION]
            )

            # Check results
            assert len(result) == 2
            assert "train_CHEMBL123" in result
            assert "valid_CHEMBL789" in result

            # Check that get_dataset_embedding was called for each dataset
            assert mock_get_features.call_count == 2

    @patch("themap.data.molecule_datasets.MoleculeDatasets.load_datasets")
    @patch("themap.utils.cache_utils.GlobalMoleculeCache")
    def test_compute_all_features_with_cache(
        self, mock_cache_class, mock_load_datasets, mock_molecule_dataset
    ):
        """Test feature computation with global cache."""
        # Setup mocks
        mock_datasets = {"train_CHEMBL123": mock_molecule_dataset}
        mock_load_datasets.return_value = mock_datasets

        # Mock cache
        mock_cache = MagicMock()
        mock_cache_class.return_value = mock_cache

        # Mock cache methods
        unique_smiles_map = {"CCO": [(0, 0)], "CCN": [(0, 1)], "CCC": [(0, 2)]}
        mock_cache.get_unique_smiles_across_datasets.return_value = unique_smiles_map

        features_map = {"CCO": np.array([1, 2, 3]), "CCN": np.array([4, 5, 6]), "CCC": np.array([7, 8, 9])}
        mock_cache.batch_compute_features.return_value = features_map
        mock_cache._canonicalize_smiles.side_effect = lambda x: x

        with tempfile.TemporaryDirectory() as cache_dir:
            datasets = MoleculeDatasets(cache_dir=cache_dir)
            datasets.global_cache = mock_cache

            result = datasets.compute_all_features_with_deduplication(
                featurizer_name="ecfp", folds=[DataFold.TRAIN]
            )

            # Check that cache methods were called
            mock_cache.get_unique_smiles_across_datasets.assert_called_once()
            mock_cache.batch_compute_features.assert_called_once()

            # Check results
            assert len(result) == 1
            assert "train_CHEMBL123" in result
            assert result["train_CHEMBL123"].shape == (3, 3)

    @patch("themap.data.molecule_datasets.MoleculeDatasets.compute_all_features_with_deduplication")
    def test_get_distance_computation_ready_features(self, mock_compute_features):
        """Test get_distance_computation_ready_features method."""
        # Setup mock return value
        mock_features = {
            "train_CHEMBL123": np.random.rand(10, 5),
            "train_CHEMBL456": np.random.rand(8, 5),
            "valid_CHEMBL789": np.random.rand(12, 5),
            "test_CHEMBL999": np.random.rand(6, 5),
        }
        mock_compute_features.return_value = mock_features

        datasets = MoleculeDatasets()

        source_features, target_features, source_names, target_names = (
            datasets.get_distance_computation_ready_features(
                featurizer_name="ecfp",
                source_fold=DataFold.TRAIN,
                target_folds=[DataFold.VALIDATION, DataFold.TEST],
            )
        )

        # Check source features (train)
        assert len(source_features) == 2
        assert len(source_names) == 2
        assert "train_CHEMBL123" in source_names
        assert "train_CHEMBL456" in source_names

        # Check target features (validation + test)
        assert len(target_features) == 2
        assert len(target_names) == 2
        assert "valid_CHEMBL789" in target_names
        assert "test_CHEMBL999" in target_names


class TestMoleculeDatasetsGlobalCache:
    """Test global cache functionality."""

    def test_get_global_cache_stats_no_cache(self):
        """Test get_global_cache_stats when no cache is enabled."""
        datasets = MoleculeDatasets()
        stats = datasets.get_global_cache_stats()
        assert stats is None

    @patch("themap.utils.cache_utils.GlobalMoleculeCache")
    def test_get_global_cache_stats_with_cache(self, mock_cache_class):
        """Test get_global_cache_stats when cache is enabled."""
        # Setup mock
        mock_cache = MagicMock()
        mock_cache_class.return_value = mock_cache

        mock_persistent_cache = MagicMock()
        mock_cache.persistent_cache = mock_persistent_cache

        # Mock cache stats
        mock_persistent_cache.get_stats.return_value = {"hits": 10, "misses": 2}
        mock_persistent_cache.get_cache_size_info.return_value = {"disk_usage_mb": 100}

        with tempfile.TemporaryDirectory() as cache_dir:
            datasets = MoleculeDatasets(cache_dir=cache_dir)
            datasets.global_cache = mock_cache

            stats = datasets.get_global_cache_stats()

            assert stats is not None
            assert "persistent_cache_stats" in stats
            assert "persistent_cache_size" in stats
            assert "loaded_datasets" in stats
            assert stats["loaded_datasets"] == 0


class TestMoleculeDatasetsEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_datasets(self):
        """Test behavior with empty datasets."""
        datasets = MoleculeDatasets()

        # Test with empty folds
        assert datasets.get_num_fold_tasks(DataFold.TRAIN) == 0
        assert datasets.get_task_names(DataFold.TRAIN) == []

        # Test loading empty datasets
        loaded = datasets.load_datasets()
        assert len(loaded) == 0

    def test_from_directory_invalid_task_list(self):
        """Test from_directory with invalid task list file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            train_dir = Path(temp_dir) / "train"
            train_dir.mkdir()
            (train_dir / "CHEMBL123.jsonl.gz").touch()

            # Create invalid JSON file
            task_list_file = Path(temp_dir) / "invalid.json"
            task_list_file.write_text("invalid json {")

            # Should fall back to treating as text file
            datasets = MoleculeDatasets.from_directory(temp_dir, task_list_file=str(task_list_file))

            # Should still work (treating invalid JSON as text)
            assert datasets.get_num_fold_tasks(DataFold.TRAIN) == 0  # No valid task names

    @patch("themap.data.molecule_datasets.MoleculeDatasets.load_datasets")
    def test_compute_features_empty_datasets(self, mock_load_datasets):
        """Test feature computation with empty datasets."""
        mock_load_datasets.return_value = {}

        datasets = MoleculeDatasets()

        # Should handle empty datasets gracefully now
        result = datasets.compute_all_features_with_deduplication(
            featurizer_name="ecfp", folds=[DataFold.TRAIN]
        )

        assert result == {}

    def test_distance_computation_ready_features_default_target_folds(self):
        """Test get_distance_computation_ready_features with default target folds."""
        with patch.object(MoleculeDatasets, "compute_all_features_with_deduplication") as mock_compute:
            mock_compute.return_value = {
                "train_CHEMBL123": np.random.rand(5, 3),
                "valid_CHEMBL456": np.random.rand(5, 3),
                "test_CHEMBL789": np.random.rand(5, 3),
            }

            datasets = MoleculeDatasets()

            # Call without specifying target_folds (should default to validation + test)
            source_features, target_features, source_names, target_names = (
                datasets.get_distance_computation_ready_features(
                    featurizer_name="ecfp", source_fold=DataFold.TRAIN
                )
            )

            # Should use validation and test as targets by default
            assert len(source_features) == 1
            assert len(target_features) == 2
            assert "train_CHEMBL123" in source_names
            assert "valid_CHEMBL456" in target_names
            assert "test_CHEMBL789" in target_names


# Integration test with real-like data
class TestMoleculeDatasetsIntegration:
    """Integration tests with more realistic scenarios."""

    def test_full_workflow_without_cache(self):
        """Test a complete workflow without caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            train_dir = Path(temp_dir) / "train"
            train_dir.mkdir()

            # Create a mock jsonl.gz file content
            mock_data = [
                {"SMILES": "CCO", "Property": "1", "RegressionProperty": "0.8"},
                {"SMILES": "CCN", "Property": "0", "RegressionProperty": "0.2"},
            ]

            # We'll need to mock the file reading since we can't create actual jsonl.gz files easily
            with patch("dpu_utils.utils.richpath.RichPath.read_by_file_suffix") as mock_read:
                mock_read.return_value = mock_data

                # Also mock the exists check
                with patch("dpu_utils.utils.richpath.RichPath.exists") as mock_exists:
                    mock_exists.return_value = True

                    # Create actual file for glob to find
                    (train_dir / "CHEMBL123.jsonl.gz").touch()

                    # Load datasets
                    datasets = MoleculeDatasets.from_directory(temp_dir)

                    # Load the datasets
                    loaded_datasets = datasets.load_datasets(folds=[DataFold.TRAIN])

                    # Should have loaded one dataset
                    assert len(loaded_datasets) == 1
                    assert "train_CHEMBL123" in loaded_datasets

                    # Check dataset content
                    dataset = loaded_datasets["train_CHEMBL123"]
                    assert len(dataset) == 2
                    assert dataset.data[0].smiles == "CCO"
                    assert dataset.data[1].smiles == "CCN"
