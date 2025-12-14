"""Tests for the MoleculeDatasets class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dpu_utils.utils import RichPath

from themap.data.molecule_dataset import MoleculeDataset
from themap.data.molecule_datasets import DataFold, MoleculeDatasets


@pytest.fixture
def mock_molecule_dataset():
    """Create a mock MoleculeDataset for testing."""
    dataset = MoleculeDataset(
        task_id="test_task_1",
        smiles_list=["CCO", "CCN", "CCC"],
        labels=np.array([1, 0, 1], dtype=np.int32),
        numeric_labels=np.array([0.8, 0.2, 0.9], dtype=np.float32),
    )
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


class TestMoleculeDatasetsLoadDatasets:
    """Test class for testing the load_datasets method of MoleculeDatasets."""

    @patch("themap.data.molecule_dataset.MoleculeDataset.load_from_file")
    def test_load_datasets_all_folds(
        self, mock_load, sample_train_paths, sample_valid_paths, sample_test_paths, mock_molecule_dataset
    ):
        """Test loading datasets from all folds."""
        mock_load.return_value = mock_molecule_dataset

        datasets = MoleculeDatasets(
            train_data_paths=sample_train_paths,
            valid_data_paths=sample_valid_paths,
            test_data_paths=sample_test_paths,
        )

        loaded_datasets = datasets.load_datasets()

        expected_count = len(sample_train_paths) + len(sample_valid_paths) + len(sample_test_paths)
        assert len(loaded_datasets) == expected_count
        assert mock_load.call_count == expected_count

        # Verify specific dataset names are present
        assert "train_CHEMBL894522" in loaded_datasets
        assert "train_CHEMBL1023359" in loaded_datasets
        assert "valid_CHEMBL2219358" in loaded_datasets
        assert "test_CHEMBL2219236" in loaded_datasets
        assert "test_CHEMBL1963831" in loaded_datasets

    @patch("themap.data.molecule_dataset.MoleculeDataset.load_from_file")
    def test_load_datasets_specific_folds(
        self, mock_load, sample_train_paths, sample_valid_paths, mock_molecule_dataset
    ):
        """Test loading datasets from specific folds."""
        mock_load.return_value = mock_molecule_dataset

        datasets = MoleculeDatasets(train_data_paths=sample_train_paths, valid_data_paths=sample_valid_paths)

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
        mock_load.return_value = mock_molecule_dataset

        datasets = MoleculeDatasets(train_data_paths=sample_train_paths)

        # Load datasets twice
        datasets.load_datasets(folds=[DataFold.TRAIN])
        datasets.load_datasets(folds=[DataFold.TRAIN])

        # Should only call load_from_file once per dataset
        assert mock_load.call_count == len(sample_train_paths)


class TestMoleculeDatasetsDistanceComputation:
    """Test distance computation helper methods."""

    @patch("themap.data.molecule_dataset.MoleculeDataset.load_from_file")
    def test_get_datasets_for_distance_computation(self, mock_load, mock_molecule_dataset):
        """Test get_datasets_for_distance_computation method."""
        mock_load.return_value = mock_molecule_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            train_dir = Path(temp_dir) / "train"
            valid_dir = Path(temp_dir) / "valid"
            test_dir = Path(temp_dir) / "test"
            train_dir.mkdir()
            valid_dir.mkdir()
            test_dir.mkdir()

            (train_dir / "CHEMBL123.jsonl.gz").touch()
            (train_dir / "CHEMBL456.jsonl.gz").touch()
            (valid_dir / "CHEMBL789.jsonl.gz").touch()
            (test_dir / "CHEMBL999.jsonl.gz").touch()

            datasets = MoleculeDatasets.from_directory(temp_dir)

            source_datasets, target_datasets, source_names, target_names = (
                datasets.get_datasets_for_distance_computation(
                    source_fold=DataFold.TRAIN,
                    target_folds=[DataFold.VALIDATION, DataFold.TEST],
                )
            )

            # Check source datasets (train)
            assert len(source_datasets) == 2
            assert len(source_names) == 2
            assert all("train_" in name for name in source_names)

            # Check target datasets (validation + test)
            assert len(target_datasets) == 2
            assert len(target_names) == 2
            assert any("valid_" in name for name in target_names)
            assert any("test_" in name for name in target_names)

    @patch("themap.data.molecule_dataset.MoleculeDataset.load_from_file")
    def test_get_all_smiles(self, mock_load):
        """Test get_all_smiles method."""
        # Create different datasets with overlapping SMILES
        dataset1 = MoleculeDataset(
            task_id="task1",
            smiles_list=["CCO", "CCN"],
            labels=np.array([1, 0], dtype=np.int32),
        )
        dataset2 = MoleculeDataset(
            task_id="task2",
            smiles_list=["CCN", "CCC"],  # CCN overlaps
            labels=np.array([1, 0], dtype=np.int32),
        )

        mock_load.side_effect = [dataset1, dataset2]

        with tempfile.TemporaryDirectory() as temp_dir:
            train_dir = Path(temp_dir) / "train"
            train_dir.mkdir()
            (train_dir / "CHEMBL1.jsonl.gz").touch()
            (train_dir / "CHEMBL2.jsonl.gz").touch()

            datasets = MoleculeDatasets.from_directory(temp_dir)
            all_smiles = datasets.get_all_smiles(folds=[DataFold.TRAIN])

            # Should deduplicate
            assert len(all_smiles) == 3
            assert set(all_smiles) == {"CCO", "CCN", "CCC"}


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

    def test_clear_loaded_datasets(self):
        """Test clear_loaded_datasets method."""
        datasets = MoleculeDatasets()
        datasets._loaded_datasets = {"test": MagicMock()}

        datasets.clear_loaded_datasets()

        assert len(datasets._loaded_datasets) == 0


class TestMoleculeDatasetsValidation:
    """Test validation methods."""

    def test_invalid_num_workers_type(self):
        """Test initialization with invalid num_workers type."""
        from themap.data.exceptions import DatasetValidationError

        with pytest.raises(DatasetValidationError):
            MoleculeDatasets(num_workers="invalid")

    def test_invalid_num_workers_zero(self):
        """Test initialization with num_workers=0."""
        from themap.data.exceptions import DatasetValidationError

        with pytest.raises(DatasetValidationError):
            MoleculeDatasets(num_workers=0)

    def test_invalid_num_workers_negative(self):
        """Test initialization with negative num_workers (except -1)."""
        from themap.data.exceptions import DatasetValidationError

        with pytest.raises(DatasetValidationError):
            MoleculeDatasets(num_workers=-5)

    def test_valid_num_workers_minus_one(self):
        """Test initialization with num_workers=-1 (valid)."""
        datasets = MoleculeDatasets(num_workers=-1)
        assert datasets._num_workers == -1


class TestMoleculeDatasetsIntegration:
    """Integration tests with more realistic scenarios."""

    def test_full_workflow(self):
        """Test a complete workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure
            train_dir = Path(temp_dir) / "train"
            train_dir.mkdir()

            # Create a mock jsonl.gz file content
            mock_data = [
                {"SMILES": "CCO", "Property": "1", "RegressionProperty": "0.8"},
                {"SMILES": "CCN", "Property": "0", "RegressionProperty": "0.2"},
            ]

            with patch("dpu_utils.utils.richpath.RichPath.read_by_file_suffix") as mock_read:
                mock_read.return_value = mock_data

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
                    assert dataset.smiles_list[0] == "CCO"
                    assert dataset.smiles_list[1] == "CCN"

    def test_task_filtering_comprehensive(self):
        """Test comprehensive task filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create complex directory structure
            train_dir = Path(temp_dir) / "train"
            valid_dir = Path(temp_dir) / "valid"
            test_dir = Path(temp_dir) / "test"

            train_dir.mkdir()
            valid_dir.mkdir()
            test_dir.mkdir()

            # Create multiple tasks per fold
            (train_dir / "CHEMBL1000.jsonl.gz").touch()
            (train_dir / "CHEMBL1001.jsonl.gz").touch()
            (train_dir / "CHEMBL1002.jsonl.gz").touch()

            (valid_dir / "CHEMBL2000.jsonl.gz").touch()
            (valid_dir / "CHEMBL2001.jsonl.gz").touch()

            (test_dir / "CHEMBL3000.jsonl.gz").touch()

            # Test with fold-specific JSON task list
            task_list_data = {
                "train": ["CHEMBL1000", "CHEMBL1001"],  # Select 2 out of 3
                "valid": ["CHEMBL2001"],  # Select 1 out of 2
                "test": [],  # Empty
            }
            task_list_file = Path(temp_dir) / "selective_tasks.json"
            task_list_file.write_text(json.dumps(task_list_data))

            datasets = MoleculeDatasets.from_directory(temp_dir, task_list_file=str(task_list_file))

            # Verify selective loading
            assert datasets.get_num_fold_tasks(DataFold.TRAIN) == 2
            assert datasets.get_num_fold_tasks(DataFold.VALIDATION) == 1
            assert datasets.get_num_fold_tasks(DataFold.TEST) == 0

            train_names = datasets.get_task_names(DataFold.TRAIN)
            assert "CHEMBL1000" in train_names
            assert "CHEMBL1001" in train_names
            assert "CHEMBL1002" not in train_names


class TestMoleculeDatasetsFileExtensions:
    """Test file extension handling."""

    def test_only_jsonl_gz_files_loaded(self):
        """Test that only .jsonl.gz files are loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_dir = Path(temp_dir) / "train"
            train_dir.mkdir()

            # Create files with different extensions
            (train_dir / "CHEMBL123.jsonl.gz").touch()
            (train_dir / "CHEMBL456.json").touch()  # Wrong extension
            (train_dir / "CHEMBL789.txt").touch()  # Wrong extension
            (train_dir / "CHEMBL999.csv").touch()  # Wrong extension

            datasets = MoleculeDatasets.from_directory(temp_dir)

            # Should only load .jsonl.gz files
            assert datasets.get_num_fold_tasks(DataFold.TRAIN) == 1
            assert "CHEMBL123" in datasets.get_task_names(DataFold.TRAIN)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=themap.data.molecule_datasets", "--cov-report=term-missing"])
