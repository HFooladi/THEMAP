"""
Tests for task splitting functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from themap.data.molecule_dataset import MoleculeDataset
from themap.data.tasks import Task
from themap.metalearning.data.task_splits import TaskSplits, create_meta_splits, create_task_folders


class TestTaskSplits:
    """Test suite for TaskSplits."""

    @pytest.fixture
    def mock_tasks(self):
        """Create mock tasks for testing."""
        tasks = []
        for i in range(10):
            task = Mock(spec=Task)
            task.task_id = f"TASK_{i:03d}"
            task.molecule_dataset = Mock(spec=MoleculeDataset)
            task.hardness = i * 0.1  # Mock hardness values
            tasks.append(task)
        return tasks

    @pytest.fixture
    def task_splits(self, mock_tasks):
        """Create TaskSplits for testing."""
        return TaskSplits(
            train_tasks=mock_tasks[:6],
            val_tasks=mock_tasks[6:8],
            test_tasks=mock_tasks[8:],
            split_info={"random_seed": 42, "train_ratio": 0.6},
        )

    def test_initialization(self, task_splits):
        """Test TaskSplits initialization."""
        assert len(task_splits.train_tasks) == 6
        assert len(task_splits.val_tasks) == 2
        assert len(task_splits.test_tasks) == 2
        assert task_splits.split_info["random_seed"] == 42

    def test_len(self, task_splits):
        """Test TaskSplits length."""
        assert len(task_splits) == 10

    def test_summary(self, task_splits):
        """Test TaskSplits summary."""
        summary = task_splits.summary()

        expected = {
            "train": 6,
            "val": 2,
            "test": 2,
            "total": 10,
        }

        assert summary == expected

    def test_save_and_load(self, task_splits, mock_tasks):
        """Test saving and loading TaskSplits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "splits.json"

            # Save splits
            task_splits.save(filepath)
            assert filepath.exists()

            # Load splits
            loaded_splits = TaskSplits.load(filepath, mock_tasks)

            # Verify loaded splits
            assert len(loaded_splits.train_tasks) == len(task_splits.train_tasks)
            assert len(loaded_splits.val_tasks) == len(task_splits.val_tasks)
            assert len(loaded_splits.test_tasks) == len(task_splits.test_tasks)

            # Check task IDs match
            train_ids_orig = [t.task_id for t in task_splits.train_tasks]
            train_ids_loaded = [t.task_id for t in loaded_splits.train_tasks]
            assert train_ids_orig == train_ids_loaded

    def test_save_file_content(self, task_splits):
        """Test saved file content structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "splits.json"
            task_splits.save(filepath)

            with open(filepath, "r") as f:
                data = json.load(f)

            expected_keys = ["train_task_ids", "val_task_ids", "test_task_ids", "split_info", "summary"]
            for key in expected_keys:
                assert key in data

            assert len(data["train_task_ids"]) == 6
            assert len(data["val_task_ids"]) == 2
            assert len(data["test_task_ids"]) == 2


class TestCreateMetaSplits:
    """Test suite for create_meta_splits function."""

    @pytest.fixture
    def mock_tasks_with_molecules(self):
        """Create mock tasks with molecule datasets for filtering tests."""
        tasks = []
        for i in range(20):
            task = Mock(spec=Task)
            task.task_id = f"TASK_{i:03d}"

            # Create mock molecule dataset
            dataset = Mock(spec=MoleculeDataset)

            # Create mock datapoints with varying class distributions
            datapoints = []
            for class_label in [0, 1]:
                num_samples = 10 if i < 15 else 3  # Some tasks have insufficient samples
                for j in range(num_samples):
                    dp = Mock()
                    dp.labels = class_label
                    datapoints.append(dp)

            dataset.datapoints = datapoints
            task.molecule_dataset = dataset
            tasks.append(task)

        return tasks

    def test_basic_splitting(self, mock_tasks_with_molecules):
        """Test basic task splitting functionality."""
        splits = create_meta_splits(
            tasks=mock_tasks_with_molecules,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
        )

        assert isinstance(splits, TaskSplits)
        assert len(splits.train_tasks) > 0
        assert len(splits.val_tasks) > 0
        assert len(splits.test_tasks) > 0

        # Check that all tasks are accounted for (after filtering)
        total_tasks = len(splits.train_tasks) + len(splits.val_tasks) + len(splits.test_tasks)
        assert total_tasks <= len(mock_tasks_with_molecules)

    def test_split_ratios(self, mock_tasks_with_molecules):
        """Test that split ratios are approximately correct."""
        splits = create_meta_splits(
            tasks=mock_tasks_with_molecules,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42,
        )

        total = len(splits)
        train_ratio = len(splits.train_tasks) / total
        val_ratio = len(splits.val_tasks) / total
        test_ratio = len(splits.test_tasks) / total

        # Allow some tolerance due to rounding
        assert abs(train_ratio - 0.7) < 0.15
        assert abs(val_ratio - 0.2) < 0.15
        assert abs(test_ratio - 0.1) < 0.15

    def test_invalid_ratios(self, mock_tasks_with_molecules):
        """Test that invalid ratios raise error."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            create_meta_splits(
                tasks=mock_tasks_with_molecules,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum = 1.1
            )

    def test_reproducibility(self, mock_tasks_with_molecules):
        """Test that splits are reproducible with same seed."""
        splits1 = create_meta_splits(
            tasks=mock_tasks_with_molecules,
            random_seed=42,
        )

        splits2 = create_meta_splits(
            tasks=mock_tasks_with_molecules,
            random_seed=42,
        )

        # Task IDs should be the same
        train_ids1 = [t.task_id for t in splits1.train_tasks]
        train_ids2 = [t.task_id for t in splits2.train_tasks]
        assert train_ids1 == train_ids2

    def test_min_samples_filtering(self, mock_tasks_with_molecules):
        """Test filtering based on minimum samples per class."""
        # Use high minimum to filter out tasks with few samples
        splits = create_meta_splits(
            tasks=mock_tasks_with_molecules,
            min_samples_per_class=8,
            random_seed=42,
        )

        # Should have fewer tasks after filtering
        assert len(splits) < len(mock_tasks_with_molecules)

        # Check split info
        assert splits.split_info["filtered_tasks"] < splits.split_info["original_tasks"]

    def test_split_info(self, mock_tasks_with_molecules):
        """Test that split info is correctly recorded."""
        splits = create_meta_splits(
            tasks=mock_tasks_with_molecules,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
            min_samples_per_class=5,
        )

        info = splits.split_info
        assert info["train_ratio"] == 0.6
        assert info["val_ratio"] == 0.2
        assert info["test_ratio"] == 0.2
        assert info["random_seed"] == 42
        assert info["min_samples_per_class"] == 5
        assert "total_tasks" in info
        assert "original_tasks" in info
        assert "filtered_tasks" in info


class TestCreateTaskFolders:
    """Test suite for create_task_folders function."""

    @pytest.fixture
    def task_splits(self):
        """Create simple TaskSplits for testing."""
        train_tasks = [Mock(spec=Task, task_id="TRAIN_001"), Mock(spec=Task, task_id="TRAIN_002")]
        val_tasks = [Mock(spec=Task, task_id="VAL_001")]
        test_tasks = [Mock(spec=Task, task_id="TEST_001")]

        return TaskSplits(
            train_tasks=train_tasks,
            val_tasks=val_tasks,
            test_tasks=test_tasks,
        )

    def test_create_folders(self, task_splits):
        """Test folder creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            folders = create_task_folders(task_splits, base_dir)

            # Check returned folders
            expected_folders = ["train", "val", "test"]
            for folder_name in expected_folders:
                assert folder_name in folders
                assert folders[folder_name].exists()
                assert folders[folder_name].is_dir()

    def test_task_list_files(self, task_splits):
        """Test that task list files are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            folders = create_task_folders(task_splits, base_dir)

            # Check task list files
            for split_name, expected_count in [("train", 2), ("val", 1), ("test", 1)]:
                task_list_file = folders[split_name] / "task_list.json"
                assert task_list_file.exists()

                with open(task_list_file, "r") as f:
                    task_ids = json.load(f)

                assert len(task_ids) == expected_count
                assert all(isinstance(task_id, str) for task_id in task_ids)

    def test_splits_file(self, task_splits):
        """Test that splits.json file is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            create_task_folders(task_splits, base_dir)

            splits_file = base_dir / "splits.json"
            assert splits_file.exists()

            with open(splits_file, "r") as f:
                splits_data = json.load(f)

            expected_keys = ["train_task_ids", "val_task_ids", "test_task_ids", "summary"]
            for key in expected_keys:
                assert key in splits_data
