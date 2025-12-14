"""
Tests for EpisodeSampler and MetaLearningDataset.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from themap.data.molecule_dataset import MoleculeDataset
from themap.data.tasks import Task
from themap.metalearning.data.episode_sampler import EpisodeSampler, MetaLearningDataset


class TestEpisodeSampler:
    """Test suite for EpisodeSampler."""

    @pytest.fixture
    def mock_tasks(self):
        """Create mock tasks for testing."""
        tasks = []

        for task_id in range(3):
            # Create datapoints using the backward-compatible format
            # The datapoints property in MoleculeDataset returns List[Dict] now
            datapoints = []
            for class_label in [0, 1]:
                for i in range(20):  # 20 samples per class
                    dp = {
                        "smiles": f"CC{'N' * i}C",
                        "labels": class_label,
                        "bool_label": bool(class_label),
                        "numeric_label": None,
                    }
                    datapoints.append(dp)

            # Create mock dataset that returns datapoints as list of dicts
            dataset = Mock(spec=MoleculeDataset)
            dataset.datapoints = datapoints
            # Also set labels as numpy array for _get_class_counts method
            dataset.labels = np.array([dp["labels"] for dp in datapoints], dtype=np.int32)

            # Create mock task
            task = Mock(spec=Task)
            task.task_id = f"TASK_{task_id}"
            task.molecule_dataset = dataset

            tasks.append(task)

        return tasks

    @pytest.fixture
    def mock_featurizer(self):
        """Mock featurizer that returns random features."""

        def featurizer(smiles):
            return np.random.randn(100)  # 100-dimensional features

        return featurizer

    @pytest.fixture
    def episode_sampler(self, mock_tasks, mock_featurizer):
        """Create EpisodeSampler with mock data."""
        with patch("themap.metalearning.data.episode_sampler.get_featurizer", return_value=mock_featurizer):
            sampler = EpisodeSampler(
                tasks=mock_tasks,
                n_way=2,
                n_support=5,
                n_query=10,
                featurizer_name="mock_featurizer",
                random_seed=42,
            )
        return sampler

    def test_initialization(self, mock_tasks, mock_featurizer):
        """Test EpisodeSampler initialization."""
        with patch("themap.metalearning.data.episode_sampler.get_featurizer", return_value=mock_featurizer):
            sampler = EpisodeSampler(
                tasks=mock_tasks,
                n_way=2,
                n_support=5,
                n_query=10,
                featurizer_name="mock_featurizer",
            )

        assert sampler.n_way == 2
        assert sampler.n_support == 5
        assert sampler.n_query == 10
        assert sampler.featurizer_name == "mock_featurizer"
        assert len(sampler.valid_tasks) > 0

    def test_filter_valid_tasks(self, mock_tasks, mock_featurizer):
        """Test task filtering based on sample requirements."""
        with patch("themap.metalearning.data.episode_sampler.get_featurizer", return_value=mock_featurizer):
            # Should accept tasks with enough samples
            sampler = EpisodeSampler(
                tasks=mock_tasks,
                n_way=2,
                n_support=5,
                n_query=10,
                featurizer_name="mock_featurizer",
            )
            assert len(sampler.valid_tasks) == len(mock_tasks)

            # Should reject tasks with insufficient samples
            sampler_large = EpisodeSampler(
                tasks=mock_tasks,
                n_way=2,
                n_support=15,  # More than available (20 per class)
                n_query=10,
                featurizer_name="mock_featurizer",
            )
            assert len(sampler_large.valid_tasks) < len(mock_tasks)

    def test_get_class_counts(self, episode_sampler):
        """Test class count computation."""
        task = episode_sampler.valid_tasks[0]
        class_counts = episode_sampler._get_class_counts(task.molecule_dataset)

        assert isinstance(class_counts, dict)
        assert len(class_counts) == 2  # Binary classification
        assert all(count > 0 for count in class_counts.values())

    def test_sample_episode_structure(self, episode_sampler):
        """Test episode sampling returns correct structure."""
        episode = episode_sampler.sample_episode()

        required_keys = ["support_features", "support_labels", "query_features", "query_labels", "task_id"]
        for key in required_keys:
            assert key in episode

        # Check tensor shapes
        n_way = episode_sampler.n_way
        n_support = episode_sampler.n_support
        n_query = episode_sampler.n_query

        assert episode["support_features"].shape == (n_way * n_support, 100)  # 100-dim features
        assert episode["support_labels"].shape == (n_way * n_support,)
        assert episode["query_features"].shape == (n_way * n_query, 100)
        assert episode["query_labels"].shape == (n_way * n_query,)

        # Check label range
        assert episode["support_labels"].min() >= 0
        assert episode["support_labels"].max() < n_way
        assert episode["query_labels"].min() >= 0
        assert episode["query_labels"].max() < n_way

    def test_sample_episode_from_specific_task(self, episode_sampler):
        """Test sampling episode from specific task."""
        specific_task = episode_sampler.valid_tasks[0]
        episode = episode_sampler.sample_episode(specific_task)

        assert episode["task_id"] == specific_task.task_id

    def test_sample_episode_invalid_task(self, episode_sampler):
        """Test sampling from invalid task raises error."""
        invalid_task = Mock(spec=Task)
        invalid_task.task_id = "INVALID_TASK"

        with pytest.raises(ValueError, match="is not valid for episode sampling"):
            episode_sampler.sample_episode(invalid_task)

    def test_sample_batch(self, episode_sampler):
        """Test batch episode sampling."""
        batch_size = 3
        batch = episode_sampler.sample_batch(batch_size)

        required_keys = ["support_features", "support_labels", "query_features", "query_labels", "task_ids"]
        for key in required_keys:
            assert key in batch

        # Check batch dimensions
        n_way = episode_sampler.n_way
        n_support = episode_sampler.n_support
        n_query = episode_sampler.n_query

        assert batch["support_features"].shape == (batch_size, n_way * n_support, 100)
        assert batch["support_labels"].shape == (batch_size, n_way * n_support)
        assert batch["query_features"].shape == (batch_size, n_way * n_query, 100)
        assert batch["query_labels"].shape == (batch_size, n_way * n_query)
        assert len(batch["task_ids"]) == batch_size

    def test_feature_caching(self, episode_sampler):
        """Test that features are cached properly."""
        # Sample two episodes to potentially reuse features
        episode1 = episode_sampler.sample_episode()
        episode2 = episode_sampler.sample_episode()

        # Check that features are computed and cached
        # (This is mainly a smoke test since mocking makes direct verification difficult)
        assert episode1["support_features"].shape[1] == 100
        assert episode2["support_features"].shape[1] == 100

    def test_extract_features(self, episode_sampler):
        """Test feature extraction from datapoint."""
        # Create a mock datapoint
        datapoint = Mock()
        datapoint.smiles = "CCN"
        datapoint._cached_features = {}  # Initialize empty dict to avoid Mock iteration error

        features = episode_sampler._extract_features(datapoint)

        assert isinstance(features, np.ndarray)
        assert features.shape == (100,)  # Mock featurizer returns 100-dim

    def test_reproducibility_with_seed(self, mock_tasks, mock_featurizer):
        """Test that episodes are reproducible with same seed."""
        with patch("themap.metalearning.data.episode_sampler.get_featurizer", return_value=mock_featurizer):
            sampler1 = EpisodeSampler(
                tasks=mock_tasks,
                n_way=2,
                n_support=5,
                n_query=10,
                featurizer_name="mock_featurizer",
                random_seed=42,
            )

            sampler2 = EpisodeSampler(
                tasks=mock_tasks,
                n_way=2,
                n_support=5,
                n_query=10,
                featurizer_name="mock_featurizer",
                random_seed=42,
            )

        # Sample episodes with same seed should have some similarity
        # (Due to mocking, we mainly test that no errors occur)
        episode1 = sampler1.sample_episode()
        episode2 = sampler2.sample_episode()

        assert episode1["support_features"].shape == episode2["support_features"].shape
        assert episode1["query_features"].shape == episode2["query_features"].shape


class TestMetaLearningDataset:
    """Test suite for MetaLearningDataset."""

    @pytest.fixture
    def episode_sampler(self, mock_tasks, mock_featurizer):
        """Create EpisodeSampler for dataset testing."""
        with patch("themap.metalearning.data.episode_sampler.get_featurizer", return_value=mock_featurizer):
            sampler = EpisodeSampler(
                tasks=mock_tasks,
                n_way=2,
                n_support=5,
                n_query=10,
                featurizer_name="mock_featurizer",
                random_seed=42,
            )
        return sampler

    @pytest.fixture
    def mock_tasks(self):
        """Create mock tasks for testing."""
        tasks = []

        for task_id in range(3):
            # Create datapoints using the backward-compatible format
            datapoints = []
            for class_label in [0, 1]:
                for i in range(20):  # 20 samples per class
                    dp = {
                        "smiles": f"CC{'N' * i}C",
                        "labels": class_label,
                        "bool_label": bool(class_label),
                        "numeric_label": None,
                    }
                    datapoints.append(dp)

            # Create mock dataset that returns datapoints as list of dicts
            dataset = Mock(spec=MoleculeDataset)
            dataset.datapoints = datapoints
            # Also set labels as numpy array for _get_class_counts method
            dataset.labels = np.array([dp["labels"] for dp in datapoints], dtype=np.int32)

            # Create mock task
            task = Mock(spec=Task)
            task.task_id = f"TASK_{task_id}"
            task.molecule_dataset = dataset

            tasks.append(task)

        return tasks

    @pytest.fixture
    def mock_featurizer(self):
        """Mock featurizer that returns random features."""

        def featurizer(smiles):
            return np.random.randn(100)  # 100-dimensional features

        return featurizer

    def test_dataset_initialization(self, episode_sampler):
        """Test MetaLearningDataset initialization."""
        dataset = MetaLearningDataset(
            episode_sampler=episode_sampler,
            num_episodes=100,
        )

        assert len(dataset) == 100
        assert dataset.episode_sampler == episode_sampler
        assert dataset.fixed_task is None

    def test_dataset_with_fixed_task(self, episode_sampler):
        """Test dataset with fixed task."""
        fixed_task = episode_sampler.valid_tasks[0]
        dataset = MetaLearningDataset(
            episode_sampler=episode_sampler,
            num_episodes=50,
            fixed_task=fixed_task,
        )

        assert dataset.fixed_task == fixed_task

        # Sample episode should be from fixed task
        episode = dataset[0]
        assert episode["task_id"] == fixed_task.task_id

    def test_dataset_getitem(self, episode_sampler):
        """Test dataset item access."""
        dataset = MetaLearningDataset(
            episode_sampler=episode_sampler,
            num_episodes=10,
        )

        episode = dataset[0]

        # Check episode structure
        required_keys = ["support_features", "support_labels", "query_features", "query_labels", "task_id"]
        for key in required_keys:
            assert key in episode

        # Check tensor types
        assert isinstance(episode["support_features"], torch.Tensor)
        assert isinstance(episode["support_labels"], torch.Tensor)
        assert isinstance(episode["query_features"], torch.Tensor)
        assert isinstance(episode["query_labels"], torch.Tensor)

    def test_dataset_multiple_access(self, episode_sampler):
        """Test multiple accesses to dataset."""
        dataset = MetaLearningDataset(
            episode_sampler=episode_sampler,
            num_episodes=5,
        )

        # Access multiple episodes
        episodes = [dataset[i] for i in range(5)]

        # All episodes should have correct structure
        for episode in episodes:
            assert "support_features" in episode
            assert "query_features" in episode
            assert episode["support_features"].shape[0] > 0
            assert episode["query_features"].shape[0] > 0
