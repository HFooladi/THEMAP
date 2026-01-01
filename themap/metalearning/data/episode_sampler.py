"""
Episode sampling for meta-learning with prototypical networks.

This module provides functionality to sample episodes (tasks) for meta-learning
training and evaluation.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ...data.molecule_dataset import MoleculeDataset
from ...data.tasks import Task
from ...utils.featurizer_utils import get_featurizer
from ...utils.logging import get_logger

logger = get_logger(__name__)


class EpisodeSampler:
    """
    Sampler for creating meta-learning episodes from tasks.

    An episode consists of:
    - Support set: Few labeled examples for each class
    - Query set: Examples to predict using the support set

    Args:
        tasks (List[Task]): List of tasks to sample from
        n_way (int): Number of classes per episode
        n_support (int): Number of support examples per class
        n_query (int): Number of query examples per class
        featurizer_name (str): Name of molecular featurizer to use
        random_seed (Optional[int]): Random seed for reproducibility
    """

    def __init__(
        self,
        tasks: List[Task],
        n_way: int = 2,
        n_support: int = 5,
        n_query: int = 15,
        featurizer_name: str = "ecfp",
        random_seed: Optional[int] = None,
    ):
        self.tasks = tasks
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.featurizer_name = featurizer_name

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Get featurizer
        self.featurizer = get_featurizer(featurizer_name)

        # Filter tasks that have enough samples
        self.valid_tasks = self._filter_valid_tasks()

        logger.info(
            f"EpisodeSampler initialized with {len(self.valid_tasks)} valid tasks "
            f"({n_way}-way, {n_support}-shot, {n_query} query)"
        )

    def _filter_valid_tasks(self) -> List[Task]:
        """Filter tasks that have enough samples for episodes."""
        valid_tasks = []
        min_samples_per_class = self.n_support + self.n_query

        for task in self.tasks:
            if task.molecule_dataset is None:
                continue

            # Get class counts using simplified dataset structure
            class_counts = self._get_class_counts(task.molecule_dataset)

            # Check if we have enough classes and samples per class
            if len(class_counts) >= self.n_way:
                sufficient_classes = sum(
                    1 for count in class_counts.values() if count >= min_samples_per_class
                )
                if sufficient_classes >= self.n_way:
                    valid_tasks.append(task)
                else:
                    logger.debug(f"Task {task.task_id} filtered: insufficient samples per class")
            else:
                logger.debug(f"Task {task.task_id} filtered: insufficient classes")

        return valid_tasks

    def _get_class_counts(self, dataset: MoleculeDataset) -> Dict[int, int]:
        """Get count of samples per class using simplified dataset structure."""
        class_counts: Dict[int, int] = {}

        # Use numpy array labels directly from simplified MoleculeDataset
        for label in dataset.labels:
            label_int = int(label)
            class_counts[label_int] = class_counts.get(label_int, 0) + 1

        return class_counts

    def sample_episode(self, task: Optional[Task] = None) -> Dict[str, torch.Tensor]:
        """
        Sample a single episode.

        Args:
            task (Optional[Task]): Specific task to sample from. If None, samples randomly.

        Returns:
            Dict[str, torch.Tensor]: Episode data containing:
                - support_features: Support set molecular features
                - support_labels: Support set labels
                - query_features: Query set molecular features
                - query_labels: Query set labels
                - task_id: Task identifier
        """
        if task is None:
            task = random.choice(self.valid_tasks)

        if task not in self.valid_tasks:
            raise ValueError(f"Task {task.task_id} is not valid for episode sampling")

        dataset = task.molecule_dataset

        # Group indices by class using simplified dataset structure
        class_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(dataset.labels):
            label_int = int(label)
            if label_int not in class_indices:
                class_indices[label_int] = []
            class_indices[label_int].append(idx)

        # Sample classes for this episode
        available_classes = [
            cls for cls, indices in class_indices.items() if len(indices) >= self.n_support + self.n_query
        ]

        if len(available_classes) < self.n_way:
            raise ValueError(f"Task {task.task_id} doesn't have enough classes with sufficient samples")

        selected_classes = random.sample(available_classes, self.n_way)

        # Sample support and query sets
        support_features = []
        support_labels = []
        query_features = []
        query_labels = []

        for class_idx, original_class in enumerate(selected_classes):
            indices = class_indices[original_class]
            sampled_indices = random.sample(indices, self.n_support + self.n_query)

            # Split into support and query
            support_indices = sampled_indices[: self.n_support]
            query_indices = sampled_indices[self.n_support :]

            # Extract features and relabel
            for idx in support_indices:
                features = self._extract_features(dataset.smiles_list[idx])
                support_features.append(features)
                support_labels.append(class_idx)  # Relabel to 0, 1, ..., n_way-1

            for idx in query_indices:
                features = self._extract_features(dataset.smiles_list[idx])
                query_features.append(features)
                query_labels.append(class_idx)

        # Convert to tensors
        episode = {
            "support_features": torch.tensor(np.array(support_features), dtype=torch.float32),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_features": torch.tensor(np.array(query_features), dtype=torch.float32),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
            "task_id": task.task_id,
        }

        return episode

    def _extract_features(self, smiles: str) -> np.ndarray:
        """Extract molecular features from a SMILES string."""
        # Compute features using the featurizer
        features = self.featurizer(smiles)

        if features is None:
            raise ValueError(f"Failed to compute features for SMILES: {smiles}")

        # Handle different featurizer return types
        if isinstance(features, np.ndarray):
            if features.ndim == 2:
                return features[0]  # Return first row if 2D
            return features

        return np.array(features).flatten()

    def sample_batch(self, batch_size: int, task: Optional[Task] = None) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of episodes.

        Args:
            batch_size (int): Number of episodes to sample
            task (Optional[Task]): Specific task to sample from

        Returns:
            Dict[str, torch.Tensor]: Batched episode data
        """
        episodes = [self.sample_episode(task) for _ in range(batch_size)]

        # Stack episodes into batches
        batch = {
            "support_features": torch.stack([ep["support_features"] for ep in episodes]),
            "support_labels": torch.stack([ep["support_labels"] for ep in episodes]),
            "query_features": torch.stack([ep["query_features"] for ep in episodes]),
            "query_labels": torch.stack([ep["query_labels"] for ep in episodes]),
            "task_ids": [ep["task_id"] for ep in episodes],
        }

        return batch


class MetaLearningDataset(Dataset):
    """
    PyTorch Dataset for meta-learning episodes.

    Args:
        episode_sampler (EpisodeSampler): Episode sampler
        num_episodes (int): Number of episodes per epoch
        fixed_task (Optional[Task]): If provided, all episodes use this task
    """

    def __init__(
        self,
        episode_sampler: EpisodeSampler,
        num_episodes: int = 1000,
        fixed_task: Optional[Task] = None,
    ):
        self.episode_sampler = episode_sampler
        self.num_episodes = num_episodes
        self.fixed_task = fixed_task

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Sample an episode."""
        return self.episode_sampler.sample_episode(self.fixed_task)


def create_episode_samplers(
    train_tasks: List[Task],
    val_tasks: List[Task],
    test_tasks: List[Task],
    n_way: int = 2,
    n_support: int = 5,
    n_query: int = 15,
    featurizer_name: str = "ecfp",
    random_seed: Optional[int] = 42,
) -> Tuple[EpisodeSampler, EpisodeSampler, EpisodeSampler]:
    """
    Create episode samplers for train/val/test splits.

    Returns:
        Tuple[EpisodeSampler, EpisodeSampler, EpisodeSampler]: Train, val, test samplers
    """
    train_sampler = EpisodeSampler(train_tasks, n_way, n_support, n_query, featurizer_name, random_seed)
    val_sampler = EpisodeSampler(val_tasks, n_way, n_support, n_query, featurizer_name, random_seed)
    test_sampler = EpisodeSampler(test_tasks, n_way, n_support, n_query, featurizer_name, random_seed)

    return train_sampler, val_sampler, test_sampler
