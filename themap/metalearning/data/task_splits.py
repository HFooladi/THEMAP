"""
Task splitting utilities for meta-learning.

This module provides functionality to split tasks into train/validation/test sets
for meta-learning experiments.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from ...data.tasks import Task
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskSplits:
    """
    Container for meta-learning task splits.

    Args:
        train_tasks (List[Task]): Tasks for meta-training
        val_tasks (List[Task]): Tasks for meta-validation
        test_tasks (List[Task]): Tasks for meta-testing
        split_info (Dict): Information about the split
    """

    train_tasks: List[Task]
    val_tasks: List[Task]
    test_tasks: List[Task]
    split_info: Optional[Dict] = None

    def __len__(self) -> int:
        """Total number of tasks across all splits."""
        return len(self.train_tasks) + len(self.val_tasks) + len(self.test_tasks)

    def summary(self) -> Dict[str, int]:
        """Get summary of task counts."""
        return {
            "train": len(self.train_tasks),
            "val": len(self.val_tasks),
            "test": len(self.test_tasks),
            "total": len(self),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save task splits to file."""
        filepath = Path(filepath)

        split_data = {
            "train_task_ids": [task.task_id for task in self.train_tasks],
            "val_task_ids": [task.task_id for task in self.val_tasks],
            "test_task_ids": [task.task_id for task in self.test_tasks],
            "split_info": self.split_info or {},
            "summary": self.summary(),
        }

        with open(filepath, "w") as f:
            json.dump(split_data, f, indent=2)

        logger.info(f"Saved task splits to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path], all_tasks: List[Task]) -> "TaskSplits":
        """Load task splits from file."""
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            split_data = json.load(f)

        # Create task ID to task mapping
        task_map = {task.task_id: task for task in all_tasks}

        # Reconstruct task lists
        train_tasks = [task_map[task_id] for task_id in split_data["train_task_ids"]]
        val_tasks = [task_map[task_id] for task_id in split_data["val_task_ids"]]
        test_tasks = [task_map[task_id] for task_id in split_data["test_task_ids"]]

        logger.info(f"Loaded task splits from {filepath}")

        return cls(
            train_tasks=train_tasks,
            val_tasks=val_tasks,
            test_tasks=test_tasks,
            split_info=split_data.get("split_info"),
        )


def create_meta_splits(
    tasks: List[Task],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: Optional[int] = 42,
    stratify_by: Optional[str] = None,
    min_samples_per_class: int = 5,
) -> TaskSplits:
    """
    Create meta-learning task splits.

    Args:
        tasks (List[Task]): List of all tasks
        train_ratio (float): Proportion of tasks for training
        val_ratio (float): Proportion of tasks for validation
        test_ratio (float): Proportion of tasks for testing
        random_seed (Optional[int]): Random seed for reproducibility
        stratify_by (Optional[str]): Task attribute to stratify by (e.g., 'hardness')
        min_samples_per_class (int): Minimum samples per class in each task

    Returns:
        TaskSplits: Container with train/val/test task splits
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    if random_seed is not None:
        random.seed(random_seed)

    # Filter tasks by minimum samples per class if needed
    filtered_tasks = []
    for task in tasks:
        if task.molecule_dataset is not None:
            if _has_min_samples_per_class(task, min_samples_per_class):
                filtered_tasks.append(task)
            else:
                logger.warning(f"Task {task.task_id} filtered out due to insufficient samples per class")
        else:
            filtered_tasks.append(task)

    logger.info(f"Using {len(filtered_tasks)} tasks after filtering (from {len(tasks)} original)")

    # Shuffle tasks
    tasks_copy = filtered_tasks.copy()
    random.shuffle(tasks_copy)

    # Calculate split sizes
    n_total = len(tasks_copy)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Ensure all tasks are used

    # Create splits
    train_tasks = tasks_copy[:n_train]
    val_tasks = tasks_copy[n_train : n_train + n_val]
    test_tasks = tasks_copy[n_train + n_val :]

    split_info = {
        "total_tasks": n_total,
        "original_tasks": len(tasks),
        "filtered_tasks": len(filtered_tasks),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_seed": random_seed,
        "stratify_by": stratify_by,
        "min_samples_per_class": min_samples_per_class,
    }

    splits = TaskSplits(
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        test_tasks=test_tasks,
        split_info=split_info,
    )

    logger.info(f"Created meta splits: {splits.summary()}")

    return splits


def _has_min_samples_per_class(task: Task, min_samples: int) -> bool:
    """Check if task has minimum samples per class."""
    if task.molecule_dataset is None:
        return True

    # Get labels
    labels = [dp.labels for dp in task.molecule_dataset.datapoints if dp.labels is not None]

    if not labels:
        return False

    # For binary classification, check both classes
    if isinstance(labels[0], (int, float, bool)):
        unique_labels, counts = zip(*[(label, labels.count(label)) for label in set(labels)])
        return all(count >= min_samples for count in counts)

    # For multi-label or more complex cases, assume it's valid
    return True


def create_task_folders(
    task_splits: TaskSplits,
    base_dir: Union[str, Path],
    copy_data: bool = False,
) -> Dict[str, Path]:
    """
    Create folder structure for meta-learning tasks.

    Args:
        task_splits (TaskSplits): Task splits
        base_dir (Union[str, Path]): Base directory for meta-learning
        copy_data (bool): Whether to copy task data to folders

    Returns:
        Dict[str, Path]: Dictionary mapping split names to folder paths
    """
    base_dir = Path(base_dir)

    folders = {
        "train": base_dir / "train",
        "val": base_dir / "val",
        "test": base_dir / "test",
    }

    # Create directories
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    # Save task lists
    for split_name, split_tasks in [
        ("train", task_splits.train_tasks),
        ("val", task_splits.val_tasks),
        ("test", task_splits.test_tasks),
    ]:
        task_list_file = folders[split_name] / "task_list.json"
        task_ids = [task.task_id for task in split_tasks]

        with open(task_list_file, "w") as f:
            json.dump(task_ids, f, indent=2)

    # Save complete split information
    split_file = base_dir / "splits.json"
    task_splits.save(split_file)

    logger.info(f"Created task folders in {base_dir}")

    return folders
