"""
Comprehensive evaluation pipeline for meta-learning models.

This module provides detailed evaluation capabilities including per-task analysis,
confidence intervals, and performance breakdown by task characteristics.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from scipy import stats

from ...data.tasks import Task
from ...utils.logging import get_logger
from ..data.episode_sampler import EpisodeSampler
from ..data.task_splits import TaskSplits
from ..models.prototypical_network import PrototypicalNetwork

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for meta-learning evaluation."""

    # Episode configuration
    n_way: int = 2
    n_support: int = 5
    n_query: int = 15
    featurizer_name: str = "ecfp"

    # Evaluation configuration
    num_episodes_per_task: int = 100
    confidence_level: float = 0.95
    temperature: float = 1.0

    # Analysis configuration
    per_task_analysis: bool = True
    task_difficulty_analysis: bool = True
    confusion_matrix: bool = True

    # Output configuration
    output_dir: str = "./evaluation_outputs"
    save_predictions: bool = True
    save_detailed_results: bool = True

    # Device configuration
    device: str = "auto"
    random_seed: int = 42

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)


class MetaLearningEvaluator:
    """
    Comprehensive evaluator for meta-learning models.

    Args:
        model (PrototypicalNetwork): Trained model to evaluate
        config (EvaluationConfig): Evaluation configuration
        task_splits (TaskSplits): Task splits containing test tasks
    """

    def __init__(
        self,
        model: PrototypicalNetwork,
        config: EvaluationConfig,
        task_splits: TaskSplits,
    ):
        self.model = model
        self.config = config
        self.task_splits = task_splits

        # Set up device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        # Set up random seeds
        self._set_random_seeds()

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize episode sampler for test tasks
        self.test_sampler = EpisodeSampler(
            tasks=task_splits.test_tasks,
            n_way=config.n_way,
            n_support=config.n_support,
            n_query=config.n_query,
            featurizer_name=config.featurizer_name,
            random_seed=config.random_seed,
        )

        logger.info(f"MetaLearningEvaluator initialized with {len(task_splits.test_tasks)} test tasks")

    def _setup_device(self) -> torch.device:
        """Set up computing device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)

        logger.info(f"Using device: {device}")
        return device

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def evaluate(self) -> Dict[str, Union[float, Dict]]:
        """
        Run comprehensive evaluation.

        Returns:
            Dict containing evaluation results and statistics
        """
        logger.info("Starting comprehensive evaluation")
        start_time = time.time()

        self.model.eval()

        # Overall evaluation
        overall_results = self._evaluate_overall()

        # Per-task evaluation
        per_task_results = {}
        if self.config.per_task_analysis:
            per_task_results = self._evaluate_per_task()

        # Task difficulty analysis
        difficulty_analysis = {}
        if self.config.task_difficulty_analysis:
            difficulty_analysis = self._analyze_task_difficulty(per_task_results)

        # Compile final results
        results = {
            "overall": overall_results,
            "per_task": per_task_results,
            "difficulty_analysis": difficulty_analysis,
            "config": self.config.to_dict(),
            "evaluation_time": time.time() - start_time,
        }

        # Save results
        self._save_results(results)

        logger.info(f"Evaluation completed in {results['evaluation_time']:.2f} seconds")
        logger.info(
            f"Overall accuracy: {overall_results['accuracy']:.4f} ± {overall_results['accuracy_ci']:.4f}"
        )

        return results

    def _evaluate_overall(self) -> Dict[str, float]:
        """Evaluate overall performance across all test tasks."""
        logger.info("Evaluating overall performance")

        all_accuracies = []
        all_losses = []
        all_predictions = []
        all_true_labels = []

        _total_episodes = len(self.task_splits.test_tasks) * self.config.num_episodes_per_task

        with torch.no_grad():
            for task_idx, task in enumerate(self.task_splits.test_tasks):
                task_accuracies = []
                task_losses = []

                for episode_idx in range(self.config.num_episodes_per_task):
                    # Sample episode from specific task
                    episode = self.test_sampler.sample_episode(task)

                    # Move to device
                    for key in ["support_features", "support_labels", "query_features", "query_labels"]:
                        episode[key] = episode[key].to(self.device)

                    # Get predictions
                    probabilities = self.model.predict_proba(
                        episode["query_features"],
                        episode["support_features"],
                        episode["support_labels"],
                        temperature=self.config.temperature,
                    )

                    # Compute metrics
                    predictions = torch.argmax(probabilities, dim=1)
                    accuracy = (predictions == episode["query_labels"]).float().mean().item()
                    loss = torch.nn.functional.cross_entropy(
                        torch.log(probabilities + 1e-8), episode["query_labels"].long()
                    ).item()

                    task_accuracies.append(accuracy)
                    task_losses.append(loss)

                    # Store for detailed analysis
                    if self.config.save_predictions:
                        all_predictions.extend(predictions.cpu().numpy())
                        all_true_labels.extend(episode["query_labels"].cpu().numpy())

                all_accuracies.extend(task_accuracies)
                all_losses.extend(task_losses)

                if (task_idx + 1) % 10 == 0:
                    logger.info(f"Evaluated {task_idx + 1}/{len(self.task_splits.test_tasks)} tasks")

        # Compute statistics
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        mean_loss = np.mean(all_losses)

        # Confidence interval
        confidence_interval = stats.t.interval(
            self.config.confidence_level,
            len(all_accuracies) - 1,
            loc=mean_accuracy,
            scale=stats.sem(all_accuracies),
        )
        accuracy_ci = (confidence_interval[1] - confidence_interval[0]) / 2

        # Confusion matrix
        confusion_matrix = None
        if self.config.confusion_matrix and all_predictions:
            confusion_matrix = self._compute_confusion_matrix(all_true_labels, all_predictions)

        return {
            "accuracy": mean_accuracy,
            "accuracy_std": std_accuracy,
            "accuracy_ci": accuracy_ci,
            "loss": mean_loss,
            "num_episodes": len(all_accuracies),
            "confusion_matrix": confusion_matrix,
        }

    def _evaluate_per_task(self) -> Dict[str, Dict[str, float]]:
        """Evaluate performance on each individual task."""
        logger.info("Evaluating per-task performance")

        per_task_results = {}

        with torch.no_grad():
            for task in self.task_splits.test_tasks:
                task_accuracies = []
                task_losses = []

                for episode_idx in range(self.config.num_episodes_per_task):
                    # Sample episode from specific task
                    episode = self.test_sampler.sample_episode(task)

                    # Move to device
                    for key in ["support_features", "support_labels", "query_features", "query_labels"]:
                        episode[key] = episode[key].to(self.device)

                    # Compute loss and accuracy
                    loss, metrics = self.model.episodic_loss(
                        episode["support_features"],
                        episode["support_labels"],
                        episode["query_features"],
                        episode["query_labels"],
                        temperature=self.config.temperature,
                    )

                    task_accuracies.append(metrics["accuracy"])
                    task_losses.append(loss.item())

                # Compute task statistics
                mean_accuracy = np.mean(task_accuracies)
                std_accuracy = np.std(task_accuracies)
                mean_loss = np.mean(task_losses)

                # Confidence interval for this task
                if len(task_accuracies) > 1:
                    confidence_interval = stats.t.interval(
                        self.config.confidence_level,
                        len(task_accuracies) - 1,
                        loc=mean_accuracy,
                        scale=stats.sem(task_accuracies),
                    )
                    accuracy_ci = (confidence_interval[1] - confidence_interval[0]) / 2
                else:
                    accuracy_ci = 0.0

                per_task_results[task.task_id] = {
                    "accuracy": mean_accuracy,
                    "accuracy_std": std_accuracy,
                    "accuracy_ci": accuracy_ci,
                    "loss": mean_loss,
                    "num_episodes": len(task_accuracies),
                    "task_hardness": getattr(task, "hardness", None),
                }

        return per_task_results

    def _analyze_task_difficulty(self, per_task_results: Dict[str, Dict]) -> Dict[str, Union[float, List]]:
        """Analyze performance relative to task difficulty."""
        if not per_task_results:
            return {}

        logger.info("Analyzing task difficulty correlation")

        # Extract accuracies and hardness values
        accuracies = []
        hardness_values = []

        for task_id, results in per_task_results.items():
            accuracy = results["accuracy"]
            hardness = results.get("task_hardness")

            if hardness is not None and not np.isnan(hardness):
                accuracies.append(accuracy)
                hardness_values.append(hardness)

        analysis = {
            "num_tasks_with_hardness": len(accuracies),
            "correlation": None,
            "correlation_pvalue": None,
            "hardness_bins": {},
        }

        if len(accuracies) > 2:
            # Compute correlation
            correlation, p_value = stats.pearsonr(hardness_values, accuracies)
            analysis["correlation"] = correlation
            analysis["correlation_pvalue"] = p_value

            # Bin analysis
            hardness_array = np.array(hardness_values)
            accuracy_array = np.array(accuracies)

            # Create bins
            num_bins = min(5, len(accuracies) // 3)
            if num_bins > 1:
                bins = np.linspace(hardness_array.min(), hardness_array.max(), num_bins + 1)
                bin_indices = np.digitize(hardness_array, bins) - 1

                for bin_idx in range(num_bins):
                    mask = bin_indices == bin_idx
                    if np.any(mask):
                        bin_accuracies = accuracy_array[mask]
                        analysis["hardness_bins"][f"bin_{bin_idx}"] = {
                            "hardness_range": [float(bins[bin_idx]), float(bins[bin_idx + 1])],
                            "mean_accuracy": float(np.mean(bin_accuracies)),
                            "std_accuracy": float(np.std(bin_accuracies)),
                            "num_tasks": int(np.sum(mask)),
                        }

        return analysis

    def _compute_confusion_matrix(
        self, true_labels: List[int], predictions: List[int]
    ) -> Dict[str, List[List[int]]]:
        """Compute confusion matrix."""
        unique_labels = sorted(set(true_labels + predictions))
        n_classes = len(unique_labels)

        matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]

        for true_label, pred_label in zip(true_labels, predictions):
            true_idx = unique_labels.index(true_label)
            pred_idx = unique_labels.index(pred_label)
            matrix[true_idx][pred_idx] += 1

        return {
            "matrix": matrix,
            "labels": unique_labels,
        }

    def _save_results(self, results: Dict) -> None:
        """Save evaluation results to files."""
        # Save main results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_numpy_types(results)
            json.dump(json_results, f, indent=2)

        # Save summary
        summary = {
            "overall_accuracy": results["overall"]["accuracy"],
            "overall_accuracy_ci": results["overall"]["accuracy_ci"],
            "num_test_tasks": len(self.task_splits.test_tasks),
            "total_episodes": results["overall"]["num_episodes"],
            "evaluation_time": results["evaluation_time"],
        }

        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved evaluation results to {self.output_dir}")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def evaluate_single_task(self, task: Task, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate performance on a single task.

        Args:
            task (Task): Task to evaluate
            num_episodes (int): Number of episodes to sample

        Returns:
            Dict[str, float]: Evaluation metrics for the task
        """
        logger.info(f"Evaluating single task: {task.task_id}")

        self.model.eval()
        accuracies = []
        losses = []

        with torch.no_grad():
            for episode_idx in range(num_episodes):
                # Sample episode
                episode = self.test_sampler.sample_episode(task)

                # Move to device
                for key in ["support_features", "support_labels", "query_features", "query_labels"]:
                    episode[key] = episode[key].to(self.device)

                # Compute metrics
                loss, metrics = self.model.episodic_loss(
                    episode["support_features"],
                    episode["support_labels"],
                    episode["query_features"],
                    episode["query_labels"],
                    temperature=self.config.temperature,
                )

                accuracies.append(metrics["accuracy"])
                losses.append(loss.item())

        # Compute statistics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_loss = np.mean(losses)

        # Confidence interval
        if len(accuracies) > 1:
            confidence_interval = stats.t.interval(
                self.config.confidence_level,
                len(accuracies) - 1,
                loc=mean_accuracy,
                scale=stats.sem(accuracies),
            )
            accuracy_ci = (confidence_interval[1] - confidence_interval[0]) / 2
        else:
            accuracy_ci = 0.0

        return {
            "task_id": task.task_id,
            "accuracy": mean_accuracy,
            "accuracy_std": std_accuracy,
            "accuracy_ci": accuracy_ci,
            "loss": mean_loss,
            "num_episodes": len(accuracies),
        }
