"""
Meta-learning trainer for prototypical networks.

This module provides a comprehensive training framework for meta-learning
with prototypical networks on molecular tasks.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from ...utils.logging import get_logger
from ..data.episode_sampler import EpisodeSampler
from ..data.task_splits import TaskSplits
from ..models.prototypical_network import PrototypicalNetwork, PrototypicalNetworkConfig

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for meta-learning training."""

    # Model configuration
    model_config: PrototypicalNetworkConfig

    # Episode configuration
    n_way: int = 2
    n_support: int = 5
    n_query: int = 15
    featurizer_name: str = "ecfp"

    # Training configuration
    num_epochs: int = 100
    episodes_per_epoch: int = 100
    val_episodes_per_epoch: int = 50
    batch_size: int = 4

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "step"  # 'step', 'cosine', 'plateau'
    lr_step_size: int = 20
    lr_gamma: float = 0.1
    gradient_clip: Optional[float] = 1.0

    # Regularization
    temperature: float = 1.0
    dropout_prob: float = 0.1

    # Validation and early stopping
    val_frequency: int = 5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Logging and checkpointing
    log_frequency: int = 10
    checkpoint_frequency: int = 20
    save_best_model: bool = True

    # Paths
    output_dir: str = "./metalearning_outputs"
    experiment_name: str = "prototypical_network"

    # Miscellaneous
    random_seed: int = 42
    device: str = "auto"  # 'auto', 'cpu', 'cuda'
    num_workers: int = 4

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        config_dict["model_config"] = self.model_config.to_dict()
        return config_dict


class MetaLearningTrainer:
    """
    Trainer for meta-learning with prototypical networks.

    Args:
        config (TrainingConfig): Training configuration
        task_splits (TaskSplits): Task splits for train/val/test
    """

    def __init__(self, config: TrainingConfig, task_splits: TaskSplits):
        self.config = config
        self.task_splits = task_splits

        # Set up device
        self.device = self._setup_device()

        # Set up random seeds
        self._set_random_seeds()

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = PrototypicalNetwork(
            input_dim=config.model_config.input_dim,
            hidden_dims=config.model_config.hidden_dims,
            output_dim=config.model_config.output_dim,
            dropout_prob=config.model_config.dropout_prob,
            activation=config.model_config.activation,
            distance_metric=config.model_config.distance_metric,
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Initialize learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Initialize episode samplers
        self.train_sampler, self.val_sampler, self.test_sampler = self._setup_samplers()

        # Initialize logging
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        else:
            self.writer = None
            logger.warning("TensorBoard not available. Install tensorboard for logging support.")

        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = []

        logger.info(f"MetaLearningTrainer initialized. Output dir: {self.output_dir}")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")

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

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Set up learning rate scheduler."""
        if self.config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        elif self.config.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
            )
        elif self.config.lr_scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.config.lr_gamma,
                patience=self.config.lr_step_size,
            )
        else:
            return None

    def _setup_samplers(self) -> Tuple[EpisodeSampler, EpisodeSampler, EpisodeSampler]:
        """Set up episode samplers for train/val/test."""
        train_sampler = EpisodeSampler(
            tasks=self.task_splits.train_tasks,
            n_way=self.config.n_way,
            n_support=self.config.n_support,
            n_query=self.config.n_query,
            featurizer_name=self.config.featurizer_name,
            random_seed=self.config.random_seed,
        )

        val_sampler = EpisodeSampler(
            tasks=self.task_splits.val_tasks,
            n_way=self.config.n_way,
            n_support=self.config.n_support,
            n_query=self.config.n_query,
            featurizer_name=self.config.featurizer_name,
            random_seed=self.config.random_seed + 1,
        )

        test_sampler = EpisodeSampler(
            tasks=self.task_splits.test_tasks,
            n_way=self.config.n_way,
            n_support=self.config.n_support,
            n_query=self.config.n_query,
            featurizer_name=self.config.featurizer_name,
            random_seed=self.config.random_seed + 2,
        )

        return train_sampler, val_sampler, test_sampler

    def train(self) -> Dict[str, List[float]]:
        """
        Run meta-learning training.

        Returns:
            Dict[str, List[float]]: Training history
        """
        logger.info("Starting meta-learning training")
        start_time = time.time()

        # Save configuration
        self._save_config()

        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch

                # Training phase
                train_metrics = self._train_epoch()

                # Validation phase
                if epoch % self.config.val_frequency == 0:
                    val_metrics = self._validate()

                    # Learning rate scheduling
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_metrics["accuracy"])
                        else:
                            self.scheduler.step()

                    # Early stopping check
                    if self._check_early_stopping(val_metrics["accuracy"]):
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                    # Save checkpoint
                    if (
                        epoch % self.config.checkpoint_frequency == 0
                        or val_metrics["accuracy"] > self.best_val_accuracy
                    ):
                        self._save_checkpoint(epoch, val_metrics["accuracy"])

                # Logging
                if epoch % self.config.log_frequency == 0:
                    self._log_metrics(
                        epoch, train_metrics, val_metrics if epoch % self.config.val_frequency == 0 else None
                    )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise

        finally:
            if self.writer is not None:
                self.writer.close()

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Save final model and history
        self._save_final_model()
        self._save_training_history()

        return {
            "train_loss": [h["train_loss"] for h in self.training_history],
            "train_accuracy": [h["train_accuracy"] for h in self.training_history],
            "val_loss": [h.get("val_loss", 0) for h in self.training_history],
            "val_accuracy": [h.get("val_accuracy", 0) for h in self.training_history],
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        num_episodes = 0

        for episode_idx in range(self.config.episodes_per_epoch):
            # Sample episode batch
            episode_batch = self.train_sampler.sample_batch(self.config.batch_size)

            # Move to device
            for key in ["support_features", "support_labels", "query_features", "query_labels"]:
                episode_batch[key] = episode_batch[key].to(self.device)

            # Forward pass for each episode in batch
            batch_loss = 0.0
            batch_accuracy = 0.0

            for batch_idx in range(self.config.batch_size):
                support_features = episode_batch["support_features"][batch_idx]
                support_labels = episode_batch["support_labels"][batch_idx]
                query_features = episode_batch["query_features"][batch_idx]
                query_labels = episode_batch["query_labels"][batch_idx]

                # Compute episodic loss
                loss, metrics = self.model.episodic_loss(
                    support_features,
                    support_labels,
                    query_features,
                    query_labels,
                    temperature=self.config.temperature,
                )

                batch_loss += loss
                batch_accuracy += metrics["accuracy"]

            # Average over batch
            batch_loss /= self.config.batch_size
            batch_accuracy /= self.config.batch_size

            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()

            # Gradient clipping
            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += batch_loss.item()
            total_accuracy += batch_accuracy
            num_episodes += 1

        return {
            "loss": total_loss / num_episodes,
            "accuracy": total_accuracy / num_episodes,
        }

    def _validate(self) -> Dict[str, float]:
        """Validate on validation tasks."""
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        num_episodes = 0

        with torch.no_grad():
            for episode_idx in range(self.config.val_episodes_per_epoch):
                # Sample episode
                episode = self.val_sampler.sample_episode()

                # Move to device
                for key in ["support_features", "support_labels", "query_features", "query_labels"]:
                    episode[key] = episode[key].to(self.device)

                # Compute loss
                loss, metrics = self.model.episodic_loss(
                    episode["support_features"],
                    episode["support_labels"],
                    episode["query_features"],
                    episode["query_labels"],
                    temperature=self.config.temperature,
                )

                total_loss += loss.item()
                total_accuracy += metrics["accuracy"]
                num_episodes += 1

        return {
            "loss": total_loss / num_episodes,
            "accuracy": total_accuracy / num_episodes,
        }

    def _check_early_stopping(self, val_accuracy: float) -> bool:
        """Check early stopping condition."""
        if val_accuracy > self.best_val_accuracy + self.config.early_stopping_min_delta:
            self.best_val_accuracy = val_accuracy
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def _log_metrics(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log training metrics."""
        # Console logging
        log_str = f"Epoch {epoch:3d}: Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['accuracy']:.4f}"
        if val_metrics is not None:
            log_str += f", Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}"
        logger.info(log_str)

        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Accuracy/Train", train_metrics["accuracy"], epoch)

            if val_metrics is not None:
                self.writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
                self.writer.add_scalar("Accuracy/Val", val_metrics["accuracy"], epoch)

            # Learning rate logging
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Store in history
        history_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        }
        if val_metrics is not None:
            history_entry.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
            )

        self.training_history.append(history_entry)

    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def _save_checkpoint(self, epoch: int, val_accuracy: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_accuracy": val_accuracy,
            "config": self.config.to_dict(),
        }

        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_accuracy > self.best_val_accuracy and self.config.save_best_model:
            best_model_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model with val accuracy: {val_accuracy:.4f}")

    def _save_final_model(self) -> None:
        """Save final model state."""
        final_model_path = self.output_dir / "final_model.pt"
        torch.save(self.model.state_dict(), final_model_path)

    def _save_training_history(self) -> None:
        """Save training history."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate on test tasks.

        Args:
            num_episodes (int): Number of episodes to evaluate

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info(f"Evaluating on {num_episodes} test episodes")

        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0

        with torch.no_grad():
            for episode_idx in range(num_episodes):
                # Sample episode
                episode = self.test_sampler.sample_episode()

                # Move to device
                for key in ["support_features", "support_labels", "query_features", "query_labels"]:
                    episode[key] = episode[key].to(self.device)

                # Compute loss
                loss, metrics = self.model.episodic_loss(
                    episode["support_features"],
                    episode["support_labels"],
                    episode["query_features"],
                    episode["query_labels"],
                    temperature=self.config.temperature,
                )

                total_loss += loss.item()
                total_accuracy += metrics["accuracy"]

        eval_metrics = {
            "test_loss": total_loss / num_episodes,
            "test_accuracy": total_accuracy / num_episodes,
        }

        logger.info(
            f"Test Results: Loss={eval_metrics['test_loss']:.4f}, "
            f"Accuracy={eval_metrics['test_accuracy']:.4f}"
        )

        # Save evaluation results
        eval_path = self.output_dir / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)

        return eval_metrics
