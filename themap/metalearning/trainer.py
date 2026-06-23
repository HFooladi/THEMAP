"""Algorithm-agnostic episodic meta-training loop.

Both :class:`~themap.metalearning.models.protonet.ProtoNet` and
:class:`~themap.metalearning.models.maml.MAMLLearner` expose the same interface
(``outer_parameters()`` and ``compute_episode_loss(episode, device)``), so this
trainer drives either one without special-casing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logging import get_logger
from ._torch import require_torch
from .config import TrainConfig
from .episodes import EpisodeSampler

logger = get_logger(__name__)


def _safe_auroc(probs: Any, labels: Any) -> float:
    """AUROC guarded against single-class query sets (returns NaN)."""
    from sklearn.metrics import roc_auc_score

    y = labels.detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs.detach().cpu().numpy()))


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` if available, else ``"cpu"``."""
    torch = require_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class MetaTrainer:
    """Runs episodic meta-training with optional validation early-stopping."""

    def __init__(
        self,
        learner: Any,
        train_sampler: EpisodeSampler,
        config: TrainConfig,
        val_sampler: Optional[EpisodeSampler] = None,
    ):
        self.torch = require_torch()
        self.config = config
        self.device = resolve_device(config.device)
        self.learner = learner.to(self.device)
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        self.torch.manual_seed(config.seed)
        self.optimizer = self.torch.optim.Adam(
            learner.outer_parameters(), lr=config.outer_lr, weight_decay=config.weight_decay
        )

    def _train_epoch(self) -> Dict[str, float]:
        self.learner.train()
        losses: List[float] = []
        aurocs: List[float] = []
        for _ in range(self.config.episodes_per_epoch):
            episodes = self.train_sampler.sample_batch(self.config.meta_batch_size)
            self.optimizer.zero_grad()
            batch_loss = 0.0
            for ep in episodes:
                loss, probs, labels = self.learner.compute_episode_loss(ep, self.device)
                batch_loss = batch_loss + loss
                aurocs.append(_safe_auroc(probs, labels))
            batch_loss = batch_loss / len(episodes)
            batch_loss.backward()
            if self.config.grad_clip > 0:
                self.torch.nn.utils.clip_grad_norm_(self.learner.outer_parameters(), self.config.grad_clip)
            self.optimizer.step()
            losses.append(float(batch_loss.detach()))
        return {"loss": float(np.mean(losses)), "auroc": float(np.nanmean(aurocs))}

    def _validate(self) -> Dict[str, float]:
        # Validation mirrors the final low-data protocol: adapt on the support set
        # and score query AUROC. We do NOT wrap this in ``no_grad`` because MAML's
        # inner loop needs gradients; ``adapt_and_predict`` manages its own context.
        self.learner.eval()
        aurocs: List[float] = []
        for _ in range(self.config.val_episodes):
            ep = self.val_sampler.sample_episode().to(self.device)  # type: ignore[union-attr]
            probs = self.learner.adapt_and_predict(ep.x_s, ep.y_s, ep.x_q)
            aurocs.append(_safe_auroc(probs, ep.y_q))
        return {"auroc": float(np.nanmean(aurocs))}

    def train(self) -> Dict[str, List[float]]:
        """Run the meta-training loop and return the metric history."""
        history: Dict[str, List[float]] = {"train_loss": [], "train_auroc": [], "val_auroc": []}
        best_val = -np.inf
        best_state = None
        patience_left = self.config.patience

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["train_auroc"].append(train_metrics["auroc"])

            val_auroc = np.nan
            if self.val_sampler is not None and self.config.val_episodes > 0:
                val_auroc = self._validate()["auroc"]
            history["val_auroc"].append(float(val_auroc))

            logger.info(
                "epoch %d/%d  train_loss=%.4f train_auroc=%.3f val_auroc=%.3f",
                epoch,
                self.config.num_epochs,
                train_metrics["loss"],
                train_metrics["auroc"],
                val_auroc,
            )

            # Early stopping on validation AUROC.
            if self.val_sampler is not None and self.config.patience > 0 and np.isfinite(val_auroc):
                if val_auroc > best_val:
                    best_val = val_auroc
                    best_state = {k: v.detach().clone() for k, v in self.learner.state_dict().items()}
                    patience_left = self.config.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        logger.info("Early stopping at epoch %d (best val_auroc=%.3f).", epoch, best_val)
                        break

        if best_state is not None:
            self.learner.load_state_dict(best_state)
        return history
