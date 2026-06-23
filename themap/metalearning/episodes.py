"""Feature extraction and episodic sampling for meta-learning.

Featurization happens **once** per dataset via the batched, SMILES-deduplicated
:class:`~themap.features.molecule.MoleculeFeaturizer` path; episodes are then
sampled cheaply from in-memory ``(X, y)`` numpy arrays. This avoids the per-SMILES
re-featurization that plagued the previous implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from ..utils.logging import get_logger
from ._torch import require_torch

logger = get_logger(__name__)


@dataclass
class TaskFeatures:
    """Cached features and labels for a single task.

    Attributes:
        task_id: Task identifier.
        X: Feature matrix of shape ``(n, dim)``, float32.
        y: Binary labels of shape ``(n,)``, int.
        pos_idx: Indices of positive (label 1) rows.
        neg_idx: Indices of negative (label 0) rows.
    """

    task_id: str
    X: NDArray[np.float32]
    y: NDArray[np.int64]
    pos_idx: NDArray[np.int64]
    neg_idx: NDArray[np.int64]

    @classmethod
    def from_arrays(cls, task_id: str, X: NDArray[np.float32], y: NDArray[Any]) -> "TaskFeatures":
        """Build a :class:`TaskFeatures`, dropping rows with non-finite features."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64).ravel()
        finite = np.isfinite(X).all(axis=1)
        dropped = int((~finite).sum())
        if dropped:
            logger.warning("Task %s: dropping %d row(s) with non-finite features.", task_id, dropped)
            X, y = X[finite], y[finite]
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        return cls(task_id=task_id, X=X, y=y, pos_idx=pos_idx, neg_idx=neg_idx)

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Episode:
    """A single n-way support/query episode as CPU tensors."""

    x_s: Any  # torch.Tensor (n_support, dim)
    y_s: Any  # torch.Tensor (n_support,)
    x_q: Any  # torch.Tensor (n_query, dim)
    y_q: Any  # torch.Tensor (n_query,)
    task_id: str

    def to(self, device: str) -> "Episode":
        """Return a copy with all tensors moved to ``device``."""
        return Episode(
            x_s=self.x_s.to(device),
            y_s=self.y_s.to(device),
            x_q=self.x_q.to(device),
            y_q=self.y_q.to(device),
            task_id=self.task_id,
        )


class FeatureBank:
    """Holds featurized ``(X, y)`` arrays for a collection of tasks."""

    def __init__(self, tasks: Dict[str, TaskFeatures]):
        self.tasks = tasks

    @classmethod
    def from_datasets(
        cls,
        datasets: Dict[str, Any],
        featurizer: str = "ecfp",
        n_jobs: int = 8,
    ) -> "FeatureBank":
        """Featurize every dataset once and build a :class:`FeatureBank`.

        Args:
            datasets: Mapping ``task_id -> MoleculeDataset``.
            featurizer: Molecular featurizer name.
            n_jobs: Parallel jobs for featurization.

        Returns:
            A :class:`FeatureBank` keyed by the same task ids.
        """
        from ..features.molecule import MoleculeFeaturizer

        # Batched, SMILES-deduplicated featurization; sets ``_features`` on each.
        MoleculeFeaturizer(featurizer_name=featurizer, n_jobs=n_jobs).featurize_datasets(
            datasets, deduplicate=True
        )

        tasks: Dict[str, TaskFeatures] = {}
        for task_id, ds in datasets.items():
            features = ds.features
            if features is None:
                logger.warning("Task %s has no features after featurization; skipping.", task_id)
                continue
            tasks[task_id] = TaskFeatures.from_arrays(task_id, features, ds.labels)
        return cls(tasks)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vectors."""
        for tf in self.tasks.values():
            return int(tf.X.shape[1])
        raise ValueError("FeatureBank is empty; cannot infer feature dimension.")

    def task_ids(self) -> List[str]:
        return list(self.tasks.keys())

    def __getitem__(self, task_id: str) -> TaskFeatures:
        return self.tasks[task_id]

    def __len__(self) -> int:
        return len(self.tasks)


class EpisodeSampler:
    """Samples balanced n-way support/query episodes from a set of tasks.

    Only binary (2-way) episodes are supported, matching molecular activity
    classification. Tasks with too few examples of either class are filtered out
    at construction time.
    """

    def __init__(
        self,
        tasks: List[TaskFeatures],
        n_support: int = 10,
        n_query: int = 15,
        n_way: int = 2,
        balanced: bool = True,
        seed: Optional[int] = None,
    ):
        if n_way != 2:
            raise ValueError("Only 2-way (binary) episodes are supported.")
        self.torch = require_torch()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.balanced = balanced
        self.rng = np.random.default_rng(seed)

        # Per-class quota (balanced split of support/query across the two classes).
        self.sup_per_class = n_support // n_way
        self.qry_per_class = n_query // n_way
        need = self.sup_per_class + self.qry_per_class

        self.tasks = [t for t in tasks if min(len(t.pos_idx), len(t.neg_idx)) >= need]
        skipped = len(tasks) - len(self.tasks)
        if skipped:
            logger.warning(
                "EpisodeSampler: skipped %d/%d task(s) with fewer than %d examples per class.",
                skipped,
                len(tasks),
                need,
            )
        if not self.tasks:
            raise ValueError(
                f"No task has enough examples for {n_support}-shot/{n_query}-query "
                f"binary episodes (need >= {need} per class)."
            )

    def __len__(self) -> int:
        return len(self.tasks)

    def _draw(self, idx_pool: NDArray[np.int64], n: int) -> NDArray[np.int64]:
        return self.rng.choice(idx_pool, size=n, replace=False)

    def sample_episode(self, task: Optional[TaskFeatures] = None) -> Episode:
        """Sample one balanced binary episode.

        Args:
            task: Specific task to sample from; if None, a random task is chosen.

        Returns:
            An :class:`Episode` with relabeled ``{0, 1}`` targets on CPU.
        """
        if task is None:
            task = self.tasks[self.rng.integers(len(self.tasks))]

        sup_idx: List[int] = []
        qry_idx: List[int] = []
        sup_lbl: List[int] = []
        qry_lbl: List[int] = []
        for cls, pool in ((0, task.neg_idx), (1, task.pos_idx)):
            chosen = self._draw(pool, self.sup_per_class + self.qry_per_class)
            sup_idx.extend(chosen[: self.sup_per_class])
            qry_idx.extend(chosen[self.sup_per_class :])
            sup_lbl.extend([cls] * self.sup_per_class)
            qry_lbl.extend([cls] * self.qry_per_class)

        torch = self.torch
        x_s = torch.from_numpy(task.X[np.asarray(sup_idx)]).float()
        x_q = torch.from_numpy(task.X[np.asarray(qry_idx)]).float()
        y_s = torch.tensor(sup_lbl, dtype=torch.long)
        y_q = torch.tensor(qry_lbl, dtype=torch.long)
        return Episode(x_s=x_s, y_s=y_s, x_q=x_q, y_q=y_q, task_id=task.task_id)

    def sample_batch(self, meta_batch_size: int) -> List[Episode]:
        """Sample a list of ``meta_batch_size`` episodes (variable shapes allowed)."""
        return [self.sample_episode() for _ in range(meta_batch_size)]
