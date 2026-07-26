"""Feature extraction and episodic sampling for meta-learning.

Featurization happens **once** per dataset via the batched, SMILES-deduplicated
:class:`~themap.features.molecule.MoleculeFeaturizer` path; episodes are then
sampled cheaply from in-memory ``(X, y)`` numpy arrays. This avoids the per-SMILES
re-featurization that plagued the previous implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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

    @classmethod
    def from_loader_cached(
        cls,
        loader: Any,
        fold: str,
        task_ids: Sequence[str],
        featurizer: str = "ecfp",
        n_jobs: int = 1,
        cache: Optional[Any] = None,
        batch_size: int = 512,
    ) -> "FeatureBank":
        """Featurize a fold's tasks, reusing anything already on disk.

        Featurizing thousands of assays takes minutes; doing it once per algorithm or per
        re-run wastes most of a benchmark's wall-clock. Cached tasks are read straight
        back; misses are featurized in batches so cross-task SMILES deduplication still
        applies within a batch, then written back.

        Raw features are cached and :meth:`TaskFeatures.from_arrays` does the non-finite
        row filtering on load, so cached and uncached paths give identical results.

        Args:
            loader: A :class:`~themap.data.loader.DatasetLoader`.
            fold: Fold to read from.
            task_ids: Tasks to featurize.
            featurizer: Molecular featurizer name.
            n_jobs: Parallel jobs for featurization. For fingerprints, 1 is typically
                fastest — the process-pool startup cost outweighs the parallelism.
            cache: Optional :class:`~themap.features.cache.FeatureCache`.
            batch_size: Number of tasks featurized per batch.

        Returns:
            A :class:`FeatureBank` holding every task that produced features.
        """
        from ..features.molecule import MoleculeFeaturizer

        tasks: Dict[str, TaskFeatures] = {}
        pending: List[str] = []

        for task_id in task_ids:
            if cache is not None:
                features, labels = cache.load_molecule_features(task_id, featurizer)
                if features is not None and labels is not None:
                    tasks[task_id] = TaskFeatures.from_arrays(task_id, features, labels)
                    continue
            pending.append(task_id)

        if pending:
            logger.info(
                "Featurizing %d/%d task(s) with '%s' (%d already cached).",
                len(pending),
                len(list(task_ids)),
                featurizer,
                len(tasks),
            )
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            datasets = loader.load_datasets(fold, task_ids=batch)
            MoleculeFeaturizer(featurizer_name=featurizer, n_jobs=n_jobs).featurize_datasets(
                datasets, deduplicate=True
            )
            for task_id, dataset in datasets.items():
                features = dataset.features
                if features is None:
                    logger.warning("Task %s produced no features; skipping.", task_id)
                    continue
                labels = np.asarray(dataset.labels)
                if cache is not None:
                    cache.save_molecule_features(
                        task_id,
                        featurizer,
                        np.asarray(features, dtype=np.float32),
                        labels.astype(np.int32),
                        metadata={"featurizer": featurizer, "n_molecules": int(len(labels))},
                    )
                tasks[task_id] = TaskFeatures.from_arrays(task_id, features, labels)
            logger.info(
                "Featurized %d/%d pending task(s).", min(start + batch_size, len(pending)), len(pending)
            )

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


def max_feasible_n_support(
    tasks: List[TaskFeatures],
    n_query: int,
    n_way: int = 2,
    quantile: Optional[float] = None,
) -> int:
    """Largest total balanced support shot a task in the pool can supply.

    A balanced binary episode needs ``n_support // n_way`` support and
    ``n_query // n_way`` query examples *per class*. For each task the support
    quota per class is bounded by ``min(#pos, #neg) - (n_query // n_way)``; the
    returned value is ``n_way`` times the best such quota across all tasks, i.e.
    the largest ``n_support`` for which at least one task can form an episode.

    Beware what "at least one task" means on a large, skewed corpus: over thousands of
    assays this reports the capacity of the single biggest one, so using it as a cap can
    leave the requested shot untouched while :class:`EpisodeSampler` silently drops most
    of the pool. Pass ``quantile=0.5`` for the median task's capacity, which is the honest
    number to log alongside :func:`usable_task_count`.

    Args:
        tasks: Candidate tasks.
        n_query: Total query size the episode must also supply.
        n_way: Number of classes (only 2 is supported elsewhere).
        quantile: If given, report this quantile of per-task capacity instead of the max.

    Returns:
        Total support size, or 0 when no task can supply even a 1-shot-per-class episode.
    """
    qry_per_class = n_query // n_way
    per_class = [max(0, min(len(t.pos_idx), len(t.neg_idx)) - qry_per_class) for t in tasks]
    if not per_class:
        return 0
    if quantile is None:
        return max(per_class) * n_way
    return int(np.quantile(per_class, quantile)) * n_way


def usable_task_count(
    tasks: List[TaskFeatures],
    n_support: int,
    n_query: int,
    n_way: int = 2,
    adaptive: bool = False,
    min_support: int = 4,
    min_query: int = 4,
) -> int:
    """How many tasks would survive :class:`EpisodeSampler`'s validity filter.

    Mirrors the filter exactly, so it can be logged before training starts instead of
    discovering the shortfall from a warning buried in the sampler.
    """
    if adaptive:
        need = max(1, min_support // n_way) + max(1, min_query // n_way)
    else:
        need = n_support // n_way + n_query // n_way
    return sum(1 for t in tasks if min(len(t.pos_idx), len(t.neg_idx)) >= need)


class EpisodeSampler:
    """Samples balanced n-way support/query episodes from a set of tasks.

    Only binary (2-way) episodes are supported, matching molecular activity
    classification. Tasks with too few examples of either class are filtered out
    at construction time.

    By default the requested shot is a *requirement*: a task must be able to supply
    the full ``n_support``/``n_query`` quota or it is dropped. On corpora of many small
    assays that discards most of the pool — on FS-Mol's 4938 training tasks (median 44
    datapoints), a 64-shot request keeps only about 17%. Setting ``adaptive=True`` treats
    the requested sizes as *maxima* instead, the way FS-Mol's own ``StratifiedTaskSampler``
    does, so small tasks contribute smaller episodes rather than being excluded.
    """

    def __init__(
        self,
        tasks: List[TaskFeatures],
        n_support: int = 10,
        n_query: int = 15,
        n_way: int = 2,
        balanced: bool = True,
        seed: Optional[int] = None,
        adaptive: bool = False,
        min_support: int = 4,
        min_query: int = 4,
    ):
        if n_way != 2:
            raise ValueError("Only 2-way (binary) episodes are supported.")
        self.torch = require_torch()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.balanced = balanced
        self.adaptive = adaptive
        self.rng = np.random.default_rng(seed)

        # Per-class quota (balanced split of support/query across the two classes).
        self.sup_per_class = n_support // n_way
        self.qry_per_class = n_query // n_way
        self.min_sup_per_class = max(1, min_support // n_way)
        self.min_qry_per_class = max(1, min_query // n_way)

        need = (
            self.min_sup_per_class + self.min_qry_per_class
            if adaptive
            else self.sup_per_class + self.qry_per_class
        )

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

    def _quota_for(self, task: TaskFeatures) -> tuple:
        """Per-class ``(support, query)`` quota for one task.

        Fixed at the requested sizes unless ``adaptive`` is set, in which case both are
        clipped to what the task can supply. The query side is protected first — it
        carries the meta-gradient — and support shrinks second, never below
        ``min_support``.
        """
        if not self.adaptive:
            return self.sup_per_class, self.qry_per_class

        avail = min(len(task.pos_idx), len(task.neg_idx))
        sup_pc = min(self.sup_per_class, max(self.min_sup_per_class, avail - self.qry_per_class))
        qry_pc = min(self.qry_per_class, avail - sup_pc)
        if qry_pc < self.min_qry_per_class:
            qry_pc = self.min_qry_per_class
            sup_pc = avail - qry_pc
        return sup_pc, qry_pc

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

        sup_per_class, qry_per_class = self._quota_for(task)

        sup_idx: List[int] = []
        qry_idx: List[int] = []
        sup_lbl: List[int] = []
        qry_lbl: List[int] = []
        for cls, pool in ((0, task.neg_idx), (1, task.pos_idx)):
            chosen = self._draw(pool, sup_per_class + qry_per_class)
            sup_idx.extend(chosen[:sup_per_class])
            qry_idx.extend(chosen[sup_per_class:])
            sup_lbl.extend([cls] * sup_per_class)
            qry_lbl.extend([cls] * qry_per_class)

        torch = self.torch
        x_s = torch.from_numpy(task.X[np.asarray(sup_idx)]).float()
        x_q = torch.from_numpy(task.X[np.asarray(qry_idx)]).float()
        y_s = torch.tensor(sup_lbl, dtype=torch.long)
        y_q = torch.tensor(qry_lbl, dtype=torch.long)
        return Episode(x_s=x_s, y_s=y_s, x_q=x_q, y_q=y_q, task_id=task.task_id)

    def sample_batch(self, meta_batch_size: int) -> List[Episode]:
        """Sample a list of ``meta_batch_size`` episodes (variable shapes allowed)."""
        return [self.sample_episode() for _ in range(meta_batch_size)]
