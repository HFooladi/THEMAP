"""Run THEMAP's meta-learners on the FS-Mol benchmark under FS-Mol's own protocol.

This exists to answer one question: does THEMAP's PyTorch re-implementation of ProtoNet
and MAML land anywhere near the published TensorFlow FS-Mol results, or is something
fundamentally wrong? It is a parity check, not a leaderboard entry.

The shape of the run differs from :class:`~themap.metalearning.runner.MetaLearnExperiment`
in one decisive way: that class meta-trains a fresh model for a single target task, which
is right for distance-guided source selection but wrong here — it would re-pay the cost of
featurizing and meta-training on thousands of source assays for every target. This module
featurizes once, meta-trains once per algorithm, and then evaluates every target task with
the same model, which is also exactly what FS-Mol does.

Protocol choices that matter, and why:

* **Query sets follow FS-Mol** (``query_mode="fsmol"``): draw a stratified support set of
  size N, evaluate on the entire remainder. THEMAP's default fractional holdout makes
  N=128 infeasible for three quarters of the FS-Mol test tasks.
* **Episodes are adaptive** (``adaptive_episodes=True``): the requested meta-training shot
  is a maximum, not a requirement, so all ~4938 FS-Mol training assays contribute. Under
  the strict default a 64-shot request would silently keep only ~17% of them.
* **Validation uses FS-Mol's own ``valid`` fold**, not a held-out source task.
* **ΔAUPRC** is already defined as FS-Mol defines it in
  :func:`themap.metalearning.evaluation._safe_metrics`, so no change was needed there.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.loader import DatasetLoader
from ..utils.logging import get_logger
from .config import Algorithm, EncoderConfig, MAMLConfig, ProtoConfig, TrainConfig
from .episodes import EpisodeSampler, FeatureBank, TaskFeatures, max_feasible_n_support, usable_task_count
from .evaluation import RESULT_COLUMNS, LowDataEvaluator
from .trainer import MetaTrainer

logger = get_logger(__name__)

#: Support sizes FS-Mol reports. 256 is only feasible for ~26% of its test tasks.
FSMOL_SUPPORT_SIZES: Tuple[int, ...] = (16, 32, 64, 128, 256)


@dataclass
class BenchmarkConfig:
    """Configuration for :class:`FSMolBenchmark`.

    Attributes:
        data_dir: FS-Mol dataset root holding ``train``/``valid``/``test`` folds.
        task_list_file: JSON file in ``data_dir`` naming the canonical fold membership.
            FS-Mol's ``fsmol_tasks_list.json`` already has the required shape.
        algorithms: Which meta-learners to benchmark.
        featurizer: Molecular featurizer name.
        support_sizes: Evaluation support-set sizes swept on each target task.
        seeds: Repeats per (task, support size). FS-Mol uses 10.
        target_ids: Explicit target task list; when None the subset file is used, and
            when that is absent every task in ``target_fold`` is evaluated.
        subset_file: Path to a subset JSON written by
            :func:`themap.metalearning.subset.save_subset`.
        n_source_tasks: Cap on meta-training source tasks (0 = all of them). Intended for
            smoke tests and debugging, not as a cost lever — featurization is cached and
            meta-training cost does not scale with pool size.
        source_sample_seed: Seed used when ``n_source_tasks`` subsamples the pool.
        min_source_tasks: Abort if fewer sources than this survive the episode filter.
        adaptive_episodes: Treat the meta-training shot as a maximum (see module docstring).
        train_shot: Support size used to build meta-training episodes.
        query_mode: ``"fsmol"`` for parity, ``"holdout"`` for THEMAP's default scheme.
        n_jobs: Featurization parallelism; 1 is usually fastest for fingerprints.
        cache_dir: Feature cache directory.
        reference_dir: Where FS-Mol's baseline CSVs are cached.
        offline: Never fetch the reference CSVs from the network.
        output_dir: Directory for all emitted artifacts.
    """

    data_dir: str = "benchmarking_datasets/fsmol_datasets"
    task_list_file: str = "fsmol_tasks_list.json"
    algorithms: List[Algorithm] = field(default_factory=lambda: ["proto", "maml"])
    featurizer: str = "ecfp"
    support_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    seeds: int = 10
    target_ids: Optional[List[str]] = None
    subset_file: Optional[str] = None
    source_fold: str = "train"
    valid_fold: str = "valid"
    target_fold: str = "test"
    n_source_tasks: int = 0
    source_sample_seed: int = 0
    min_source_tasks: int = 50
    adaptive_episodes: bool = True
    train_shot: int = 64
    query_mode: str = "fsmol"
    n_jobs: int = 1
    cache_dir: str = "feature_cache/fsmol"
    reference_dir: str = "benchmarking_datasets/fsmol_reference"
    offline: bool = False
    output_dir: str = "fsmol_benchmark_out"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    proto: ProtoConfig = field(default_factory=ProtoConfig)
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _git_sha() -> str:
    """Short git SHA of the working tree, or ``"unknown"`` outside a repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


class FSMolBenchmark:
    """Featurize once, meta-train once per algorithm, evaluate every target task."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = DatasetLoader(config.data_dir, task_list_file=config.task_list_file)
        self._histories: Dict[str, Any] = {}
        self._pool_stats: Dict[str, Any] = {}

    # --- id resolution ----------------------------------------------------

    def _fold_ids(self, fold: str) -> List[str]:
        """Task ids for a fold, from the task list when present, else from disk."""
        if self.loader.task_list and fold in self.loader.task_list:
            return list(self.loader.task_list[fold])
        fold_dir = Path(self.config.data_dir) / fold
        return sorted(p.name[: -len(".jsonl.gz")] for p in fold_dir.glob("*.jsonl.gz"))

    def resolve_source_ids(self) -> List[str]:
        """Meta-training source tasks, optionally subsampled."""
        ids = self._fold_ids(self.config.source_fold)
        if self.config.n_source_tasks and self.config.n_source_tasks < len(ids):
            rng = np.random.default_rng(self.config.source_sample_seed)
            ids = sorted(rng.choice(ids, size=self.config.n_source_tasks, replace=False).tolist())
        return ids

    def resolve_target_ids(self) -> List[str]:
        """Target tasks: explicit list, else the saved subset, else the whole fold."""
        if self.config.target_ids:
            return list(self.config.target_ids)
        if self.config.subset_file:
            from .subset import load_subset

            return load_subset(self.config.subset_file)
        return self._fold_ids(self.config.target_fold)

    # --- featurization ----------------------------------------------------

    def _feature_cache(self) -> Any:
        from ..features.cache import FeatureCache

        return FeatureCache(self.config.cache_dir)

    def _bank(self, fold: str, task_ids: Sequence[str]) -> FeatureBank:
        return FeatureBank.from_loader_cached(
            self.loader,
            fold,
            task_ids,
            featurizer=self.config.featurizer,
            n_jobs=self.config.n_jobs,
            cache=self._feature_cache(),
        )

    def featurize(self) -> Dict[str, FeatureBank]:
        """Featurize the source, validation and target folds, populating the cache."""
        cfg = self.config
        banks = {
            "source": self._bank(cfg.source_fold, self.resolve_source_ids()),
            "valid": self._bank(cfg.valid_fold, self._fold_ids(cfg.valid_fold)),
            "target": self._bank(cfg.target_fold, self.resolve_target_ids()),
        }
        for name, bank in banks.items():
            logger.info("Featurized %s fold: %d task(s).", name, len(bank))
        return banks

    # --- meta-training ----------------------------------------------------

    def _build_learner(self, algorithm: Algorithm, feature_dim: int) -> Any:
        cfg = self.config
        if algorithm == "proto":
            from .models.protonet import ProtoNet

            return ProtoNet(feature_dim, cfg.encoder, cfg.proto)
        if algorithm == "maml":
            from .models.maml import MAMLLearner

            return MAMLLearner(feature_dim, cfg.encoder, cfg.maml)
        raise ValueError(f"Unknown algorithm '{algorithm}' (expected 'proto' or 'maml').")

    def _sampler(self, tasks: List[TaskFeatures], seed: int) -> EpisodeSampler:
        cfg = self.config
        return EpisodeSampler(
            tasks,
            n_support=cfg.train_shot,
            n_query=cfg.train.n_query,
            seed=seed,
            adaptive=cfg.adaptive_episodes,
        )

    def meta_train(self, algorithm: Algorithm, source: FeatureBank, valid: FeatureBank) -> Any:
        """Meta-train one learner on the full source pool, validating on FS-Mol's valid fold."""
        cfg = self.config
        source_tasks = list(source.tasks.values())
        valid_tasks = list(valid.tasks.values())

        # Report the pool honestly. max_feasible_n_support over thousands of assays
        # reports the single biggest task's capacity, so it is useless as a cap here --
        # the median is the number that describes the pool.
        n_usable = usable_task_count(
            source_tasks,
            cfg.train_shot,
            cfg.train.n_query,
            adaptive=cfg.adaptive_episodes,
        )
        median_capacity = max_feasible_n_support(source_tasks, cfg.train.n_query, quantile=0.5)
        logger.info(
            "Meta-training '%s': shot=%d n_query=%d adaptive=%s -> %d/%d source task(s) usable "
            "(median task capacity %d).",
            algorithm,
            cfg.train_shot,
            cfg.train.n_query,
            cfg.adaptive_episodes,
            n_usable,
            len(source_tasks),
            median_capacity,
        )
        self._pool_stats[algorithm] = {
            "n_source_tasks": len(source_tasks),
            "n_usable_tasks": n_usable,
            "median_task_capacity": median_capacity,
            "train_shot_requested": cfg.train_shot,
            "adaptive_episodes": cfg.adaptive_episodes,
        }
        if n_usable < cfg.min_source_tasks:
            raise ValueError(
                f"Only {n_usable} source task(s) can supply a {cfg.train_shot}-shot/"
                f"{cfg.train.n_query}-query episode, below min_source_tasks="
                f"{cfg.min_source_tasks}. Lower --train-shot, enable adaptive episodes, "
                f"or lower --min-source-tasks."
            )

        train_sampler = self._sampler(source_tasks, cfg.train.seed)
        val_sampler = self._sampler(valid_tasks, cfg.train.seed + 1) if valid_tasks else None

        realized = [train_sampler.sample_episode() for _ in range(min(64, len(train_sampler)))]
        self._pool_stats[algorithm]["realized_mean_support"] = float(
            np.mean([e.x_s.shape[0] for e in realized])
        )
        self._pool_stats[algorithm]["realized_mean_query"] = float(
            np.mean([e.x_q.shape[0] for e in realized])
        )
        logger.info(
            "Realized episode sizes for '%s': support %.1f, query %.1f (requested %d/%d).",
            algorithm,
            self._pool_stats[algorithm]["realized_mean_support"],
            self._pool_stats[algorithm]["realized_mean_query"],
            cfg.train_shot,
            cfg.train.n_query,
        )

        learner = self._build_learner(algorithm, source.feature_dim)
        trainer = MetaTrainer(learner, train_sampler, cfg.train, val_sampler=val_sampler)
        self._histories[algorithm] = trainer.train()
        return learner

    # --- evaluation -------------------------------------------------------

    def evaluate(self, algorithm: Algorithm, learner: Any, targets: FeatureBank) -> pd.DataFrame:
        """Sweep every target task with one already-trained learner."""
        cfg = self.config
        frames: List[pd.DataFrame] = []
        for i, (task_id, task) in enumerate(sorted(targets.tasks.items()), start=1):
            evaluator = LowDataEvaluator(
                learner=learner,
                target=task,
                input_dim=targets.feature_dim,
                encoder_config=cfg.encoder,
                algorithm=algorithm,
                support_sizes=cfg.support_sizes,
                seeds=cfg.seeds,
                device=cfg.train.device,
                query_mode=cfg.query_mode,  # type: ignore[arg-type]
            )
            result = evaluator.evaluate()
            result.insert(0, "task_id", task_id)
            frames.append(result)
            if i % 5 == 0 or i == len(targets.tasks):
                logger.info("Evaluated %d/%d target task(s) for '%s'.", i, len(targets.tasks), algorithm)
        if not frames:
            return pd.DataFrame(columns=["task_id", *RESULT_COLUMNS])
        return pd.concat(frames, ignore_index=True)

    # --- orchestration ----------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the whole benchmark and write every artifact to ``output_dir``."""
        cfg = self.config
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        banks = self.featurize()
        frames: List[pd.DataFrame] = []
        for algorithm in cfg.algorithms:
            learner = self.meta_train(algorithm, banks["source"], banks["valid"])
            frames.append(self.evaluate(algorithm, learner, banks["target"]))

        results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._save(out, results, banks)
        return results

    def _save(self, out: Path, results: pd.DataFrame, banks: Dict[str, FeatureBank]) -> None:
        results.to_csv(out / "per_task_results.csv", index=False)

        from .report import build_report

        report = build_report(results, self.config)
        for name, frame in report.tables.items():
            frame.to_csv(out / f"{name}.csv", index=False)
        (out / "report.md").write_text(report.markdown)
        if report.figure_bytes is not None:
            (out / "comparison.png").write_bytes(report.figure_bytes)

        with open(out / "config.json", "w") as handle:
            json.dump(
                {
                    "config": asdict(self.config),
                    "resolved": {
                        "source_task_ids": sorted(banks["source"].task_ids()),
                        "valid_task_ids": sorted(banks["valid"].task_ids()),
                        "target_task_ids": sorted(banks["target"].task_ids()),
                    },
                    "pool_stats": self._pool_stats,
                    "provenance": {
                        "git_sha": _git_sha(),
                        "python": platform.python_version(),
                    },
                },
                handle,
                indent=2,
                default=str,
            )
        with open(out / "history.json", "w") as handle:
            json.dump(self._histories, handle, indent=2, default=str)
        logger.info("Wrote FS-Mol benchmark artifacts to %s", out)
