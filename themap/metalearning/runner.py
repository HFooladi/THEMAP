"""End-to-end distance-guided meta-learning experiment.

Glues the whole workflow together: select the k-nearest source datasets from a
saved distance file, featurize sources + target once, meta-train the chosen
algorithm on the sources, and evaluate the low-data gain on the target.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..data.loader import DatasetLoader
from ..utils.logging import get_logger
from .config import ExperimentConfig
from .episodes import EpisodeSampler, FeatureBank, max_feasible_n_support
from .evaluation import METRICS, LowDataEvaluator
from .selection import select_k_nearest_sources
from .trainer import MetaTrainer

logger = get_logger(__name__)


class MetaLearnExperiment:
    """Runs the select → featurize → meta-train → low-data-evaluate pipeline."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _load_datasets(self, source_ids: List[str]) -> Tuple[Dict, object]:
        loader = DatasetLoader(self.config.data_dir)
        sources = loader.load_datasets(self.config.source_fold, task_ids=source_ids)
        missing = [s for s in source_ids if s not in sources]
        if missing:
            raise ValueError(f"Source dataset(s) not found in fold '{self.config.source_fold}': {missing}")
        targets = loader.load_datasets(self.config.target_fold, task_ids=[self.config.target_id])
        if self.config.target_id not in targets:
            raise ValueError(
                f"Target '{self.config.target_id}' not found in fold '{self.config.target_fold}'."
            )
        return sources, targets[self.config.target_id]

    def _build_learner(self, feature_dim: int):
        cfg = self.config
        if cfg.algorithm == "proto":
            from .models.protonet import ProtoNet

            return ProtoNet(feature_dim, cfg.encoder, cfg.proto)
        if cfg.algorithm == "maml":
            from .models.maml import MAMLLearner

            return MAMLLearner(feature_dim, cfg.encoder, cfg.maml)
        raise ValueError(f"Unknown algorithm '{cfg.algorithm}' (expected 'proto' or 'maml').")

    def run(self) -> pd.DataFrame:
        """Execute the experiment and return the long-form results DataFrame."""
        cfg = self.config

        # 1. Select k-nearest sources from the distance file.
        selected = select_k_nearest_sources(cfg.distance_file, cfg.target_id, cfg.k)
        source_ids = [s for s, _ in selected]
        logger.info("Selected %d source(s) for target '%s': %s", len(source_ids), cfg.target_id, selected)

        # 2. Load datasets, 3. featurize sources + target once (shared dedup).
        sources, target_ds = self._load_datasets(source_ids)
        all_datasets = dict(sources)
        all_datasets[cfg.target_id] = target_ds
        bank = FeatureBank.from_datasets(all_datasets, featurizer=cfg.featurizer, n_jobs=cfg.n_jobs)
        feature_dim = bank.feature_dim

        source_tasks = [bank[s] for s in source_ids if s in bank.tasks]
        if not source_tasks:
            raise ValueError("No source tasks survived featurization/validity filtering.")
        target_task = bank[cfg.target_id]
        train_tasks, val_tasks = self._split_sources(source_tasks)

        # 4-6. Meta-train and low-data-evaluate.
        frames: List[pd.DataFrame] = []
        histories: Dict[str, list] = {}
        if cfg.train_shot_mode == "fixed":
            # FS-Mol single-model protocol: train one model once and evaluate it
            # across every support size in a single sweep.
            train_shot = self._train_shot_for_n(0, train_tasks, cfg)
            if train_shot >= 2:
                learner, history = self._meta_train(feature_dim, train_tasks, val_tasks, train_shot)
                histories["all"] = history
                frames.append(self._evaluate(learner, target_task, feature_dim, cfg.support_sizes))
            else:
                logger.warning("Sources cannot supply a balanced episode; no model trained.")
        else:
            # "match": a fresh model per support size, with the training shot tracking N.
            for n in cfg.support_sizes:
                train_shot = self._train_shot_for_n(n, train_tasks, cfg)
                if train_shot < 2:
                    logger.warning("Skipping support_size=%d: sources cannot supply a balanced episode.", n)
                    continue
                learner, history = self._meta_train(feature_dim, train_tasks, val_tasks, train_shot)
                histories[str(n)] = history
                frames.append(self._evaluate(learner, target_task, feature_dim, [n]))

        results = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["algorithm", "support_size", "seed", "method", *METRICS])
        )
        summary = LowDataEvaluator.summarize(results)

        # 7. Persist outputs.
        if cfg.output_dir:
            self._save(cfg, selected, results, summary, histories)

        return results

    def _train_shot_for_n(self, n: int, train_tasks: List, cfg: ExperimentConfig) -> int:
        """Pick the meta-training support shot for an eval support size ``n``.

        In ``"match"`` mode the training shot tracks ``n``; in ``"fixed"`` (FS-Mol
        single-model) mode it is ``TrainConfig.n_support``. Either way it is capped to
        what the source datasets can supply, so :class:`EpisodeSampler` never raises.
        """
        desired = cfg.train.n_support if cfg.train_shot_mode == "fixed" else n
        feasible = max_feasible_n_support(train_tasks, cfg.train.n_query)
        return min(desired, feasible)

    def _evaluate(self, learner, target_task, feature_dim: int, support_sizes: List[int]):
        """Run the low-data evaluator for one learner over the given support sizes."""
        cfg = self.config
        evaluator = LowDataEvaluator(
            learner=learner,
            target=target_task,
            input_dim=feature_dim,
            encoder_config=cfg.encoder,
            algorithm=cfg.algorithm,
            support_sizes=support_sizes,
            seeds=cfg.seeds,
            device=cfg.train.device,
            query_fraction=cfg.query_fraction,
        )
        return evaluator.evaluate()

    def _meta_train(self, feature_dim: int, train_tasks: List, val_tasks: List, n_support: int):
        """Build a fresh learner and meta-train it at the given support shot."""
        cfg = self.config
        train_sampler = EpisodeSampler(
            train_tasks, n_support=n_support, n_query=cfg.train.n_query, seed=cfg.train.seed
        )
        val_sampler = (
            EpisodeSampler(
                val_tasks,
                n_support=n_support,
                n_query=cfg.train.n_query,
                seed=cfg.train.seed + 1,
            )
            if val_tasks
            else None
        )
        learner = self._build_learner(feature_dim)
        trainer = MetaTrainer(learner, train_sampler, cfg.train, val_sampler=val_sampler)
        history = trainer.train()
        return learner, history

    def _split_sources(self, source_tasks: List) -> Tuple[List, List]:
        """Hold out one source task for validation when at least 3 are available."""
        if len(source_tasks) >= 3:
            return source_tasks[:-1], source_tasks[-1:]
        return source_tasks, []

    def _save(self, cfg, selected, results, summary, history) -> None:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results.to_csv(out / "results.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)
        with open(out / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        with open(out / "selected_sources.json", "w") as f:
            json.dump({"target_id": cfg.target_id, "sources": selected}, f, indent=2)
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Saved meta-learning results to %s", out)
