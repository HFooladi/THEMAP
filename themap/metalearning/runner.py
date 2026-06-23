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
from .episodes import EpisodeSampler, FeatureBank
from .evaluation import LowDataEvaluator
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

        # 4. Source episode sampler (+ optional held-out source for validation).
        train_tasks, val_tasks = self._split_sources(source_tasks)
        train_sampler = EpisodeSampler(
            train_tasks, n_support=cfg.train.n_support, n_query=cfg.train.n_query, seed=cfg.train.seed
        )
        val_sampler = (
            EpisodeSampler(
                val_tasks,
                n_support=cfg.train.n_support,
                n_query=cfg.train.n_query,
                seed=cfg.train.seed + 1,
            )
            if val_tasks
            else None
        )

        # 5. Build learner and meta-train.
        learner = self._build_learner(feature_dim)
        trainer = MetaTrainer(learner, train_sampler, cfg.train, val_sampler=val_sampler)
        history = trainer.train()

        # 6. Low-data evaluation on the target.
        evaluator = LowDataEvaluator(
            learner=learner,
            target=target_task,
            input_dim=feature_dim,
            encoder_config=cfg.encoder,
            algorithm=cfg.algorithm,
            support_sizes=cfg.support_sizes,
            seeds=cfg.seeds,
            device=cfg.train.device,
        )
        results = evaluator.evaluate()
        summary = LowDataEvaluator.summarize(results)

        # 7. Persist outputs.
        if cfg.output_dir:
            self._save(cfg, selected, results, summary, history)

        return results

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
