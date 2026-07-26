"""Distance-vs-random source-selection comparison — THEMAP's headline hypothesis.

For a target task, THEMAP's central claim is that selecting the ``k`` source datasets
that are *nearest* by dataset distance (especially OTDD) and meta-training on them
beats selecting ``k`` sources *at random*. This module runs both arms end to end and
reports the head-to-head, reusing :class:`~themap.metalearning.runner.MetaLearnExperiment`
unchanged for each arm (only the source-selection step differs).

The distance arm runs once; the random arm runs over several seeds so its performance
carries a confidence interval (any single random draw is noisy). If distance-based
selection consistently sits above the random band, the hypothesis is supported.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .config import Algorithm, EncoderConfig, ExperimentConfig, MAMLConfig, ProtoConfig, TrainConfig
from .evaluation import _ci95
from .runner import MetaLearnExperiment

logger = get_logger(__name__)

# Metrics compared between the two arms (headline metric first).
_COMPARE_METRICS = ("auroc", "delta_auprc")


@dataclass
class CompareConfig:
    """Configuration for :class:`SelectionComparison`.

    Carries the same knobs as :class:`~themap.metalearning.config.ExperimentConfig`
    plus the comparison-specific ones: how many random draws to average, whether to
    run the self-contained ``demo`` scenario, and which distance to auto-compute when
    no ``distance_file`` is supplied.

    Attributes:
        data_dir: Directory with ``train``/``valid``/``test`` folds.
        target_id: Target task id (required unless ``demo`` is True).
        distance_file: Saved distance file; auto-computed if None (and not ``demo``).
        k: Number of source datasets each arm selects.
        random_seeds: Number of random-selection draws to average over.
        demo: If True, build a controlled close-vs-distant scenario so the effect is
            reliably visible on the bundled datasets (see :meth:`_build_demo`).
        distance_method: Distance to auto-compute for the real path (``"otdd"`` is the
            paper's headline distance; needs the ``[otdd]`` extra). The demo always
            uses a fast Euclidean distance internally.
        output_dir: Where comparison outputs (and any auto-computed distances) are written.
    """

    data_dir: str
    target_id: Optional[str] = None
    distance_file: Optional[str] = None
    k: int = 3
    algorithm: Algorithm = "proto"
    featurizer: str = "ecfp"
    support_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    train_shot_mode: Literal["match", "fixed"] = "match"
    query_fraction: float = 0.5
    seeds: int = 3
    random_seeds: int = 5
    demo: bool = False
    distance_method: str = "otdd"
    n_jobs: int = 8
    source_fold: str = "train"
    target_fold: str = "test"
    output_dir: str = "metalearn_compare_out"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    proto: ProtoConfig = field(default_factory=ProtoConfig)
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


class SelectionComparison:
    """Runs the distance arm and the random arm, then reports the head-to-head."""

    # Bundled assay sharded into the controlled demo target + close source shards.
    _DEMO_BASE_ASSAY = "CHEMBL1613776"
    # Unrelated bundled datasets mixed in as distant decoys.
    _DEMO_DECOYS = ("CHEMBL1023359", "CHEMBL2218944", "CHEMBL2219012", "CHEMBL1614274", "CHEMBL3705844")

    def __init__(self, config: CompareConfig):
        self.config = config

    def run(self) -> pd.DataFrame:
        """Run both arms and return the combined long-form results DataFrame.

        The returned frame is the standard meta-learning results schema with two extra
        columns: ``selection`` (``"distance"``/``"random"``) and ``selection_seed``
        (``-1`` for the single distance run, ``0..random_seeds-1`` for the random ones).
        """
        cfg = self.config
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        data_dir, distance_file, target_id, target_fold = self._prepare_inputs(out)

        frames: List[pd.DataFrame] = []
        logger.info("Distance arm: k=%d nearest sources for target '%s'.", cfg.k, target_id)
        dist_res = self._run_arm(data_dir, distance_file, target_id, target_fold, "distance", 0)
        dist_res["selection"] = "distance"
        dist_res["selection_seed"] = -1
        frames.append(dist_res)

        for s in range(cfg.random_seeds):
            logger.info("Random arm draw %d/%d for target '%s'.", s + 1, cfg.random_seeds, target_id)
            rand_res = self._run_arm(data_dir, distance_file, target_id, target_fold, "random", s)
            rand_res["selection"] = "random"
            rand_res["selection_seed"] = s
            frames.append(rand_res)

        results = pd.concat(frames, ignore_index=True)
        summary = self.summarize(results)
        self._save(out, results, summary, target_id, str(distance_file))
        return results

    def _prepare_inputs(self, out: Path) -> Tuple[str, str, str, str]:
        """Resolve (data_dir, distance_file, target_id, target_fold) for both arms."""
        cfg = self.config
        if cfg.demo:
            return self._build_demo(out)
        if not cfg.target_id:
            raise ValueError("target_id is required unless demo=True.")
        if cfg.distance_file:
            if not Path(cfg.distance_file).exists():
                raise FileNotFoundError(f"Distance file not found: {cfg.distance_file}")
            return cfg.data_dir, cfg.distance_file, cfg.target_id, cfg.target_fold
        distance_file = self._compute_distances(cfg.data_dir, out / "distances", cfg.distance_method)
        return cfg.data_dir, distance_file, cfg.target_id, cfg.target_fold

    def _compute_distances(self, data_dir: str | Path, out_dir: Path, method: str) -> str:
        """Compute a molecule distance file via ``quick_distance``.

        Falls back to Euclidean only for the ``demo`` scenario (a demonstration should
        never fail to produce a picture); real runs surface the error so the requested
        science is not silently changed.
        """
        from ..pipeline import quick_distance

        cfg = self.config
        logger.info("Computing '%s' distances over %s ...", method, data_dir)
        try:
            quick_distance(
                str(data_dir),
                output_dir=str(out_dir),
                molecule_featurizer=cfg.featurizer,
                molecule_method=method,
                n_jobs=cfg.n_jobs,
                device=cfg.train.device,
            )
        except Exception as exc:  # noqa: BLE001 - re-raise or fall back with context
            if cfg.demo and method != "euclidean":
                logger.warning("Demo '%s' distance failed (%s); falling back to euclidean.", method, exc)
                return self._compute_distances(data_dir, out_dir, "euclidean")
            raise RuntimeError(
                f"Could not compute '{method}' distances ({exc}). Install the OTDD extra "
                "with `pip install 'themap[otdd]'`, or pass distance_method='euclidean' "
                "(or supply a precomputed distance_file)."
            ) from exc

        path = Path(out_dir) / "molecule_distances.csv"
        if not path.exists():
            raise RuntimeError(f"Distance computation did not produce {path}.")
        return str(path)

    def _run_arm(
        self,
        data_dir: str,
        distance_file: str,
        target_id: str,
        target_fold: str,
        strategy: str,
        seed: int,
    ) -> pd.DataFrame:
        """Run one MetaLearnExperiment arm and return its long-form results."""
        cfg = self.config
        exp = ExperimentConfig(
            data_dir=str(data_dir),
            distance_file=str(distance_file),
            target_id=target_id,
            k=cfg.k,
            algorithm=cfg.algorithm,
            selection_strategy=strategy,  # type: ignore[arg-type]
            selection_seed=seed,
            featurizer=cfg.featurizer,
            support_sizes=cfg.support_sizes,
            train_shot_mode=cfg.train_shot_mode,
            query_fraction=cfg.query_fraction,
            seeds=cfg.seeds,
            n_jobs=cfg.n_jobs,
            output_dir=None,
            source_fold=cfg.source_fold,
            target_fold=target_fold,
            encoder=cfg.encoder,
            proto=cfg.proto,
            maml=cfg.maml,
            train=cfg.train,
        )
        return MetaLearnExperiment(exp).run()

    @staticmethod
    def summarize(results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the head-to-head per support size.

        For each metric in :data:`_COMPARE_METRICS` and each support size, reports the
        meta-learned score for the distance arm, the mean ± 95% CI across random draws,
        and ``gap = distance − random_mean`` (positive ⇒ distance selection wins).
        """
        meta = results[results["method"] == "meta"]
        rows: List[dict] = []
        for n, group in meta.groupby("support_size"):
            row: dict = {"support_size": int(n)}
            dist = group[group["selection"] == "distance"]
            rand = group[group["selection"] == "random"]
            for metric in _COMPARE_METRICS:
                d_val = float(np.nanmean(dist[metric].to_numpy())) if len(dist) else float("nan")
                if len(rand):
                    per_draw = rand.groupby("selection_seed")[metric].mean().to_numpy()
                else:
                    per_draw = np.array([])
                r_mean = float(np.nanmean(per_draw)) if len(per_draw) else float("nan")
                row[f"distance_{metric}"] = d_val
                row[f"random_{metric}_mean"] = r_mean
                row[f"random_{metric}_ci95"] = _ci95(per_draw)
                row[f"gap_{metric}"] = d_val - r_mean
            rows.append(row)
        return pd.DataFrame(rows).sort_values("support_size").reset_index(drop=True)

    @staticmethod
    def verdict(summary: pd.DataFrame) -> float:
        """Mean AUROC advantage of distance-based over random selection across sizes."""
        if not len(summary):
            return float("nan")
        return float(np.nanmean(summary["gap_auroc"].to_numpy()))

    def _build_demo(self, out: Path) -> Tuple[str, str, str, str]:
        """Materialize a controlled close-vs-distant scenario under ``out/demo``.

        Shards one balanced bundled assay via ``StratifiedKFold`` into a held-out target
        plus same-distribution ("close") source shards, then mixes in unrelated bundled
        datasets as distant decoys and computes a real (Euclidean) distance file. By
        construction the close shards rank nearest, so the distance arm selects them and
        the random arm mostly picks decoys — making the hypothesis visible with one command.
        """
        from sklearn.model_selection import StratifiedKFold

        from ..data.loader import DatasetLoader

        cfg = self.config
        demo = out / "demo"
        demo_data = demo / "datasets"
        for sub in ("train", "test"):
            (demo_data / sub).mkdir(parents=True, exist_ok=True)

        base_path = Path(cfg.data_dir) / cfg.source_fold / f"{self._DEMO_BASE_ASSAY}.jsonl.gz"
        if not base_path.exists():
            raise FileNotFoundError(
                f"Demo base assay '{self._DEMO_BASE_ASSAY}' not found at {base_path}. The --demo "
                f"scenario is built from the bundled datasets; run it against the repository's "
                f"`datasets/` directory, or use a real --target-id instead of --demo."
            )
        base = DatasetLoader(cfg.data_dir).load_dataset(cfg.source_fold, self._DEMO_BASE_ASSAY)
        smiles = np.array(base.smiles_list)
        labels = np.array(base.labels)
        folds = [
            te for _, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(smiles, labels)
        ]

        self._write_jsonl(
            demo_data / "test" / "CLOSE_TARGET.jsonl.gz", smiles[folds[0]], labels[folds[0]], "CLOSE_TARGET"
        )
        for i in range(1, 4):
            self._write_jsonl(
                demo_data / "train" / f"CLOSE_S{i}.jsonl.gz",
                smiles[folds[i]],
                labels[folds[i]],
                f"CLOSE_S{i}",
            )
        for tid in self._DEMO_DECOYS:
            src = Path(cfg.data_dir) / cfg.source_fold / f"{tid}.jsonl.gz"
            if src.exists():
                shutil.copy(src, demo_data / "train" / f"{tid}.jsonl.gz")
            else:
                logger.warning("Demo decoy %s not found in %s; skipping.", tid, cfg.data_dir)

        distance_file = self._compute_distances(demo_data, demo / "output", "euclidean")
        return str(demo_data), distance_file, "CLOSE_TARGET", "test"

    @staticmethod
    def _write_jsonl(path: Path, smiles, labels, assay: str) -> None:
        import gzip

        with gzip.open(path, "wt") as fh:
            for s, y in zip(smiles, labels):
                fh.write(json.dumps({"SMILES": str(s), "Property": f"{float(y)}", "Assay_ID": assay}) + "\n")

    def _save(
        self, out: Path, results: pd.DataFrame, summary: pd.DataFrame, target_id: str, distance_file: str
    ) -> None:
        results.to_csv(out / "comparison.csv", index=False)
        summary.to_csv(out / "comparison_summary.csv", index=False)
        with open(out / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        with open(out / "selected_sources.json", "w") as f:
            json.dump(
                {"target_id": target_id, "distance_file": distance_file, "k": self.config.k}, f, indent=2
            )
        logger.info("Saved distance-vs-random comparison to %s", out)
