"""Low-data evaluation: how much does meta-learning help the target dataset?

For a range of small support-set sizes ``N`` drawn (stratified) from the target,
we compare the meta-learned model adapted to those ``N`` samples against an MLP of
identical architecture trained from scratch on the same ``N`` samples, scoring both
on the held-out remainder of the target. We report AUROC, AUPRC (average precision),
and FS-Mol's headline metric ΔAUPRC (``average_precision − fraction_positive``).
Repeating over seeds yields mean ± 95% confidence intervals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ._torch import require_torch
from .config import EncoderConfig
from .episodes import TaskFeatures
from .trainer import resolve_device

logger = get_logger(__name__)

# Metrics reported per (support_size, seed, method); used to build the result schema.
METRICS = ("auroc", "avg_precision", "delta_auprc")


def _ci95(values: np.ndarray) -> float:
    """Half-width of the 95% t-confidence interval for a 1-D sample."""
    from scipy import stats

    vals = values[np.isfinite(values)]
    if len(vals) < 2:
        return float("nan")
    sem = stats.sem(vals)
    return float(sem * stats.t.ppf(0.975, len(vals) - 1))


def _safe_metrics(probs: Any, labels: Any) -> Dict[str, float]:
    """AUROC, AUPRC and ΔAUPRC for query predictions; NaN if the query is single-class.

    ΔAUPRC follows FS-Mol: ``average_precision − fraction_positive(query)``, i.e. the
    advantage over a constant predictor that always outputs the positive prevalence.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = labels.detach().cpu().numpy()
    p = probs.detach().cpu().numpy()
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "avg_precision": float("nan"), "delta_auprc": float("nan")}
    ap = float(average_precision_score(y, p))
    return {
        "auroc": float(roc_auc_score(y, p)),
        "avg_precision": ap,
        "delta_auprc": ap - float(y.mean()),
    }


def _train_baseline(
    x_sup: Any,
    y_sup: Any,
    x_qry: Any,
    input_dim: int,
    encoder_config: EncoderConfig,
    device: str,
    seed: int,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
) -> np.ndarray:
    """Train a fresh MLP on the support set and return query positive-probs."""
    torch = require_torch()
    from .models.encoder import MLPEncoder

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        MLPEncoder(input_dim, encoder_config),
        torch.nn.Linear(encoder_config.embed_dim, 2),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x_sup), y_sup)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(x_qry), dim=-1)[:, 1].cpu().numpy()


class LowDataEvaluator:
    """Evaluates a meta-learned model against a from-scratch baseline on a target."""

    def __init__(
        self,
        learner: Any,
        target: TaskFeatures,
        input_dim: int,
        encoder_config: EncoderConfig,
        algorithm: str,
        support_sizes: List[int],
        seeds: int = 5,
        device: str = "auto",
        query_fraction: float = 0.5,
    ):
        self.torch = require_torch()
        self.learner = learner
        self.target = target
        self.input_dim = input_dim
        self.encoder_config = encoder_config
        self.algorithm = algorithm
        self.support_sizes = support_sizes
        self.seeds = seeds
        self.device = resolve_device(device)
        self.query_fraction = query_fraction

    def _build_seed_pools(self, seed: int) -> Optional[dict]:
        """Carve a fixed query set, leaving ordered per-class support pools.

        For a given seed this is computed once and shared by every support size:
        a class-balanced query set of size ``round(len(y) * query_fraction)`` is
        held out, and the remaining indices form per-class pools shuffled into a
        fixed order. Sampling the first ``n // 2`` per class from those pools makes
        smaller support sets prefixes of larger ones (nested), while the query set
        stays identical across all ``N`` so AUROC is directly comparable.

        Returns ``None`` when the target cannot support a 2-class query plus at
        least a 1-shot-per-class support draw.
        """
        from sklearn.model_selection import train_test_split

        y = self.target.y
        if len(np.unique(y)) < 2 or min(int((y == 0).sum()), int((y == 1).sum())) < 2:
            return None

        idx = np.arange(len(y))
        q = int(round(len(y) * self.query_fraction))
        # Keep at least 1 per class on each side of the split.
        q = max(2, min(q, len(y) - 2))
        try:
            pool_idx, qry_idx = train_test_split(idx, test_size=q, stratify=y, random_state=seed)
        except ValueError:
            return None

        rng = np.random.default_rng(seed)
        pool_pos = pool_idx[y[pool_idx] == 1]
        pool_neg = pool_idx[y[pool_idx] == 0]
        rng.shuffle(pool_pos)
        rng.shuffle(pool_neg)
        if len(pool_pos) < 1 or len(pool_neg) < 1:
            return None

        max_support = 2 * min(len(pool_pos), len(pool_neg))
        return {
            "qry_idx": qry_idx,
            "pos_pool": pool_pos,
            "neg_pool": pool_neg,
            "max_support": max_support,
        }

    @staticmethod
    def _support_for_n(pools: dict, n: int) -> Optional[np.ndarray]:
        """First ``n // 2`` indices per class from the fixed-order pools.

        Returns ``None`` when ``n`` exceeds what the support pools can supply.
        """
        per_class = n // 2
        if per_class < 1 or n > pools["max_support"]:
            return None
        return np.concatenate([pools["pos_pool"][:per_class], pools["neg_pool"][:per_class]])

    def evaluate(self) -> pd.DataFrame:
        """Run the full support-size × seed sweep and return long-form results.

        Returns:
            DataFrame with columns ``[algorithm, support_size, seed, method, auroc]``
            where ``method`` is ``"meta"`` or ``"baseline"``.
        """
        torch = self.torch
        self.learner.to(self.device).eval()
        rows: List[dict] = []

        for seed in range(self.seeds):
            pools = self._build_seed_pools(seed)
            if pools is None:
                logger.warning("Skipping seed=%d (target too small for a stratified split).", seed)
                continue
            qry_idx = pools["qry_idx"]
            for n in self.support_sizes:
                sup_idx = self._support_for_n(pools, n)
                if sup_idx is None:
                    logger.warning(
                        "Skipping support_size=%d seed=%d (only %d support example(s) available).",
                        n,
                        seed,
                        pools["max_support"],
                    )
                    continue
                x_sup = torch.from_numpy(self.target.X[sup_idx]).float().to(self.device)
                y_sup = torch.from_numpy(self.target.y[sup_idx]).long().to(self.device)
                x_qry = torch.from_numpy(self.target.X[qry_idx]).float().to(self.device)
                y_qry = torch.from_numpy(self.target.y[qry_idx]).long().to(self.device)

                meta_probs = self.learner.adapt_and_predict(x_sup, y_sup, x_qry)
                meta_metrics = _safe_metrics(meta_probs, y_qry)

                base_probs = _train_baseline(
                    x_sup, y_sup, x_qry, self.input_dim, self.encoder_config, self.device, seed
                )
                base_metrics = _safe_metrics(torch.from_numpy(base_probs), y_qry)

                for method, metrics in (("meta", meta_metrics), ("baseline", base_metrics)):
                    rows.append(
                        {
                            "algorithm": self.algorithm,
                            "support_size": n,
                            "seed": seed,
                            "method": method,
                            **metrics,
                        }
                    )

        return pd.DataFrame(rows, columns=["algorithm", "support_size", "seed", "method", *METRICS])

    @staticmethod
    def summarize(results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate long-form results to mean ± 95% CI per (method, support_size).

        Each metric in :data:`METRICS` (``auroc``, ``avg_precision``, ``delta_auprc``)
        gets ``<metric>_mean`` and ``<metric>_ci95`` columns; ``n_seeds`` counts the
        finite AUROC repeats.
        """
        records: List[dict] = []
        for (algo, method, n), group in results.groupby(["algorithm", "method", "support_size"]):
            record: dict = {"algorithm": algo, "method": method, "support_size": n}
            for metric in METRICS:
                vals = group[metric].to_numpy()
                record[f"{metric}_mean"] = float(np.nanmean(vals))
                record[f"{metric}_ci95"] = _ci95(vals)
            record["n_seeds"] = int(np.isfinite(group["auroc"].to_numpy()).sum())
            records.append(record)
        return (
            pd.DataFrame(records).sort_values(["algorithm", "method", "support_size"]).reset_index(drop=True)
        )
