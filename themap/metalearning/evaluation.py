"""Low-data evaluation: how much does meta-learning help the target dataset?

For a range of small support-set sizes ``N`` drawn (stratified) from the target,
we compare the meta-learned model adapted to those ``N`` samples against an MLP of
identical architecture trained from scratch on the same ``N`` samples, scoring both
by AUROC on the held-out remainder of the target. Repeating over seeds yields
mean ± 95% confidence intervals.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ._torch import require_torch
from .config import EncoderConfig
from .episodes import TaskFeatures
from .trainer import _safe_auroc, resolve_device

logger = get_logger(__name__)


def _ci95(values: np.ndarray) -> float:
    """Half-width of the 95% t-confidence interval for a 1-D sample."""
    from scipy import stats

    vals = values[np.isfinite(values)]
    if len(vals) < 2:
        return float("nan")
    sem = stats.sem(vals)
    return float(sem * stats.t.ppf(0.975, len(vals) - 1))


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

    def _split(self, n: int, seed: int) -> Optional[tuple]:
        """Stratified support/query split; None if infeasible."""
        from sklearn.model_selection import StratifiedShuffleSplit

        y = self.target.y
        if n >= len(y) or len(np.unique(y)) < 2:
            return None
        if min(int((y == 0).sum()), int((y == 1).sum())) < 2:
            return None
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
        sup_idx, qry_idx = next(splitter.split(self.target.X, y))
        return sup_idx, qry_idx

    def evaluate(self) -> pd.DataFrame:
        """Run the full support-size × seed sweep and return long-form results.

        Returns:
            DataFrame with columns ``[algorithm, support_size, seed, method, auroc]``
            where ``method`` is ``"meta"`` or ``"baseline"``.
        """
        torch = self.torch
        self.learner.to(self.device).eval()
        rows: List[dict] = []

        for n in self.support_sizes:
            for seed in range(self.seeds):
                split = self._split(n, seed)
                if split is None:
                    logger.warning("Skipping support_size=%d seed=%d (infeasible split).", n, seed)
                    continue
                sup_idx, qry_idx = split
                x_sup = torch.from_numpy(self.target.X[sup_idx]).float().to(self.device)
                y_sup = torch.from_numpy(self.target.y[sup_idx]).long().to(self.device)
                x_qry = torch.from_numpy(self.target.X[qry_idx]).float().to(self.device)
                y_qry = torch.from_numpy(self.target.y[qry_idx]).long().to(self.device)

                meta_probs = self.learner.adapt_and_predict(x_sup, y_sup, x_qry)
                meta_auroc = _safe_auroc(meta_probs, y_qry)

                base_probs = _train_baseline(
                    x_sup, y_sup, x_qry, self.input_dim, self.encoder_config, self.device, seed
                )
                base_auroc = _safe_auroc(torch.from_numpy(base_probs), y_qry)

                rows.append(
                    {
                        "algorithm": self.algorithm,
                        "support_size": n,
                        "seed": seed,
                        "method": "meta",
                        "auroc": meta_auroc,
                    }
                )
                rows.append(
                    {
                        "algorithm": self.algorithm,
                        "support_size": n,
                        "seed": seed,
                        "method": "baseline",
                        "auroc": base_auroc,
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def summarize(results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate long-form results to mean ± 95% CI per (method, support_size)."""
        records: List[dict] = []
        for (algo, method, n), group in results.groupby(["algorithm", "method", "support_size"]):
            aurocs = group["auroc"].to_numpy()
            records.append(
                {
                    "algorithm": algo,
                    "method": method,
                    "support_size": n,
                    "auroc_mean": float(np.nanmean(aurocs)),
                    "auroc_ci95": _ci95(aurocs),
                    "n_seeds": int(np.isfinite(aurocs).sum()),
                }
            )
        return (
            pd.DataFrame(records).sort_values(["algorithm", "method", "support_size"]).reset_index(drop=True)
        )
