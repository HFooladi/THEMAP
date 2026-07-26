"""Turn benchmark results into a side-by-side comparison with FS-Mol's published numbers.

Three things here are easy to get subtly wrong, so they are called out explicitly:

**Aggregation order.** FS-Mol's ``±0.008`` on ProtoNet is the spread *across the 157 test
tasks*, not across its ten within-task repeats. Averaging seeds and tasks together in one
pass produces error bars several times too tight. :func:`aggregate_runs` therefore averages
seeds within a task first, then takes mean ± standard error over tasks.

**Which reference to compare against.** Recomputing ΔAUPRC from FS-Mol's per-task CSVs
lands uniformly ``+0.005`` above the numbers printed in the paper, for every method — a
detail of how the paper averaged the prevalence term. Comparisons here are
recomputed-against-recomputed so the offset cancels.

**Rank correlation, not just means.** THEMAP's encoder (an MLP over ECFP) is weaker than
FS-Mol's (GNN+ECFP+FC with Mahalanobis distance), so a lower mean is expected and says
little on its own. Whether the two agree *per task* is the signal that separates "correct
but weaker" from "broken": a broken implementation has no reason to find the same tasks
easy.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .fsmol_reference import PROTONET_METHOD, aggregate_reference, load_reference_table

logger = get_logger(__name__)

#: Offset between ΔAUPRC recomputed from FS-Mol's per-task CSVs and the paper's Table 2.
PAPER_OFFSET = 0.005

#: The FS-Mol baseline each THEMAP algorithm is the counterpart of.
COUNTERPARTS = {"proto": PROTONET_METHOD, "maml": "GNN-MAML"}

_THEMAP_PREFIX = "themap"


@dataclass
class BenchmarkReport:
    """Everything the benchmark emits for human consumption."""

    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    markdown: str = ""
    figure_bytes: Optional[bytes] = None
    verdict: Dict[str, Any] = field(default_factory=dict)


def _label(algorithm: str, method: str) -> str:
    """Row label for a THEMAP arm; the ``baseline`` arm is not an FS-Mol baseline."""
    if method == "baseline":
        return f"{_THEMAP_PREFIX}-baseline-mlp"
    return f"{_THEMAP_PREFIX}-{algorithm}"


def per_task_scores(results: pd.DataFrame, metric: str = "delta_auprc") -> pd.DataFrame:
    """Collapse seeds within each (arm, task, support size).

    Returns:
        DataFrame ``[arm, task_id, support_size, <metric>, n_seeds]``.
    """
    if results.empty:
        return pd.DataFrame(columns=["arm", "task_id", "support_size", metric, "n_seeds"])
    frame = results.copy()
    frame["arm"] = [_label(a, m) for a, m in zip(frame["algorithm"], frame["method"])]
    grouped = frame.groupby(["arm", "task_id", "support_size"])[metric]
    out = grouped.mean().reset_index()
    out["n_seeds"] = grouped.count().reset_index()[metric].to_numpy()
    return out


def aggregate_runs(results: pd.DataFrame, metric: str = "delta_auprc") -> pd.DataFrame:
    """Mean ± standard error over *tasks*, after averaging seeds within each task.

    Returns:
        DataFrame ``[arm, support_size, mean, sem, n_tasks]``.
    """
    per_task = per_task_scores(results, metric=metric)
    if per_task.empty:
        return pd.DataFrame(columns=["arm", "support_size", "mean", "sem", "n_tasks"])

    records: List[dict] = []
    for (arm, size), group in per_task.groupby(["arm", "support_size"]):
        values = group[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        records.append(
            {
                "arm": arm,
                "support_size": int(size),
                "mean": float(values.mean()) if n else float("nan"),
                "sem": float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                "n_tasks": n,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["arm", "support_size"]).reset_index(drop=True)


def comparison_table(
    results: pd.DataFrame,
    reference: Optional[pd.DataFrame],
    task_ids: Sequence[str],
) -> pd.DataFrame:
    """THEMAP arms and FS-Mol baselines side by side, on the same tasks."""
    ours = aggregate_runs(results)
    if reference is None:
        return ours

    theirs = aggregate_reference(reference, task_ids=task_ids).rename(
        columns={"method": "arm", "delta_auprc_mean": "mean", "delta_auprc_sem": "sem"}
    )
    theirs["arm"] = "fsmol-" + theirs["arm"].astype(str)
    combined = pd.concat([ours, theirs[["arm", "support_size", "mean", "sem", "n_tasks"]]], ignore_index=True)
    return combined.sort_values(["support_size", "mean"], ascending=[True, False]).reset_index(drop=True)


def correlations(
    results: pd.DataFrame,
    reference: pd.DataFrame,
    algorithm: str = "proto",
    reference_method: str = PROTONET_METHOD,
) -> pd.DataFrame:
    """Per-task agreement between a THEMAP arm and an FS-Mol baseline, per support size.

    A high rank correlation with a negative offset is the signature of a correct
    implementation behind a weaker encoder; a correlation near zero is not.

    Returns:
        DataFrame ``[support_size, spearman, spearman_p, pearson, pearson_p, slope,
        intercept, mean_offset, n_tasks]``.
    """
    from scipy import stats

    ours = per_task_scores(results)
    ours = ours[ours["arm"] == f"{_THEMAP_PREFIX}-{algorithm}"]
    theirs = reference[reference["method"] == reference_method]

    records: List[dict] = []
    for size in sorted(ours["support_size"].unique()):
        left = ours[ours["support_size"] == size][["task_id", "delta_auprc"]]
        right = theirs[theirs["support_size"] == size][["task_id", "delta_auprc"]]
        merged = left.merge(right, on="task_id", suffixes=("_themap", "_fsmol")).dropna()
        n = len(merged)
        record: dict = {"support_size": int(size), "n_tasks": n}
        if n >= 3:
            x = merged["delta_auprc_fsmol"].to_numpy(dtype=float)
            y = merged["delta_auprc_themap"].to_numpy(dtype=float)
            rho, rho_p = stats.spearmanr(x, y)
            r, r_p = stats.pearsonr(x, y)
            fit = stats.linregress(x, y)
            record.update(
                spearman=float(rho),
                spearman_p=float(rho_p),
                pearson=float(r),
                pearson_p=float(r_p),
                slope=float(fit.slope),
                intercept=float(fit.intercept),
                mean_offset=float(np.mean(y - x)),
            )
        else:
            record.update(
                spearman=float("nan"),
                spearman_p=float("nan"),
                pearson=float("nan"),
                pearson_p=float("nan"),
                slope=float("nan"),
                intercept=float("nan"),
                mean_offset=float("nan"),
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def correlation_context(
    reference: pd.DataFrame,
    task_ids: Sequence[str],
    reference_method: str = PROTONET_METHOD,
    support_size: int = 16,
) -> pd.DataFrame:
    """How well FS-Mol's *own* baselines agree with the reference method, on these tasks.

    Without this control a high correlation is uninterpretable: tasks differ in intrinsic
    difficulty, so any two competent models agree to some extent. Comparing THEMAP's
    correlation against the spread among FS-Mol's own baselines says whether the agreement
    is specific to the implementation or just the tasks.

    Returns:
        DataFrame ``[method, spearman, n_tasks]``, excluding the reference method itself.
    """
    from scipy import stats

    frame = reference[
        (reference["support_size"] == support_size) & (reference["task_id"].isin(list(task_ids)))
    ]
    anchor = frame[frame["method"] == reference_method].set_index("task_id")["delta_auprc"]

    records: List[dict] = []
    for method, group in frame.groupby("method"):
        if method == reference_method:
            continue
        other = group.set_index("task_id")["delta_auprc"]
        joined = pd.concat([anchor, other], axis=1).dropna()
        if len(joined) >= 3:
            rho, _ = stats.spearmanr(joined.iloc[:, 0], joined.iloc[:, 1])
        else:
            rho = float("nan")
        records.append({"method": f"fsmol-{method}", "spearman": float(rho), "n_tasks": len(joined)})
    return pd.DataFrame.from_records(records).sort_values("spearman", ascending=False).reset_index(drop=True)


def evaluate_criteria(
    comparison: pd.DataFrame,
    corr: pd.DataFrame,
    algorithm: str = "proto",
    headline_size: int = 16,
) -> pd.DataFrame:
    """Score the run against the acceptance criteria.

    The criteria are designed to separate "sound implementation, weaker encoder" from
    "broken". The load-bearing one is the comparison against FS-Mol's random forest: RF
    consumes the same ECFP fingerprints THEMAP does, so the encoder gap cannot explain a
    loss to it.
    """

    def arm_mean(arm: str, size: int) -> float:
        row = comparison[(comparison["arm"] == arm) & (comparison["support_size"] == size)]
        return float(row["mean"].iloc[0]) if len(row) else float("nan")

    meta = f"{_THEMAP_PREFIX}-{algorithm}"
    counterpart = COUNTERPARTS.get(algorithm, PROTONET_METHOD)
    ours = arm_mean(meta, headline_size)
    baseline = arm_mean(f"{_THEMAP_PREFIX}-baseline-mlp", headline_size)
    rf = arm_mean("fsmol-RF", headline_size)
    pn = arm_mean(f"fsmol-{counterpart}", headline_size)
    rho_row = corr[corr["support_size"] == headline_size]
    rho = float(rho_row["spearman"].iloc[0]) if len(rho_row) else float("nan")

    sizes = sorted(s for s in comparison["support_size"].unique() if s <= 128)
    curve = [arm_mean(meta, s) for s in sizes]
    monotone = all((np.isnan(b) or np.isnan(a) or b >= a - 0.01) for a, b in zip(curve, curve[1:]))

    checks = [
        ("A1 meta beats a trivial predictor", "delta_auprc > 0.03", ours, ours > 0.03),
        ("A2 meta beats its own from-scratch MLP", "> themap-baseline-mlp", ours - baseline, ours > baseline),
        ("A3 meta beats FS-Mol RF on the same features", f"> {rf:.3f}", ours, ours > rf),
        (f"A4 within reach of FS-Mol {counterpart}", f">= 0.60 x {pn:.3f}", ours, ours >= 0.60 * pn),
        (
            "A5 improves with support size",
            "non-decreasing to N=128",
            curve[-1] if curve else np.nan,
            monotone,
        ),
        (f"A6 agrees per task with FS-Mol {counterpart}", "spearman >= 0.5", rho, rho >= 0.5),
    ]
    return pd.DataFrame(
        [
            {"criterion": name, "threshold": thr, "observed": obs, "passed": bool(ok)}
            for name, thr, obs, ok in checks
        ]
    )


def _figure(comparison: pd.DataFrame, corr_scatter: Optional[pd.DataFrame]) -> Optional[bytes]:
    """Two-panel PNG: curves vs support size, and the per-task scatter at N=16."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - plotting must never break a run
        logger.warning("Skipping plot (matplotlib unavailable: %s).", exc)
        return None

    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        for arm, group in comparison.groupby("arm"):
            group = group.sort_values("support_size")
            style = "-" if str(arm).startswith(_THEMAP_PREFIX) else "--"
            width = 2.2 if str(arm).startswith(_THEMAP_PREFIX) else 1.2
            ax.errorbar(
                group["support_size"],
                group["mean"],
                yerr=group["sem"],
                label=str(arm),
                linestyle=style,
                linewidth=width,
                marker="o",
                markersize=4,
                capsize=2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("support set size")
        ax.set_ylabel(r"mean $\Delta$AUPRC")
        ax.set_title("THEMAP (solid) vs FS-Mol published (dashed)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        ax = axes[1]
        if corr_scatter is not None and len(corr_scatter) >= 3:
            x = corr_scatter["delta_auprc_fsmol"].to_numpy()
            y = corr_scatter["delta_auprc_themap"].to_numpy()
            ax.scatter(x, y, alpha=0.75)
            lo = float(min(x.min(), y.min())) - 0.02
            hi = float(max(x.max(), y.max())) + 0.02
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(r"FS-Mol ProtoNet $\Delta$AUPRC")
            ax.set_ylabel(r"THEMAP $\Delta$AUPRC")
            ax.set_title("per-task agreement at N=16")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        else:
            ax.axis("off")

        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, dpi=150, format="png")
        plt.close(fig)
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping plot (rendering failed: %s).", exc)
        return None


def _md_table(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown`` so the report does not pull in
    ``tabulate``, which is not among THEMAP's dependencies.
    """
    if frame.empty:
        return "_(no rows)_"
    work = frame.reset_index() if index else frame

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        if isinstance(value, (bool, np.bool_)):
            return "PASS" if value else "**FAIL**"
        return str(value)

    headers = [str(c) for c in work.columns]
    rows = [[cell(v) for v in record] for record in work.itertuples(index=False)]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    out = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    out += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join(out)


def _markdown(
    comparison: pd.DataFrame,
    corr: pd.DataFrame,
    criteria: pd.DataFrame,
    representativeness: Optional[pd.DataFrame],
    n_tasks: int,
    context: Optional[pd.DataFrame] = None,
) -> str:
    lines = [
        "# FS-Mol parity check",
        "",
        f"THEMAP's meta-learners evaluated on {n_tasks} FS-Mol test task(s) under FS-Mol's",
        "protocol (stratified support of size N, query = the remainder, ΔAUPRC = AUPRC minus",
        "query positive rate). FS-Mol rows are its published per-task results restricted to the",
        "same tasks, so the two sides are directly comparable.",
        "",
        f"FS-Mol numbers are recomputed from the released per-task CSVs and sit ~{PAPER_OFFSET:+.3f}",
        "above the paper's printed Table 2 for every method; comparing recomputed to recomputed",
        "cancels that offset.",
        "",
        "## Mean ΔAUPRC (± standard error across tasks)",
        "",
        _md_table(comparison.pivot(index="arm", columns="support_size", values="mean").round(3), index=True),
        "",
        "## Per-task agreement with FS-Mol ProtoNet",
        "",
        _md_table(corr.round(3)),
        "",
        "## Acceptance criteria",
        "",
        _md_table(criteria.round(3)),
        "",
    ]
    if context is not None and not context.empty:
        lines += [
            "## Is that correlation just task difficulty?",
            "",
            "Tasks differ in intrinsic difficulty, so any two competent models agree somewhat.",
            "For scale, here is how FS-Mol's own baselines correlate with FS-Mol ProtoNet across",
            "the same tasks at N=16. THEMAP's arm should be read against this spread, not against 1.0.",
            "",
            _md_table(context.round(3)),
            "",
        ]
    if representativeness is not None:
        lines += [
            "## Subset representativeness",
            "",
            "Reference means on the evaluated subset vs the full 157-task benchmark.",
            "",
            _md_table(
                representativeness.pivot(index="method", columns="support_size", values="abs_delta").round(3),
                index=True,
            ),
            "",
        ]
    passed = int(criteria["passed"].sum())
    lines += [
        f"**Verdict: {passed}/{len(criteria)} acceptance criteria passed.**",
        "",
    ]
    return "\n".join(lines)


def build_report(results: pd.DataFrame, config: Any) -> BenchmarkReport:
    """Assemble every comparison artifact from a completed benchmark run."""
    report = BenchmarkReport()
    if results.empty:
        report.markdown = "# FS-Mol parity check\n\nNo results were produced.\n"
        return report

    task_ids = sorted(results["task_id"].unique())
    reference: Optional[pd.DataFrame] = None
    try:
        reference = load_reference_table(config.reference_dir, offline=config.offline)
        known = set(reference["task_id"])
        if not set(task_ids) & known:
            logger.warning("No evaluated task appears in the FS-Mol reference; skipping comparison.")
            reference = None
    except Exception as exc:  # noqa: BLE001 - a missing reference must not lose the results
        logger.warning("FS-Mol reference unavailable (%s); reporting THEMAP results only.", exc)

    comparison = comparison_table(results, reference, task_ids)
    report.tables["comparison_summary"] = comparison
    report.tables["per_task_summary"] = per_task_scores(results)

    algorithms = sorted(set(results["algorithm"].astype(str)))
    # ProtoNet is the headline arm when present: it is FS-Mol's strongest and
    # best-documented baseline.
    algorithm = "proto" if "proto" in algorithms else algorithms[0]

    if reference is not None:
        frames = []
        for algo in algorithms:
            frame = correlations(
                results, reference, algorithm=algo, reference_method=COUNTERPARTS.get(algo, PROTONET_METHOD)
            )
            frame.insert(0, "algorithm", algo)
            frame.insert(1, "vs", COUNTERPARTS.get(algo, PROTONET_METHOD))
            frames.append(frame)
        corr_all = pd.concat(frames, ignore_index=True)
        report.tables["correlations"] = corr_all
        corr = corr_all[corr_all["algorithm"] == algorithm].drop(columns=["algorithm", "vs"])
        report.tables["reference_per_task"] = reference[reference["task_id"].isin(task_ids)]

        from .subset import subset_representativeness

        representativeness = subset_representativeness(reference, task_ids)
        report.tables["representativeness"] = representativeness

        context = correlation_context(
            reference, task_ids, reference_method=COUNTERPARTS.get(algorithm, PROTONET_METHOD)
        )
        report.tables["correlation_context"] = context

        ours = per_task_scores(results)
        ours = ours[(ours["arm"] == f"{_THEMAP_PREFIX}-{algorithm}") & (ours["support_size"] == 16)]
        theirs = reference[(reference["method"] == PROTONET_METHOD) & (reference["support_size"] == 16)][
            ["task_id", "delta_auprc"]
        ]
        scatter = (
            ours[["task_id", "delta_auprc"]]
            .merge(theirs, on="task_id", suffixes=("_themap", "_fsmol"))
            .dropna()
        )
    else:
        corr = pd.DataFrame(columns=["support_size", "spearman"])
        representativeness = None
        context = None
        scatter = None

    criteria = evaluate_criteria(comparison, corr, algorithm=algorithm)
    report.tables["acceptance_criteria"] = criteria
    report.verdict = {
        "passed": int(criteria["passed"].sum()),
        "total": int(len(criteria)),
        "failed": criteria[~criteria["passed"]]["criterion"].tolist(),
    }
    report.markdown = _markdown(comparison, corr, criteria, representativeness, len(task_ids), context)
    report.figure_bytes = _figure(comparison, scatter)
    return report
