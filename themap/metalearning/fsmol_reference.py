"""Published FS-Mol baseline results, as a table THEMAP runs can be compared against.

The original FS-Mol release ships per-task evaluation summaries for its seven baselines
under ``baselines/*_summary.csv`` in the microsoft/FS-Mol repository. Each file covers
exactly the 157 FS-Mol test tasks and records, per support-set size, the mean AUPRC over
FS-Mol's ten stratified repeats.

Two details matter when reading them:

* The cells hold **AUPRC**, not FS-Mol's headline ΔAUPRC. The paper defines
  ``ΔAUPRC = AUPRC − |actives| / |query|``, so the per-task ``fraction_positive_test``
  column has to be subtracted. :func:`load_reference_table` does that.
* A cell is **empty** when the task was too small for that support size (FS-Mol skips the
  evaluation point rather than shrinking it). Those become ``NaN`` and must stay ``NaN`` —
  reading them as ``0.0`` would drag the mean at large support sizes toward zero.

Recomputing the means this way lands uniformly ``+0.005`` above the numbers printed in
Table 2 of the paper, for every method. The offset is a detail of how the paper averaged
the prevalence term, so comparisons should be recomputed-against-recomputed rather than
against the printed table.

This module is deliberately torch-free so the CLI can build the comparison without
importing torch.
"""

from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

_RAW_BASE = "https://raw.githubusercontent.com/microsoft/FS-Mol/main/baselines"

#: Display name -> file name under ``baselines/`` in the FS-Mol repository.
FSMOL_BASELINE_FILES: Dict[str, str] = {
    "PN": "ProtoNet-gnn+ecfp+fc-Support64_summary.csv",
    "GNN-MAML": "MAML-Support16_summary.csv",
    "GNN-MT": "GNN-Multitask_summary.csv",
    "RF": "random_forest_summary.csv",
    "kNN": "kNN_summary.csv",
    "MAT": "MAT_summary.csv",
    "GNN-ST": "GNN-ST_summary.csv",
}

#: The method whose per-task scores THEMAP's ProtoNet is correlated against.
PROTONET_METHOD = "PN"

#: Support-set sizes present as ``<N>_train`` columns in every summary file.
REFERENCE_SUPPORT_SIZES: Tuple[int, ...] = (16, 32, 64, 128, 256)

_CELL_RE = re.compile(r"^\s*(-?[\d.eE+-]+)\s*(?:\+/-\s*(-?[\d.eE+-]+))?\s*$")


def _parse_mean_std(cell: object) -> Tuple[float, float]:
    """Parse a ``"0.747+/-0.049"`` summary cell into ``(mean, std)``.

    Empty, missing or unparseable cells yield ``(nan, nan)`` so that support sizes FS-Mol
    skipped propagate as missing data rather than as zeros.

    Args:
        cell: Raw cell value from the summary CSV.

    Returns:
        Tuple of ``(mean, std)``; ``std`` is ``nan`` when the cell carries no ``+/-`` part.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return float("nan"), float("nan")
    text = str(cell).strip()
    if not text or text.lower() in {"nan", "none"}:
        return float("nan"), float("nan")
    match = _CELL_RE.match(text)
    if match is None:
        logger.warning("Unparseable FS-Mol summary cell %r; treating as missing.", text)
        return float("nan"), float("nan")
    mean = float(match.group(1))
    std = float(match.group(2)) if match.group(2) is not None else float("nan")
    return mean, std


def download_reference_csvs(
    cache_dir: str | Path,
    offline: bool = False,
    force: bool = False,
) -> Dict[str, Path]:
    """Ensure every FS-Mol baseline summary CSV is present in ``cache_dir``.

    The files total roughly 150 KB. They are fetched once and reused, so only the first
    run needs network access.

    Args:
        cache_dir: Directory to hold the downloaded CSVs.
        offline: If True, never touch the network; a missing file is an error.
        force: Re-download even when a cached copy exists.

    Returns:
        Mapping of method display name to the local CSV path.

    Raises:
        FileNotFoundError: If ``offline`` is set and a file is not already cached.
        RuntimeError: If a download fails.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    for method, filename in FSMOL_BASELINE_FILES.items():
        target = cache / filename
        if target.exists() and not force:
            paths[method] = target
            continue
        if offline:
            raise FileNotFoundError(
                f"FS-Mol reference file for '{method}' not cached at {target} and offline=True. "
                f"Run once without --offline to populate {cache}."
            )
        url = f"{_RAW_BASE}/{filename}"
        logger.info("Downloading FS-Mol reference '%s' -> %s", method, target)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Could not download FS-Mol reference from {url}: {exc}") from exc
        paths[method] = target

    return paths


def reference_checksums(cache_dir: str | Path) -> Dict[str, str]:
    """SHA-256 of each cached reference CSV, for recording run provenance."""
    cache = Path(cache_dir)
    out: Dict[str, str] = {}
    for method, filename in FSMOL_BASELINE_FILES.items():
        path = cache / filename
        if path.exists():
            out[method] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def parse_reference_csv(path: str | Path, method: str) -> pd.DataFrame:
    """Parse one ``*_summary.csv`` into long form with ΔAUPRC derived.

    Args:
        path: Path to the summary CSV.
        method: Display name to tag the rows with.

    Returns:
        DataFrame with columns ``[method, task_id, support_size, auprc, auprc_std,
        fraction_positive, delta_auprc]``. Support sizes the task was too small for are
        present as rows with ``NaN`` scores.
    """
    frame = pd.read_csv(path, dtype=str)
    if "TASK_ID" not in frame.columns:
        raise ValueError(f"{path} is missing the TASK_ID column; got {list(frame.columns)}")

    records: List[dict] = []
    for _, row in frame.iterrows():
        task_id = str(row["TASK_ID"]).strip()
        frac_pos = float(row["fraction_positive_test"])
        for size in REFERENCE_SUPPORT_SIZES:
            column = f"{size}_train"
            if column not in frame.columns:
                continue
            mean, std = _parse_mean_std(row[column])
            records.append(
                {
                    "method": method,
                    "task_id": task_id,
                    "support_size": size,
                    "auprc": mean,
                    "auprc_std": std,
                    "fraction_positive": frac_pos,
                    "delta_auprc": mean - frac_pos,
                }
            )
    return pd.DataFrame.from_records(records)


def load_reference_table(
    cache_dir: str | Path,
    offline: bool = False,
    methods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load every FS-Mol baseline into a single long-form ΔAUPRC table.

    Args:
        cache_dir: Directory holding (or to receive) the summary CSVs.
        offline: Refuse to hit the network; requires the files to be cached already.
        methods: Restrict to these display names; defaults to all of
            :data:`FSMOL_BASELINE_FILES`.

    Returns:
        Concatenated DataFrame as documented on :func:`parse_reference_csv`.
    """
    wanted = list(methods) if methods else list(FSMOL_BASELINE_FILES)
    unknown = [m for m in wanted if m not in FSMOL_BASELINE_FILES]
    if unknown:
        raise ValueError(f"Unknown FS-Mol reference method(s) {unknown}; known: {list(FSMOL_BASELINE_FILES)}")

    paths = download_reference_csvs(cache_dir, offline=offline)
    frames = [parse_reference_csv(paths[method], method) for method in wanted]
    table = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded FS-Mol reference: %d method(s), %d task(s), %d row(s).",
        table["method"].nunique(),
        table["task_id"].nunique(),
        len(table),
    )
    return table


def reference_task_ids(table: pd.DataFrame) -> List[str]:
    """Task ids present for **every** method in ``table``, sorted."""
    per_method = [set(group["task_id"]) for _, group in table.groupby("method")]
    if not per_method:
        return []
    common = set.intersection(*per_method)
    return sorted(common)


def aggregate_reference(
    table: pd.DataFrame,
    task_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Mean ΔAUPRC ± standard error per (method, support size), optionally on a subset.

    The standard error is taken **across tasks**, matching how the FS-Mol paper reports
    its error bars — not across FS-Mol's ten within-task repeats.

    Args:
        table: Long-form reference table from :func:`load_reference_table`.
        task_ids: Restrict the aggregation to these tasks; defaults to all of them.

    Returns:
        DataFrame with ``[method, support_size, delta_auprc_mean, delta_auprc_sem, n_tasks]``.
    """
    frame = table
    if task_ids is not None:
        frame = frame[frame["task_id"].isin(list(task_ids))]

    records: List[dict] = []
    for (method, size), group in frame.groupby(["method", "support_size"]):
        values = group["delta_auprc"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        records.append(
            {
                "method": method,
                "support_size": int(size),
                "delta_auprc_mean": float(values.mean()) if n else float("nan"),
                "delta_auprc_sem": float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                "n_tasks": n,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["method", "support_size"]).reset_index(drop=True)


def reference_pivot(table: pd.DataFrame, task_ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Method x support-size matrix of mean ΔAUPRC, for at-a-glance reporting."""
    summary = aggregate_reference(table, task_ids=task_ids)
    return summary.pivot(index="method", columns="support_size", values="delta_auprc_mean")
