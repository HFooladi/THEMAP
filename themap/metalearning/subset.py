"""Pick a small, representative slice of the FS-Mol test set to benchmark against.

Evaluating all 157 FS-Mol test tasks is affordable but slow to iterate on, so the parity
benchmark runs on a stratified subset. Because FS-Mol publishes *per-task* baseline scores
(see :mod:`themap.metalearning.fsmol_reference`), the reference means can be recomputed on
exactly the same subset — the comparison stays apples-to-apples regardless of which tasks
are chosen. Choosing them well simply keeps the subset means close to the full-benchmark
means, which :func:`subset_representativeness` verifies.

Selection is fully deterministic — no random number generator is involved — so a run is
reproducible from the saved subset file alone. Tasks are stratified along three axes that
are not independent of one another (task size correlates negatively with FS-Mol ProtoNet
performance, so stratifying on difficulty alone would skew the size distribution):

1. **EC super-class** of the protein target, proportionally allocated. FS-Mol's test set is
   dominated by transferases (125 of 157 are kinases).
2. **Difficulty**, as terciles of FS-Mol ProtoNet ΔAUPRC at support size 16.
3. **Task size**, as terciles of the molecule count.

Within each (difficulty, size) cell the *medoid* task is taken — the one closest to the
cell centre — rather than an arbitrary member, so each pick is typical of its cell.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .fsmol_reference import PROTONET_METHOD

logger = get_logger(__name__)

_PROPERTY_RE = re.compile(rb'"Property":\s*"([\d.eE+-]+)"')


@dataclass
class SubsetSpec:
    """Knobs for :func:`select_benchmark_subset`.

    Attributes:
        n_tasks: How many tasks to select.
        min_large: Minimum number of selected tasks with more molecules than
            ``large_task_threshold``, so the largest support sizes have enough tasks to
            average over. Only 43 of the 157 test tasks can support N=256 at all.
        large_task_threshold: Molecule count above which a task counts as "large".
        ec_column: Column of the protein metadata CSV holding the EC super-class name.
        ec_collapse_below: EC classes with fewer tasks than this fold into ``"other"``.
        difficulty_method: Reference method whose per-task ΔAUPRC defines difficulty.
        difficulty_support_size: Support size at which difficulty is read.
        n_difficulty_bins: Number of difficulty strata within each EC class.
        n_size_bins: Number of task-size strata within each EC class.
    """

    n_tasks: int = 20
    min_large: int = 6
    large_task_threshold: int = 256
    ec_column: str = "EC_super_class_name"
    ec_collapse_below: int = 5
    difficulty_method: str = PROTONET_METHOD
    difficulty_support_size: int = 16
    n_difficulty_bins: int = 3
    n_size_bins: int = 3


def task_label_counts(
    data_dir: str | Path,
    fold: str = "test",
    task_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Count molecules and class balance for each task, without full JSON parsing.

    Reads only the ``Property`` field out of each ``.jsonl.gz`` line, which is roughly an
    order of magnitude cheaper than decoding FS-Mol's per-molecule graph and fingerprint
    payloads.

    Args:
        data_dir: Root directory holding the ``train``/``valid``/``test`` folds.
        fold: Fold to scan.
        task_ids: Restrict to these task ids; defaults to every file in the fold.

    Returns:
        DataFrame with ``[task_id, n, n_pos, n_neg, min_class, frac_pos]``.
    """
    fold_dir = Path(data_dir) / fold
    if not fold_dir.is_dir():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    if task_ids is None:
        ids = sorted(p.name[: -len(".jsonl.gz")] for p in fold_dir.glob("*.jsonl.gz"))
    else:
        ids = list(task_ids)

    records: List[dict] = []
    for task_id in ids:
        path = fold_dir / f"{task_id}.jsonl.gz"
        if not path.exists():
            logger.warning("Task file missing, skipping: %s", path)
            continue
        n_pos = n_neg = 0
        with gzip.open(path, "rb") as handle:
            for line in handle:
                match = _PROPERTY_RE.search(line)
                if match is None:
                    continue
                if int(float(match.group(1))) == 1:
                    n_pos += 1
                else:
                    n_neg += 1
        total = n_pos + n_neg
        records.append(
            {
                "task_id": task_id,
                "n": total,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "min_class": min(n_pos, n_neg),
                "frac_pos": (n_pos / total) if total else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records)


def _load_ec_classes(proteins_csv: str | Path, column: str) -> Dict[str, str]:
    """Map task id -> EC super-class name from FS-Mol's protein metadata CSV."""
    frame = pd.read_csv(proteins_csv, dtype=str)
    if "chembl_id" not in frame.columns or column not in frame.columns:
        raise ValueError(f"{proteins_csv} must contain 'chembl_id' and '{column}'; got {list(frame.columns)}")
    return {
        str(row["chembl_id"]).strip(): str(row[column]).strip()
        for _, row in frame.iterrows()
        if isinstance(row[column], str) and row[column].strip()
    }


def _allocate(counts: Dict[str, int], total: int) -> Dict[str, int]:
    """Proportional allocation with a floor of one per stratum, largest remainder first."""
    strata = sorted(counts, key=lambda key: (-counts[key], key))
    pool = sum(counts.values())
    if pool == 0:
        return {}
    # Floor of 1 per stratum, but never more strata than tasks requested.
    strata = strata[:total]
    exact = {key: total * counts[key] / pool for key in strata}
    alloc = {key: max(1, int(np.floor(exact[key]))) for key in strata}

    # Trim or top up to hit `total` exactly, ordered by largest fractional remainder.
    order = sorted(strata, key=lambda key: (-(exact[key] - np.floor(exact[key])), key))
    while sum(alloc.values()) > total:
        for key in reversed(order):
            if sum(alloc.values()) <= total:
                break
            if alloc[key] > 1:
                alloc[key] -= 1
    idx = 0
    while sum(alloc.values()) < total:
        key = order[idx % len(order)]
        if alloc[key] < counts[key]:
            alloc[key] += 1
        idx += 1
        if idx > 10 * total:  # pragma: no cover - defensive
            break
    return alloc


def _bin_ranks(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to one of ``n_bins`` equal-count bins by rank (ties broken by order)."""
    if len(values) == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(len(values))
    return np.minimum((ranks * n_bins) // max(len(values), 1), n_bins - 1)


def _pick_from_stratum(frame: pd.DataFrame, k: int, spec: SubsetSpec) -> List[str]:
    """Deterministically take ``k`` medoid tasks spread over difficulty x size cells."""
    if k >= len(frame):
        return sorted(frame["task_id"].tolist())

    work = frame.copy()
    work["difficulty_bin"] = _bin_ranks(work["difficulty"].to_numpy(), spec.n_difficulty_bins)
    work["size_bin"] = _bin_ranks(work["n"].to_numpy().astype(float), spec.n_size_bins)

    cells: Dict[tuple, pd.DataFrame] = {
        key: group for key, group in work.groupby(["difficulty_bin", "size_bin"])
    }
    # Visit the most populous cells first so a small k still covers the bulk of the stratum.
    cell_order = sorted(cells, key=lambda key: (-len(cells[key]), key))

    chosen: List[str] = []
    taken: set = set()
    while len(chosen) < k:
        progressed = False
        for key in cell_order:
            if len(chosen) >= k:
                break
            candidates = cells[key][~cells[key]["task_id"].isin(taken)]
            if candidates.empty:
                continue
            centre_d = candidates["difficulty_pct"].mean()
            centre_s = candidates["size_pct"].mean()
            distance = (candidates["difficulty_pct"] - centre_d) ** 2 + (
                candidates["size_pct"] - centre_s
            ) ** 2
            best = candidates.assign(_d=distance).sort_values(["_d", "task_id"]).iloc[0]["task_id"]
            chosen.append(str(best))
            taken.add(str(best))
            progressed = True
        if not progressed:  # pragma: no cover - defensive
            break
    return chosen


def select_benchmark_subset(
    data_dir: str | Path,
    reference: pd.DataFrame,
    proteins_csv: str | Path,
    spec: SubsetSpec = SubsetSpec(),
    candidate_ids: Optional[Sequence[str]] = None,
    counts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Choose ``spec.n_tasks`` representative FS-Mol test tasks.

    Args:
        data_dir: FS-Mol dataset root (needs a ``test/`` fold).
        reference: Long-form reference table from
            :func:`themap.metalearning.fsmol_reference.load_reference_table`.
        proteins_csv: FS-Mol test protein metadata CSV (for EC super-classes).
        spec: Selection knobs.
        candidate_ids: Restrict the universe to these ids; defaults to every task that has
            both a reference score and a file on disk.
        counts: Precomputed :func:`task_label_counts` output, to avoid rescanning the
            ``.jsonl.gz`` files when selecting repeatedly.

    Returns:
        DataFrame describing the selected tasks, one row each, with the stratification
        columns used to pick them.

    Raises:
        ValueError: If the universe is smaller than ``spec.n_tasks``.
    """
    difficulty = reference[
        (reference["method"] == spec.difficulty_method)
        & (reference["support_size"] == spec.difficulty_support_size)
    ][["task_id", "delta_auprc"]].rename(columns={"delta_auprc": "difficulty"})
    difficulty = difficulty[np.isfinite(difficulty["difficulty"])]

    ids = list(candidate_ids) if candidate_ids is not None else difficulty["task_id"].tolist()
    if counts is None:
        counts = task_label_counts(data_dir, fold="test", task_ids=ids)
    else:
        counts = counts[counts["task_id"].isin(ids)]

    universe = counts.merge(difficulty, on="task_id", how="inner")
    ec_map = _load_ec_classes(proteins_csv, spec.ec_column)
    universe["ec_class_raw"] = universe["task_id"].map(ec_map).fillna("unknown")

    class_counts = universe["ec_class_raw"].value_counts().to_dict()
    universe["ec_class"] = [
        name if class_counts.get(name, 0) >= spec.ec_collapse_below else "other"
        for name in universe["ec_class_raw"]
    ]

    if len(universe) < spec.n_tasks:
        raise ValueError(f"Only {len(universe)} candidate task(s) available; requested {spec.n_tasks}.")

    # Percentile ranks are computed once, globally, so cell centres are comparable.
    universe = universe.sort_values("task_id").reset_index(drop=True)
    universe["difficulty_pct"] = universe["difficulty"].rank(pct=True, method="average")
    universe["size_pct"] = universe["n"].rank(pct=True, method="average")

    allocation = _allocate(
        {name: int(count) for name, count in universe["ec_class"].value_counts().items()},
        spec.n_tasks,
    )
    logger.info("Subset allocation by EC class: %s", allocation)

    chosen: List[str] = []
    for ec_class, k in sorted(allocation.items(), key=lambda item: (-item[1], item[0])):
        stratum = universe[universe["ec_class"] == ec_class]
        chosen.extend(_pick_from_stratum(stratum, k, spec))

    chosen = _enforce_large_quota(chosen, universe, spec)
    selected = universe[universe["task_id"].isin(chosen)].copy()
    selected["difficulty_bin"] = _bin_ranks(selected["difficulty"].to_numpy(), spec.n_difficulty_bins)
    selected["size_bin"] = _bin_ranks(selected["n"].to_numpy().astype(float), spec.n_size_bins)
    return selected.sort_values("task_id").reset_index(drop=True)


def _enforce_large_quota(chosen: List[str], universe: pd.DataFrame, spec: SubsetSpec) -> List[str]:
    """Swap in large tasks until ``spec.min_large`` of the picks can support big N."""
    sizes = universe.set_index("task_id")["n"].to_dict()
    ec = universe.set_index("task_id")["ec_class"].to_dict()
    large = [t for t in chosen if sizes.get(t, 0) > spec.large_task_threshold]
    if len(large) >= spec.min_large:
        return chosen

    dominant = universe["ec_class"].value_counts().idxmax()
    spare = sorted(
        (t for t in universe["task_id"] if t not in chosen and sizes.get(t, 0) > spec.large_task_threshold),
        key=lambda t: (-sizes[t], t),
    )
    # Drop the smallest picks from the dominant class first, so EC proportions barely move.
    droppable = sorted(
        (t for t in chosen if sizes.get(t, 0) <= spec.large_task_threshold and ec.get(t) == dominant),
        key=lambda t: (sizes[t], t),
    )
    result = list(chosen)
    needed = spec.min_large - len(large)
    for replacement, victim in zip(spare[:needed], droppable[:needed]):
        result[result.index(victim)] = replacement
        logger.info(
            "Large-task quota: swapped %s (n=%d) for %s (n=%d).",
            victim,
            sizes[victim],
            replacement,
            sizes[replacement],
        )
    return result


def subset_representativeness(reference: pd.DataFrame, subset_ids: Sequence[str]) -> pd.DataFrame:
    """Compare reference means on the subset against the full benchmark.

    Returns:
        DataFrame ``[method, support_size, subset_mean, full_mean, abs_delta, n_subset, n_full]``.
    """
    from .fsmol_reference import aggregate_reference

    subset = aggregate_reference(reference, task_ids=subset_ids).rename(
        columns={"delta_auprc_mean": "subset_mean", "n_tasks": "n_subset"}
    )[["method", "support_size", "subset_mean", "n_subset"]]
    full = aggregate_reference(reference).rename(
        columns={"delta_auprc_mean": "full_mean", "n_tasks": "n_full"}
    )[["method", "support_size", "full_mean", "n_full"]]
    merged = subset.merge(full, on=["method", "support_size"], how="outer")
    merged["abs_delta"] = (merged["subset_mean"] - merged["full_mean"]).abs()
    return merged.sort_values(["method", "support_size"]).reset_index(drop=True)


def save_subset(selected: pd.DataFrame, path: str | Path, spec: SubsetSpec, provenance: dict) -> None:
    """Write the chosen task ids plus the metadata needed to reproduce them."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_ids": sorted(selected["task_id"].tolist()),
        "spec": asdict(spec),
        "per_task": selected.to_dict(orient="records"),
        "provenance": provenance,
    }
    with open(target, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Saved %d-task benchmark subset to %s", len(selected), target)


def load_subset(path: str | Path) -> List[str]:
    """Read back the task ids written by :func:`save_subset`."""
    with open(path) as handle:
        payload = json.load(handle)
    ids = payload.get("task_ids")
    if not ids:
        raise ValueError(f"No task_ids found in subset file {path}")
    return [str(task_id) for task_id in ids]
