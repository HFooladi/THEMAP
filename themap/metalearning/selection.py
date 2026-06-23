"""Load saved dataset-distance files and select the k-nearest source datasets.

THEMAP's pipeline saves an N×M dataset-distance matrix as a nested dict
``target_id -> source_id -> distance`` in one of three formats (see
``themap/pipeline/orchestrator.py``):

* **JSON** — the nested dict verbatim.
* **NPZ** — arrays ``distances`` (shape ``(n_targets, n_sources)``),
  ``source_ids`` and ``target_ids``.
* **CSV** — a dense matrix with **rows = targets, columns = sources** (note: the
  in-code comment in the orchestrator claims the opposite; the actual output and
  ``output/molecule_distances.csv`` have targets as rows).

This module reads any of those back and answers the question that drives the
meta-learning workflow: *which source datasets are closest to a given target?*

It is intentionally torch-free so the CLI can use it without importing torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..utils.logging import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


def load_distance_matrix(
    path: PathLike,
    target_hint: str = "",
) -> Tuple[NDArray[np.float64], List[str], List[str]]:
    """Load a saved distance file into a dense ``target × source`` matrix.

    Args:
        path: Path to a ``.json``, ``.csv`` or ``.npz`` distance file.
        target_hint: A known target id, used to disambiguate CSV orientation
            when the saved matrix could be read either way.

    Returns:
        Tuple ``(D, target_ids, source_ids)`` where ``D[i, j]`` is the distance
        from ``target_ids[i]`` to ``source_ids[j]``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is unsupported or the content is empty.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Distance file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        return _load_json(p)
    if suffix == ".npz":
        return _load_npz(p)
    if suffix == ".csv":
        return _load_csv(p, target_hint)
    raise ValueError(f"Unsupported distance file format '{suffix}' (expected .json/.csv/.npz)")


def _load_json(p: Path) -> Tuple[NDArray[np.float64], List[str], List[str]]:
    with open(p) as f:
        nested = json.load(f)
    if not nested:
        raise ValueError(f"Distance file is empty: {p}")
    target_ids = list(nested.keys())
    source_ids = list(nested[target_ids[0]].keys())
    matrix = np.array(
        [[float(nested[t].get(s, np.nan)) for s in source_ids] for t in target_ids],
        dtype=np.float64,
    )
    return matrix, target_ids, source_ids


def _load_npz(p: Path) -> Tuple[NDArray[np.float64], List[str], List[str]]:
    data = np.load(p, allow_pickle=True)
    matrix = np.asarray(data["distances"], dtype=np.float64)
    source_ids = [str(s) for s in data["source_ids"]]
    target_ids = [str(t) for t in data["target_ids"]]
    return matrix, target_ids, source_ids


def _load_csv(p: Path, target_hint: str) -> Tuple[NDArray[np.float64], List[str], List[str]]:
    import pandas as pd

    df = pd.read_csv(p, index_col=0)
    if df.empty:
        raise ValueError(f"Distance file is empty: {p}")
    row_ids = [str(i) for i in df.index]
    col_ids = [str(c) for c in df.columns]

    # Orchestrator convention is rows=targets, cols=sources. If a target hint is
    # given and only the columns contain it, the file was transposed -> flip it.
    if target_hint and target_hint not in row_ids and target_hint in col_ids:
        logger.debug("Distance CSV appears transposed (target in columns); transposing.")
        df = df.T
        row_ids, col_ids = col_ids, row_ids

    matrix = df.to_numpy(dtype=np.float64)
    return matrix, row_ids, col_ids


def select_k_nearest_sources(
    distance_path: PathLike,
    target_id: str,
    k: int,
    exclude_self: bool = True,
) -> List[Tuple[str, float]]:
    """Return the ``k`` source datasets closest to ``target_id``.

    Args:
        distance_path: Path to a saved distance file (JSON/CSV/NPZ).
        target_id: Target task id whose nearest sources are requested.
        k: Number of nearest sources to return.
        exclude_self: If True, drop a source whose id equals ``target_id``.

    Returns:
        List of ``(source_id, distance)`` pairs sorted by ascending distance.
        Sources with NaN/inf distances are dropped. If fewer than ``k`` valid
        sources remain, all of them are returned (with a warning).

    Raises:
        ValueError: If ``target_id`` is not present in the distance file, or if
            ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    matrix, target_ids, source_ids = load_distance_matrix(distance_path, target_hint=target_id)
    if target_id not in target_ids:
        raise ValueError(
            f"Target id '{target_id}' not found among targets in {distance_path}. "
            f"Available targets: {target_ids}"
        )

    row = matrix[target_ids.index(target_id)]
    candidates: List[Tuple[str, float]] = []
    for source_id, dist in zip(source_ids, row):
        if exclude_self and source_id == target_id:
            continue
        if not np.isfinite(dist):
            continue
        candidates.append((source_id, float(dist)))

    candidates.sort(key=lambda pair: pair[1])

    if len(candidates) < k:
        logger.warning(
            "Only %d valid source(s) available for target '%s'; requested k=%d.",
            len(candidates),
            target_id,
            k,
        )
    if not candidates:
        raise ValueError(f"No valid source datasets found for target '{target_id}'.")

    return candidates[:k]
