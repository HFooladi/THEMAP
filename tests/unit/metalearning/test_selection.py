"""Tests for the distance-file loader and k-nearest source selector.

These are torch-free and mirror the orchestrator's JSON/CSV/NPZ output formats.
"""

import json

import numpy as np
import pandas as pd
import pytest

from themap.metalearning.selection import load_distance_matrix, select_k_nearest_sources

# target -> source -> distance (orchestrator nested-dict convention).
NESTED = {
    "T1": {"S1": 5.0, "S2": 1.0, "S3": 3.0},
    "T2": {"S1": 2.0, "S2": 4.0, "S3": 0.5},
}


def _write_json(path):
    path.write_text(json.dumps(NESTED))
    return path


def _write_csv(path, transposed=False):
    # Orchestrator writes rows=targets, cols=sources.
    df = pd.DataFrame(NESTED).T
    if transposed:
        df = df.T
    df.to_csv(path)
    return path


def _write_npz(path):
    target_ids = list(NESTED)
    source_ids = list(NESTED[target_ids[0]])
    arr = np.array([[NESTED[t][s] for s in source_ids] for t in target_ids])
    np.savez(path, distances=arr, source_ids=source_ids, target_ids=target_ids)
    return path


@pytest.mark.unit
class TestLoadDistanceMatrix:
    def test_json_roundtrip(self, tmp_path):
        D, targets, sources = load_distance_matrix(_write_json(tmp_path / "d.json"))
        assert targets == ["T1", "T2"]
        assert sources == ["S1", "S2", "S3"]
        assert D[targets.index("T1"), sources.index("S2")] == 1.0

    def test_npz_roundtrip(self, tmp_path):
        D, targets, sources = load_distance_matrix(_write_npz(tmp_path / "d.npz"))
        assert set(targets) == {"T1", "T2"}
        assert D[targets.index("T2"), sources.index("S3")] == 0.5

    def test_csv_rows_are_targets(self, tmp_path):
        D, targets, sources = load_distance_matrix(_write_csv(tmp_path / "d.csv"), target_hint="T1")
        assert "T1" in targets and "S1" in sources
        assert D[targets.index("T1"), sources.index("S2")] == 1.0

    def test_csv_transposed_autodetect(self, tmp_path):
        # File saved transposed (targets in columns); hint flips it back.
        D, targets, sources = load_distance_matrix(
            _write_csv(tmp_path / "dt.csv", transposed=True), target_hint="T1"
        )
        assert "T1" in targets and "S1" in sources
        assert D[targets.index("T1"), sources.index("S2")] == 1.0

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_distance_matrix(tmp_path / "nope.json")

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("x")
        with pytest.raises(ValueError):
            load_distance_matrix(p)


@pytest.mark.unit
class TestSelectKNearest:
    def test_ascending_order(self, tmp_path):
        result = select_k_nearest_sources(_write_json(tmp_path / "d.json"), "T1", k=2)
        assert [s for s, _ in result] == ["S2", "S3"]  # 1.0 then 3.0
        assert result[0][1] == 1.0

    def test_self_exclusion(self, tmp_path):
        nested = {"T1": {"T1": 0.0, "S2": 1.0, "S3": 3.0}}
        (tmp_path / "self.json").write_text(json.dumps(nested))
        result = select_k_nearest_sources(tmp_path / "self.json", "T1", k=5)
        assert "T1" not in [s for s, _ in result]

    def test_drops_nonfinite(self, tmp_path):
        nested = {"T1": {"S1": float("nan"), "S2": 1.0, "S3": float("inf")}}
        (tmp_path / "nan.json").write_text(json.dumps(nested))
        result = select_k_nearest_sources(tmp_path / "nan.json", "T1", k=5)
        assert [s for s, _ in result] == ["S2"]

    def test_fewer_than_k_warns(self, tmp_path, caplog):
        result = select_k_nearest_sources(_write_json(tmp_path / "d.json"), "T1", k=10)
        assert len(result) == 3
        assert any("valid source" in r.message for r in caplog.records)

    def test_unknown_target_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            select_k_nearest_sources(_write_json(tmp_path / "d.json"), "NOPE", k=1)

    def test_nonpositive_k_raises(self, tmp_path):
        with pytest.raises(ValueError, match="k must be positive"):
            select_k_nearest_sources(_write_json(tmp_path / "d.json"), "T1", k=0)
