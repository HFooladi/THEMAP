"""Tests for parsing FS-Mol's published per-task baseline summaries."""

import io

import numpy as np
import pandas as pd
import pytest

from themap.metalearning.fsmol_reference import (
    FSMOL_BASELINE_FILES,
    REFERENCE_SUPPORT_SIZES,
    _parse_mean_std,
    aggregate_reference,
    download_reference_csvs,
    load_reference_table,
    parse_reference_csv,
    reference_task_ids,
)

# Two tasks; the second has no 256-support result, exactly as upstream encodes
# "this task was too small for that support size".
_CSV = (
    "TASK_ID,fraction_positive_train,fraction_positive_test,fraction_positive_test_std,"
    "fraction_positive_train_std,16_train,32_train,64_train,128_train,256_train\n"
    "CHEMBL1119333,0.5,0.5,0.0,0.0,0.747+/-0.049,0.777+/-0.049,0.811+/-0.034,0.871+/-0.012,0.883+/-0.032\n"
    "CHEMBL1006005,0.5,0.4,0.0,0.0,0.598+/-0.053,0.613+/-0.038,0.624+/-0.039,0.635+/-0.059,\n"
)


def _write(tmp_path, name, text=_CSV):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.mark.unit
class TestParseMeanStd:
    @pytest.mark.parametrize(
        "cell,mean,std",
        [
            ("0.747+/-0.049", 0.747, 0.049),
            ("0.6+/-0.037", 0.6, 0.037),
            (" 0.5 ", 0.5, None),
            ("-0.012+/-0.004", -0.012, 0.004),
        ],
    )
    def test_parses_values(self, cell, mean, std):
        got_mean, got_std = _parse_mean_std(cell)
        assert got_mean == pytest.approx(mean)
        if std is None:
            assert np.isnan(got_std)
        else:
            assert got_std == pytest.approx(std)

    @pytest.mark.parametrize("cell", ["", "   ", None, float("nan"), "nan", "not-a-number"])
    def test_missing_becomes_nan(self, cell):
        mean, std = _parse_mean_std(cell)
        assert np.isnan(mean) and np.isnan(std)


@pytest.mark.unit
class TestParseReferenceCsv:
    def test_delta_auprc_subtracts_query_prevalence(self, tmp_path):
        table = parse_reference_csv(_write(tmp_path, "x.csv"), "PN")
        row = table[(table.task_id == "CHEMBL1006005") & (table.support_size == 16)].iloc[0]
        # AUPRC 0.598 on a task whose query set is 40% actives.
        assert row["auprc"] == pytest.approx(0.598)
        assert row["fraction_positive"] == pytest.approx(0.4)
        assert row["delta_auprc"] == pytest.approx(0.198)

    def test_empty_cell_stays_nan_not_zero(self, tmp_path):
        # Reading a blank cell as 0.0 would silently drag the N=256 mean toward zero.
        table = parse_reference_csv(_write(tmp_path, "x.csv"), "PN")
        row = table[(table.task_id == "CHEMBL1006005") & (table.support_size == 256)].iloc[0]
        assert np.isnan(row["auprc"])
        assert np.isnan(row["delta_auprc"])

    def test_covers_every_support_size(self, tmp_path):
        table = parse_reference_csv(_write(tmp_path, "x.csv"), "PN")
        assert sorted(table.support_size.unique()) == sorted(REFERENCE_SUPPORT_SIZES)
        assert set(table.method) == {"PN"}

    def test_missing_task_id_column_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="TASK_ID"):
            parse_reference_csv(bad, "PN")


@pytest.mark.unit
class TestAggregate:
    def test_nan_cells_excluded_from_mean_and_count(self, tmp_path):
        table = parse_reference_csv(_write(tmp_path, "x.csv"), "PN")
        agg = aggregate_reference(table)
        at_16 = agg[agg.support_size == 16].iloc[0]
        at_256 = agg[agg.support_size == 256].iloc[0]
        assert at_16["n_tasks"] == 2
        assert at_16["delta_auprc_mean"] == pytest.approx((0.247 + 0.198) / 2)
        # Only one task has a 256 result, so the mean is over that task alone.
        assert at_256["n_tasks"] == 1
        assert at_256["delta_auprc_mean"] == pytest.approx(0.383)

    def test_restricting_to_subset(self, tmp_path):
        table = parse_reference_csv(_write(tmp_path, "x.csv"), "PN")
        agg = aggregate_reference(table, task_ids=["CHEMBL1119333"])
        at_16 = agg[agg.support_size == 16].iloc[0]
        assert at_16["n_tasks"] == 1
        assert at_16["delta_auprc_mean"] == pytest.approx(0.247)

    def test_reference_task_ids_is_the_intersection(self, tmp_path):
        one = parse_reference_csv(_write(tmp_path, "a.csv"), "PN")
        two = parse_reference_csv(_write(tmp_path, "b.csv"), "RF")
        two = two[two.task_id != "CHEMBL1006005"]
        common = reference_task_ids(pd.concat([one, two], ignore_index=True))
        assert common == ["CHEMBL1119333"]


@pytest.mark.unit
class TestCaching:
    def test_uses_cache_without_network(self, tmp_path, monkeypatch):
        for filename in FSMOL_BASELINE_FILES.values():
            (tmp_path / filename).write_text(_CSV)

        def explode(*args, **kwargs):  # pragma: no cover - must never be called
            raise AssertionError("network access attempted despite a populated cache")

        monkeypatch.setattr("urllib.request.urlopen", explode)
        table = load_reference_table(tmp_path)
        assert set(table.method) == set(FSMOL_BASELINE_FILES)

    def test_offline_with_empty_cache_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="offline"):
            download_reference_csvs(tmp_path, offline=True)

    def test_unknown_method_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown FS-Mol reference method"):
            load_reference_table(tmp_path, methods=["NoSuchModel"])


@pytest.mark.unit
def test_parse_handles_stringio_roundtrip():
    """The upstream files are plain CSV; guard against dtype coercion of the cells."""
    frame = pd.read_csv(io.StringIO(_CSV), dtype=str)
    assert frame.loc[1, "256_train"] != frame.loc[1, "256_train"] or frame.loc[1, "256_train"] == ""
