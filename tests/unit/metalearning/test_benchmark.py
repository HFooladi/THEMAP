"""Tests for the FS-Mol parity benchmark harness and its report."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from themap.metalearning.benchmark import BenchmarkConfig, FSMolBenchmark  # noqa: E402
from themap.metalearning.config import TrainConfig  # noqa: E402
from themap.metalearning.episodes import FeatureBank, TaskFeatures  # noqa: E402
from themap.metalearning.report import (  # noqa: E402
    aggregate_runs,
    comparison_table,
    correlations,
    evaluate_criteria,
    per_task_scores,
)


def _bank(prefix, n_tasks, n_per_class=60, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    tasks = {}
    for i in range(n_tasks):
        task_id = f"{prefix}{i}"
        pos = rng.normal(1.0, 0.5, (n_per_class, dim)).astype(np.float32)
        neg = rng.normal(-1.0, 0.5, (n_per_class, dim)).astype(np.float32)
        X = np.vstack([pos, neg])
        y = np.array([1] * n_per_class + [0] * n_per_class)
        tasks[task_id] = TaskFeatures.from_arrays(task_id, X, y)
    return FeatureBank(tasks)


@pytest.fixture
def tiny_config(tmp_path):
    return BenchmarkConfig(
        data_dir=str(tmp_path),
        task_list_file="",
        algorithms=["proto"],
        support_sizes=[8, 16],
        seeds=2,
        train_shot=8,
        min_source_tasks=2,
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        reference_dir=str(tmp_path / "ref"),
        offline=True,
        train=TrainConfig(
            num_epochs=1,
            episodes_per_epoch=2,
            meta_batch_size=2,
            n_query=8,
            val_episodes=2,
            patience=0,
            device="cpu",
        ),
    )


@pytest.mark.unit
class TestBenchmarkRun:
    def test_end_to_end_with_stubbed_features(self, tiny_config, monkeypatch, tmp_path):
        (tmp_path / "test").mkdir()
        banks = {
            "source": _bank("SRC", 6, seed=1),
            "valid": _bank("VAL", 2, seed=2),
            "target": _bank("TGT", 3, seed=3),
        }
        monkeypatch.setattr(FSMolBenchmark, "featurize", lambda self: banks)

        results = FSMolBenchmark(tiny_config).run()

        # 3 targets x 2 support sizes x 2 seeds x (meta + baseline).
        assert len(results) == 3 * 2 * 2 * 2
        assert set(results["task_id"]) == {"TGT0", "TGT1", "TGT2"}
        assert set(results["method"]) == {"meta", "baseline"}
        assert np.isfinite(results["delta_auprc"]).all()

    def test_writes_every_artifact(self, tiny_config, monkeypatch, tmp_path):
        (tmp_path / "test").mkdir()
        banks = {
            "source": _bank("SRC", 6, seed=1),
            "valid": _bank("VAL", 2, seed=2),
            "target": _bank("TGT", 3, seed=3),
        }
        monkeypatch.setattr(FSMolBenchmark, "featurize", lambda self: banks)
        FSMolBenchmark(tiny_config).run()

        out = tmp_path / "out"
        for name in (
            "per_task_results.csv",
            "comparison_summary.csv",
            "config.json",
            "history.json",
            "report.md",
            "acceptance_criteria.csv",
        ):
            assert (out / name).exists(), name

    def test_missing_reference_does_not_lose_results(self, tiny_config, monkeypatch, tmp_path):
        """An unreachable FS-Mol reference must degrade to THEMAP-only reporting."""
        (tmp_path / "test").mkdir()
        banks = {
            "source": _bank("SRC", 6, seed=1),
            "valid": _bank("VAL", 2, seed=2),
            "target": _bank("TGT", 3, seed=3),
        }
        monkeypatch.setattr(FSMolBenchmark, "featurize", lambda self: banks)
        results = FSMolBenchmark(tiny_config).run()
        assert not results.empty
        assert (tmp_path / "out" / "report.md").exists()

    def test_too_few_usable_sources_fails_loudly(self, tiny_config, monkeypatch, tmp_path):
        (tmp_path / "test").mkdir()
        banks = {
            "source": _bank("SRC", 6, n_per_class=6, seed=1),
            "valid": _bank("VAL", 2, seed=2),
            "target": _bank("TGT", 3, seed=3),
        }
        monkeypatch.setattr(FSMolBenchmark, "featurize", lambda self: banks)
        tiny_config.min_source_tasks = 100
        with pytest.raises(ValueError, match="min_source_tasks"):
            FSMolBenchmark(tiny_config).run()

    def test_records_pool_statistics(self, tiny_config, monkeypatch, tmp_path):
        (tmp_path / "test").mkdir()
        banks = {
            "source": _bank("SRC", 6, seed=1),
            "valid": _bank("VAL", 2, seed=2),
            "target": _bank("TGT", 3, seed=3),
        }
        monkeypatch.setattr(FSMolBenchmark, "featurize", lambda self: banks)
        bench = FSMolBenchmark(tiny_config)
        bench.run()
        stats = bench._pool_stats["proto"]
        assert stats["n_source_tasks"] == 6
        assert stats["realized_mean_support"] <= tiny_config.train_shot


def _results_frame():
    """Two tasks, one support size, two seeds - with a deliberate spread across tasks."""
    rows = []
    for task_id, seed_values in (("A", [0.30, 0.32]), ("B", [0.10, 0.12])):
        for seed, value in enumerate(seed_values):
            rows.append(
                dict(
                    task_id=task_id,
                    algorithm="proto",
                    support_size=16,
                    seed=seed,
                    method="meta",
                    auroc=0.7,
                    avg_precision=value + 0.5,
                    delta_auprc=value,
                    n_support_actual=16,
                    n_query_actual=100,
                    frac_pos_query=0.5,
                )
            )
            rows.append(
                dict(
                    task_id=task_id,
                    algorithm="proto",
                    support_size=16,
                    seed=seed,
                    method="baseline",
                    auroc=0.6,
                    avg_precision=value + 0.45,
                    delta_auprc=value - 0.05,
                    n_support_actual=16,
                    n_query_actual=100,
                    frac_pos_query=0.5,
                )
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestReportAggregation:
    def test_seeds_collapse_within_task_first(self):
        per_task = per_task_scores(_results_frame())
        meta = per_task[per_task["arm"] == "themap-proto"].set_index("task_id")
        assert meta.loc["A", "delta_auprc"] == pytest.approx(0.31)
        assert meta.loc["B", "delta_auprc"] == pytest.approx(0.11)
        assert (meta["n_seeds"] == 2).all()

    def test_error_bar_is_across_tasks_not_seeds(self):
        """Pooling seeds and tasks in one pass would give a far tighter interval."""
        agg = aggregate_runs(_results_frame())
        row = agg[agg["arm"] == "themap-proto"].iloc[0]
        assert row["mean"] == pytest.approx(0.21)
        assert row["n_tasks"] == 2
        # SE over the two per-task means {0.31, 0.11} is 0.1; over the four raw seed
        # values it would be ~0.054.
        assert row["sem"] == pytest.approx(0.1, abs=1e-6)

    def test_arms_are_labelled_distinctly(self):
        agg = aggregate_runs(_results_frame())
        assert set(agg["arm"]) == {"themap-proto", "themap-baseline-mlp"}

    def test_empty_results_are_handled(self):
        empty = pd.DataFrame(
            columns=["task_id", "algorithm", "support_size", "seed", "method", "delta_auprc"]
        )
        assert aggregate_runs(empty).empty
        assert per_task_scores(empty).empty


def _reference_frame(values):
    return pd.DataFrame(
        [
            dict(
                method="PN",
                task_id=task_id,
                support_size=16,
                auprc=np.nan,
                auprc_std=np.nan,
                fraction_positive=0.5,
                delta_auprc=value,
            )
            for task_id, value in values.items()
        ]
    )


@pytest.mark.unit
class TestComparisonAndCriteria:
    def test_reference_rows_are_prefixed_and_merged(self):
        combined = comparison_table(_results_frame(), _reference_frame({"A": 0.4, "B": 0.2}), ["A", "B"])
        assert "fsmol-PN" in set(combined["arm"])
        assert "themap-proto" in set(combined["arm"])

    def test_correlation_detects_matching_task_ordering(self):
        results = _results_frame()
        reference = _reference_frame({"A": 0.45, "B": 0.15, "C": 0.9})
        corr = correlations(results, reference).iloc[0]
        # Only A and B overlap, and THEMAP ranks them the same way FS-Mol does.
        assert corr["n_tasks"] == 2
        assert np.isnan(corr["spearman"])  # fewer than 3 paired tasks

    def test_correlation_with_enough_tasks(self):
        rows = []
        for i, value in enumerate([0.05, 0.15, 0.25, 0.35, 0.45]):
            rows.append(
                dict(
                    task_id=f"T{i}",
                    algorithm="proto",
                    support_size=16,
                    seed=0,
                    method="meta",
                    auroc=0.7,
                    avg_precision=value + 0.5,
                    delta_auprc=value,
                    n_support_actual=16,
                    n_query_actual=100,
                    frac_pos_query=0.5,
                )
            )
        reference = _reference_frame({f"T{i}": v + 0.1 for i, v in enumerate([0.05, 0.15, 0.25, 0.35, 0.45])})
        corr = correlations(pd.DataFrame(rows), reference).iloc[0]
        assert corr["spearman"] == pytest.approx(1.0)
        assert corr["slope"] == pytest.approx(1.0, abs=1e-6)
        assert corr["mean_offset"] == pytest.approx(-0.1, abs=1e-6)

    def test_criteria_flag_a_meta_arm_that_loses_to_its_baseline(self):
        comparison = pd.DataFrame(
            [
                dict(arm="themap-proto", support_size=16, mean=0.02, sem=0.01, n_tasks=20),
                dict(arm="themap-baseline-mlp", support_size=16, mean=0.10, sem=0.01, n_tasks=20),
                dict(arm="fsmol-RF", support_size=16, mean=0.098, sem=0.007, n_tasks=20),
                dict(arm="fsmol-PN", support_size=16, mean=0.211, sem=0.009, n_tasks=20),
            ]
        )
        corr = pd.DataFrame([dict(support_size=16, spearman=0.05)])
        criteria = evaluate_criteria(comparison, corr)
        assert not criteria["passed"].all()
        failed = set(criteria[~criteria["passed"]]["criterion"])
        assert any("from-scratch MLP" in name for name in failed)
        assert any("RF" in name for name in failed)
        assert any("per task" in name for name in failed)

    def test_criteria_pass_for_a_sound_but_weaker_encoder(self):
        comparison = pd.DataFrame(
            [
                dict(arm="themap-proto", support_size=16, mean=0.150, sem=0.02, n_tasks=20),
                dict(arm="themap-proto", support_size=64, mean=0.190, sem=0.02, n_tasks=20),
                dict(arm="themap-proto", support_size=128, mean=0.210, sem=0.02, n_tasks=20),
                dict(arm="themap-baseline-mlp", support_size=16, mean=0.060, sem=0.02, n_tasks=20),
                dict(arm="fsmol-RF", support_size=16, mean=0.098, sem=0.007, n_tasks=20),
                dict(arm="fsmol-PN", support_size=16, mean=0.211, sem=0.009, n_tasks=20),
            ]
        )
        corr = pd.DataFrame([dict(support_size=16, spearman=0.68)])
        criteria = evaluate_criteria(comparison, corr)
        assert criteria["passed"].all(), criteria[~criteria["passed"]]["criterion"].tolist()
