"""Tests for the low-data evaluator (meta vs from-scratch baseline)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from themap.metalearning.config import EncoderConfig  # noqa: E402
from themap.metalearning.episodes import TaskFeatures  # noqa: E402
from themap.metalearning.evaluation import LowDataEvaluator  # noqa: E402
from themap.metalearning.models import ProtoNet  # noqa: E402


def _target(dim=32, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(2.0, 0.5, (80, dim)).astype(np.float32)
    neg = rng.normal(-2.0, 0.5, (80, dim)).astype(np.float32)
    X = np.vstack([pos, neg])
    y = np.array([1] * 80 + [0] * 80)
    return TaskFeatures.from_arrays("target", X, y)


@pytest.mark.unit
class TestLowDataEvaluator:
    def test_results_schema_and_methods(self):
        target = _target()
        learner = ProtoNet(input_dim=32)
        evaluator = LowDataEvaluator(
            learner=learner,
            target=target,
            input_dim=32,
            encoder_config=EncoderConfig(),
            algorithm="proto",
            support_sizes=[16, 32],
            seeds=2,
            device="cpu",
        )
        results = evaluator.evaluate()
        assert set(results.columns) == {
            "algorithm",
            "support_size",
            "seed",
            "method",
            "auroc",
            "avg_precision",
            "delta_auprc",
        }
        assert set(results["method"]) == {"meta", "baseline"}
        # 2 sizes x 2 seeds x 2 methods.
        assert len(results) == 8
        assert set(results["support_size"]) == {16, 32}

    def test_delta_auprc_definition(self):
        target = _target()  # balanced 80/80
        evaluator = LowDataEvaluator(
            learner=ProtoNet(input_dim=32),
            target=target,
            input_dim=32,
            encoder_config=EncoderConfig(),
            algorithm="proto",
            support_sizes=[16],
            seeds=1,
            device="cpu",
        )
        results = evaluator.evaluate()
        # delta_auprc == average_precision - positive fraction of the query set.
        row = results.iloc[0]
        assert np.isfinite(row["delta_auprc"])
        assert -1.0 <= row["delta_auprc"] <= 1.0
        np.testing.assert_allclose(row["delta_auprc"], row["avg_precision"] - 0.5, atol=1e-6)

    def test_summarize_has_ci(self):
        target = _target()
        evaluator = LowDataEvaluator(
            learner=ProtoNet(input_dim=32),
            target=target,
            input_dim=32,
            encoder_config=EncoderConfig(),
            algorithm="proto",
            support_sizes=[16],
            seeds=3,
            device="cpu",
        )
        summary = LowDataEvaluator.summarize(evaluator.evaluate())
        assert {"auroc_mean", "auroc_ci95", "n_seeds"} <= set(summary.columns)
        assert (summary["n_seeds"] == 3).all()

    def test_skips_infeasible_support_size(self):
        target = _target()  # 160 samples
        evaluator = LowDataEvaluator(
            learner=ProtoNet(input_dim=32),
            target=target,
            input_dim=32,
            encoder_config=EncoderConfig(),
            algorithm="proto",
            support_sizes=[16, 1000],  # 1000 > max_support -> skipped
            seeds=1,
            device="cpu",
        )
        results = evaluator.evaluate()
        assert set(results["support_size"]) == {16}

    def test_query_set_fixed_and_support_nested_across_sizes(self):
        target = _target()  # 80 pos / 80 neg
        evaluator = LowDataEvaluator(
            learner=ProtoNet(input_dim=32),
            target=target,
            input_dim=32,
            encoder_config=EncoderConfig(),
            algorithm="proto",
            support_sizes=[16, 32, 64],
            seeds=1,
            device="cpu",
            query_fraction=0.5,
        )
        pools = evaluator._build_seed_pools(seed=0)
        assert pools is not None
        # Query set is shared across all support sizes (computed once per seed).
        q16 = pools["qry_idx"]
        # Support sets are nested: smaller N is a prefix of larger N.
        s16 = evaluator._support_for_n(pools, 16)
        s32 = evaluator._support_for_n(pools, 32)
        s64 = evaluator._support_for_n(pools, 64)
        per16, per32 = 16 // 2, 32 // 2
        # Each half (pos then neg) of s16 is a prefix of the matching half of s32.
        assert np.array_equal(s16[:per16], s32[:per16])
        assert np.array_equal(s16[per16:], s32[per32 : per32 + per16])
        # Support and query never overlap.
        assert not (set(s64.tolist()) & set(q16.tolist()))

    def test_skips_when_target_too_small(self, caplog):
        # 6 samples, query_fraction 0.5 -> query 3, pool 3; only tiny support feasible.
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (6, 8)).astype(np.float32)
        y = np.array([1, 1, 1, 0, 0, 0])
        target = TaskFeatures.from_arrays("tiny", X, y)
        evaluator = LowDataEvaluator(
            learner=ProtoNet(input_dim=8),
            target=target,
            input_dim=8,
            encoder_config=EncoderConfig(),
            algorithm="proto",
            support_sizes=[64],  # far larger than the pool -> skipped
            seeds=1,
            device="cpu",
        )
        results = evaluator.evaluate()
        assert results.empty
