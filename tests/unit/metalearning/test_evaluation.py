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
        assert set(results.columns) == {"algorithm", "support_size", "seed", "method", "auroc"}
        assert set(results["method"]) == {"meta", "baseline"}
        # 2 sizes x 2 seeds x 2 methods.
        assert len(results) == 8
        assert set(results["support_size"]) == {16, 32}

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
            support_sizes=[16, 1000],  # 1000 >= 160 -> skipped
            seeds=1,
            device="cpu",
        )
        results = evaluator.evaluate()
        assert set(results["support_size"]) == {16}
