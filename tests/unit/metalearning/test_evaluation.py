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
            "n_support_actual",
            "n_query_actual",
            "frac_pos_query",
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


def _skewed_target(n_pos=70, n_neg=30, dim=32, seed=0):
    """A deliberately imbalanced task, to tell proportional from forced-balanced draws."""
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [
            rng.normal(2.0, 0.5, (n_pos, dim)).astype(np.float32),
            rng.normal(-2.0, 0.5, (n_neg, dim)).astype(np.float32),
        ]
    )
    y = np.array([1] * n_pos + [0] * n_neg)
    return TaskFeatures.from_arrays("skewed", X, y)


def _evaluator(target, support_sizes, seeds=1, query_mode="fsmol", dim=32):
    return LowDataEvaluator(
        learner=ProtoNet(input_dim=dim),
        target=target,
        input_dim=dim,
        encoder_config=EncoderConfig(),
        algorithm="proto",
        support_sizes=support_sizes,
        seeds=seeds,
        device="cpu",
        query_mode=query_mode,
    )


@pytest.mark.unit
class TestFSMolQueryMode:
    def test_query_is_the_entire_remainder(self):
        target = _target()  # 160 molecules
        ev = _evaluator(target, [16])
        sup_idx, qry_idx = ev._split_for_n(seed=0, n=16, pools=None)
        assert len(sup_idx) == 16
        assert len(qry_idx) == 160 - 16
        assert set(sup_idx.tolist()) | set(qry_idx.tolist()) == set(range(160))

    def test_support_and_query_are_disjoint(self):
        ev = _evaluator(_target(), [32])
        sup_idx, qry_idx = ev._split_for_n(seed=3, n=32, pools=None)
        assert not (set(sup_idx.tolist()) & set(qry_idx.tolist()))

    def test_support_is_proportionally_stratified_not_forced_balanced(self):
        # 70/30 task: FS-Mol draws ~70% actives into the support set, unlike _support_for_n
        # which always takes n // 2 per class.
        target = _skewed_target(n_pos=70, n_neg=30)
        ev = _evaluator(target, [20])
        sup_idx, _ = ev._split_for_n(seed=0, n=20, pools=None)
        n_pos = int(target.y[sup_idx].sum())
        assert abs(n_pos - 14) <= 1
        assert n_pos != 10

    def test_larger_support_sizes_stay_feasible(self):
        # 160 molecules: N=128 is impossible under the 50% holdout but fine for FS-Mol.
        target = _target()
        assert _evaluator(target, [128], query_mode="holdout")._split_for_n(0, 128, None) is None
        holdout = _evaluator(target, [128], query_mode="holdout").evaluate()
        assert holdout.empty
        fsmol = _evaluator(target, [128]).evaluate()
        assert set(fsmol["support_size"]) == {128}

    def test_infeasible_size_is_skipped_not_raised(self):
        results = _evaluator(_target(), [16, 1000]).evaluate()
        assert set(results["support_size"]) == {16}

    def test_seeds_reproduce_and_differ(self):
        ev = _evaluator(_target(), [16])
        a1, _ = ev._split_for_n(seed=0, n=16, pools=None)
        a2, _ = ev._split_for_n(seed=0, n=16, pools=None)
        b, _ = ev._split_for_n(seed=1, n=16, pools=None)
        assert np.array_equal(a1, a2)
        assert not np.array_equal(a1, b)

    def test_delta_auprc_uses_the_actual_query_prevalence(self):
        results = _evaluator(_skewed_target(), [16]).evaluate()
        row = results.iloc[0]
        np.testing.assert_allclose(
            row["delta_auprc"], row["avg_precision"] - row["frac_pos_query"], atol=1e-6
        )

    def test_recorded_sizes_match_the_draw(self):
        results = _evaluator(_target(), [16, 32], seeds=2).evaluate()
        assert (results["n_support_actual"] == results["support_size"]).all()
        assert (results["n_query_actual"] == 160 - results["support_size"]).all()

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="query_mode"):
            _evaluator(_target(), [16], query_mode="nonsense")


@pytest.mark.unit
def test_holdout_remains_the_default():
    ev = LowDataEvaluator(
        learner=ProtoNet(input_dim=32),
        target=_target(),
        input_dim=32,
        encoder_config=EncoderConfig(),
        algorithm="proto",
        support_sizes=[16],
        seeds=1,
        device="cpu",
    )
    assert ev.query_mode == "holdout"
    pools = ev._build_seed_pools(seed=0)
    sup_idx, qry_idx = ev._split_for_n(seed=0, n=16, pools=pools)
    # Shared query set, forced-balanced support - the pre-existing behaviour.
    assert np.array_equal(qry_idx, pools["qry_idx"])
    assert int(_target().y[sup_idx].sum()) == 8
