"""Tests for FeatureBank and EpisodeSampler."""

import numpy as np
import pytest

from themap.metalearning.episodes import (
    EpisodeSampler,
    TaskFeatures,
    max_feasible_n_support,
    usable_task_count,
)

torch = pytest.importorskip("torch")


def _make_task(task_id="t", n_pos=40, n_neg=40, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_pos + n_neg, dim)).astype(np.float32)
    y = np.array([1] * n_pos + [0] * n_neg)
    return TaskFeatures.from_arrays(task_id, X, y)


@pytest.mark.unit
class TestTaskFeatures:
    def test_class_indices(self):
        tf = _make_task(n_pos=10, n_neg=7)
        assert len(tf.pos_idx) == 10
        assert len(tf.neg_idx) == 7

    def test_drops_nonfinite_rows(self):
        X = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([1, 0, 1])
        tf = TaskFeatures.from_arrays("t", X, y)
        assert len(tf) == 2
        assert np.isfinite(tf.X).all()


@pytest.mark.unit
class TestMaxFeasibleNSupport:
    def test_uses_most_capable_task(self):
        tasks = [_make_task(n_pos=5, n_neg=5), _make_task(n_pos=40, n_neg=40)]
        # Best task: min(40,40) - (n_query//2) = 40 - 5 = 35 per class -> 70 total.
        assert max_feasible_n_support(tasks, n_query=10) == 70

    def test_limited_by_minority_class(self):
        tasks = [_make_task(n_pos=8, n_neg=40)]
        # min(8,40) - 5 = 3 per class -> 6 total.
        assert max_feasible_n_support(tasks, n_query=10) == 6

    def test_zero_when_no_task_can_supply(self):
        tasks = [_make_task(n_pos=3, n_neg=3)]
        # min(3,3) - 5 = -2 -> clamped to 0.
        assert max_feasible_n_support(tasks, n_query=10) == 0


@pytest.mark.unit
class TestEpisodeSampler:
    def test_balanced_sizes(self):
        sampler = EpisodeSampler([_make_task()], n_support=10, n_query=20, seed=1)
        ep = sampler.sample_episode()
        assert ep.x_s.shape[0] == 10
        assert ep.x_q.shape[0] == 20
        # Balanced across the two classes.
        assert int((ep.y_s == 0).sum()) == int((ep.y_s == 1).sum()) == 5

    def test_support_query_disjoint(self):
        # With a single task and distinct rows, support and query rows must differ.
        sampler = EpisodeSampler([_make_task(dim=8)], n_support=8, n_query=10, seed=3)
        ep = sampler.sample_episode()
        sup = {tuple(r.tolist()) for r in ep.x_s}
        qry = {tuple(r.tolist()) for r in ep.x_q}
        assert sup.isdisjoint(qry)

    def test_labels_relabeled_binary(self):
        sampler = EpisodeSampler([_make_task()], n_support=6, n_query=6, seed=2)
        ep = sampler.sample_episode()
        assert set(ep.y_s.tolist()) <= {0, 1}
        assert set(ep.y_q.tolist()) <= {0, 1}

    def test_validity_filter_skips_small_tasks(self):
        small = _make_task("small", n_pos=2, n_neg=2)
        big = _make_task("big", n_pos=40, n_neg=40)
        sampler = EpisodeSampler([small, big], n_support=10, n_query=20, seed=0)
        assert len(sampler) == 1
        assert sampler.tasks[0].task_id == "big"

    def test_all_tasks_too_small_raises(self):
        with pytest.raises(ValueError):
            EpisodeSampler([_make_task(n_pos=2, n_neg=2)], n_support=10, n_query=20)

    def test_determinism(self):
        task = _make_task()
        ep1 = EpisodeSampler([task], n_support=8, n_query=8, seed=42).sample_episode()
        ep2 = EpisodeSampler([task], n_support=8, n_query=8, seed=42).sample_episode()
        assert np.array_equal(ep1.x_s.numpy(), ep2.x_s.numpy())

    def test_rejects_non_binary(self):
        with pytest.raises(ValueError):
            EpisodeSampler([_make_task()], n_way=3)

    def test_sample_batch_length(self):
        sampler = EpisodeSampler([_make_task()], n_support=6, n_query=6, seed=0)
        assert len(sampler.sample_batch(4)) == 4


@pytest.mark.unit
class TestAdaptiveEpisodes:
    """Adaptive mode treats the requested shot as a maximum, the way FS-Mol's sampler does."""

    @staticmethod
    def _task(task_id, n_per_class, dim=8):
        rng = np.random.default_rng(abs(hash(task_id)) % 2**32)
        X = rng.normal(size=(2 * n_per_class, dim)).astype(np.float32)
        y = np.array([1] * n_per_class + [0] * n_per_class)
        return TaskFeatures.from_arrays(task_id, X, y)

    def test_keeps_small_tasks_the_strict_filter_drops(self):
        tasks = [self._task("small", 12)]
        with pytest.raises(ValueError):
            EpisodeSampler(tasks, n_support=64, n_query=32)  # 32 + 16 per class required
        sampler = EpisodeSampler(tasks, n_support=64, n_query=32, adaptive=True)
        assert len(sampler) == 1

    def test_episode_shrinks_to_task_capacity(self):
        sampler = EpisodeSampler([self._task("small", 12)], n_support=64, n_query=32, adaptive=True, seed=0)
        ep = sampler.sample_episode()
        # 12 per class total: query keeps its 16-per-class request clipped, support takes the rest.
        assert ep.x_s.shape[0] + ep.x_q.shape[0] == 24
        assert ep.x_s.shape[0] >= 2 and ep.x_q.shape[0] >= 2
        assert ep.x_s.shape[0] <= 64 and ep.x_q.shape[0] <= 32

    def test_episodes_stay_balanced_and_disjoint(self):
        sampler = EpisodeSampler([self._task("small", 12)], n_support=64, n_query=32, adaptive=True, seed=1)
        ep = sampler.sample_episode()
        assert int((ep.y_s == 0).sum()) == int((ep.y_s == 1).sum())
        assert int((ep.y_q == 0).sum()) == int((ep.y_q == 1).sum())

    def test_large_task_still_gets_the_full_request(self):
        sampler = EpisodeSampler([self._task("big", 200)], n_support=64, n_query=32, adaptive=True, seed=0)
        ep = sampler.sample_episode()
        assert ep.x_s.shape[0] == 64
        assert ep.x_q.shape[0] == 32

    def test_still_filters_below_the_floor(self):
        # 2 per class cannot satisfy min_support=8 (4/class) + min_query=8 (4/class).
        with pytest.raises(ValueError):
            EpisodeSampler(
                [self._task("tiny", 2)],
                n_support=64,
                n_query=32,
                adaptive=True,
                min_support=8,
                min_query=8,
            )

    def test_deterministic_given_a_seed(self):
        tasks = [self._task("a", 12), self._task("b", 30)]
        first = EpisodeSampler(tasks, n_support=64, n_query=32, adaptive=True, seed=7).sample_episode()
        second = EpisodeSampler(tasks, n_support=64, n_query=32, adaptive=True, seed=7).sample_episode()
        assert first.task_id == second.task_id
        assert torch.equal(first.x_s, second.x_s)

    def test_default_is_unchanged(self):
        tasks = [self._task("a", 12), self._task("big", 200)]
        strict = EpisodeSampler(tasks, n_support=64, n_query=32)
        assert len(strict) == 1  # only the big task survives, exactly as before
        ep = strict.sample_episode()
        assert ep.x_s.shape[0] == 64 and ep.x_q.shape[0] == 32


@pytest.mark.unit
class TestUsableTaskCount:
    @staticmethod
    def _task(task_id, n_per_class, dim=8):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2 * n_per_class, dim)).astype(np.float32)
        y = np.array([1] * n_per_class + [0] * n_per_class)
        return TaskFeatures.from_arrays(task_id, X, y)

    def test_matches_the_sampler_in_both_modes(self):
        tasks = [self._task("a", 5), self._task("b", 25), self._task("c", 100)]
        for adaptive in (False, True):
            expected = len(EpisodeSampler(tasks, n_support=32, n_query=16, adaptive=adaptive))
            assert usable_task_count(tasks, 32, 16, adaptive=adaptive) == expected

    def test_adaptive_retains_more(self):
        tasks = [self._task("a", 5), self._task("b", 25), self._task("c", 100)]
        assert usable_task_count(tasks, 64, 32, adaptive=True) > usable_task_count(tasks, 64, 32)

    def test_quantile_reports_the_median_task_not_the_biggest(self):
        tasks = [self._task("a", 10), self._task("b", 20), self._task("c", 500)]
        assert max_feasible_n_support(tasks, n_query=4) == 2 * (500 - 2)
        assert max_feasible_n_support(tasks, n_query=4, quantile=0.5) == 2 * (20 - 2)
