"""Tests for FeatureBank and EpisodeSampler."""

import numpy as np
import pytest

from themap.metalearning.episodes import EpisodeSampler, TaskFeatures, max_feasible_n_support

pytest.importorskip("torch")


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
