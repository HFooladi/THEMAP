"""Tests for the from-scratch functional MAML learner."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.metrics import roc_auc_score  # noqa: E402

from themap.metalearning.config import MAMLConfig  # noqa: E402
from themap.metalearning.episodes import EpisodeSampler, TaskFeatures  # noqa: E402
from themap.metalearning.models import MAMLLearner  # noqa: E402


def _separable_sampler(dim=32, seed=0, shift=3.0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(shift, 0.4, (60, dim)).astype(np.float32)
    neg = rng.normal(-shift, 0.4, (60, dim)).astype(np.float32)
    X = np.vstack([pos, neg])
    y = np.array([1] * 60 + [0] * 60)
    return EpisodeSampler([TaskFeatures.from_arrays("t", X, y)], n_support=10, n_query=20, seed=seed)


@pytest.mark.unit
class TestMAML:
    @pytest.mark.parametrize("first_order", [True, False])
    def test_outer_grads_reach_every_param(self, first_order):
        """meta_loss.backward() must populate grads on all meta-parameters."""
        sampler = _separable_sampler()
        m = MAMLLearner(input_dim=32, maml_config=MAMLConfig(inner_steps=3, first_order=first_order))
        loss, _, _ = m.compute_episode_loss(sampler.sample_episode(), "cpu")
        loss.backward()
        missing = [name for name, p in m.named_parameters() if p.grad is None]
        assert missing == []

    def test_inner_loop_reduces_support_loss(self):
        sampler = _separable_sampler()
        m = MAMLLearner(input_dim=32, maml_config=MAMLConfig(inner_steps=5, inner_lr=0.05))
        ep = sampler.sample_episode()
        params, buffers = m._init_fast_params(detach=True)
        before = torch.nn.functional.cross_entropy(
            torch.func.functional_call(m, (params, buffers), (ep.x_s,)), ep.y_s
        )
        with torch.enable_grad():
            fast = m._adapt(params, buffers, ep.x_s, ep.y_s, inner_steps=5, first_order=True)
        after = torch.nn.functional.cross_entropy(
            torch.func.functional_call(m, (fast, buffers), (ep.x_s,)), ep.y_s
        )
        assert float(after) < float(before)

    def test_meta_training_improves_target_auroc(self):
        sampler = _separable_sampler()
        m = MAMLLearner(input_dim=32, maml_config=MAMLConfig(inner_steps=3))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(60):
            loss, _, _ = m.compute_episode_loss(sampler.sample_episode(), "cpu")
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        ep = sampler.sample_episode()
        probs = m.adapt_and_predict(ep.x_s, ep.y_s, ep.x_q).numpy()
        assert roc_auc_score(ep.y_q.numpy(), probs) > 0.9

    def test_adapt_and_predict_does_not_mutate_model(self):
        sampler = _separable_sampler()
        m = MAMLLearner(input_dim=32)
        before = {k: v.detach().clone() for k, v in m.named_parameters()}
        ep = sampler.sample_episode()
        m.adapt_and_predict(ep.x_s, ep.y_s, ep.x_q)
        for k, v in m.named_parameters():
            assert torch.equal(before[k], v)
