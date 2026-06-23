"""Tests for the Prototypical Network."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.metrics import roc_auc_score  # noqa: E402

from themap.metalearning.episodes import EpisodeSampler, TaskFeatures  # noqa: E402
from themap.metalearning.models import ProtoNet  # noqa: E402


def _separable_sampler(dim=32, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(3.0, 0.4, (60, dim)).astype(np.float32)
    neg = rng.normal(-3.0, 0.4, (60, dim)).astype(np.float32)
    X = np.vstack([pos, neg])
    y = np.array([1] * 60 + [0] * 60)
    return EpisodeSampler([TaskFeatures.from_arrays("t", X, y)], n_support=10, n_query=20, seed=seed)


@pytest.mark.unit
class TestProtoNet:
    def test_prototype_shape(self):
        net = ProtoNet(input_dim=16)
        emb = torch.randn(8, net.encoder.embed_dim)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        protos = net.compute_prototypes(emb, labels)
        assert protos.shape == (2, net.encoder.embed_dim)

    def test_identical_embeddings_loss_is_log2(self):
        # If the encoder is the identity and all inputs are equal, prototypes
        # collapse and softmax is uniform -> CE == log(2). This guards against
        # the historical bug of feeding log-probs into cross_entropy.
        dim = 8

        class IdProto(ProtoNet):
            def forward(self, x):
                return x

        net = IdProto(input_dim=dim)
        z = torch.zeros(4, dim)
        loss, _ = net.episodic_loss(z, torch.tensor([0, 0, 1, 1]), z, torch.tensor([0, 1, 0, 1]))
        assert math.isclose(float(loss), math.log(2), rel_tol=1e-4)

    def test_learns_separable_task(self):
        sampler = _separable_sampler()
        net = ProtoNet(input_dim=32)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(50):
            ep = sampler.sample_episode()
            loss, _ = net.episodic_loss(ep.x_s, ep.y_s, ep.x_q, ep.y_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        ep = sampler.sample_episode()
        probs = net.predict_proba(ep.x_s, ep.y_s, ep.x_q).numpy()
        assert roc_auc_score(ep.y_q.numpy(), probs) > 0.9

    def test_compute_episode_loss_interface(self):
        sampler = _separable_sampler()
        net = ProtoNet(input_dim=32)
        loss, probs, labels = net.compute_episode_loss(sampler.sample_episode(), "cpu")
        assert loss.requires_grad
        assert probs.shape == labels.shape
