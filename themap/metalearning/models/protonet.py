"""Prototypical Network for binary molecular activity classification.

The encoder maps molecules to embeddings; class prototypes are the per-class mean
support embeddings; query molecules are classified by (negative) distance to each
prototype. The episodic loss feeds **raw logits** into ``cross_entropy`` — fixing
the previous implementation that incorrectly applied ``cross_entropy`` to
log-probabilities (a double log-softmax).
"""

from __future__ import annotations

from typing import Any, Tuple

from .._torch import require_torch
from ..config import EncoderConfig, ProtoConfig
from .encoder import MLPEncoder

torch = require_torch()
nn = torch.nn
F = torch.nn.functional


class ProtoNet(nn.Module):
    """Prototypical Network with an MLP encoder (2-way / binary)."""

    def __init__(
        self,
        input_dim: int,
        encoder_config: EncoderConfig | None = None,
        proto_config: ProtoConfig | None = None,
    ):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, encoder_config)
        self.config = proto_config or ProtoConfig()

    def forward(self, x: Any) -> Any:
        return self.encoder(x)

    def compute_prototypes(self, support_emb: Any, support_labels: Any) -> Any:
        """Mean support embedding per class, ordered ``[class 0, class 1]``.

        Uses explicit class indices (not ``torch.unique``) so the output is always
        a ``(2, embed_dim)`` tensor in a fixed class order even if a class is
        missing from the support set (in which case its prototype is all-zeros).
        """
        protos = []
        for cls in (0, 1):
            mask = support_labels == cls
            if mask.any():
                protos.append(support_emb[mask].mean(dim=0))
            else:
                protos.append(torch.zeros(support_emb.shape[1], device=support_emb.device))
        return torch.stack(protos, dim=0)

    def compute_logits(self, query_emb: Any, prototypes: Any) -> Any:
        """Logits of shape ``(n_query, 2)`` from query embeddings and prototypes."""
        if self.config.distance_metric == "cosine":
            q = F.normalize(query_emb, dim=-1)
            p = F.normalize(prototypes, dim=-1)
            sims = q @ p.t()
            return sims / self.config.temperature
        # Squared euclidean distance -> negative distance as logit.
        dists = torch.cdist(query_emb, prototypes) ** 2
        return -dists / self.config.temperature

    def episodic_forward(self, x_s: Any, y_s: Any, x_q: Any) -> Any:
        """Return query logits for a single episode."""
        support_emb = self.encoder(x_s)
        query_emb = self.encoder(x_q)
        prototypes = self.compute_prototypes(support_emb, y_s)
        return self.compute_logits(query_emb, prototypes)

    def episodic_loss(self, x_s: Any, y_s: Any, x_q: Any, y_q: Any) -> Tuple[Any, Any]:
        """Cross-entropy loss on raw query logits plus the logits themselves."""
        logits = self.episodic_forward(x_s, y_s, x_q)
        loss = F.cross_entropy(logits, y_q)
        return loss, logits

    @torch.no_grad()
    def predict_proba(self, x_s: Any, y_s: Any, x_q: Any) -> Any:
        """Positive-class probability for each query molecule."""
        logits = self.episodic_forward(x_s, y_s, x_q)
        return F.softmax(logits, dim=-1)[:, 1]

    # --- Unified interface used by MetaTrainer -----------------------------
    def outer_parameters(self) -> Any:
        return self.parameters()

    def compute_episode_loss(self, episode: Any, device: str) -> Tuple[Any, Any, Any]:
        """Return ``(loss, query_probs, query_labels)`` for one episode."""
        ep = episode.to(device)
        loss, logits = self.episodic_loss(ep.x_s, ep.y_s, ep.x_q, ep.y_q)
        probs = F.softmax(logits, dim=-1)[:, 1].detach()
        return loss, probs, ep.y_q

    @torch.no_grad()
    def adapt_and_predict(self, x_s: Any, y_s: Any, x_q: Any) -> Any:
        """Evaluation-time prediction: prototypes from support, classify query."""
        return self.predict_proba(x_s, y_s, x_q)
