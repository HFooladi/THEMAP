"""From-scratch functional MAML for binary molecular activity classification.

The inner loop adapts a copy of the meta-parameters with plain SGD using
``torch.func.functional_call`` (no in-place optimizer, no external dependency).
The outer loss is computed on the query set with the *adapted* parameters; because
the adapted parameters are a differentiable function of the meta-parameters,
``meta_loss.backward()`` accumulates the meta-gradient directly onto the model's
parameters, which the :class:`MetaTrainer` then steps with Adam.

* ``first_order=True``  → FOMAML (drops the second-order term; cheap, CPU-friendly).
* ``first_order=False`` → full MAML (differentiates through the inner SGD).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .._torch import require_torch
from ..config import EncoderConfig, MAMLConfig
from .encoder import MLPEncoder

torch = require_torch()
nn = torch.nn
F = torch.nn.functional
functional_call = torch.func.functional_call


class MAMLLearner(nn.Module):
    """MAML with an MLP encoder + linear binary head."""

    def __init__(
        self,
        input_dim: int,
        encoder_config: EncoderConfig | None = None,
        maml_config: MAMLConfig | None = None,
    ):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, encoder_config)
        self.head = nn.Linear(self.encoder.embed_dim, 2)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.config = maml_config or MAMLConfig()

    def forward(self, x: Any) -> Any:
        return self.head(self.encoder(x))

    def _init_fast_params(self, detach: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return ``(params, buffers)`` dicts to seed the inner loop.

        For meta-training (``detach=False``) the actual leaf parameters are used so
        the meta-gradient flows back to them. For evaluation (``detach=True``) fresh
        leaves are created so inner-loop adaptation never touches the model state.
        """
        if detach:
            params = {k: v.detach().clone().requires_grad_(True) for k, v in self.named_parameters()}
        else:
            params = {k: v for k, v in self.named_parameters()}
        buffers = {k: v for k, v in self.named_buffers()}
        return params, buffers

    def _adapt(
        self,
        params: Dict[str, Any],
        buffers: Dict[str, Any],
        x_s: Any,
        y_s: Any,
        inner_steps: int,
        first_order: bool,
    ) -> Dict[str, Any]:
        """Run the inner-loop SGD adaptation and return the adapted parameters."""
        fast = dict(params)
        for _ in range(inner_steps):
            logits = functional_call(self, (fast, buffers), (x_s,))
            loss = F.cross_entropy(logits, y_s)
            grads = torch.autograd.grad(
                loss,
                list(fast.values()),
                create_graph=not first_order,
            )
            fast = {
                name: weight - self.config.inner_lr * grad
                for (name, weight), grad in zip(fast.items(), grads)
            }
        return fast

    # --- Unified interface used by MetaTrainer -----------------------------
    def outer_parameters(self) -> Any:
        return self.parameters()

    def compute_episode_loss(self, episode: Any, device: str) -> Tuple[Any, Any, Any]:
        """Adapt on support, return ``(query_loss, query_probs, query_labels)``."""
        ep = episode.to(device)
        params, buffers = self._init_fast_params(detach=False)
        fast = self._adapt(params, buffers, ep.x_s, ep.y_s, self.config.inner_steps, self.config.first_order)
        q_logits = functional_call(self, (fast, buffers), (ep.x_q,))
        loss = F.cross_entropy(q_logits, ep.y_q)
        probs = F.softmax(q_logits, dim=-1)[:, 1].detach()
        return loss, probs, ep.y_q

    def adapt_and_predict(self, x_s: Any, y_s: Any, x_q: Any) -> Any:
        """Evaluation-time adaptation: adapt on support, predict on query.

        Uses ``eval_inner_steps`` and first-order updates; never mutates the model.
        Returns positive-class probabilities for each query molecule.
        """
        params, buffers = self._init_fast_params(detach=True)
        with torch.enable_grad():
            fast = self._adapt(params, buffers, x_s, y_s, self.config.eval_inner_steps, first_order=True)
        with torch.no_grad():
            q_logits = functional_call(self, (fast, buffers), (x_q,))
            return F.softmax(q_logits, dim=-1)[:, 1]
