"""Shared MLP encoder used by both ProtoNet and MAML."""

from __future__ import annotations

from typing import Any, List

from .._torch import require_torch
from ..config import EncoderConfig

torch = require_torch()
nn = torch.nn


_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class MLPEncoder(nn.Module):
    """Feed-forward encoder mapping feature vectors to embeddings.

    Args:
        input_dim: Dimensionality of the input feature vectors.
        config: Encoder configuration (hidden dims, embed dim, dropout, activation).
    """

    def __init__(self, input_dim: int, config: EncoderConfig | None = None):
        super().__init__()
        config = config or EncoderConfig()
        activation = _ACTIVATIONS[config.activation]

        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(activation())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev = hidden
        layers.append(nn.Linear(prev, config.embed_dim))

        self.net = nn.Sequential(*layers)
        self.embed_dim = config.embed_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Any) -> Any:
        """Encode a batch of feature vectors into embeddings."""
        return self.net(x)
