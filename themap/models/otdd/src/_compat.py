"""Compatibility shims for OTDD.

The vendored OTDD reference implementation calls ``geomloss.utils.distances``
and ``geomloss.utils.squared_distances``. As of geomloss 0.3.x these helpers
were moved to ``geomloss._legacy.utils``, so ``geomloss.utils`` raises
``AttributeError`` and OTDD silently fails (the caller catches the exception
and returns ``inf``).

We define drop-in replacements here in pure torch so that OTDD does not
depend on geomloss's private layout. The semantics match geomloss's helpers
with ``use_keops=False``.
"""

from __future__ import annotations

import torch


def squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances.

    Returns an ``(N, M)`` tensor (or ``(B, N, M)`` for batched inputs) of
    ``||x_i - y_j||^2`` values.
    """
    if x.dim() == 2:
        D_xx = (x * x).sum(-1).unsqueeze(1)  # (N, 1)
        D_xy = torch.matmul(x, y.permute(1, 0))  # (N, M)
        D_yy = (y * y).sum(-1).unsqueeze(0)  # (1, M)
    elif x.dim() == 3:  # batched
        D_xx = (x * x).sum(-1).unsqueeze(2)  # (B, N, 1)
        D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B, N, M)
        D_yy = (y * y).sum(-1).unsqueeze(1)  # (B, 1, M)
    else:
        raise ValueError(f"Expected 2D or 3D tensors, got {x.dim()}D")
    return D_xx - 2 * D_xy + D_yy


def distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances. Matches ``geomloss._legacy.utils.distances``."""
    return torch.sqrt(torch.clamp_min(squared_distances(x, y), 1e-8))
