"""Lazy torch import guard for the meta-learning subpackage.

``torch`` is an optional dependency in THEMAP (installed via the ``ml`` extra).
The meta-learning models require it, but the lightweight utilities
(:mod:`themap.metalearning.selection`, :mod:`themap.metalearning.config`) do
not. Every torch-using module obtains torch through :func:`require_torch` so a
missing install produces a clear, actionable error instead of an opaque
``ModuleNotFoundError`` at an arbitrary call site.
"""

from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    """Return the imported ``torch`` module or raise an informative error.

    Returns:
        The ``torch`` module.

    Raises:
        ImportError: If torch is not installed, with an install hint.
    """
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "Meta-learning requires PyTorch, which is not installed. "
            "Install it with: pip install 'themap[ml]'  (or pip install torch)."
        ) from exc
