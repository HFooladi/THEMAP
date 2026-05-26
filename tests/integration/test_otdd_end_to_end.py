"""End-to-end integration tests for OTDD.

The existing OTDD tests mock the inner DatasetDistance class, so silent
regressions in the actual vendored OTDD code (e.g. external API renames
like ``geomloss.utils`` -> ``geomloss._legacy.utils``, or numerical
instability in the Gaussian inner-OT eigh) slip through. This module
runs OTDD on tiny synthetic datasets so any future regression in the
real OTDD pipeline gets caught.

The tests skip cleanly when the OTDD extras (``geomloss`` and ``pot``)
are not installed, so they cost nothing in core-only CI.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _have(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _have("geomloss"), reason="geomloss not installed"),
    pytest.mark.skipif(not _have("ot"), reason="POT (pot) not installed"),
]


def _gaussian_dataset(n: int, d: int, mean_shift: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Make a (features, labels) pair with two Gaussian-distributed classes."""
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos
    pos = rng.normal(loc=mean_shift, scale=1.0, size=(n_pos, d)).astype(np.float32)
    neg = rng.normal(loc=-mean_shift, scale=1.0, size=(n_neg, d)).astype(np.float32)
    features = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)], axis=0).astype(np.int32)
    return features, labels


def test_otdd_returns_finite_distance() -> None:
    """OTDD must produce finite, non-trivial distances on well-behaved data.

    Regression guard for the chain of issues we hit in May 2026: geomloss
    API rename masked by a try/except returning inf, and Gaussian inner-OT
    eigh failures on degenerate covariance matrices.
    """
    from themap.distance import compute_dataset_distance_matrix

    src_feat, src_lab = _gaussian_dataset(n=200, d=8, mean_shift=0.0, seed=0)
    tgt_feat, tgt_lab = _gaussian_dataset(n=200, d=8, mean_shift=1.5, seed=1)

    result = compute_dataset_distance_matrix(
        source_features=[src_feat],
        source_labels=[src_lab],
        target_features=[tgt_feat],
        target_labels=[tgt_lab],
        source_ids=["src"],
        target_ids=["tgt"],
        method="otdd",
        device="auto",
        maxsamples=100,
    )
    value = result["tgt"]["src"]
    assert np.isfinite(value), f"OTDD returned non-finite distance: {value!r}"
    assert value > 0, f"OTDD distance should be positive on disjoint Gaussians, got {value!r}"


def test_otdd_handles_high_dim_low_sample_covariance() -> None:
    """OTDD must not crash on under-sampled high-dim data.

    With more feature dims than per-class samples, the empirical covariance
    matrices passed to the Gaussian inner-OT approximation are nearly
    singular. Without ridge regularization in our sqrtm wrapper, eigh
    raises ``_LinAlgError`` and OTDD returns inf. This test exercises the
    regularization path. d=50 with ~50 per-class samples is the realistic
    regime for desc2D / mordred descriptors on small ChEMBL assays.
    """
    from themap.distance import compute_dataset_distance_matrix

    src_feat, src_lab = _gaussian_dataset(n=100, d=50, mean_shift=0.0, seed=0)
    tgt_feat, tgt_lab = _gaussian_dataset(n=100, d=50, mean_shift=0.5, seed=1)

    result = compute_dataset_distance_matrix(
        source_features=[src_feat],
        source_labels=[src_lab],
        target_features=[tgt_feat],
        target_labels=[tgt_lab],
        source_ids=["src"],
        target_ids=["tgt"],
        method="otdd",
        device="auto",
        maxsamples=80,
    )
    value = result["tgt"]["src"]
    assert np.isfinite(value), f"OTDD failed on undersampled high-dim data: {value!r}"
