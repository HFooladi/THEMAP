"""Integration tests for featurizer compatibility with molfeat.

Tests that all featurizers in AVAILABLE_FEATURIZERS produce valid output
on real SMILES molecules.
"""

import numpy as np
import pytest

from themap.utils.featurizer_utils import (
    AVAILABLE_FEATURIZERS,
    COUNT_FINGERPRINT_FEATURIZERS,
    DESCRIPTOR_FEATURIZERS,
    DGL_FEATURIZERS,
    FINGERPRINT_FEATURIZERS,
    HF_FEATURIZERS,
    get_featurizer,
)

TEST_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "c1ccccc1",  # benzene
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "c1ccc2ccccc2c1",  # naphthalene
]

# Fast featurizers (fingerprints + count fingerprints + descriptors)
FAST_FEATURIZERS = FINGERPRINT_FEATURIZERS + COUNT_FINGERPRINT_FEATURIZERS + DESCRIPTOR_FEATURIZERS


@pytest.mark.integration
@pytest.mark.parametrize("featurizer_name", FAST_FEATURIZERS)
def test_fast_featurizer(featurizer_name: str) -> None:
    """Test that fast featurizers (FP/descriptors) produce valid output."""
    transformer = get_featurizer(featurizer_name, n_jobs=1)
    result = transformer(TEST_SMILES)
    arr = np.array(result, dtype=np.float32)

    # Should have one row per molecule
    assert arr.shape[0] == len(TEST_SMILES), f"Expected {len(TEST_SMILES)} rows, got {arr.shape[0]}"
    # Should have at least 1 feature dimension
    assert arr.shape[1] > 0, "Feature dimension should be > 0"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("featurizer_name", HF_FEATURIZERS)
def test_hf_featurizer(featurizer_name: str) -> None:
    """Test that HuggingFace featurizers produce valid output."""
    transformer = get_featurizer(featurizer_name, n_jobs=1)
    result = transformer(TEST_SMILES)
    arr = np.array(result, dtype=np.float32)

    assert arr.shape[0] == len(TEST_SMILES)
    assert arr.shape[1] > 0
    assert not np.all(np.isnan(arr)), "All values are NaN"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("featurizer_name", DGL_FEATURIZERS)
def test_dgl_featurizer(featurizer_name: str) -> None:
    """Test that DGL featurizers produce valid output."""
    transformer = get_featurizer(featurizer_name, n_jobs=1)
    result = transformer(TEST_SMILES)
    arr = np.array(result, dtype=np.float32)

    assert arr.shape[0] == len(TEST_SMILES)
    assert arr.shape[1] > 0
    assert not np.all(np.isnan(arr)), "All values are NaN"


@pytest.mark.integration
def test_all_featurizers_in_available_list() -> None:
    """Verify every featurizer in AVAILABLE_FEATURIZERS can be routed by get_featurizer."""
    for name in AVAILABLE_FEATURIZERS:
        transformer = get_featurizer(name, n_jobs=1)
        assert transformer is not None, f"get_featurizer returned None for '{name}'"
