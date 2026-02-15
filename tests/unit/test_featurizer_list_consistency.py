"""Unit tests for featurizer list consistency across the codebase.

Ensures that all files referencing featurizer lists stay in sync
via the single source of truth in themap/utils/featurizer_utils.py.
"""

import pytest

from themap.utils.featurizer_utils import (
    AVAILABLE_FEATURIZERS,
    COUNT_FINGERPRINT_FEATURIZERS,
    DESCRIPTOR_FEATURIZERS,
    DGL_FEATURIZERS,
    FINGERPRINT_FEATURIZERS,
    HF_FEATURIZERS,
    NEURAL_FEATURIZERS,
)


@pytest.mark.unit
class TestFeaturizerListConsistency:
    """Tests for featurizer list consistency."""

    def test_categories_form_partition(self) -> None:
        """Category lists should partition AVAILABLE_FEATURIZERS with no overlaps."""
        all_from_categories = (
            FINGERPRINT_FEATURIZERS
            + COUNT_FINGERPRINT_FEATURIZERS
            + DESCRIPTOR_FEATURIZERS
            + NEURAL_FEATURIZERS
        )
        assert sorted(all_from_categories) == sorted(AVAILABLE_FEATURIZERS)

    def test_no_duplicates_in_available(self) -> None:
        """AVAILABLE_FEATURIZERS should have no duplicates."""
        assert len(AVAILABLE_FEATURIZERS) == len(set(AVAILABLE_FEATURIZERS))

    def test_no_duplicates_in_categories(self) -> None:
        """Each category list should have no duplicates."""
        for name, lst in [
            ("FINGERPRINT_FEATURIZERS", FINGERPRINT_FEATURIZERS),
            ("COUNT_FINGERPRINT_FEATURIZERS", COUNT_FINGERPRINT_FEATURIZERS),
            ("DESCRIPTOR_FEATURIZERS", DESCRIPTOR_FEATURIZERS),
            ("HF_FEATURIZERS", HF_FEATURIZERS),
            ("DGL_FEATURIZERS", DGL_FEATURIZERS),
        ]:
            assert len(lst) == len(set(lst)), f"Duplicates found in {name}"

    def test_no_overlap_between_categories(self) -> None:
        """No featurizer should appear in more than one category."""
        categories = [
            FINGERPRINT_FEATURIZERS,
            COUNT_FINGERPRINT_FEATURIZERS,
            DESCRIPTOR_FEATURIZERS,
            HF_FEATURIZERS,
            DGL_FEATURIZERS,
        ]
        all_items = []
        for cat in categories:
            all_items.extend(cat)
        assert len(all_items) == len(set(all_items)), "Overlap detected between category lists"

    def test_neural_is_hf_plus_dgl(self) -> None:
        """NEURAL_FEATURIZERS should be HF_FEATURIZERS + DGL_FEATURIZERS."""
        assert NEURAL_FEATURIZERS == HF_FEATURIZERS + DGL_FEATURIZERS

    def test_molecule_py_imports_from_featurizer_utils(self) -> None:
        """themap/features/molecule.py should use the same list as featurizer_utils."""
        from themap.features.molecule import MOLECULE_FEATURIZERS

        assert MOLECULE_FEATURIZERS is AVAILABLE_FEATURIZERS

    def test_config_py_imports_from_featurizer_utils(self) -> None:
        """themap/config.py should use the same list as featurizer_utils."""
        from themap.config import MOLECULE_FEATURIZERS

        assert MOLECULE_FEATURIZERS is AVAILABLE_FEATURIZERS

    def test_features_init_exports(self) -> None:
        """themap/features/__init__.py should export key constants."""
        from themap.features import (
            COUNT_FINGERPRINT_FEATURIZERS as exported_count,
        )
        from themap.features import (
            DESCRIPTOR_FEATURIZERS as exported_desc,
        )
        from themap.features import (
            FINGERPRINT_FEATURIZERS as exported_fp,
        )
        from themap.features import (
            MOLECULE_FEATURIZERS as exported_mol,
        )
        from themap.features import (
            NEURAL_FEATURIZERS as exported_neural,
        )

        assert exported_mol is AVAILABLE_FEATURIZERS
        assert exported_fp is FINGERPRINT_FEATURIZERS
        assert exported_count is COUNT_FINGERPRINT_FEATURIZERS
        assert exported_desc is DESCRIPTOR_FEATURIZERS
        assert exported_neural is NEURAL_FEATURIZERS

    def test_get_featurizer_routes_all(self) -> None:
        """get_featurizer() should accept every name in AVAILABLE_FEATURIZERS."""
        from themap.utils.featurizer_utils import get_featurizer

        for name in AVAILABLE_FEATURIZERS:
            transformer = get_featurizer(name, n_jobs=1)
            assert transformer is not None
