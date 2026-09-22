"""Regression tests for optional molfeat transformer imports.

molfeat 1.0.0 removed ``GraphormerTransformer`` and ``PretrainedDGLTransformer``
(the ``graphormer`` and ``dgl_pretrained`` submodules were dropped when molfeat
narrowed its scope to small-molecule featurization). Those classes must only be
imported by the branch that actually needs them -- importing them eagerly takes
every other featurizer down with them, so ``ecfp`` would fail on an ImportError
for a Graphormer class it never uses.

See https://github.com/datamol-io/molfeat/pull/124.
"""

import pytest

from themap.utils.featurizer_utils import get_featurizer

# Names removed in molfeat 1.0.0, and the featurizer that must keep working anyway.
REMOVED_IN_MOLFEAT_1 = ("GraphormerTransformer", "PretrainedDGLTransformer")


@pytest.fixture
def molfeat_without_deep_transformers(monkeypatch):
    """Make molfeat look like 1.0.0, where the deep-learning transformers are gone."""
    pretrained = pytest.importorskip("molfeat.trans.pretrained")
    for name in REMOVED_IN_MOLFEAT_1:
        monkeypatch.delattr(pretrained, name, raising=False)
    return pretrained


@pytest.mark.unit
class TestOptionalMolfeatTransformers:
    """Featurizers that don't need the removed classes must not import them."""

    @pytest.mark.parametrize("featurizer", ["ecfp", "maccs", "desc2D"])
    def test_fingerprints_work_without_removed_classes(
        self, featurizer, molfeat_without_deep_transformers
    ) -> None:
        """A fingerprint/descriptor featurizer must not touch Graphormer or DGL."""
        transformer = get_featurizer(featurizer, n_jobs=1)
        assert transformer is not None

    def test_dgl_featurizer_reports_missing_support(self, molfeat_without_deep_transformers) -> None:
        """A DGL featurizer should fail with a message naming molfeat, not a bare ImportError."""
        with pytest.raises(ImportError, match="molfeat"):
            get_featurizer("gin_supervised_infomax", n_jobs=1)

    def test_unknown_featurizer_still_raises_value_error(self, molfeat_without_deep_transformers) -> None:
        """Routing for unknown names must be unaffected by the import change."""
        with pytest.raises(ValueError, match="not found"):
            get_featurizer("definitely-not-a-featurizer", n_jobs=1)
