"""Unit tests for the notebook plotting style helper."""

import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from themap.utils import set_plot_style  # noqa: E402


@pytest.mark.unit
def test_set_plot_style_configures_serif_and_palette():
    palette = set_plot_style()

    assert plt.rcParams["font.family"] == ["serif"]
    serif = list(plt.rcParams["font.serif"])
    # Guaranteed fallbacks are always present at the tail.
    assert "DejaVu Serif" in serif
    assert serif[-1] == "serif"
    # Set2 is an 8-color qualitative palette.
    assert len(palette) == 8


@pytest.mark.unit
def test_set_plot_style_applies_professional_rcparams():
    set_plot_style(font_size=20)

    assert plt.rcParams["font.size"] == 20
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False
    assert plt.rcParams["mathtext.fontset"] == "cm"
    # No LaTeX dependency.
    assert plt.rcParams["text.usetex"] is False


@pytest.mark.unit
def test_set_plot_style_falls_back_without_cmu(monkeypatch):
    """Missing CMU/Computer Modern fonts must not raise; DejaVu remains."""

    class _Font:
        def __init__(self, name):
            self.name = name

    # Simulate a machine with no serif fonts discovered at all.
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [_Font("Arial")])

    set_plot_style()  # should not raise

    serif = list(plt.rcParams["font.serif"])
    assert "CMU Serif" not in serif
    assert "DejaVu Serif" in serif
    assert serif[-1] == "serif"


@pytest.mark.unit
def test_set_plot_style_renders_without_error(tmp_path):
    set_plot_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title(r"correlation $\rho$")
    out = tmp_path / "smoke.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
