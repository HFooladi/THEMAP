"""Notebook plotting style helpers for THEMAP.

Provides a single entry point, :func:`set_plot_style`, that configures
matplotlib + seaborn with a professional, publication-friendly look: the
seaborn ``Set2`` qualitative palette and the *CMU Serif* font family (the
Computer Modern look), with graceful fallback to the default serif font when
CMU Serif is not installed (for example on Google Colab).

``matplotlib`` and ``seaborn`` are optional dependencies, so they are imported
lazily inside :func:`set_plot_style`. Importing this module (and therefore
``themap.utils``) never requires a plotting stack.
"""

from __future__ import annotations

# Preferred serif families, in order. ``matplotlib`` silently falls through to
# the next entry whenever a font is missing, so listing several keeps things
# error-free on machines without CMU Serif / Computer Modern installed.
# ``DejaVu Serif`` ships with matplotlib and is the guaranteed final fallback.
_SERIF_PREFERENCES = (
    "CMU Serif",
    "Computer Modern Roman",
    "Latin Modern Roman",
    "DejaVu Serif",
)


def set_plot_style(
    palette: str = "Set2",
    style: str = "whitegrid",
    font_size: float = 12,
    font_scale: float = 1.0,
) -> list[tuple[float, float, float]]:
    """Apply a professional matplotlib/seaborn style for THEMAP notebooks.

    Configures the seaborn ``Set2`` palette and a CMU Serif (Computer Modern)
    font family. If CMU Serif is not available, matplotlib gracefully falls
    back through ``Computer Modern Roman``, ``Latin Modern Roman`` and finally
    the always-present ``DejaVu Serif`` — so calling this never raises on a
    machine that lacks the preferred fonts.

    Math text (``$\\rho$``, ``$\\Delta$``, ...) is rendered with the Computer
    Modern math fontset, which does not require a LaTeX installation.

    Args:
        palette: Seaborn palette name applied as the default color cycle.
        style: Seaborn base style (e.g. ``"whitegrid"``, ``"ticks"``).
        font_size: Base font size in points.
        font_scale: Seaborn font scaling factor applied on top of ``font_size``.

    Returns:
        The resolved color palette as a list of RGB tuples, handy for assigning
        explicit colors in a plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    serif_stack = [f for f in _SERIF_PREFERENCES if f in available]
    if "DejaVu Serif" not in serif_stack:
        serif_stack.append("DejaVu Serif")
    serif_stack.append("serif")

    # seaborn theme first — it resets font.family to sans-serif, so the
    # rcParams override below must come after it.
    sns.set_theme(style=style, palette=palette, font_scale=font_scale)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_stack,
            "font.size": font_size,
            "mathtext.fontset": "cm",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )
    return sns.color_palette(palette)
