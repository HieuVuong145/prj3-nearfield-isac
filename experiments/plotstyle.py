"""Shared figure style: fixed colour per algorithm (identity never depends on rank/order)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots of the validated reference palette, assigned once per entity.
ALGO_COLOR = {
    "P1": "#2a78d6",
    "B1-r": "#eb6834",
    "B1": "#1baf7a",
    "B3": "#eda100",
    "SBL": "#e87ba4",
    "SOMP": "#008300",
    "B0": "#6250d6",
    "B2": "#e34948",
}
ALGO_MARKER = {"P1": "o", "B1-r": "s", "B1": "^", "B3": "D", "SBL": "v", "SOMP": "P", "B0": "X", "B2": "*"}
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
# Sequential single-hue (blue) ramp, light -> dark, for heatmaps.
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.markersize": 7, "legend.frameon": False, "font.size": 10, "figure.dpi": 150,
    })


def algo_line(ax, x, y, name: str, **kw):
    return ax.plot(x, y, color=ALGO_COLOR[name], marker=ALGO_MARKER[name], label=name,
                   markeredgecolor=SURFACE, markeredgewidth=1.0, **kw)


def ref_line(ax, x, y, label: str, style: str = "--"):
    """Reference curves (CRB, grid floor) in ink, not in a series colour."""
    return ax.plot(x, y, style, color=INK if style == "--" else INK2, lw=1.5, label=label)
