from __future__ import annotations

from typing import Any

PAPER_COLORS = {
    "tcn": "#0072B2",
    "moment_rbf": "#D55E00",
    "moment_full": "#CC79A7",
    "random_forest": "#009E73",
    "logistic_regression": "#7F7F7F",
    "raw": "#56B4E9",
    "statistical": "#009E73",
    "neutral": "#7F7F7F",
    "black": "#222222",
}


def apply_paper_style(plt: Any) -> None:
    """Apply the shared font, grid, and export defaults used by manuscript figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.0,
            "savefig.dpi": 300,
            "svg.fonttype": "path",
        }
    )
