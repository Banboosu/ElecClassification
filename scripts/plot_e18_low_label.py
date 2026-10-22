"""Plot E18 unified low-label curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tcn_moment.paper_style import apply_paper_style

ROOT = Path("artifacts/improvements/e18_low_label/e18_low_label_20261020_v1")
OUT = Path("docs/figures")
TAGS = ("1p", "5p", "10p", "20p", "40p")
X = np.asarray([1, 5, 10, 20, 40])
LABELS = {
    "f15_hgb": "F15+HGB",
    "dyn_hgb": "DYN+HGB",
    "moment_logit": "MOMENT orig",
    "diff_logit": "MOMENT ΔP",
    "moment_diff_logit": "MOMENT dual",
    "tcn": "TCN",
    "tcn_dyn": "TCN+DYN",
    "dyn_moment": "DYN+MOMENT",
    "dyn_moment_diff": "DYN+MOMENT/ΔP",
    "full": "Full",
    "multirocket": "MultiRocket",
    "minirocket": "MiniRocket",
}
COLORS = {
    "F15+HGB": "#999999",
    "DYN+HGB": "#0072B2",
    "MOMENT orig": "#CC79A7",
    "MOMENT ΔP": "#E69F00",
    "MOMENT dual": "#D55E00",
    "TCN": "#009E73",
    "TCN+DYN": "#56B4E9",
    "DYN+MOMENT": "#7570B3",
    "DYN+MOMENT/ΔP": "#A6761D",
    "Full": "#000000",
    "MultiRocket": "#B2182B",
    "MiniRocket": "#EF8A62",
}


def load():
    data = json.loads((ROOT / "summary.json").read_text())
    rows = {(row["tag"], row["method"]): row for row in data["aggregate"]}
    return rows


def curve(rows, method):
    return np.asarray([rows[(tag, method)]["mean"] for tag in TAGS]), np.asarray(
        [rows[(tag, method)]["sample_sd"] for tag in TAGS]
    )


def line(ax, rows, method, *, dashed=False):
    mean, sd = curve(rows, method)
    ax.errorbar(
        X,
        mean,
        yerr=sd,
        marker="o",
        markersize=4,
        linewidth=1.8,
        capsize=2,
        color=COLORS[LABELS[method]],
        linestyle="--" if dashed else "-",
        label=LABELS[method],
    )


def main():
    apply_paper_style(plt)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()

    # Main methods (without Rocket baselines)
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for method in (
        "f15_hgb",
        "dyn_hgb",
        "moment_logit",
        "moment_diff_logit",
        "tcn",
        "tcn_dyn",
        "dyn_moment_diff",
        "full",
    ):
        line(ax, rows, method)
    ax.set_xscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels([f"{value}%" for value in X])
    ax.set_xlabel("Training-label budget")
    ax.set_ylabel("Validation Macro-F1 (%)")
    ax.set_ylim(60, 100)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"e18_macro_f1_main.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Rocket comparison
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for method in ("dyn_hgb", "tcn", "tcn_dyn", "full", "multirocket", "minirocket"):
        line(ax, rows, method)
    ax.set_xscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels([f"{value}%" for value in X])
    ax.set_xlabel("Training-label budget")
    ax.set_ylabel("Validation Macro-F1 (%)")
    ax.set_ylim(60, 100)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"e18_macro_f1_rockets.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Difference vs Full
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    full = curve(rows, "full")[0]
    for method in ("multirocket", "minirocket", "tcn", "tcn_dyn", "dyn_hgb"):
        mean = curve(rows, method)[0]
        ax.plot(
            X,
            mean - full,
            marker="o",
            linewidth=1.8,
            color=COLORS[LABELS[method]],
            label=f"{LABELS[method]} − Full",
        )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels([f"{value}%" for value in X])
    ax.set_xlabel("Training-label budget")
    ax.set_ylabel("Δ Macro-F1 (pp)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"e18_delta_vs_full.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Battery recall/FPR
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True)
    for method in ("dyn_hgb", "tcn", "tcn_dyn", "full", "multirocket"):
        recall = np.asarray([rows[(tag, method)]["battery_recall"] for tag in TAGS])
        fpr = np.asarray([rows[(tag, method)]["battery_fpr"] for tag in TAGS])
        axes[0].plot(X, recall, marker="o", linewidth=1.6, color=COLORS[LABELS[method]], label=LABELS[method])
        axes[1].plot(X, fpr, marker="o", linewidth=1.6, color=COLORS[LABELS[method]])
    for axis, ylabel in ((axes[0], "Battery recall (%)"), (axes[1], "Battery FPR (%)")):
        axis.set_xscale("log")
        axis.set_xticks(X)
        axis.set_xticklabels([f"{value}%" for value in X])
        axis.set_xlabel("Training-label budget")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"e18_battery_recall_fpr.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("wrote E18 figures")


if __name__ == "__main__":
    main()
