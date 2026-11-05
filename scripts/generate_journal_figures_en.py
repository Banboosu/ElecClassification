"""Generate the English journal figures (vector PDF + 300-dpi PNG) for manuscript_iop_en.tex.

The wording and layout follow the supervisor's English draft (docs/导师英文稿.pdf);
the data sources and plotting conventions are those of
``generate_journal_figures.py`` (the Chinese review draft), so both language
versions show the same numbers. Differences from the supervisor's artwork:

* figures are vector PDFs drawn at the printed width instead of raster images;
* Fig. 7(a) spells out every contrast and keeps its legend outside the data
  area, and Fig. 7(b) labels the 10% and 40% ticks and adds MultiRocket
  (todo1102 items 2d/6d);
* Fig. 5(b) also identifies the three input streams in its legend;
* Fig. 6(b) uses the group names of the DYN feature table.

Run from the project root:  PYTHONPATH=src python scripts/generate_journal_figures_en.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_journal_figures import (
    FIG_W,
    GRID,
    INK,
    LIGHT,
    MID,
    PANEL_H,
    load_json,
    panel_label,
    smooth,
)

OUT_DIRS = [ROOT / "docs" / "ioplatextemplate" / "figures_en", ROOT / "docs" / "figures" / "en"]
LABEL_NAMES_EN = {"0": "Normal", "1": "Battery anomaly", "2": "Charger fault"}
BUDGET_XLABEL = "Training label fraction (log scale)"
VAL_F1 = "Validation Macro-F1 (%)"


def setup_english_journal_style() -> None:
    """Print-ready rcParams with Arial text and matching math glyphs."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.unicode_minus": True,
            "font.size": 9.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 7.6,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.handletextpad": 0.6,
            "legend.labelspacing": 0.32,
            "legend.borderaxespad": 0.3,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def tidy_en(ax: plt.Axes) -> None:
    # Unlike the shared tidy(), keep per-axis tick label sizes set by the caller.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_en(fig: plt.Figure, stem: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(d / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
        print(f"Saved {d / stem}.pdf and .png")
    plt.close(fig)


def log_budget_axis(ax: plt.Axes, ticks: list[int], xlim: tuple[float, float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{v}%" for v in ticks])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=7.8, pad=4.0)
    ax.set_xlim(*xlim)
    ax.set_xlabel(BUDGET_XLABEL)


# --------------------------------------------------------------------------- #
# Fig. 1 - dataset overview
# --------------------------------------------------------------------------- #
def plot_fig1_dataset_overview() -> None:
    setup_english_journal_style()
    split_candidates = [
        ROOT / "artifacts/improvements/e01/e01_20261017_v2_pilot_seed42/split_manifest.json",
        ROOT
        / "artifacts/imports/e01_validation_20261017/runs/e01_20261017_v2_pilot_seed42/split_manifest.json",
    ]
    split_path = next((path for path in split_candidates if path.exists()), None)
    if split_path is None:
        raise FileNotFoundError("Formal seed42 split manifest required for class examples")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split["protocol"]["random_state"] != 42:
        raise ValueError("Dataset example figures require formal seed42")
    train_ids = set(split["splits"]["train"])
    if len(train_ids) != 24569:
        raise ValueError("Unexpected training split size")
    by_class: dict[str, list[list[float]]] = {"0": [], "1": [], "2": []}
    csv.field_size_limit(10**8)
    with (ROOT / "data/raw/最新多.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=" "):
            if row["charging_station_id"] not in train_ids:
                continue
            values = [
                float(v)
                for v in row["charging_powers_str"].replace('"', "").split(",")
                if v.strip()
            ]
            by_class[str(row["InsertedColumn"]).strip()].append(values)

    fig = plt.figure(figsize=(FIG_W, 4.35))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.55, wspace=0.30)

    # Same seed and draw order as the Chinese figure, so the examples match.
    rng = np.random.default_rng(42)
    sample_styles = [(INK, "-", "Sample 1"), (MID, "--", "Sample 2"), (LIGHT, "-.", "Sample 3")]
    for col, cls in enumerate(("0", "1", "2")):
        ax = fig.add_subplot(gs[0, col])
        seqs = by_class[cls]
        chosen = rng.choice(len(seqs), size=3, replace=False)
        for idx, (sc, sls, slabel) in zip(chosen, sample_styles):
            ax.plot(
                np.arange(len(seqs[idx])),
                seqs[idx],
                color=sc,
                linestyle=sls,
                linewidth=1.0,
                label=slabel,
            )
        ax.set_xlabel("Sample index")
        ax.set_title(f"({'abc'[col]}) {LABEL_NAMES_EN[cls]}", loc="left", pad=4.0)
        if col == 0:
            ax.set_ylabel("Charging power (recorded units)")
        tidy_en(ax)
        if col == 2:
            ax.legend(loc="upper right", fontsize=6.8)

    ax = fig.add_subplot(gs[1, :])
    class_styles = {"0": (INK, "-", "o"), "1": (MID, "--", "s"), "2": (LIGHT, "-.", "^")}
    for cls in ("0", "1", "2"):
        lengths = np.sort(np.asarray([len(s) for s in by_class[cls]]))
        ecdf = np.arange(1, len(lengths) + 1) / len(lengths) * 100.0
        c_col, c_ls, c_mk = class_styles[cls]
        med = int(np.median(lengths))
        ax.plot(
            lengths,
            ecdf,
            linewidth=1.3,
            color=c_col,
            linestyle=c_ls,
            marker=c_mk,
            markevery=max(1, len(lengths) // 8),
            markersize=3.4,
            markerfacecolor="white" if c_ls != "-" else c_col,
            markeredgecolor=c_col,
            markeredgewidth=0.9,
            label=f"{LABEL_NAMES_EN[cls]} (median {med})",
        )
        ax.axvline(med, color=c_col, linestyle=":", linewidth=0.8, alpha=0.8)

    ax.axhline(50.0, color=LIGHT, linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 830)
    ax.set_ylim(0, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Valid sequence length $L$ (samples)")
    ax.set_ylabel("Cumulative probability (%)")
    ax.legend(loc="lower right", ncol=1)
    ax.set_title("(d) Empirical CDFs of valid sequence length", loc="left", pad=5.0)
    tidy_en(ax)
    save_en(fig, "fig1_dataset_overview")


# --------------------------------------------------------------------------- #
# Fig. 2 - overall framework (layout of the supervisor's diagram)
# --------------------------------------------------------------------------- #
def plot_fig2_overall_framework() -> None:
    setup_english_journal_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    w, h = 6.2, 3.32
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    edge = "#3A3A3A"
    lw_box = 1.3

    def box(x0: float, y0: float, x1: float, y1: float, text: str, size: float = 10.5) -> None:
        ax.add_patch(
            Rectangle(
                (x0, y0), x1 - x0, y1 - y0, facecolor="white", edgecolor=edge, linewidth=lw_box
            )
        )
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight="bold",
            color=INK,
            linespacing=1.15,
        )

    def container(y0: float, y1: float) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (1.205, y0),
                3.49,
                y1 - y0,
                boxstyle="round,pad=0,rounding_size=0.11",
                facecolor="white",
                edgecolor=INK,
                linewidth=1.5,
            )
        )

    def arrow(p0: tuple[float, float], p1: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.3,
                color=INK,
                shrinkA=0,
                shrinkB=0,
            )
        )

    def prob(x: float, y: float, sup: str) -> None:
        ax.text(
            x,
            y,
            rf"$\mathbfit{{p}}^{{\mathbfit{{{sup}}}}}$",
            ha="left",
            va="center",
            fontsize=14,
            color=INK,
        )

    # Branch containers (top to bottom: MOMENT dual stream, dynamics, TCN).
    yI, yII, yIII = (2.58, 3.28), (1.57, 2.27), (0.54, 1.24)
    for y0, y1 in (yI, yII, yIII):
        container(y0, y1)
    cI, cIII = sum(yI) / 2, sum(yIII) / 2

    # Input and the trunk that feeds the three branches.
    box(0.03, 1.62, 0.81, 2.24, "", size=10)
    ax.text(
        0.42, 2.04, "Input", ha="center", va="center", fontsize=15, fontweight="bold", color=INK
    )
    ax.text(
        0.42,
        1.78,
        r"$\mathbfit{x}_i\quad\mathbfit{m}_i$",
        ha="center",
        va="center",
        fontsize=15,
        color=INK,
    )
    ax.plot([0.81, 0.97], [1.93, 1.93], color=INK, lw=1.3)
    ax.plot([0.97, 0.97], [cIII, cI], color=INK, lw=1.3)
    for yc in (cI, 1.93, cIII):
        arrow((0.97, yc), (1.2, yc))

    # Branch I: P and dP share the frozen encoder.
    ax.text(1.43, 3.07, r"$\mathbfit{P}$", ha="center", va="center", fontsize=14, color=INK)
    ax.text(
        1.40,
        2.79,
        r"$\mathbf{\Delta}\mathbfit{P}$",
        ha="center",
        va="center",
        fontsize=14,
        color=INK,
    )
    arrow((1.53, 3.07), (1.77, 3.07))
    arrow((1.55, 2.79), (1.77, 2.79))
    box(1.77, 2.67, 2.72, 3.19, "Frozen\nMOMENT")
    arrow((2.72, 2.93), (2.89, 2.93))
    box(2.89, 2.67, 3.89, 3.19, "Mask-aware\npooling")
    arrow((3.89, 2.93), (4.08, 2.93))
    prob(4.12, 2.95, "MD")

    # Branch II: raw-scale dynamics features + HGB.
    box(1.67, 1.66, 2.62, 2.18, "60-D\ndynamics")
    arrow((2.62, 1.92), (2.79, 1.92))
    box(2.79, 1.66, 3.59, 2.18, "HGB")
    arrow((3.59, 1.92), (3.84, 1.92))
    prob(3.88, 1.94, "DYN")

    # Branch III: masked TCN.
    box(2.27, 0.62, 3.22, 1.16, "Masked\nTCN")
    arrow((3.22, 0.89), (3.42, 0.89))
    prob(3.46, 0.91, "TCN")

    # Probability bus into the cross-fitted stacker.
    xb = 4.85
    ax.plot([4.695, xb], [cI, cI], color=INK, lw=1.3)
    ax.plot([4.695, xb], [cIII, cIII], color=INK, lw=1.3)
    ax.plot([xb, xb], [cIII, cI], color=INK, lw=1.3)
    arrow((4.695, 1.92), (5.09, 1.92))

    box(5.09, 1.66, 6.15, 2.18, "5-fold OOF\nstacking")
    arrow((5.62, 1.66), (5.62, 1.42))
    box(5.09, 0.88, 6.15, 1.42, "Logistic\nmeta-learner")
    arrow((5.62, 0.88), (5.62, 0.64))
    box(5.09, 0.04, 6.15, 0.64, "3-class\noutput", size=11)

    save_en(fig, "fig2_overall_framework")


# --------------------------------------------------------------------------- #
# Fig. 3 - pooling repair and frozen-head convergence
# --------------------------------------------------------------------------- #
def plot_fig3_pooling_and_head_stabilization() -> None:
    setup_english_journal_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H + 0.1))

    stages = [
        "Default\npooling",
        "Masked\npooling",
        "Unfreeze\n2 layers",
        "RBF-SVM\nhead",
        "Full\nfine-tune",
    ]
    values = np.array([69.46, 77.41, 81.50, 85.17, 95.37])
    sds = np.array([0.76, 0.89, 1.06, 1.03, 0.39])
    x = np.arange(len(stages))
    ax1.axhline(95.99, color=LIGHT, linestyle="--", linewidth=1.0, zorder=1)
    ax1.axhline(88.89, color=LIGHT, linestyle=":", linewidth=1.0, zorder=1)
    ax1.errorbar(
        x,
        values,
        yerr=sds,
        color=INK,
        marker="o",
        markerfacecolor=INK,
        markeredgecolor=INK,
        linewidth=1.6,
        markersize=4.6,
        capsize=2.5,
        elinewidth=0.8,
        zorder=3,
    )
    ax1.set_xticks(x, stages)
    ax1.tick_params(axis="x", labelsize=7.0)
    ax1.set_xlim(-0.55, 4.55)
    ax1.set_ylim(64, 100)
    ax1.set_xlabel("Adaptation stage (full labels)")
    ax1.set_ylabel("Test Macro-F1 (%)")
    ax1.grid(axis="x", visible=False)
    handles = [
        Line2D(
            [], [], color=INK, marker="o", linewidth=1.6, markersize=4.6, label="MOMENT-1-large"
        ),
        Line2D([], [], color=LIGHT, linestyle="--", linewidth=1.0, label="TCN baseline 95.99"),
        Line2D([], [], color=LIGHT, linestyle=":", linewidth=1.0, label="F15 + RF 88.89"),
    ]
    ax1.legend(
        handles=handles,
        loc="lower right",
        fontsize=7.0,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
    )
    panel_label(ax1, "(a)")

    root_e03d = ROOT / "artifacts/imports/e03d_head_validation_20261018"
    specs = [
        ("fusion", "fanin_gain", INK, "-", "o"),
        ("fusion", "legacy_fresh", INK, "--", "o"),
        ("statistics", "fanin_gain", MID, "-", "s"),
        ("statistics", "legacy_fresh", MID, "--", "s"),
        ("moment", "fanin_gain", LIGHT, "-", "^"),
        ("moment", "legacy_fresh", LIGHT, "--", "^"),
    ]
    for branch, mode, color, ls, marker in specs:
        hist_path = root_e03d / f"e03d_head_diagnostic_20261018_v2_{branch}_{mode}/history.json"
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        epochs = np.array([r["epoch"] for r in history])
        f1s = smooth([r["validation_macro_f1"] * 100 for r in history], 3)
        ax2.plot(
            epochs,
            f1s,
            color=color,
            linestyle=ls,
            linewidth=1.3 if mode == "fanin_gain" else 1.0,
            marker=marker,
            markevery=40,
            markersize=3.6,
            markerfacecolor="white" if ls == "--" else color,
            markeredgecolor=color,
            markeredgewidth=0.9,
        )
    ax2.set_xlim(1, 200)
    ax2.set_ylim(71, 103.5)
    ax2.set_yticks([75, 80, 85, 90, 95])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(VAL_F1)
    branch_handles = [
        Line2D([], [], color=c, marker=m, linestyle="none", markersize=4.2, label=lab)
        for c, m, lab in ((INK, "o", "Fusion"), (MID, "s", "Statistics"), (LIGHT, "^", "MOMENT"))
    ]
    style_handles = [
        Line2D([], [], color=INK, linestyle="-", linewidth=1.3, label="Scale compensation"),
        Line2D([], [], color=INK, linestyle="--", linewidth=1.0, label="Original scale"),
    ]
    leg = ax2.legend(
        handles=branch_handles,
        loc="upper left",
        ncol=3,
        fontsize=6.8,
        columnspacing=1.0,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    ax2.add_artist(leg)
    ax2.legend(
        handles=style_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.915),
        ncol=2,
        fontsize=6.8,
        columnspacing=1.0,
        handlelength=2.0,
        borderaxespad=0.2,
    )
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy_en(ax)
    fig.tight_layout(w_pad=1.8)
    save_en(fig, "fig3_pooling_and_head_stabilization")


# --------------------------------------------------------------------------- #
# Fig. 4 - TCN architecture and length strata
# --------------------------------------------------------------------------- #
def plot_fig4_tcn_and_length_strata() -> None:
    setup_english_journal_style()
    e05c = load_json(
        "artifacts/imports/e05c_tcn_replication_20261019/terminal_artifacts/summary.json"
    )
    n2 = load_json(
        "artifacts/imports/n2_tcn_low_label_20261020/n2_tcn_low_label_20261020_v1/summary.json"
    )
    e18_deep = load_json("artifacts/analysis/e18_deep_analysis_semantics_v2/summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H + 0.1))

    agg05 = {(r["condition"], r["fraction"]): r for r in e05c["aggregate"]}
    agg_n2 = {r["fraction"]: r for r in n2["aggregate"]}
    x_10_100 = [10, 40, 100]
    fracs = [0.1, 0.4, 1.0]
    x_full = [1, 5, 10, 40, 100]
    masked_means = [
        agg_n2[0.01]["macro_f1_mean"],
        agg_n2[0.05]["macro_f1_mean"],
        *(agg05[("legacy_masked50", f)]["macro_f1_mean_percent"] for f in fracs),
    ]
    masked_sds = [
        agg_n2[0.01]["macro_f1_sd"],
        agg_n2[0.05]["macro_f1_sd"],
        *(agg05[("legacy_masked50", f)]["macro_f1_sample_sd_percent"] for f in fracs),
    ]
    curves = [
        (x_full, masked_means, masked_sds, "6-layer TCN, masked pooling", INK, "o", "-", INK, 1.6),
        (
            x_10_100,
            [agg05[("legacy_unmasked50", f)]["macro_f1_mean_percent"] for f in fracs],
            [agg05[("legacy_unmasked50", f)]["macro_f1_sample_sd_percent"] for f in fracs],
            "6-layer TCN, mean pooling",
            MID,
            "^",
            "--",
            "white",
            1.1,
        ),
        (
            x_10_100,
            [agg05[("current_fixed50", f)]["macro_f1_mean_percent"] for f in fracs],
            [agg05[("current_fixed50", f)]["macro_f1_sample_sd_percent"] for f in fracs],
            "4-block residual TCN, masked pooling",
            LIGHT,
            "s",
            ":",
            "white",
            1.1,
        ),
    ]
    for xs, means, sds, label, color, marker, ls, mfc, lw in curves:
        ax1.errorbar(
            xs,
            means,
            yerr=sds,
            marker=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle=ls,
            color=color,
            linewidth=lw,
            markersize=4.3,
            capsize=2.0,
            elinewidth=0.8,
            label=label,
        )
    log_budget_axis(ax1, x_full, (0.75, 135))
    ax1.set_ylim(34, 106)
    ax1.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax1.set_ylabel(VAL_F1)
    ax1.legend(loc="lower right", fontsize=6.6, handlelength=2.0, borderaxespad=0.2)
    panel_label(ax1, "(a)")

    strata = ["Short\n($L\\leq266$)", "Medium\n(266\u2013399)", "Long\n($L>399$)"]
    x2 = np.arange(3)
    specs = [
        ("40p", "full", "40% full fusion", INK, "o", "-"),
        ("40p", "tcn", "40% masked TCN", INK, "s", "--"),
        ("10p", "full", "10% full fusion", MID, "D", "-"),
        ("10p", "tcn", "10% masked TCN", MID, "^", "--"),
    ]
    for t_key, m_key, label, color, marker, ls in specs:
        rec = e18_deep["length_strata"][t_key][m_key]
        ax2.errorbar(
            x2,
            rec["mean"],
            yerr=rec["sd"],
            marker=marker,
            markerfacecolor=color if ls == "-" else "white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle=ls,
            color=color,
            linewidth=1.4 if ls == "-" else 1.0,
            markersize=4.2,
            capsize=2.0,
            elinewidth=0.8,
            label=label,
        )
    ax2.set_xticks(x2, strata)
    ax2.set_xlim(-0.3, 2.3)
    ax2.set_ylim(77.5, 101)
    ax2.set_xlabel("Valid-length group")
    ax2.set_ylabel("Group Macro-F1 (%)")
    ax2.legend(
        loc="lower left",
        ncol=2,
        columnspacing=0.8,
        fontsize=7.0,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
    )
    ax2.grid(axis="x", visible=False)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy_en(ax)
    fig.tight_layout(w_pad=1.6)
    save_en(fig, "fig4_tcn_and_length_strata")


# --------------------------------------------------------------------------- #
# Fig. 5 - dual stream and pretraining attribution
# --------------------------------------------------------------------------- #
def plot_fig5_dual_stream_and_pretraining() -> None:
    setup_english_journal_style()
    e09 = load_json(
        "artifacts/imports/e09_analysis_corrected_20261020/analysis_four_seed_corrected/summary.json"
    )
    e16 = load_json(
        "artifacts/imports/e16_random_attribution_20261020/e16_random_attribution_20261020_v1/summary.json"
    )
    e16b = load_json(
        "artifacts/imports/e16b_random_low_label_20261020/e16b_random_low_label_20261020_v1/summary.json"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H + 0.1))

    tags = ["1p", "10p", "40p"]
    x_pct = [1, 10, 40]
    e09_map = {(r["method"], r["tag"]): r for r in e09["aggregate"]}
    streams = [
        ("moment_logit", "Original sequence $P$", LIGHT, "s", ":"),
        ("diff_logit", "Difference stream $\\Delta P$", MID, "^", "--"),
        ("moment_diff_logit", "Dual stream $P+\\Delta P$", INK, "o", "-"),
        ("stack_moment_diff", "Dual stream + DYN stacking", INK, "D", "-"),
    ]
    for key, label, color, marker, ls in streams:
        ax1.errorbar(
            x_pct,
            [e09_map[(key, t)]["mean"] for t in tags],
            yerr=[e09_map[(key, t)]["sample_sd"] for t in tags],
            label=label,
            color=color,
            marker=marker,
            markerfacecolor="white" if ls in (":", "--") else color,
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle=ls,
            linewidth=1.5 if ls == "-" else 1.1,
            markersize=4.4,
            capsize=2.0,
            elinewidth=0.8,
        )
    log_budget_axis(ax1, x_pct, (0.75, 55))
    ax1.set_ylim(65, 100)
    ax1.set_ylabel(VAL_F1)
    ax1.legend(loc="lower right", fontsize=7.0)
    panel_label(ax1, "(a)")

    e16_map = {
        (r["tag"], r["initialization"], r["stream"]): r
        for r in [*e16["aggregate"], *e16b["aggregate"]]
    }
    tags_b = ["1p", "5p", "10p", "20p", "40p"]
    x_b = [1, 5, 10, 20, 40]
    stream_specs = [
        ("dual", "o", "Dual $P+\\Delta P$", INK),
        ("diff", "^", "Difference $\\Delta P$", MID),
        ("orig", "s", "Original $P$", LIGHT),
    ]
    for stream, marker, _, color in stream_specs:
        for init, ls, filled in (("pretrained", "-", True), ("random", "--", False)):
            ax2.errorbar(
                x_b,
                [e16_map[(t, init, stream)]["mean"] for t in tags_b],
                yerr=[e16_map[(t, init, stream)]["sample_sd"] for t in tags_b],
                color=color,
                marker=marker,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                linestyle=ls,
                linewidth=1.4 if filled else 1.0,
                markersize=4.2 if filled else 3.8,
                capsize=1.8,
                elinewidth=0.7,
            )
    log_budget_axis(ax2, x_b, (0.75, 55))
    ax2.set_ylim(42, 106)
    ax2.set_yticks([50, 60, 70, 80, 90])
    ax2.set_ylabel(VAL_F1)
    init_handles = [
        Line2D([], [], color=INK, linestyle="-", label="Pretrained weights"),
        Line2D([], [], color=INK, linestyle="--", label="Random initialization"),
    ]
    stream_handles = [
        Line2D([], [], color=color, marker=marker, linestyle="none", markersize=4.4, label=label)
        for _, marker, label, color in stream_specs
    ]
    leg = ax2.legend(handles=init_handles, loc="lower right", fontsize=7.0, handlelength=2.4)
    ax2.add_artist(leg)
    ax2.legend(
        handles=stream_handles,
        loc="upper center",
        ncol=3,
        fontsize=6.8,
        columnspacing=0.8,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy_en(ax)
    fig.tight_layout(w_pad=1.8)
    save_en(fig, "fig5_dual_stream_and_pretraining")


# --------------------------------------------------------------------------- #
# Fig. 6 - dynamics features
# --------------------------------------------------------------------------- #
def plot_fig6_dyn_feature_ablation() -> None:
    setup_english_journal_style()
    e08 = load_json(
        "artifacts/imports/e08_strong_features_20261020/e08_strong_features_20261020_v1/analysis/summary.json"
    )
    agg = {(r["method"], r["tag"]): r for r in e08["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H + 0.1))

    methods = [
        ("dyn_hgb", "DYN + HGB", INK, "o", "-", ["1p", "5p", "10p", "40p", "100p"]),
        ("dyn_svm", "DYN + RBF-SVM", MID, "s", "-.", ["1p", "10p", "40p"]),
        ("f15_hgb", "F15 + HGB", MID, "^", "--", ["1p", "5p", "10p", "40p", "100p"]),
        ("f15_svm", "F15 + RBF-SVM", LIGHT, "v", ":", ["1p", "10p", "40p"]),
    ]
    for key, label, color, marker, ls, m_tags in methods:
        ax1.errorbar(
            [int(t[:-1]) for t in m_tags],
            [agg[(key, t)]["macro_f1_mean"] for t in m_tags],
            yerr=[agg[(key, t)]["macro_f1_sd"] for t in m_tags],
            label=label,
            color=color,
            marker=marker,
            markerfacecolor="white" if ls in ("--", ":") else color,
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle=ls,
            linewidth=1.5 if key == "dyn_hgb" else 1.1,
            markersize=4.4,
            capsize=2.0,
            elinewidth=0.8,
        )
    log_budget_axis(ax1, [1, 5, 10, 40, 100], (0.75, 135))
    ax1.set_ylim(72, 101)
    ax1.set_ylabel(VAL_F1)
    ax1.legend(loc="lower right", fontsize=7.0)
    panel_label(ax1, "(a)")

    grp = {r["group"]: r for r in e08["group_ablation_40p"]}
    order = [
        ("roughness", "Roughness and\nturning points"),
        ("trend", "Segment trends"),
        ("amplitude", "Amplitude and\nquantiles"),
        ("spectral", "Spectrum and\nautocorrelation"),
        ("shape", "Shape"),
        ("length", "Length"),
    ]
    y = np.arange(len(order))
    only_vals = [grp[g]["only_group_f1"] for g, _ in order]
    drop_vals = [grp[g]["drop_group_f1"] for g, _ in order]
    ax2.axvline(95.00, color=LIGHT, linestyle=":", linewidth=1.0, zorder=1)
    for yi, ov, dv in zip(y, only_vals, drop_vals):
        ax2.plot([ov, dv], [yi, yi], color="#CCCCCC", linewidth=1.4, zorder=1)
    ax2.plot(
        only_vals,
        y,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=MID,
        markeredgewidth=1.0,
        linestyle="--",
        color=MID,
        linewidth=1.1,
        markersize=4.6,
        label="Keep only this group",
        zorder=3,
    )
    ax2.plot(
        drop_vals,
        y,
        marker="D",
        markerfacecolor=INK,
        markeredgecolor=INK,
        linestyle="-",
        color=INK,
        linewidth=1.1,
        markersize=4.0,
        label="Drop this group",
        zorder=3,
    )
    ax2.set_yticks(y, [label for _, label in order])
    ax2.tick_params(axis="y", labelsize=7.6)
    ax2.set_xlim(35, 103)
    ax2.set_ylim(5.6, -1.2)
    ax2.set_xlabel("Validation Macro-F1 at 40% labels (%)")
    ax2.legend(loc="upper left", fontsize=7.0)
    ax2.grid(axis="y", visible=False)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy_en(ax)
    fig.tight_layout(w_pad=1.4)
    save_en(fig, "fig6_dyn_feature_ablation")


# --------------------------------------------------------------------------- #
# Fig. 7 - fusion ablation and label-budget curves
# --------------------------------------------------------------------------- #
def plot_fig7_stacking_ablation_and_low_label() -> None:
    setup_english_journal_style()
    e12b = load_json(
        "artifacts/imports/e12b_fusion_ablation_corrected_20261020/e12b_fusion_ablation_corrected/summary.json"
    )
    per_seed = {(r["tag"], r["method"]): np.asarray(r["per_seed"]) for r in e12b["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_W, PANEL_H + 0.75), gridspec_kw={"width_ratios": [0.82, 1.18]}
    )

    pairs = [
        ("tcn_meta", "tcn", "TCN + meta-learner\n\u2212 TCN"),
        ("full", "tcn_meta", "Full fusion\n\u2212 TCN + meta-learner"),
        ("tcn+dyn", "tcn", "TCN + DYN \u2212 TCN"),
        ("tcn+md", "tcn", "TCN + MOMENT/\u0394P\n\u2212 TCN"),
        ("dyn+md", "dyn", "DYN + MOMENT/\u0394P\n\u2212 DYN"),
        ("full", "dyn+md", "TCN marginal\ncontribution"),
        ("full", "tcn+md", "DYN marginal\ncontribution"),
        ("full", "tcn+dyn", "MOMENT/\u0394P marginal\ncontribution"),
    ]
    y = np.arange(len(pairs))
    off = 0.17
    for k, (tag, color, marker, mfc, ls, label) in enumerate(
        [("10p", INK, "o", INK, "-", "10% labels"), ("40p", MID, "s", "white", "--", "40% labels")]
    ):
        diffs = [per_seed[(tag, left)] - per_seed[(tag, right)] for left, right, _ in pairs]
        ax1.errorbar(
            [float(d.mean()) for d in diffs],
            y - off if k == 0 else y + off,
            xerr=[float(d.std(ddof=1)) for d in diffs],
            fmt=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle=ls,
            color=color,
            linewidth=0.9,
            markersize=4.0,
            capsize=1.8,
            elinewidth=0.8,
            label=label,
        )
    ax1.axvline(0, color=LIGHT, linestyle=":", linewidth=1.0)
    ax1.axhline(4.5, color=GRID, linewidth=0.8)
    ax1.set_yticks(y, [p[2] for p in pairs])
    ax1.tick_params(axis="y", labelsize=7.0)
    ax1.set_ylim(7.6, -0.6)
    ax1.set_xlim(-0.9, 5.9)
    ax1.set_xlabel("Paired Macro-F1 difference (pp)")
    ax1.legend(loc="lower right", ncol=1, fontsize=7.0, handlelength=1.8, borderaxespad=0.4)
    ax1.grid(axis="y", visible=False)
    panel_label(ax1, "(a)")

    e18 = load_json(
        "artifacts/imports/e18_low_label_20261020/e18_low_label/e18_low_label_20261020_v1/summary.json"
    )
    e18_100 = load_json(
        "artifacts/imports/e18_low_label_100p_20261020/e18_low_label_100p/e18_low_label_100p_20261020_v1/summary.json"
    )
    agg18 = {(r["tag"], r["method"]): r for r in [*e18["aggregate"], *e18_100["aggregate"]]}
    tags = ["1p", "5p", "10p", "20p", "40p", "100p"]
    x_pct = [1, 5, 10, 20, 40, 100]
    series = [
        ("moment_logit", "MOMENT, original stream", LIGHT, "s", ":", "white"),
        ("moment_diff_logit", "MOMENT, dual stream", MID, "D", "--", "white"),
        ("dyn_hgb", "DYN + HGB", MID, "^", "-.", MID),
        ("tcn", "Masked TCN", INK, "o", "--", "white"),
        ("dyn_moment_diff", "DYN + MOMENT/\u0394P", MID, "X", "-", MID),
        ("multirocket", "MultiRocket", LIGHT, "P", "-", LIGHT),
        ("full", "Full fusion", INK, "*", "-", INK),
    ]
    for key, label, color, marker, ls, mfc in series:
        ax2.errorbar(
            x_pct,
            [agg18[(t, key)]["mean"] for t in tags],
            yerr=[agg18[(t, key)]["sample_sd"] for t in tags],
            label=label,
            color=color,
            marker=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle=ls,
            linewidth=1.8 if key == "full" else 1.0,
            markersize=6.0 if key == "full" else 4.0,
            capsize=1.6,
            elinewidth=0.7,
            zorder=4 if key == "full" else 3,
        )
    log_budget_axis(ax2, x_pct, (0.8, 125))
    ax2.tick_params(axis="x", labelsize=7.2)
    ax2.set_ylim(48, 101)
    ax2.set_ylabel(VAL_F1)
    ax2.legend(
        loc="lower right",
        fontsize=6.7,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
        handlelength=1.8,
        labelspacing=0.25,
    )
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy_en(ax)
    fig.tight_layout(w_pad=1.2)
    save_en(fig, "fig7_stacking_ablation_and_low_label")


# --------------------------------------------------------------------------- #
# Fig. 8 - noise sensitivity and accuracy-latency trade-off
# --------------------------------------------------------------------------- #
def plot_fig8_robustness_and_latency() -> None:
    setup_english_journal_style()
    e13 = load_json(
        "artifacts/imports/e13_robustness_20261020/e13_robustness/e13b_robustness_20261020_v1/summary.json"
    )
    agg13 = {(r["tag"], r["method"], r["perturbation"]): r for r in e13["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H + 0.1))

    levels = ["clean", "noise_0p1pct", "noise_0p25pct", "noise_0p5pct", "noise_1pct"]
    x = np.arange(len(levels))
    series = [
        ("full", "Full fusion", INK, "o", "-"),
        ("tcn", "Masked TCN", INK, "s", "--"),
        ("tcn_dyn", "TCN + DYN", MID, "^", "-."),
        ("md_logit", "MOMENT/\u0394P", MID, "D", "--"),
        ("dyn_hgb", "DYN + HGB", LIGHT, "v", ":"),
    ]
    for key, label, color, marker, ls in series:
        ax1.errorbar(
            x,
            [agg13[("40p", key, p)]["mean"] for p in levels],
            yerr=[agg13[("40p", key, p)]["sample_sd"] for p in levels],
            label=label,
            color=color,
            marker=marker,
            markerfacecolor=color if ls == "-" else "white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle=ls,
            linewidth=1.6 if key == "full" else 1.1,
            markersize=4.2,
            capsize=2.0,
            elinewidth=0.8,
        )
    ax1.set_xticks(x, ["0", "0.10", "0.25", "0.50", "1.00"])
    ax1.set_ylim(0, 108)
    ax1.set_yticks([20, 40, 60, 80, 100])
    ax1.set_xlabel("Additive Gaussian noise level (% RMS)")
    ax1.set_ylabel(VAL_F1)
    ax1.legend(loc="lower left", ncol=2, columnspacing=0.8, fontsize=6.7, handlelength=1.7)
    ax1.grid(axis="x", visible=False)
    panel_label(ax1, "(a)")

    points = [
        ("DYN+HGB", 1.990, 11.430, 95.00, LIGHT, "v"),
        ("TCN", 0.039, 0.882, 97.53, INK, "s"),
        ("TCN+DYN", 2.092, 12.595, 97.93, MID, "^"),
        ("Full fusion", 11.128, 46.103, 98.09, INK, "o"),
    ]
    ax2.plot([0.039, 2.092, 11.128], [97.53, 97.93, 98.09], color=GRID, linewidth=1.2, zorder=1)
    for _, lat32, lat1, f1, color, marker in points:
        ax2.plot([lat32, lat1], [f1, f1], color=GRID, linestyle="--", linewidth=0.8, zorder=1)
        ax2.scatter(lat32, f1, color=color, marker=marker, s=30, zorder=4)
        ax2.scatter(
            lat1,
            f1,
            facecolors="white",
            edgecolors=color,
            marker=marker,
            s=24,
            linewidths=0.9,
            zorder=4,
        )
    labels = {
        "TCN": ("TCN  97.53", 8, -12, "left", "top"),
        "TCN+DYN": ("TCN + DYN  97.93", 0, 11, "center", "bottom"),
        "Full fusion": ("Full fusion\n98.09", 7, -6, "left", "top"),
        "DYN+HGB": ("DYN + HGB  95.00", 7, 4, "left", "bottom"),
    }
    for name, (text, dx, dy, ha, va) in labels.items():
        lat32, f1 = next((p[1], p[3]) for p in points if p[0] == name)
        ax2.annotate(
            text,
            xy=(lat32, f1),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.0,
            fontweight="bold" if name == "Full fusion" else "normal",
            color=INK if name != "DYN+HGB" else LIGHT,
        )
    ax2.set_xscale("log")
    ax2.grid(False, which="minor")
    ax2.grid(True, which="major")
    ax2.set_xlim(0.015, 200)
    ax2.set_ylim(93.9, 99.0)
    ax2.set_yticks([94.5, 95.5, 96.5, 97.5, 98.5])
    ax2.set_xlabel("Per-record inference latency (ms, log scale)")
    ax2.set_ylabel("Validation Macro-F1 at 40% labels (%)")
    handles = [
        Line2D(
            [], [], color=INK, marker="s", linestyle="none", markersize=4.2, label="Batch size 32"
        ),
        Line2D(
            [],
            [],
            color=INK,
            marker="s",
            linestyle="none",
            markersize=4.2,
            markerfacecolor="white",
            label="Batch size 1",
        ),
    ]
    ax2.legend(handles=handles, loc="lower right", fontsize=7.0, handlelength=1.0)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy_en(ax)
    fig.tight_layout(w_pad=1.8)
    save_en(fig, "fig8_robustness_and_latency")


if __name__ == "__main__":
    plot_fig1_dataset_overview()
    plot_fig2_overall_framework()
    plot_fig3_pooling_and_head_stabilization()
    plot_fig4_tcn_and_length_strata()
    plot_fig5_dual_stream_and_pretraining()
    plot_fig6_dyn_feature_ablation()
    plot_fig7_stacking_ablation_and_low_label()
    plot_fig8_robustness_and_latency()
