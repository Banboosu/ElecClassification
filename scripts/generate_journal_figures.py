"""Generate standardized Chinese academic journal figures (PNG 300 DPI + SVG) from audited JSON records.

Design rules (IOP-style, single text width = 153 mm):

* every figure is laid out at its final printed width (6.0 in ~ 153 mm) and
  included at ``0.96\\textwidth``, so the scale factor is ~1 and a 8 pt label
  really prints at ~7.7 pt instead of being shrunk to 4-5 pt;
* the artwork carries only axis labels, tick labels and short legend entries;
  numbers, panel descriptions and mechanism notes belong to the caption;
* no per-point value labels except a handful of places where the value itself
  is the message, no annotation boxes, no arrows with explanation text.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from tcn_moment.label_semantics import LABEL_NAMES_ZH

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
TEX_FIG_DIR = ROOT / "docs" / "ioplatextemplate"

# 153 mm text width of iopjournal.cls, in inches.
FIG_W = 153.0 / 25.4
# Gutter between the two panels of a wide figure.
PANEL_H = 2.62

INK = "#111111"
MID = "#4D4D4D"
LIGHT = "#8C8C8C"
GRID = "#D9D9D9"


def setup_chinese_journal_style() -> None:
    """Register Windows Chinese fonts and configure print-ready matplotlib rcParams."""
    font_candidates = [
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        Path("/mnt/c/Windows/Fonts/simsun.ttc"),
    ]
    for fp in font_candidates:
        if fp.exists():
            fm.fontManager.addfont(str(fp))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 7.9,
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
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, text: str) -> None:
    """Left-aligned bold panel tag placed above the axes, journal style."""
    ax.set_title(text, loc="left", pad=5.0)


def tidy(ax: plt.Axes) -> None:
    """Remove the top/right spines and soften the tick marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8.6)


def load_json(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def smooth(values: list[float], window: int = 3) -> np.ndarray:
    """Centred moving average used only to tame epoch-to-epoch jitter."""
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size < window:
        return arr
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def save_both(fig: plt.Figure, stem: str, *, extra_dir: Path | None = None) -> None:
    # The IOP LaTeX draft includes figures from its own directory, so the fresh
    # raster copies have to land there as well, otherwise the PDF keeps the
    # stale artwork.
    dirs = [FIG_DIR, TEX_FIG_DIR]
    if extra_dir is not None:
        dirs.append(extra_dir)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        out = d / stem
        for ext in (".png", ".svg"):
            fig.savefig(out.with_suffix(ext), bbox_inches="tight", pad_inches=0.06)
        print(f"Saved {out}.png and .svg")
    plt.close(fig)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = "#FFFFFF",
    ec: str = "#111111",
    lw: float = 0.9,
    ls: str = "-",
    fontsize: float = 8.5,
    fontweight: str = "normal",
    radius: float = 0.02,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=INK,
        zorder=3,
        linespacing=1.3,
    )


def draw_arrow(
    ax: plt.Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 0.9,
    ls: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            linestyle=ls,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=4,
        )
    )


# --------------------------------------------------------------------------- #
# Fig. 2 - overall framework
# --------------------------------------------------------------------------- #
def _rounded_path(
    ax: plt.Axes, pts: list[tuple[float, float]], *, color: str = INK, lw: float = 0.9
) -> None:
    """Polyline with small rounded corners, used for the probability bus."""
    rad = 0.008
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=4, solid_joinstyle="round")


def plot_fig1_overall_framework() -> None:
    """Clean three-branch framework diagram.

    The figure carries only the minimum labels needed to read the flow.
    Branch details, dimensions and training settings belong in the LaTeX
    caption, not in the artwork.
    """
    setup_chinese_journal_style()
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Input
    draw_box(
        ax,
        0.012,
        0.31,
        0.145,
        0.38,
        "Input\n$x_i,\\ m_i$",
        fc="#F2F2F2",
        ec="#111111",
        lw=1.4,
        fontsize=9.8,
        fontweight="bold",
    )

    # Branch containers
    containers = [
        (0.74, 0.22, "-", "#111111"),
        (0.40, 0.22, "--", "#333333"),
        (0.06, 0.22, "-.", "#555555"),
    ]
    for y0, h, ls, ec in containers:
        draw_box(ax, 0.19, y0, 0.55, h, "", fc="#FAFAFA", ec=ec, lw=1.2, ls=ls, radius=0.015)

    # Branch I: P and dP -> frozen MOMENT -> mask-aware pooling -> p_MD
    draw_box(
        ax, 0.205, 0.855, 0.085, 0.075, "$P$", fc="#FFFFFF", ec="#222222", lw=1.0, fontsize=10.5
    )
    draw_box(
        ax,
        0.205,
        0.765,
        0.085,
        0.075,
        "$\\Delta P$",
        fc="#FFFFFF",
        ec="#222222",
        lw=1.0,
        ls="--",
        fontsize=10.5,
    )
    draw_box(
        ax,
        0.315,
        0.765,
        0.135,
        0.165,
        "Frozen\nMOMENT",
        fc="#E9E9E9",
        ec="#111111",
        lw=1.2,
        fontsize=10.5,
        fontweight="bold",
    )
    draw_box(
        ax,
        0.475,
        0.765,
        0.135,
        0.165,
        "Mask-aware\npooling",
        fc="#FFFFFF",
        ec="#111111",
        lw=1.2,
        fontsize=10.0,
    )
    draw_box(
        ax,
        0.635,
        0.765,
        0.085,
        0.165,
        "$p^{MD}$",
        fc="#FFFFFF",
        ec="#111111",
        lw=1.2,
        fontsize=11.0,
        fontweight="bold",
    )

    # Branch II: 60-D dynamics -> HGB -> p_DYN
    draw_box(
        ax,
        0.205,
        0.425,
        0.195,
        0.165,
        "60-D\ndynamics",
        fc="#EDEDED",
        ec="#333333",
        lw=1.0,
        fontsize=10.5,
    )
    draw_box(
        ax,
        0.430,
        0.425,
        0.120,
        0.165,
        "HGB",
        fc="#FFFFFF",
        ec="#333333",
        lw=1.0,
        fontsize=11.0,
        fontweight="bold",
    )
    draw_box(
        ax,
        0.635,
        0.425,
        0.085,
        0.165,
        "$p^{DYN}$",
        fc="#FFFFFF",
        ec="#333333",
        lw=1.0,
        fontsize=11.0,
        fontweight="bold",
    )

    # Branch III: masked TCN -> p_TCN
    draw_box(
        ax,
        0.205,
        0.095,
        0.195,
        0.165,
        "Masked\nTCN",
        fc="#E5E5E5",
        ec="#555555",
        lw=1.0,
        ls="-",
        fontsize=10.5,
    )
    draw_box(
        ax,
        0.635,
        0.095,
        0.085,
        0.165,
        "$p^{TCN}$",
        fc="#FFFFFF",
        ec="#555555",
        lw=1.0,
        fontsize=11.0,
        fontweight="bold",
    )

    # Fusion column
    draw_box(
        ax,
        0.775,
        0.615,
        0.185,
        0.140,
        "5-fold OOF\nstacking",
        fc="#F2F2F2",
        ec="#111111",
        lw=1.2,
        fontsize=10.0,
    )
    draw_box(
        ax,
        0.775,
        0.405,
        0.185,
        0.140,
        "Logistic\nmeta-learner",
        fc="#FFFFFF",
        ec="#111111",
        lw=1.2,
        fontsize=10.0,
    )
    draw_box(
        ax,
        0.775,
        0.195,
        0.185,
        0.140,
        "3-class\noutput",
        fc="#EDEDED",
        ec="#111111",
        lw=1.4,
        fontsize=10.5,
        fontweight="bold",
    )

    # Input -> branches
    for y in (0.855, 0.505, 0.175):
        draw_arrow(ax, (0.157, 0.50), (0.19, y), color="#222222", lw=1.2)
    # Branch I
    draw_arrow(ax, (0.292, 0.893), (0.315, 0.865), color="#111111", lw=1.1)
    draw_arrow(ax, (0.292, 0.803), (0.315, 0.803), color="#111111", lw=1.1, ls="--")
    draw_arrow(ax, (0.450, 0.848), (0.475, 0.848), color="#111111", lw=1.1)
    draw_arrow(ax, (0.610, 0.848), (0.635, 0.848), color="#111111", lw=1.1)
    # Branch II
    draw_arrow(ax, (0.400, 0.507), (0.430, 0.507), color="#333333", lw=1.1)
    draw_arrow(ax, (0.550, 0.507), (0.635, 0.507), color="#333333", lw=1.1)
    # Branch III
    draw_arrow(ax, (0.400, 0.177), (0.635, 0.177), color="#555555", lw=1.1, ls="-.")
    # Branch outputs share a short bus before entering the meta-learner.
    for y in (0.848, 0.507, 0.177):
        ax.plot([0.720, 0.755], [y, y], color="#222222", lw=1.2, zorder=4)
    ax.plot([0.755, 0.755], [0.177, 0.848], color="#222222", lw=1.2, zorder=4)
    draw_arrow(ax, (0.755, 0.680), (0.775, 0.680), color="#111111", lw=1.3)
    # Fusion internal
    draw_arrow(ax, (0.867, 0.615), (0.867, 0.545), color="#111111", lw=1.1)
    draw_arrow(ax, (0.867, 0.405), (0.867, 0.335), color="#111111", lw=1.1)

    out_dirs = [FIG_DIR, FIG_DIR / "label_semantics_v2" / "paper"]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig2_overall_framework.pdf", bbox_inches="tight")
    save_both(fig, "fig2_overall_framework", extra_dir=FIG_DIR / "label_semantics_v2" / "paper")


# --------------------------------------------------------------------------- #
# Fig. 3 - pooling repair and frozen head convergence
# --------------------------------------------------------------------------- #
def plot_fig2_pooling_and_head_stabilization() -> None:
    setup_chinese_journal_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H))

    # (a) adaptation ladder
    stages = ["默认\n池化", "掩码\n池化", "解冻\n2 层", "RBF\n-SVM", "完全\n微调"]
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
    ax1.set_xlim(-0.45, 4.45)
    ax1.set_ylim(64, 100)
    ax1.set_xlabel("适配阶段（全量标注）")
    ax1.set_ylabel("测试集 Macro-F1 (%)")
    ax1.grid(axis="x", visible=False)
    handles = [
        Line2D(
            [], [], color=INK, marker="o", linewidth=1.6, markersize=4.6, label="MOMENT-1-large"
        ),
        Line2D([], [], color=LIGHT, linestyle="--", linewidth=1.0, label="TCN 基线 95.99"),
        Line2D([], [], color=LIGHT, linestyle=":", linewidth=1.0, label="F15+RF 88.89"),
    ]
    ax1.legend(
        handles=handles,
        loc="lower right",
        fontsize=7.0,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
    )  # type: ignore[arg-type]
    panel_label(ax1, "(a)")

    # (b) convergence of the frozen head
    root_e03d = ROOT / "artifacts/imports/e03d_head_validation_20261018"
    specs = [
        ("fusion", "fanin_gain", "融合 · 尺度补偿", INK, "-", "o"),
        ("fusion", "legacy_fresh", "融合 · 原尺度", INK, "--", "o"),
        ("statistics", "fanin_gain", "统计 · 尺度补偿", MID, "-", "s"),
        ("statistics", "legacy_fresh", "统计 · 原尺度", MID, "--", "s"),
        ("moment", "fanin_gain", "MOMENT · 尺度补偿", LIGHT, "-", "^"),
        ("moment", "legacy_fresh", "MOMENT · 原尺度", LIGHT, "--", "^"),
    ]
    for branch, mode, label, color, ls, marker in specs:
        hist_path = root_e03d / f"e03d_head_diagnostic_20261018_v2_{branch}_{mode}/history.json"
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        epochs = np.array([r["epoch"] for r in history])
        f1s = smooth([r["validation_macro_f1"] * 100 for r in history], 3)
        ax2.plot(
            epochs,
            f1s,
            label=label,
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
    ax2.set_ylim(71, 104.0)
    ax2.set_yticks([75, 80, 85, 90, 95])
    ax2.set_xlabel("训练轮次 (epoch)")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    ax2.legend(
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        fontsize=6.8,
        frameon=False,
        handlelength=1.6,
    )
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy(ax)
    fig.tight_layout(w_pad=1.8)
    save_both(fig, "fig3_pooling_and_head_stabilization")


# --------------------------------------------------------------------------- #
# Fig. 5 - dual stream and pretraining attribution
# --------------------------------------------------------------------------- #
def plot_fig3_dual_stream_and_pretraining() -> None:
    setup_chinese_journal_style()
    e09 = load_json(
        "artifacts/imports/e09_analysis_corrected_20261020/analysis_four_seed_corrected/summary.json"
    )
    e16 = load_json(
        "artifacts/imports/e16_random_attribution_20261020/e16_random_attribution_20261020_v1/summary.json"
    )
    e16b = load_json(
        "artifacts/imports/e16b_random_low_label_20261020/e16b_random_low_label_20261020_v1/summary.json"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H))

    # (a) input streams at three label budgets
    tags = ["1p", "10p", "40p"]
    x_pct = [1, 10, 40]
    e09_map = {(r["method"], r["tag"]): r for r in e09["aggregate"]}
    streams = [
        ("moment_logit", "原序列 $P$", LIGHT, "s", ":"),
        ("diff_logit", "差分流 $\\Delta P$", MID, "^", "--"),
        ("moment_diff_logit", "双流 $P+\\Delta P$", INK, "o", "-"),
        ("stack_moment_diff", "双流 + DYN 堆叠", INK, "D", "-"),
    ]
    for key, label, color, marker, ls in streams:
        means = [e09_map[(key, t)]["mean"] for t in tags]
        sds = [e09_map[(key, t)]["sample_sd"] for t in tags]
        ax1.errorbar(
            x_pct,
            means,
            yerr=sds,
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

    ax1.set_xscale("log")
    ax1.set_xticks(x_pct, ["1%", "10%", "40%"])
    ax1.minorticks_off()
    ax1.tick_params(axis="x", labelsize=7.8, pad=4.0)
    ax1.set_xlim(0.75, 55)
    ax1.set_ylim(65, 100)
    ax1.set_xlabel("训练标注比例（对数坐标）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    ax1.legend(loc="lower right", fontsize=7.2)
    panel_label(ax1, "(a)")

    # (b) pretrained vs random initialisation, same architecture
    e16_map = {
        (r["tag"], r["initialization"], r["stream"]): r
        for r in [*e16["aggregate"], *e16b["aggregate"]]
    }
    tags_b = ["1p", "5p", "10p", "20p", "40p"]
    x_b = [1, 5, 10, 20, 40]
    stream_specs = [
        ("dual", "o", "双流 $P+\\Delta P$", INK),
        ("diff", "^", "差分流 $\\Delta P$", MID),
        ("orig", "s", "原序列 $P$", LIGHT),
    ]
    for stream, marker, label, color in stream_specs:
        for init, ls, filled in (("pretrained", "-", True), ("random", "--", False)):
            means = [e16_map[(t, init, stream)]["mean"] for t in tags_b]
            sds = [e16_map[(t, init, stream)]["sample_sd"] for t in tags_b]
            ax2.errorbar(
                x_b,
                means,
                yerr=sds,
                label=label if init == "pretrained" else None,
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

    ax2.set_xscale("log")
    ax2.set_xticks(x_b, [f"{v}%" for v in x_b])
    ax2.minorticks_off()
    ax2.tick_params(axis="x", labelsize=7.8, pad=4.0)
    ax2.set_xlim(0.75, 55)
    ax2.set_ylim(42, 99)
    ax2.set_xlabel("训练标注比例（对数坐标）")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    handles = [
        Line2D([], [], color=INK, linestyle="-", marker="o", markersize=4.0, label="预训练权重"),
        Line2D(
            [],
            [],
            color=INK,
            linestyle="--",
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            label="随机初始化",
        ),
    ]
    ax2.legend(handles=handles, loc="lower right", fontsize=7.2, handlelength=2.4)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy(ax)
    fig.tight_layout(w_pad=1.8)
    save_both(fig, "fig5_dual_stream_and_pretraining")


# --------------------------------------------------------------------------- #
# Fig. 6 - dynamics features
# --------------------------------------------------------------------------- #
def plot_fig4_dyn_feature_ablation() -> None:
    setup_chinese_journal_style()
    e08 = load_json(
        "artifacts/imports/e08_strong_features_20261020/e08_strong_features_20261020_v1/analysis/summary.json"
    )
    agg = {(r["method"], r["tag"]): r for r in e08["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_W, PANEL_H + 0.1), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )

    # (a) F15 vs DYN across budgets
    methods = [
        ("dyn_hgb", "DYN + HGB", INK, "o", "-", ["1p", "5p", "10p", "40p", "100p"]),
        ("dyn_svm", "DYN + RBF-SVM", MID, "s", "-.", ["1p", "10p", "40p"]),
        ("f15_hgb", "F15 + HGB", MID, "^", "--", ["1p", "5p", "10p", "40p", "100p"]),
        ("f15_svm", "F15 + RBF-SVM", LIGHT, "v", ":", ["1p", "10p", "40p"]),
    ]
    for key, label, color, marker, ls, m_tags in methods:
        x_m = [int(t[:-1]) for t in m_tags]
        means = [agg[(key, t)]["macro_f1_mean"] for t in m_tags]
        sds = [agg[(key, t)]["macro_f1_sd"] for t in m_tags]
        ax1.errorbar(
            x_m,
            means,
            yerr=sds,
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

    x_pct = [1, 5, 10, 40, 100]
    ax1.set_xscale("log")
    ax1.set_xticks(x_pct, [f"{v}%" for v in x_pct])
    ax1.minorticks_off()
    ax1.tick_params(axis="x", labelsize=7.8, pad=4.0)
    ax1.set_xlim(0.75, 135)
    ax1.set_ylim(72, 101)
    ax1.set_xlabel("训练标注比例（对数坐标）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    ax1.legend(loc="lower right", fontsize=7.2)
    panel_label(ax1, "(a)")

    # (b) group ablation profile at 40% labels
    grp = {r["group"]: r for r in e08["group_ablation_40p"]}
    order = [
        ("roughness", "粗糙度与转折"),
        ("trend", "分段趋势"),
        ("amplitude", "幅值与分位"),
        ("spectral", "频谱与自相关"),
        ("shape", "形态与高阶矩"),
        ("length", "有效长度占比"),
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
        label="仅保留该组",
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
        label="剔除该组",
        zorder=3,
    )
    ax2.set_yticks(y, [label for _, label in order])
    ax2.invert_yaxis()
    ax2.set_xlim(35, 103)
    ax2.set_ylim(5.6, -1.0)
    ax2.set_xlabel("40% 标注下验证集 Macro-F1 (%)")
    ax2.legend(loc="upper left", fontsize=7.2, frameon=False)
    ax2.grid(axis="y", visible=False)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy(ax)
    fig.tight_layout(w_pad=1.8)
    save_both(fig, "fig6_dyn_feature_ablation")


# --------------------------------------------------------------------------- #
# Fig. 7 - fusion ablation and label-budget curves
# --------------------------------------------------------------------------- #
def plot_fig5_stacking_ablation_and_low_label() -> None:
    setup_chinese_journal_style()
    e12b = load_json(
        "artifacts/imports/e12b_fusion_ablation_corrected_20261020/e12b_fusion_ablation_corrected/summary.json"
    )
    per_seed = {(r["tag"], r["method"]): np.asarray(r["per_seed"]) for r in e12b["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_W, PANEL_H + 0.70), gridspec_kw={"width_ratios": [1.08, 1.0]}
    )

    # (a) seed-paired differences
    pairs = [
        ("tcn_meta", "tcn", "TCN+元 − TCN"),
        ("full", "tcn_meta", "完整融合 − TCN+元"),
        ("tcn+dyn", "tcn", "+DYN"),
        ("tcn+md", "tcn", "+MOMENT/ΔP"),
        ("dyn+md", "dyn", "MOMENT/ΔP 配 DYN"),
        ("full", "dyn+md", "TCN 边际"),
        ("full", "tcn+md", "DYN 边际"),
        ("full", "tcn+dyn", "MOMENT/ΔP 边际"),
    ]
    y = np.arange(len(pairs))
    h = 0.17
    for k, (tag, color, marker, mfc, ls, label) in enumerate(
        [
            ("10p", INK, "o", INK, "-", "10% 标注"),
            ("40p", MID, "s", "white", "--", "40% 标注"),
        ]
    ):
        diffs = [per_seed[(tag, left)] - per_seed[(tag, right)] for left, right, _ in pairs]
        means = [float(d.mean()) for d in diffs]
        sds = [float(d.std(ddof=1)) for d in diffs]
        offs = y - h if k == 0 else y + h
        ax1.errorbar(
            means,
            offs,
            xerr=sds,
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
    ax1.set_yticks(y, [p[2] for p in pairs], fontsize=7.9)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.9, 5.9)
    ax1.set_xlabel("配对 Macro-F1 差值 (pp)")
    ax1.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        ncol=2,
        fontsize=7.0,
        columnspacing=0.9,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
        handlelength=1.6,
    )
    ax1.grid(axis="y", visible=False)
    panel_label(ax1, "(a)")

    # (b) unified-protocol label budgets
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
        ("moment_logit", "MOMENT 原序列", LIGHT, "s", ":", "white"),
        ("moment_diff_logit", "MOMENT 双流", MID, "D", "--", "white"),
        ("dyn_hgb", "DYN + HGB", MID, "^", "-.", MID),
        ("tcn", "Masked TCN", INK, "o", "--", "white"),
        ("dyn_moment_diff", "DYN+MOMENT/ΔP", MID, "X", "-", MID),
        ("full", "完整融合", INK, "*", "-", INK),
    ]
    for key, label, color, marker, ls, mfc in series:
        means = [agg18[(t, key)]["mean"] for t in tags]
        sds = [agg18[(t, key)]["sample_sd"] for t in tags]
        ax2.errorbar(
            x_pct,
            means,
            yerr=sds,
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
        )

    ax2.set_xscale("log")
    tick_pct = [1, 5, 20, 100]
    ax2.set_xticks(tick_pct, [f"{v}%" for v in tick_pct])
    ax2.minorticks_off()
    ax2.set_xlim(0.75, 430)
    ax2.set_ylim(48, 101)
    ax2.set_xlabel("训练标注比例（对数坐标）")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    ax2.legend(
        loc="lower right",
        ncol=1,
        fontsize=6.9,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
        handlelength=1.8,
    )
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy(ax)
    fig.tight_layout(w_pad=1.8)
    save_both(fig, "fig7_stacking_ablation_and_low_label")


# --------------------------------------------------------------------------- #
# Fig. 8 - noise sensitivity and accuracy-latency trade-off
# --------------------------------------------------------------------------- #
def plot_fig6_robustness_and_latency() -> None:
    setup_chinese_journal_style()
    e13 = load_json(
        "artifacts/imports/e13_robustness_20261020/e13_robustness/e13b_robustness_20261020_v1/summary.json"
    )
    agg13 = {(r["tag"], r["method"], r["perturbation"]): r for r in e13["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H + 0.1))

    # (a) additive Gaussian noise
    levels = ["clean", "noise_0p1pct", "noise_0p25pct", "noise_0p5pct", "noise_1pct"]
    x = np.arange(len(levels))

    series = [
        ("full", "完整融合", INK, "o", "-"),
        ("tcn", "Masked TCN", INK, "s", "--"),
        ("tcn_dyn", "TCN + DYN", MID, "^", "-."),
        ("md_logit", "MOMENT 双流", MID, "D", "--"),
        ("dyn_hgb", "DYN + HGB", LIGHT, "v", ":"),
    ]
    for key, label, color, marker, ls in series:
        means = [agg13[("40p", key, p)]["mean"] for p in levels]
        sds = [agg13[("40p", key, p)]["sample_sd"] for p in levels]
        ax1.errorbar(
            x,
            means,
            yerr=sds,
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
    ax1.set_xlabel("加性高斯噪声强度 (% RMS)")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    ax1.set_yticks([20, 40, 60, 80, 100])
    ax1.legend(
        loc="lower left", ncol=2, columnspacing=0.8, fontsize=6.9, handlelength=1.7, frameon=False
    )
    ax1.grid(axis="x", visible=False)
    panel_label(ax1, "(a)")

    # (b) accuracy vs single-sample latency (Pareto view, labels only where they matter)
    points = [
        ("DYN+HGB", 1.990, 11.430, 95.00, LIGHT, "v"),
        ("TCN", 0.039, 0.882, 97.53, INK, "s"),
        ("TCN+DYN", 2.092, 12.595, 97.93, MID, "^"),
        ("完整融合", 11.128, 46.103, 98.09, INK, "o"),
    ]
    ax2.plot([0.039, 2.092, 11.128], [97.53, 97.93, 98.09], color=GRID, linewidth=1.2, zorder=1)
    for name, lat32, lat1, f1, color, marker in points:
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
        "TCN+DYN": ("TCN+DYN  97.93", 0, 11, "center", "bottom"),
        "完整融合": ("完整融合  98.09", 8, -12, "left", "top"),
        "DYN+HGB": ("DYN+HGB  95.00", 7, 4, "left", "bottom"),
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
            fontweight="bold" if name == "完整融合" else "normal",
            color=INK if name != "DYN+HGB" else LIGHT,
        )

    ax2.set_xscale("log")
    ax2.grid(False, which="minor")
    ax2.grid(True, which="major")
    ax2.set_xlim(0.015, 200)
    ax2.set_ylim(93.9, 99.0)
    ax2.set_xlabel("单样本推理时延 (ms，对数坐标)")
    ax2.set_ylabel("40% 标注下验证集 Macro-F1 (%)")
    ax2.set_yticks([94.5, 95.5, 96.5, 97.5, 98.5])
    handles = [
        Line2D([], [], color=INK, marker="s", linestyle="none", markersize=4.2, label="批大小 32"),
        Line2D(
            [],
            [],
            color=INK,
            marker="s",
            linestyle="none",
            markersize=4.2,
            markerfacecolor="white",
            label="批大小 1",
        ),
    ]
    ax2.legend(handles=handles, loc="lower right", fontsize=7.0, handlelength=1.0)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy(ax)
    fig.tight_layout(w_pad=1.8)
    save_both(fig, "fig8_robustness_and_latency")


# --------------------------------------------------------------------------- #
# Fig. 4 - TCN architecture and length strata
# --------------------------------------------------------------------------- #
def plot_fig7_tcn_and_calibration() -> None:
    setup_chinese_journal_style()
    e05c = load_json(
        "artifacts/imports/e05c_tcn_replication_20261019/terminal_artifacts/summary.json"
    )
    n2 = load_json(
        "artifacts/imports/n2_tcn_low_label_20261020/n2_tcn_low_label_20261020_v1/summary.json"
    )
    e18_deep = load_json("artifacts/analysis/e18_deep_analysis_semantics_v2/summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H + 0.1))

    # (a) TCN variants across label budgets
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
    unmasked_means = [agg05[("legacy_unmasked50", f)]["macro_f1_mean_percent"] for f in fracs]
    unmasked_sds = [agg05[("legacy_unmasked50", f)]["macro_f1_sample_sd_percent"] for f in fracs]
    res_means = [agg05[("current_fixed50", f)]["macro_f1_mean_percent"] for f in fracs]
    res_sds = [agg05[("current_fixed50", f)]["macro_f1_sample_sd_percent"] for f in fracs]

    ax1.errorbar(
        x_full,
        masked_means,
        yerr=masked_sds,
        marker="o",
        markerfacecolor=INK,
        markeredgecolor=INK,
        linestyle="-",
        color=INK,
        linewidth=1.6,
        markersize=4.4,
        capsize=2.0,
        elinewidth=0.8,
        label="6 层 TCN，掩码池化",
    )
    ax1.errorbar(
        x_10_100,
        unmasked_means,
        yerr=unmasked_sds,
        marker="^",
        markerfacecolor="white",
        markeredgecolor=MID,
        markeredgewidth=0.9,
        linestyle="--",
        color=MID,
        linewidth=1.1,
        markersize=4.2,
        capsize=2.0,
        elinewidth=0.8,
        label="6 层 TCN，均值池化",
    )
    ax1.errorbar(
        x_10_100,
        res_means,
        yerr=res_sds,
        marker="s",
        markerfacecolor="white",
        markeredgecolor=LIGHT,
        markeredgewidth=0.9,
        linestyle=":",
        color=LIGHT,
        linewidth=1.1,
        markersize=4.0,
        capsize=2.0,
        elinewidth=0.8,
        label="4 块残差 TCN，掩码池化",
    )

    ax1.set_xscale("log")
    ax1.set_xticks(x_full, ["1%", "5%", "10%", "40%", "100%"])
    ax1.minorticks_off()
    ax1.tick_params(axis="x", labelsize=7.8, pad=4.0)
    ax1.set_xlim(0.75, 135)
    ax1.set_ylim(40, 108)
    ax1.set_xlabel("训练标注比例（对数坐标）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    ax1.legend(loc="lower right", fontsize=7.2, frameon=True, framealpha=1.0, edgecolor="none")
    panel_label(ax1, "(a)")

    # (b) length-stratified performance
    strata = ["短（$L\\leq266$）", "中（266–399）", "长（$L>399$）"]
    x2 = np.arange(3)
    specs = [
        ("40p", "full", "40% 完整融合", INK, "o", "-"),
        ("40p", "tcn", "40% Masked TCN", INK, "s", "--"),
        ("10p", "full", "10% 完整融合", MID, "D", "-"),
        ("10p", "tcn", "10% Masked TCN", MID, "^", "--"),
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
    ax2.set_xlabel("")
    ax2.set_ylabel("分组 Macro-F1 (%)")
    ax2.legend(
        loc="lower left",
        ncol=2,
        columnspacing=0.8,
        fontsize=7.3,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
    )
    ax2.grid(axis="x", visible=False)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy(ax)
    fig.tight_layout(w_pad=1.6)
    save_both(fig, "fig4_tcn_and_length_strata")


# --------------------------------------------------------------------------- #
# Fig. 1 - dataset overview
# --------------------------------------------------------------------------- #
def plot_fig1_dataset_overview() -> None:
    """Training split only: class-wise example curves and valid-length ECDF (monochrome)."""
    import csv

    setup_chinese_journal_style()
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

    names = LABEL_NAMES_ZH
    fig = plt.figure(figsize=(FIG_W, 4.35))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.55, wspace=0.30)

    # (a)-(c) three random examples per class, one shared legend
    rng = np.random.default_rng(42)
    sample_styles = [
        (INK, "-", "样例 1"),
        (MID, "--", "样例 2"),
        (LIGHT, "-.", "样例 3"),
    ]
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
        ax.set_xlabel("采样点序号")
        ax.set_title(f"({'abc'[col]}) {names[cls]}", loc="left", pad=4.0)
        if col == 0:
            ax.set_ylabel("充电功率（记录单位）")
        tidy(ax)
        if col == 2:
            ax.legend(loc="upper right", fontsize=7.0)

    # (d) ECDF of valid lengths
    ax = fig.add_subplot(gs[1, :])
    class_styles = {
        "0": (INK, "-", "o"),
        "1": (MID, "--", "s"),
        "2": (LIGHT, "-.", "^"),
    }
    for cls in ("0", "1", "2"):
        lengths = np.sort(np.asarray([len(s) for s in by_class[cls]]))
        ecdf = np.arange(1, len(lengths) + 1) / len(lengths) * 100.0
        c_col, c_ls, c_mk = class_styles[cls]
        med = int(np.median(lengths))
        step = max(1, len(lengths) // 8)
        ax.plot(
            lengths,
            ecdf,
            linewidth=1.3,
            color=c_col,
            linestyle=c_ls,
            marker=c_mk,
            markevery=step,
            markersize=3.4,
            markerfacecolor="white" if c_ls != "-" else c_col,
            markeredgecolor=c_col,
            markeredgewidth=0.9,
            label=f"{names[cls]}（中位数 {med}）",
        )
        ax.axvline(med, color=c_col, linestyle=":", linewidth=0.8, alpha=0.8)

    ax.axhline(50.0, color=LIGHT, linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 830)
    ax.set_ylim(0, 103)
    ax.set_xlabel("有效序列长度 $L$（采样点）")
    ax.set_ylabel("经验累积概率 (%)")
    ax.legend(loc="lower right", ncol=1)
    ax.set_title("(d) 各类有效长度经验累积分布", loc="left", pad=5.0)
    tidy(ax)

    save_both(fig, "fig1_dataset_overview", extra_dir=FIG_DIR / "label_semantics_v2" / "paper")


if __name__ == "__main__":
    plot_fig1_dataset_overview()
    plot_fig1_overall_framework()
    plot_fig2_pooling_and_head_stabilization()
    plot_fig3_dual_stream_and_pretraining()
    plot_fig4_dyn_feature_ablation()
    plot_fig5_stacking_ablation_and_low_label()
    plot_fig6_robustness_and_latency()
    plot_fig7_tcn_and_calibration()
