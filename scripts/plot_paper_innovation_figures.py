"""Generate publication-quality figures for the journal manuscript from audited JSON summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tcn_moment.paper_style import PAPER_COLORS, apply_paper_style

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"


def load_json(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def plot_e09_e16_dual_stream_and_attribution() -> None:
    apply_paper_style(plt)
    e09 = load_json(
        "artifacts/imports/e09_analysis_corrected_20261020/analysis_four_seed_corrected/summary.json"
    )
    e16 = load_json(
        "artifacts/imports/e16_random_attribution_20261020/e16_random_attribution_20261020_v1/summary.json"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))

    # Panel A: E09 streams across budgets (1%, 10%, 40%)
    tags = ["1p", "10p", "40p"]
    tag_labels = ["1% (244)", "10% (2,456)", "40% (9,827)"]
    e09_map = {(r["method"], r["tag"]): r for r in e09["aggregate"]}

    methods_a = [
        ("moment_logit", "Original stream (P)", "#7F7F7F"),
        ("diff_logit", "First-order diff (ΔP)", "#56B4E9"),
        ("moment_diff_logit", "Dual-stream (P + ΔP)", "#D55E00"),
        ("stack_moment_diff", "DYN + Dual-stream Stack", "#009E73"),
    ]
    x = np.arange(len(tags))
    width = 0.19
    for i, (m_key, m_label, color) in enumerate(methods_a):
        means = [e09_map[(m_key, t)]["mean"] for t in tags]
        sds = [e09_map[(m_key, t)]["sample_sd"] for t in tags]
        offset = (i - 1.5) * width
        bars = ax1.bar(
            x + offset,
            means,
            width,
            yerr=sds,
            capsize=3,
            label=m_label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        for bar, val, sd_val in zip(bars, means, sds):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + sd_val + 0.6,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222222",
            )

    ax1.set_xticks(x, tag_labels)
    ax1.set_ylim(60, 102)
    ax1.set_xlabel("Training label budget (samples)")
    ax1.set_ylabel("Validation Macro-F1 (%)")
    ax1.set_title("(a) Dual-stream MOMENT & DYN stacking (E09)", pad=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.2)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.grid(False, axis="x")

    # Panel B: E16 Pretrained vs Random initialization (C=1.0 matched)
    e16_map = {(r["tag"], r["initialization"], r["stream"]): r for r in e16["aggregate"]}
    groups = [
        ("10p", "orig", "10% Orig"),
        ("10p", "diff", "10% ΔP"),
        ("10p", "dual", "10% Dual"),
        ("40p", "orig", "40% Orig"),
        ("40p", "diff", "40% ΔP"),
        ("40p", "dual", "40% Dual"),
    ]
    x2 = np.arange(len(groups))
    w2 = 0.34
    pre_means = [e16_map[(t, "pretrained", s)]["mean"] for t, s, _ in groups]
    pre_sds = [e16_map[(t, "pretrained", s)]["sample_sd"] for t, s, _ in groups]
    rnd_means = [e16_map[(t, "random", s)]["mean"] for t, s, _ in groups]
    rnd_sds = [e16_map[(t, "random", s)]["sample_sd"] for t, s, _ in groups]

    b_pre = ax2.bar(
        x2 - w2 / 2,
        pre_means,
        w2,
        yerr=pre_sds,
        capsize=3,
        label="Pretrained MOMENT-1-large",
        color=PAPER_COLORS["tcn"],
        edgecolor="white",
        linewidth=0.6,
    )
    b_rnd = ax2.bar(
        x2 + w2 / 2,
        rnd_means,
        w2,
        yerr=rnd_sds,
        capsize=3,
        label="Random-initialized encoder",
        color="#B0B0B0",
        edgecolor="white",
        linewidth=0.6,
    )

    for bar_p, bar_r, vp, vr, sp, sr in zip(b_pre, b_rnd, pre_means, rnd_means, pre_sds, rnd_sds):
        diff = vp - vr
        ax2.text(
            bar_p.get_x() + bar_p.get_width() / 2,
            vp + sp + 0.6,
            f"{vp:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=PAPER_COLORS["tcn"],
            fontweight="bold",
        )
        ax2.text(
            bar_r.get_x() + bar_r.get_width() / 2,
            vr + sr + 0.6,
            f"{vr:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )
        ax2.annotate(
            f"+{diff:.1f}pp",
            xy=((bar_p.get_x() + bar_r.get_x() + bar_r.get_width()) / 2, max(vp + sp, vr + sr) + 3.4),
            ha="center",
            fontsize=7.5,
            color="#D55E00",
            fontweight="bold",
        )

    ax2.set_xticks(x2, [g[2] for g in groups])
    ax2.set_ylim(50, 102)
    ax2.set_xlabel("Budget and input stream (matched C = 1.0)")
    ax2.set_ylabel("Validation Macro-F1 (%)")
    ax2.set_title("(b) Pretrained vs. random encoder attribution (E16)", pad=10)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.2)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.grid(False, axis="x")

    fig.tight_layout()
    out = FIG_DIR / "paper_e09_e16_dual_stream_attribution_20261021"
    for ext in (".png", ".svg"):
        fig.savefig(out.with_suffix(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.png/.svg")


def plot_e12b_fusion_ablation() -> None:
    apply_paper_style(plt)
    e12b = load_json(
        "artifacts/imports/e12b_fusion_ablation_corrected_20261020/e12b_fusion_ablation_corrected/summary.json"
    )
    agg = {(r["tag"], r["method"]): r for r in e12b["aggregate"]}

    methods = [
        ("tcn", "TCN (fixed 50ep)"),
        ("tcn_meta", "TCN + Meta head"),
        ("md", "MOMENT/ΔP (C=1.0)"),
        ("dyn", "DYN-60 + HGB"),
        ("tcn+md", "TCN + MOMENT/ΔP"),
        ("dyn+md", "DYN + MOMENT/ΔP"),
        ("tcn+dyn", "TCN + DYN"),
        ("full", "Full 3-Way Stack"),
    ]
    colors = [
        "#7F7F7F",
        "#A0A0A0",
        "#CC79A7",
        "#56B4E9",
        "#E69F00",
        "#009E73",
        "#0072B2",
        "#D55E00",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    y = np.arange(len(methods))

    for ax, tag, title, xlim in [
        (ax1, "10p", "(a) 10% Label Budget (2,456 train samples)", (86.0, 97.6)),
        (ax2, "40p", "(b) 40% Label Budget (9,827 train samples)", (91.0, 99.8)),
    ]:
        means = [agg[(tag, m)]["mean"] for m, _ in methods]
        sds = [agg[(tag, m)]["sample_sd"] for m, _ in methods]
        bars = ax.barh(
            y,
            means,
            xerr=sds,
            capsize=3,
            color=colors,
            edgecolor="white",
            height=0.68,
            alpha=0.92,
        )
        for bar, m_val, sd_val in zip(bars, means, sds):
            ax.text(
                m_val + sd_val + 0.12,
                bar.get_y() + bar.get_height() / 2,
                f"{m_val:.2f} ± {sd_val:.2f}%",
                va="center",
                ha="left",
                fontsize=8.0,
                color="#222222",
                fontweight="bold" if m_val == max(means) else "normal",
            )
        ax.set_xlim(*xlim)
        ax.set_xlabel("Validation Macro-F1 (%)")
        ax.set_title(title, pad=10)
        ax.grid(True, axis="x", alpha=0.25)
        ax.grid(False, axis="y")

    ax1.set_yticks(y, [label for _, label in methods])
    ax1.invert_yaxis()

    fig.tight_layout()
    out = FIG_DIR / "paper_e12b_fusion_ablation_20261021"
    for ext in (".png", ".svg"):
        fig.savefig(out.with_suffix(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.png/.svg")


def plot_e13_e17_robustness_and_cost() -> None:
    apply_paper_style(plt)
    e13 = load_json(
        "artifacts/imports/e13_robustness_20261020/e13_robustness/e13b_robustness_20261020_v1/summary.json"
    )
    agg13 = {(r["tag"], r["method"], r["perturbation"]): r for r in e13["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))

    # Panel A: E13 Noise robustness at 40% budget
    noise_levels = ["clean", "noise_0p1pct", "noise_0p25pct", "noise_0p5pct", "noise_1pct"]
    noise_labels = ["Clean\n(0)", "0.10%\n(0.19u)", "0.25%\n(0.48u)", "0.50%\n(0.96u)", "1.00%\n(1.92u)"]
    x = np.arange(len(noise_levels))

    series = [
        ("full", "Full 3-Way Stack", "#D55E00", "o"),
        ("tcn", "TCN (single branch)", "#0072B2", "s"),
        ("tcn_dyn", "TCN + DYN", "#009E73", "^"),
        ("md_logit", "MOMENT + ΔP", "#CC79A7", "D"),
        ("dyn_hgb", "DYN + HGB", "#7F7F7F", "v"),
    ]
    for m_key, label, color, marker in series:
        means = [agg13[("40p", m_key, p)]["mean"] for p in noise_levels]
        sds = [agg13[("40p", m_key, p)]["sample_sd"] for p in noise_levels]
        ax1.errorbar(
            x,
            means,
            yerr=sds,
            label=label,
            color=color,
            marker=marker,
            capsize=3,
            linewidth=2.0,
            markersize=5.5,
        )

    ax1.set_xticks(x, noise_labels)
    ax1.set_ylim(15, 103)
    ax1.set_xlabel("Additive Gaussian noise (% of global RMS / raw units)")
    ax1.set_ylabel("Validation Macro-F1 (%)")
    ax1.set_title("(a) Noise sensitivity at 40% label budget (E13b)", pad=10)
    ax1.legend(loc="lower left", frameon=True, fontsize=8.2)
    ax1.grid(True, alpha=0.25)

    # Panel B: E17 Accuracy vs Latency trade-off on RTX 4080 (batch=32 & batch=1)
    # Data from E17 summary and E12b 40p validation F1
    points = [
        ("DYN + HGB", 1.990, 11.430, 95.00, "#7F7F7F", "v"),
        ("TCN", 0.039, 0.882, 97.53, "#0072B2", "s"),
        ("TCN + DYN", 2.092, 12.595, 97.93, "#009E73", "^"),
        ("Full Stack\n(TCN+DYN+MOMENT/ΔP)", 11.128, 46.103, 98.09, "#D55E00", "o"),
    ]
    for name, lat32, lat1, f1, color, marker in points:
        ax2.scatter(lat32, f1, color=color, marker=marker, s=90, zorder=4, label=f"{name.splitlines()[0]} (bs=32)")
        ax2.scatter(lat1, f1, color=color, marker=marker, s=55, facecolors="none", linewidths=1.8, zorder=4)
        ax2.plot([lat32, lat1], [f1, f1], color=color, linestyle="--", linewidth=1.1, alpha=0.6)
        offset_y = -0.95 if "TCN + DYN" in name else (0.26 if "Full" in name else 0.18)
        ax2.annotate(
            f"{name.splitlines()[0]}\n({f1:.2f}%, {lat32:.2f}ms @bs32)",
            xy=(lat32, f1),
            xytext=(6, offset_y * 28),
            textcoords="offset points",
            fontsize=8.0,
            color=color,
            fontweight="bold" if "Full" in name or name == "TCN + DYN" else "normal",
        )

    ax2.set_xscale("log")
    ax2.set_xlim(0.02, 160)
    ax2.set_ylim(94.4, 98.6)
    ax2.set_xlabel("Online latency per sample (ms, log scale; filled=bs32, hollow=bs1)")
    ax2.set_ylabel("40%-Label Validation Macro-F1 (%)")
    ax2.set_title("(b) Accuracy–latency trade-off on RTX 4080 (E17)", pad=10)
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    out = FIG_DIR / "paper_e13_e17_robustness_cost_20261021"
    for ext in (".png", ".svg"):
        fig.savefig(out.with_suffix(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.png/.svg")


if __name__ == "__main__":
    plot_e09_e16_dual_stream_and_attribution()
    plot_e12b_fusion_ablation()
    plot_e13_e17_robustness_and_cost()
