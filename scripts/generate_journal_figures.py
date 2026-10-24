"""Generate standardized Chinese academic journal figures (PNG 300 DPI + SVG) from audited JSON records."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"


def setup_chinese_journal_style() -> None:
    """Register Windows Chinese fonts and configure publication-grade matplotlib rcParams."""
    font_candidates = [
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        Path("/mnt/c/Windows/Fonts/simsun.ttc"),
    ]
    for fp in font_candidates:
        if fp.exists():
            fm.fontManager.addfont(str(fp))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.0,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
        }
    )


def panel_caption(ax: plt.Axes, text: str, y: float = -0.2) -> None:
    """Place a journal-style sub-figure caption such as "(a) xxx" below the axes."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=10.5)


def load_json(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def save_both(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / stem
    for ext in (".png", ".svg"):
        fig.savefig(out.with_suffix(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}.png and .svg")


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = "#F8FAFC",
    ec: str = "#334155",
    lw: float = 1.3,
    fontsize: float = 9.2,
    fontweight: str = "normal",
    text_color: str = "#0F172A",
    radius: float = 0.018,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
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
        color=text_color,
        zorder=3,
        linespacing=1.35,
    )


def draw_arrow(
    ax: plt.Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    color: str = "#475569",
    lw: float = 1.4,
    style: str = "-|>",
) -> None:
    arrow = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        zorder=4,
    )
    ax.add_patch(arrow)


def plot_fig1_overall_framework() -> None:
    setup_chinese_journal_style()
    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Branch containers
    containers = [
        (0.66, 0.31, "#FFF7ED", "#D55E00", "#B45309", "分支Ⅰ  预训练时序大模型双流表征（创新 1、创新 2）"),
        (0.35, 0.27, "#F0FDF4", "#009E73", "#047857", "分支Ⅱ  原尺度动力学特征旁路（创新 3）"),
        (0.04, 0.27, "#EFF6FF", "#0072B2", "#1D4ED8", "分支Ⅲ  任务专用掩码 TCN"),
    ]
    for y0, h, fc, ec, tc, title in containers:
        draw_box(ax, 0.20, y0, 0.53, h, "", fc=fc, ec=ec, lw=1.4, radius=0.02)
        ax.text(0.212, y0 + h - 0.018, title, fontsize=9.6, fontweight="bold", color=tc, va="top")

    draw_box(
        ax,
        0.015,
        0.30,
        0.15,
        0.40,
        "变长充电功率序列\nx ∈ R^L\n（18 ≤ L ≤ 816）\n右侧补零至长度 1024\n────────────\n有效观测掩码\nm ∈ {0, 1}^1024",
        fc="#F1F5F9",
        ec="#334155",
        lw=1.5,
        fontsize=8.8,
        fontweight="bold",
    )

    # Branch I
    draw_box(ax, 0.215, 0.795, 0.13, 0.10, "原序列流 P\n逐序列 z-score", fc="#FFFFFF", ec="#D55E00", fontsize=8.5)
    draw_box(ax, 0.215, 0.675, 0.13, 0.10, "一阶差分流 ΔP\nΔP_t = P_t − P_(t−1)", fc="#FFFFFF", ec="#D55E00", fontsize=8.5)
    draw_box(
        ax,
        0.372,
        0.675,
        0.17,
        0.22,
        "冻结 MOMENT-1-large\n（24 层编码器，patch = 8）\n────────────\n创新 1：掩码感知 patch 池化\n仅聚合完整有效 patch",
        fc="#FFEDD5",
        ec="#C2410C",
        fontsize=8.3,
        fontweight="bold",
    )
    draw_box(
        ax,
        0.567,
        0.675,
        0.148,
        0.22,
        "创新 2：双流表征拼接\n[h_P ‖ h_ΔP]（2048 维）\n───────────\n多项逻辑回归\n输出概率 p^MD",
        fc="#FFFFFF",
        ec="#C2410C",
        fontsize=8.2,
    )

    # Branch II
    draw_box(
        ax,
        0.215,
        0.38,
        0.31,
        0.18,
        "60 维原尺度动力学特征（仅在 m_t = 1 区间计算）\n"
        "幅值与分位 17 · 粗糙度与转折 12 · 分段趋势 18\n"
        "频谱与自相关 7 · 形状 4 · 长度 2",
        fc="#DCFCE7",
        ec="#059669",
        fontsize=8.3,
    )
    draw_box(
        ax,
        0.55,
        0.38,
        0.165,
        0.18,
        "直方图梯度提升树\n（HGB）\n──────────\n输出概率 p^DYN",
        fc="#FFFFFF",
        ec="#059669",
        fontsize=8.5,
        fontweight="bold",
    )

    # Branch III
    draw_box(
        ax,
        0.215,
        0.07,
        0.31,
        0.18,
        "掩码因果膨胀卷积网络（Masked TCN）\n"
        "6 层，k = 3，d = 1, 2, 4, …, 32，128 通道\n"
        "感受野 127；掩码均值池化（与补零长度无关）",
        fc="#DBEAFE",
        ec="#1D4ED8",
        fontsize=8.3,
    )
    draw_box(
        ax,
        0.55,
        0.07,
        0.165,
        0.18,
        "Softmax 分类层\n固定 50 轮余弦退火\n──────────\n输出概率 p^TCN",
        fc="#FFFFFF",
        ec="#1D4ED8",
        fontsize=8.5,
        fontweight="bold",
    )

    # Meta learner
    draw_box(
        ax,
        0.758,
        0.17,
        0.232,
        0.66,
        "创新 4：交叉拟合概率堆叠\n"
        "══════════════\n"
        "训练子集内 5 折 OOF 概率\n"
        "z = [p^TCN, p^DYN, p^MD]\nz ∈ R^9\n"
        "──────────────\n"
        "多项逻辑回归元学习器\n"
        "（可选温度缩放 T）\n"
        "──────────────\n"
        "三分类输出\n"
        "正常 / 充电器故障 / 电池异常",
        fc="#FEF2F2",
        ec="#B91C1C",
        lw=1.6,
        fontsize=8.6,
        fontweight="bold",
        text_color="#7F1D1D",
    )

    # Input -> branches
    draw_arrow(ax, (0.172, 0.62), (0.21, 0.845), color="#D55E00", lw=1.5)
    draw_arrow(ax, (0.172, 0.58), (0.21, 0.725), color="#D55E00", lw=1.5)
    draw_arrow(ax, (0.172, 0.50), (0.21, 0.47), color="#009E73", lw=1.5)
    draw_arrow(ax, (0.172, 0.38), (0.21, 0.16), color="#0072B2", lw=1.5)
    # Inside branches (left -> right)
    draw_arrow(ax, (0.349, 0.845), (0.368, 0.82), color="#C2410C")
    draw_arrow(ax, (0.349, 0.725), (0.368, 0.75), color="#C2410C")
    draw_arrow(ax, (0.546, 0.785), (0.563, 0.785), color="#C2410C")
    draw_arrow(ax, (0.529, 0.47), (0.546, 0.47), color="#059669")
    draw_arrow(ax, (0.529, 0.16), (0.546, 0.16), color="#1D4ED8")
    # Branch outputs -> meta learner
    draw_arrow(ax, (0.72, 0.785), (0.76, 0.66), color="#B91C1C", lw=1.6)
    draw_arrow(ax, (0.72, 0.47), (0.76, 0.50), color="#B91C1C", lw=1.6)
    draw_arrow(ax, (0.72, 0.16), (0.76, 0.34), color="#B91C1C", lw=1.6)

    save_both(fig, "fig1_overall_framework")


def plot_fig2_pooling_and_head_stabilization() -> None:
    setup_chinese_journal_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # Panel (a): five-seed test Macro-F1 (docs/experiment_records/v100_results_20260720.md, sections 3 and 10;
    # v1 has 4 completed seeds). Same frozen MOMENT-1-large backbone, only pooling / downstream classifier differ.
    categories = [
        "默认池化（含补零 patch）\n+ 线性头",
        "掩码感知池化\n+ 线性头",
        "掩码感知池化\n+ RBF-SVM",
    ]
    values = [69.46, 77.41, 85.17]
    sds = [0.76, 0.89, 1.03]
    colors = ["#94A3B8", "#0072B2", "#D55E00"]

    x = np.arange(len(categories))
    bars = ax1.bar(x, values, width=0.52, yerr=sds, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val, sd_val in zip(bars, values, sds):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + sd_val + 0.8,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.annotate(
        "",
        xy=(0.74, 78.2),
        xytext=(0.12, 71.6),
        arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.2, "connectionstyle": "arc3,rad=-0.25"},
    )
    ax1.text(0.3, 79.2, "+7.95 pp", fontsize=9, color="#0072B2", ha="center")
    ax1.set_xticks(x, categories, fontsize=9)
    ax1.set_ylim(50, 92)
    ax1.set_ylabel("测试集 Macro-F1 (%)")
    panel_caption(ax1, "(a) 池化方式与下游分类器", y=-0.2)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.grid(False, axis="x")

    # Panel (b): E03D convergence trajectories (Original scale vs Fan-in gain compensation)
    root_e03d = ROOT / "artifacts/imports/e03d_head_validation_20261018"
    curve_specs = [
        ("fusion", "fanin_gain", "MOMENT+统计，尺度补偿（γ=1.00）", "#D55E00", "-"),
        ("fusion", "legacy_fresh", "MOMENT+统计，原尺度", "#D55E00", "--"),
        ("statistics", "fanin_gain", "统计特征，尺度补偿（γ=8.32）", "#009E73", "-"),
        ("statistics", "legacy_fresh", "统计特征，原尺度", "#009E73", "--"),
        ("moment", "fanin_gain", "MOMENT，尺度补偿（γ=1.01）", "#0072B2", "-"),
        ("moment", "legacy_fresh", "MOMENT，原尺度", "#0072B2", "--"),
    ]
    for branch, mode, label, color, ls in curve_specs:
        hist_path = root_e03d / f"e03d_head_diagnostic_20261018_v2_{branch}_{mode}/history.json"
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        epochs = [r["epoch"] for r in history]
        f1s = [r["validation_macro_f1"] * 100 for r in history]
        ax2.plot(
            epochs,
            f1s,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.8 if ls == "-" else 1.2,
            alpha=0.95 if ls == "-" else 0.72,
        )

    ax2.axvline(30, color="#64748B", linestyle=":", linewidth=1.1)
    ax2.text(33, 72.8, "第 30 轮：尺度补偿使\n融合头 +6.58 pp", fontsize=8.0, color="#475569")
    ax2.set_xlim(1, 200)
    ax2.set_ylim(71, 93.5)
    ax2.set_xlabel("训练轮次 (Epoch)")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax2, "(b) 冻结分类头收敛轨迹", y=-0.2)
    ax2.legend(loc="lower right", frameon=True, fontsize=7.8, ncol=1)
    ax2.grid(True, alpha=0.25)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig2_pooling_and_head_stabilization")


def plot_fig3_dual_stream_and_pretraining() -> None:
    setup_chinese_journal_style()
    e09 = load_json(
        "artifacts/imports/e09_analysis_corrected_20261020/analysis_four_seed_corrected/summary.json"
    )
    e16 = load_json(
        "artifacts/imports/e16_random_attribution_20261020/e16_random_attribution_20261020_v1/summary.json"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # Panel (a): Dual-stream MOMENT & DYN stacking across label budgets
    tags = ["1p", "10p", "40p"]
    tag_labels = ["1% 标注 (244)", "10% 标注 (2,456)", "40% 标注 (9,827)"]
    e09_map = {(r["method"], r["tag"]): r for r in e09["aggregate"]}

    methods_a = [
        ("moment_logit", "原始功率流 (P)", "#7F7F7F"),
        ("diff_logit", "一阶差分流 (ΔP)", "#56B4E9"),
        ("moment_diff_logit", "双流拼接 (P + ΔP)", "#D55E00"),
        ("stack_moment_diff", "双流 + DYN 概率堆叠", "#009E73"),
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
                fontsize=7.6,
                color="#222222",
                fontweight="bold" if m_key in ("moment_diff_logit", "stack_moment_diff") else "normal",
            )

    ax1.set_xticks(x, tag_labels)
    ax1.set_ylim(60, 102)
    ax1.set_xlabel("训练标注预算（样本数）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax1, "(a) 输入流与 DYN 堆叠", y=-0.24)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8.0, columnspacing=0.8, handletextpad=0.4)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.grid(False, axis="x")

    # Panel (b): Pretrained vs Random initialization (C=1.0 matched)
    e16_map = {(r["tag"], r["initialization"], r["stream"]): r for r in e16["aggregate"]}
    groups = [
        ("10p", "orig", "10% 原始流"),
        ("10p", "diff", "10% 差分流"),
        ("10p", "dual", "10% 双流"),
        ("40p", "orig", "40% 原始流"),
        ("40p", "diff", "40% 差分流"),
        ("40p", "dual", "40% 双流"),
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
        label="预训练 MOMENT-1-large",
        color="#0072B2",
        edgecolor="white",
        linewidth=0.6,
    )
    b_rnd = ax2.bar(
        x2 + w2 / 2,
        rnd_means,
        w2,
        yerr=rnd_sds,
        capsize=3,
        label="随机初始化同构编码器",
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
            color="#0072B2",
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
            fontsize=7.6,
            color="#D55E00",
            fontweight="bold",
        )

    ax2.set_xticks(x2, [g[2] for g in groups], fontsize=8.4)
    ax2.set_ylim(50, 102)
    ax2.set_xlabel("标注预算与输入流分支（固定 C = 1.0 对照协议）")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax2, "(b) 预训练权重与随机初始化", y=-0.24)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.0, columnspacing=1.0)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.grid(False, axis="x")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig3_dual_stream_and_pretraining")


def plot_fig4_dyn_feature_ablation() -> None:
    setup_chinese_journal_style()
    e08 = load_json(
        "artifacts/imports/e08_strong_features_20261020/e08_strong_features_20261020_v1/analysis/summary.json"
    )
    agg = {(r["method"], r["tag"]): r for r in e08["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # Panel (a): F15 vs DYN-60 across 1%, 10%, 40% budgets (SVM & HGB)
    tags = ["1p", "10p", "40p"]
    tag_labels = ["1% 标注 (244)", "10% 标注 (2,456)", "40% 标注 (9,827)"]
    methods = [
        ("f15_svm", "15维基础特征 + SVM", "#94A3B8"),
        ("f15_hgb", "15维基础特征 + HGB", "#56B4E9"),
        ("dyn_svm", "60维动态特征 (DYN) + SVM", "#E69F00"),
        ("dyn_hgb", "60维动态特征 (DYN) + HGB", "#009E73"),
    ]
    x = np.arange(len(tags))
    width = 0.19
    for i, (m_key, m_label, color) in enumerate(methods):
        means = [agg[(m_key, t)]["macro_f1_mean"] for t in tags]
        sds = [agg[(m_key, t)]["macro_f1_sd"] for t in tags]
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
                val + sd_val + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold" if m_key == "dyn_hgb" else "normal",
                color="#047857" if m_key == "dyn_hgb" else "#222222",
            )

    ax1.set_xticks(x, tag_labels)
    ax1.set_ylim(68, 100)
    ax1.set_xlabel("训练标注预算（样本数）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax1, "(a) F15 与 DYN 特征对比", y=-0.24)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.0, columnspacing=1.0)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.grid(False, axis="x")

    # Panel (b): Group ablation at 40% labels (only_group_f1 vs drop_group_f1)
    grp_map = {r["group"]: r for r in e08["group_ablation_40p"]}
    group_order = [
        ("roughness", "局部粗糙度与拐点 (12维)"),
        ("trend", "分段趋势与斜率 (18维)"),
        ("amplitude", "原尺度幅值与分位数 (17维)"),
        ("spectral", "FFT频谱与自相关 (7维)"),
        ("shape", "形态回撤与高阶矩 (4维)"),
        ("length", "有效观测长度占比 (2维)"),
    ]
    y = np.arange(len(group_order))
    only_vals = [grp_map[g]["only_group_f1"] for g, _ in group_order]
    drop_vals = [grp_map[g]["drop_group_f1"] for g, _ in group_order]
    deltas = [grp_map[g]["drop_delta_pp"] for g, _ in group_order]

    h = 0.36
    b1 = ax2.barh(
        y + h / 2,
        only_vals,
        height=h,
        label="仅保留该组特征 (Keep-One)",
        color="#0072B2",
        edgecolor="white",
        alpha=0.9,
    )
    b2 = ax2.barh(
        y - h / 2,
        drop_vals,
        height=h,
        label="剔除该组特征 (Drop-One)",
        color="#D55E00",
        edgecolor="white",
        alpha=0.88,
    )
    ax2.axvline(95.003, color="#009E73", linestyle="--", linewidth=1.3, label="完整60维 DYN 基线 (95.00%)")

    for bar, val in zip(b1, only_vals):
        if val > 88.0:
            ax2.text(
                val - 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%",
                va="center",
                ha="right",
                fontsize=7.6,
                color="white",
                fontweight="bold",
            )
        else:
            ax2.text(
                val + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%",
                va="center",
                ha="left",
                fontsize=7.6,
                color="#0072B2",
            )
    for bar, val, d_pp in zip(b2, drop_vals, deltas):
        delta_txt = f" ({d_pp:+.2f}pp)" if abs(d_pp) >= 0.1 else ""
        if val < 92.0:
            ax2.text(
                val - 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%{delta_txt}",
                va="center",
                ha="right",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )
        else:
            ax2.text(
                max(val, 95.0) + 0.75,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%{delta_txt}",
                va="center",
                ha="left",
                fontsize=7.5,
                color="#D55E00",
                fontweight="bold" if d_pp < -0.3 else "normal",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )

    ax2.set_yticks(y, [label for _, label in group_order], fontsize=8.4)
    ax2.invert_yaxis()
    ax2.set_xlim(35, 108)
    ax2.set_xlabel("40% 标注预算验证集 Macro-F1 (%)")
    panel_caption(ax2, "(b) DYN 特征分组消融（40% 标注）", y=-0.24)
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.48, 1.01),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.grid(False, axis="y")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig4_dyn_feature_ablation")


def plot_fig5_stacking_ablation_and_low_label() -> None:
    setup_chinese_journal_style()
    e12b = load_json(
        "artifacts/imports/e12b_fusion_ablation_corrected_20261020/e12b_fusion_ablation_corrected/summary.json"
    )
    per_seed = {(r["tag"], r["method"]): np.asarray(r["per_seed"]) for r in e12b["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [1.05, 1.0]})

    # Panel (a): seed-paired net gains (E12b corrected; seeds 42/44/45/46 in identical order)
    pairs = [
        ("tcn_meta", "tcn", "同型元学习器自身：TCN+meta − TCN"),
        ("full", "tcn_meta", "完整融合 − TCN+meta"),
        ("tcn+dyn", "tcn", "加入 DYN：TCN+DYN − TCN"),
        ("tcn+md", "tcn", "加入 MOMENT/ΔP：TCN+MD − TCN"),
        ("dyn+md", "dyn", "加入 MOMENT/ΔP：DYN+MD − DYN"),
        ("full", "dyn+md", "TCN 边际：Full − (DYN+MD)"),
        ("full", "tcn+md", "DYN 边际：Full − (TCN+MD)"),
        ("full", "tcn+dyn", "MOMENT/ΔP 边际：Full − (TCN+DYN)"),
    ]
    y = np.arange(len(pairs))
    h = 0.36
    for k, (tag, color, label) in enumerate(
        [("10p", "#56B4E9", "10% 标注 (2,456)"), ("40p", "#D55E00", "40% 标注 (9,827)")]
    ):
        diffs = [per_seed[(tag, l)] - per_seed[(tag, r)] for l, r, _ in pairs]
        means = [float(d.mean()) for d in diffs]
        sds = [float(d.std(ddof=1)) for d in diffs]
        pos = [int((d > 0).sum()) for d in diffs]
        offs = y - h / 2 if k == 0 else y + h / 2
        ax1.barh(offs, means, xerr=sds, height=h, capsize=2.5, color=color, edgecolor="white", label=label)
        for yy, m, sd_val, n_pos in zip(offs, means, sds, pos):
            ax1.text(m + sd_val + 0.12, yy, f"{m:+.2f}（{n_pos}/4）", va="center", fontsize=7.4, color="#333333")
    ax1.axvline(0, color="#475569", linewidth=0.9)
    ax1.set_yticks(y, [p[2] for p in pairs], fontsize=8.2)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.3, 7.6)
    ax1.set_xlabel("配对 Macro-F1 差值 (pp，均值 ± 种子间 SD)")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.0)
    ax1.grid(True, axis="x", alpha=0.25)
    ax1.grid(False, axis="y")
    panel_caption(ax1, "(a) 支路贡献的配对拆解（E12b）", y=-0.16)

    # Panel (b): unified-protocol label-budget curves (E18 rerun; fixed hyper-parameters, no validation selection)
    e18 = load_json("artifacts/imports/e18_low_label_20261020/e18_low_label/e18_low_label_20261020_v1/summary.json")
    agg18 = {(r["tag"], r["method"]): r for r in e18["aggregate"]}
    tags = ["1p", "5p", "10p", "20p", "40p"]
    x_pct = [1, 5, 10, 20, 40]
    series = [
        ("moment_logit", "MOMENT 原序列", "#CC79A7", "s", ":"),
        ("moment_diff_logit", "MOMENT 双流（P+ΔP）", "#E69F00", "D", "--"),
        ("f15_hgb", "F15 + HGB", "#999999", "v", ":"),
        ("dyn_hgb", "DYN + HGB", "#009E73", "^", "--"),
        ("tcn", "Masked TCN", "#0072B2", "o", "-"),
        ("tcn_dyn", "TCN + DYN", "#56B4E9", "P", "-"),
        ("dyn_moment_diff", "DYN + MOMENT/ΔP", "#A6761D", "X", "-"),
        ("full", "三路融合（本文）", "#B91C1C", "*", "-"),
    ]
    for key, label, color, marker, ls in series:
        m = [agg18[(t, key)]["mean"] for t in tags]
        sd = [agg18[(t, key)]["sample_sd"] for t in tags]
        ax2.errorbar(
            x_pct,
            m,
            yerr=sd,
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            capsize=2.5,
            linewidth=2.3 if key == "full" else 1.5,
            markersize=8 if key == "full" else 5,
        )
    ax2.set_xscale("log")
    ax2.set_xticks(x_pct, [f"{v}%" for v in x_pct])
    ax2.minorticks_off()
    ax2.set_xlim(0.8, 50)
    ax2.set_ylim(50, 100)
    ax2.set_xlabel("训练标注比例（对数坐标）")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    ax2.legend(loc="lower right", frameon=True, fontsize=7.6, ncol=2)
    ax2.grid(True, which="major", alpha=0.25)
    panel_caption(ax2, "(b) 统一协议下的标注比例曲线（E18）", y=-0.16)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig5_stacking_ablation_and_low_label")


def plot_fig6_robustness_and_latency() -> None:
    setup_chinese_journal_style()
    e13 = load_json(
        "artifacts/imports/e13_robustness_20261020/e13_robustness/e13b_robustness_20261020_v1/summary.json"
    )
    agg13 = {(r["tag"], r["method"], r["perturbation"]): r for r in e13["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # Panel (a): Noise sensitivity at 40% label budget
    noise_levels = ["clean", "noise_0p1pct", "noise_0p25pct", "noise_0p5pct", "noise_1pct"]
    noise_labels = [
        "无噪声\n(0)",
        "0.10% RMS\n(0.19 u)",
        "0.25% RMS\n(0.48 u)",
        "0.50% RMS\n(0.96 u)",
        "1.00% RMS\n(1.92 u)",
    ]
    x = np.arange(len(noise_levels))

    series = [
        ("full", "三路异构堆叠 (Full Stack)", "#D55E00", "o"),
        ("tcn", "TCN 单分支", "#0072B2", "s"),
        ("tcn_dyn", "TCN + DYN 双路堆叠", "#009E73", "^"),
        ("md_logit", "双流 MOMENT + ΔP", "#CC79A7", "D"),
        ("dyn_hgb", "60维动态特征 (DYN + HGB)", "#7F7F7F", "v"),
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

    ax1.set_xticks(x, noise_labels, fontsize=8.4)
    ax1.set_ylim(15, 103)
    ax1.set_xlabel("有效观测区间加性高斯噪声强度（全局 RMS 百分比 / 原尺度单位）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax1, "(a) 加性高斯噪声敏感性（40% 标注）", y=-0.27)
    ax1.legend(loc="lower left", frameon=True, fontsize=8.0)
    ax1.grid(True, alpha=0.25)

    # Panel (b): Accuracy vs Latency trade-off on RTX 4080 (batch=32 & batch=1)
    points = [
        ("DYN + HGB", 1.990, 11.430, 95.00, "#7F7F7F", "v"),
        ("TCN 单分支", 0.039, 0.882, 97.53, "#0072B2", "s"),
        ("TCN + DYN", 2.092, 12.595, 97.93, "#009E73", "^"),
        ("三路异构堆叠 (Full Stack)", 11.128, 46.103, 98.09, "#D55E00", "o"),
    ]
    for name, lat32, lat1, f1, color, marker in points:
        ax2.scatter(lat32, f1, color=color, marker=marker, s=90, zorder=4, label=f"{name} (bs=32)")
        ax2.scatter(lat1, f1, color=color, marker=marker, s=55, facecolors="none", linewidths=1.8, zorder=4)
        ax2.plot([lat32, lat1], [f1, f1], color=color, linestyle="--", linewidth=1.1, alpha=0.6)
        offset_y = -24 if name == "TCN + DYN" else (8 if "Full" in name else 6)
        ax2.annotate(
            f"{name}\n({f1:.2f}%, {lat32:.2f} ms @bs=32)",
            xy=(lat32, f1),
            xytext=(6, offset_y),
            textcoords="offset points",
            fontsize=8.0,
            color=color,
            fontweight="bold" if "Full" in name or name == "TCN + DYN" else "normal",
        )

    ax2.set_xscale("log")
    ax2.set_xlim(0.02, 160)
    ax2.set_ylim(94.4, 98.6)
    ax2.set_xlabel("单样本在线推理时延 (ms，对数坐标；实心=批大小32，空心=批大小1)")
    ax2.set_ylabel("40% 标注预算验证集 Macro-F1 (%)")
    panel_caption(ax2, "(b) 精度—推理时延权衡（RTX 4080）", y=-0.2)
    ax2.grid(True, which="both", alpha=0.25)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig6_robustness_and_latency")


def plot_fig7_tcn_and_calibration() -> None:
    setup_chinese_journal_style()
    e05c = load_json(
        "artifacts/imports/e05c_tcn_replication_20261019/terminal_artifacts/summary.json"
    )
    e10 = load_json(
        "artifacts/imports/e10_calibration_safety_20261020/e10_calibration_safety/summary.json"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel (a): E05C TCN architecture & masking replication across 10%, 40%, 100%
    agg05 = {(r["condition"], r["fraction"]): r for r in e05c["aggregate"]}
    fracs = [0.1, 0.4, 1.0]
    frac_labels = ["10% 标注 (2,456)", "40% 标注 (9,827)", "100% 标注 (24,569)"]
    conds = [
        ("current_fixed50", "残差 TCN（RF=61，掩码池化）", "#94A3B8"),
        ("legacy_unmasked50", "6 层因果 TCN（RF=127，无掩码池化）", "#56B4E9"),
        ("legacy_masked50", "6 层因果 TCN（RF=127，掩码池化）", "#0072B2"),
    ]
    x = np.arange(len(fracs))
    w = 0.25
    for i, (c_key, c_label, color) in enumerate(conds):
        means = [agg05[(c_key, f)]["macro_f1_mean_percent"] for f in fracs]
        sds = [agg05[(c_key, f)]["macro_f1_sample_sd_percent"] for f in fracs]
        bars = ax1.bar(
            x + (i - 1) * w,
            means,
            w,
            yerr=sds,
            capsize=3,
            label=c_label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        for bar, val, sd_val in zip(bars, means, sds):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + sd_val + 0.5,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold" if c_key == "legacy_masked50" else "normal",
                color="#0072B2" if c_key == "legacy_masked50" else "#333333",
            )

    ax1.text(
        0.02,
        0.96,
        "补零长度 1024 → 1536 时：\n"
        "无掩码池化：最大 logit 漂移 418～531，预测改变 100/33 条\n"
        "掩码池化：logit 漂移为 0",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        color="#1E3A8A",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#EFF6FF", "edgecolor": "#93C5FD", "alpha": 0.94},
    )
    ax1.set_xticks(x, frac_labels)
    ax1.set_ylim(62, 113)
    ax1.set_xlabel("训练标注预算（样本数）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax1, "(a) TCN 结构与池化方式", y=-0.24)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.8)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.grid(False, axis="x")

    # Panel (b): E10 Length-stratified comparison & calibration summary
    strata_labels = ["短序列组", "中等序列组", "长序列组"]
    x2 = np.arange(3)
    w2 = 0.19
    seeds = ["42", "44", "45", "46"]

    def get_strata_stats(tag: str, method: str) -> tuple[list[float], list[float]]:
        means, sds = [], []
        for g in range(3):
            vals = [e10["tags"][tag]["length_strata"][s]["by_method"][method][g]["macro_f1"] * 100 for s in seeds]
            means.append(float(np.mean(vals)))
            sds.append(float(np.std(vals, ddof=1)))
        return means, sds

    s_specs = [
        ("10p", "tcn_full", "10% TCN（固定 50 轮）", "#94A3B8"),
        ("10p", "stack_tcn", "10% 三路异构堆叠", "#56B4E9"),
        ("40p", "tcn_full", "40% TCN（固定 50 轮）", "#0072B2"),
        ("40p", "stack_tcn", "40% 三路异构堆叠", "#D55E00"),
    ]
    for i, (t_key, m_key, s_label, color) in enumerate(s_specs):
        means, sds = get_strata_stats(t_key, m_key)
        bars = ax2.bar(
            x2 + (i - 1.5) * w2,
            means,
            w2,
            yerr=sds,
            capsize=2.5,
            label=s_label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        for bar, val, sd_val in zip(bars, means, sds):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + sd_val + 0.4,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold" if "stack" in m_key else "normal",
                color=color if "stack" in m_key else "#333333",
            )

    long10 = get_strata_stats("10p", "stack_tcn")[0][2] - get_strata_stats("10p", "tcn_full")[0][2]
    long40 = get_strata_stats("40p", "stack_tcn")[0][2] - get_strata_stats("40p", "tcn_full")[0][2]
    ax2.annotate(
        f"长序列组相对 TCN：\n10% 标注 {long10:+.2f} pp\n40% 标注 {long40:+.2f} pp",
        xy=(2 - 0.5 * w2, 95.0),
        xytext=(1.42, 99.8),
        arrowprops=None,
        fontsize=7.5,
        fontweight="bold",
        color="#0369A1",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F0F9FF", "edgecolor": "#BAE6FD", "alpha": 0.94},
    )
    ax2.text(
        0.02,
        0.96,
        "温度缩放（元学习器 OOF 拟合）：\n"
        "10% 堆叠 T=0.853，ECE 3.52% → 0.94%\n"
        "40% 堆叠 T=1.001，ECE 0.56%",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=7.4,
        color="#7F1D1D",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#FEF2F2", "edgecolor": "#FECACA", "alpha": 0.94},
    )

    ax2.set_xticks(x2, strata_labels)
    ax2.set_ylim(76, 105.5)
    ax2.set_xlabel("有效序列长度分组（各 seed 训练集三分位点，约 266 / 399）")
    ax2.set_ylabel("验证集分组 Macro-F1 (%)")
    panel_caption(ax2, "(b) 按序列长度分层的性能", y=-0.27)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.8)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.grid(False, axis="x")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig7_tcn_and_calibration")


if __name__ == "__main__":
    plot_fig1_overall_framework()
    plot_fig2_pooling_and_head_stabilization()
    plot_fig3_dual_stream_and_pretraining()
    plot_fig4_dyn_feature_ablation()
    plot_fig5_stacking_ablation_and_low_label()
    plot_fig6_robustness_and_latency()
    plot_fig7_tcn_and_calibration()
