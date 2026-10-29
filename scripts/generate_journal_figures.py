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

from tcn_moment.label_semantics import LABEL_NAMES_ZH

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
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Serif CJK SC", "DejaVu Sans"],
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


def save_both(fig: plt.Figure, stem: str, *, extra_dir: Path | None = None) -> None:
    dirs = [FIG_DIR]
    if extra_dir is not None:
        dirs.append(extra_dir)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        out = d / stem
        for ext in (".png", ".svg"):
            fig.savefig(out.with_suffix(ext), dpi=300, bbox_inches="tight")
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
    fc: str = "#FAFAFA",
    ec: str = "#1F2937",
    lw: float = 1.3,
    ls: str = "-",
    fontsize: float = 9.2,
    fontweight: str = "normal",
    text_color: str = "#111111",
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
        color=text_color,
        zorder=3,
        linespacing=1.35,
    )


def draw_arrow(
    ax: plt.Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    color: str = "#222222",
    lw: float = 1.4,
    ls: str = "-",
    style: str = "-|>",
) -> None:
    arrow = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        linestyle=ls,
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

    # Branch containers (pure monochrome grayscale + distinct border line styles)
    containers = [
        (0.66, 0.31, "#F7F7F7", "#111111", "-", "分支 I  预训练时序大模型双流表征（创新 1、创新 2）"),
        (0.35, 0.27, "#FAFAFA", "#333333", "--", "分支 II  原尺度动力学特征旁路（创新 3）"),
        (0.04, 0.27, "#F2F2F2", "#555555", "-.", "分支 III  任务专用掩码 TCN"),
    ]
    for y0, h, fc, ec, ls, title in containers:
        draw_box(ax, 0.20, y0, 0.53, h, "", fc=fc, ec=ec, lw=1.4, ls=ls, radius=0.02)
        ax.text(0.212, y0 + h - 0.018, title, fontsize=9.6, fontweight="bold", color="#111111", va="top")

    draw_box(
        ax,
        0.015,
        0.30,
        0.15,
        0.40,
        "变长充电功率序列\nx ∈ R^L\n（18 ≤ L ≤ 816）\n右侧补零至长度 1024\n────────────\n有效观测掩码\nm ∈ {0, 1}^1024",
        fc="#EDEDED",
        ec="#111111",
        lw=1.5,
        fontsize=8.8,
        fontweight="bold",
    )

    # Branch I
    draw_box(ax, 0.215, 0.795, 0.13, 0.10, "原序列流 P\n逐序列 z-score", fc="#FFFFFF", ec="#222222", fontsize=8.5)
    draw_box(ax, 0.215, 0.675, 0.13, 0.10, "一阶差分流 ΔP\nΔP_t = P_t − P_(t−1)", fc="#FFFFFF", ec="#222222", ls="--", fontsize=8.5)
    draw_box(
        ax,
        0.372,
        0.675,
        0.17,
        0.22,
        "冻结 MOMENT-1-large\n（24 层编码器，patch = 8）\n────────────\n创新 1：掩码感知 patch 池化\n仅聚合完整有效 patch",
        fc="#E8E8E8",
        ec="#111111",
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
        ec="#111111",
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
        fc="#ECECEC",
        ec="#333333",
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
        ec="#333333",
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
        fc="#E5E5E5",
        ec="#444444",
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
        ec="#444444",
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
        "（可选温度缩放 τ）\n"
        "──────────────\n"
        "三分类输出（原始标签顺序）\n"
        "0：正常\n1：电池异常\n2：充电器故障",
        fc="#EFEFEF",
        ec="#000000",
        lw=1.8,
        fontsize=8.6,
        fontweight="bold",
        text_color="#000000",
    )

    # Input -> branches
    draw_arrow(ax, (0.172, 0.62), (0.21, 0.845), color="#111111", lw=1.5, ls="-")
    draw_arrow(ax, (0.172, 0.58), (0.21, 0.725), color="#222222", lw=1.5, ls="--")
    draw_arrow(ax, (0.172, 0.50), (0.21, 0.47), color="#333333", lw=1.5, ls="-.")
    draw_arrow(ax, (0.172, 0.38), (0.21, 0.16), color="#444444", lw=1.5, ls=":")
    # Inside branches (left -> right)
    draw_arrow(ax, (0.349, 0.845), (0.368, 0.82), color="#111111")
    draw_arrow(ax, (0.349, 0.725), (0.368, 0.75), color="#222222", ls="--")
    draw_arrow(ax, (0.546, 0.785), (0.563, 0.785), color="#111111")
    draw_arrow(ax, (0.529, 0.47), (0.546, 0.47), color="#333333")
    draw_arrow(ax, (0.529, 0.16), (0.546, 0.16), color="#444444")
    # Branch outputs -> meta learner
    draw_arrow(ax, (0.72, 0.785), (0.76, 0.66), color="#000000", lw=1.6, ls="-")
    draw_arrow(ax, (0.72, 0.47), (0.76, 0.50), color="#222222", lw=1.6, ls="--")
    draw_arrow(ax, (0.72, 0.16), (0.76, 0.34), color="#444444", lw=1.6, ls="-.")

    save_both(fig, "fig2_overall_framework", extra_dir=FIG_DIR / "label_semantics_v2" / "paper")


def plot_fig2_pooling_and_head_stabilization() -> None:
    setup_chinese_journal_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # Panel (a): five-seed test Macro-F1 progression curve across adaptation stages (monochrome)
    categories = [
        "默认池化\n+ 线性头",
        "掩码池化\n+ 线性头",
        "掩码池化\n+ 解冻2层",
        "掩码池化\n+ RBF-SVM",
        "掩码池化\n+ 完全微调",
    ]
    values = [69.46, 77.41, 81.50, 85.17, 95.37]
    sds = [0.76, 0.89, 1.06, 1.03, 0.39]

    x = np.arange(len(categories))
    ax1.fill_between(
        x,
        np.array(values) - np.array(sds),
        np.array(values) + np.array(sds),
        color="#666666",
        alpha=0.18,
    )
    ax1.plot(
        x,
        values,
        color="#111111",
        marker="o",
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        linewidth=2.2,
        markersize=6.5,
        label="MOMENT-1-large 适配路线",
    )
    ax1.errorbar(x, values, yerr=sds, fmt="none", ecolor="#111111", capsize=3.5, elinewidth=1.2)
    ax1.axhline(95.99, color="#333333", linestyle="--", linewidth=1.5, label="原始尺度 TCN 基线 (95.99%)")
    ax1.axhline(88.89, color="#666666", linestyle=":", linewidth=1.4, label="F15 + 随机森林 (88.89%)")

    for xi, val, sd_val in zip(x, values, sds):
        offset_y = -4.2 if xi == 4 else (sd_val + 1.2)
        ax1.text(
            xi,
            val + offset_y,
            f"{val:.2f}%",
            ha="center",
            va="bottom" if offset_y > 0 else "top",
            fontsize=8.2,
            fontweight="bold" if xi in (1, 3) else "normal",
            color="#111111",
        )
    ax1.annotate(
        "掩码池化修复\n+7.95 pp",
        xy=(1.0, 77.41),
        xytext=(0.15, 83.2),
        arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.3, "connectionstyle": "arc3,rad=-0.2"},
        fontsize=8.2,
        fontweight="bold",
        color="#111111",
        ha="center",
    )
    ax1.set_xticks(x, categories, fontsize=8.4)
    ax1.set_xlim(-0.35, 4.35)
    ax1.set_ylim(64, 100)
    ax1.set_xlabel("MOMENT 池化与适配阶段（全量标注）")
    ax1.set_ylabel("测试集 Macro-F1 (%)")
    panel_caption(ax1, "(a) 池化修复与适配方式演进曲线", y=-0.24)
    ax1.legend(loc="lower right", frameon=True, fontsize=7.8)
    ax1.grid(True, alpha=0.25)

    # Panel (b): E03D convergence trajectories (monochrome with distinct line styles + sparse markers)
    root_e03d = ROOT / "artifacts/imports/e03d_head_validation_20261018"
    curve_specs = [
        ("fusion", "fanin_gain", "MOMENT+统计，尺度补偿（γ=1.00）", "#000000", "-", "o", "#000000"),
        ("fusion", "legacy_fresh", "MOMENT+统计，原尺度", "#222222", "--", "o", "#FFFFFF"),
        ("statistics", "fanin_gain", "统计特征，尺度补偿（γ=8.32）", "#444444", "-.", "s", "#444444"),
        ("statistics", "legacy_fresh", "统计特征，原尺度", "#555555", ":", "s", "#FFFFFF"),
        ("moment", "fanin_gain", "MOMENT，尺度补偿（γ=1.01）", "#666666", "-", "^", "#666666"),
        ("moment", "legacy_fresh", "MOMENT，原尺度", "#777777", "--", "^", "#FFFFFF"),
    ]
    for branch, mode, label, color, ls, marker, mfc in curve_specs:
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
            marker=marker,
            markevery=25,
            markersize=4.8,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.2,
            linewidth=1.9 if mode == "fanin_gain" else 1.3,
        )

    ax2.axvline(30, color="#444444", linestyle=":", linewidth=1.1)
    ax2.text(33, 72.8, "第 30 轮：尺度补偿使\n融合头 +6.58 pp", fontsize=8.0, color="#111111")
    ax2.set_xlim(1, 200)
    ax2.set_ylim(71, 93.5)
    ax2.set_xlabel("训练轮次 (Epoch)")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax2, "(b) 冻结分类头收敛轨迹", y=-0.2)
    ax2.legend(loc="lower right", frameon=True, fontsize=7.6, ncol=1)
    ax2.grid(True, alpha=0.25)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig3_pooling_and_head_stabilization")


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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

    # Panel (a): Dual-stream MOMENT & DYN stacking curves across label budgets (monochrome)
    tags = ["1p", "10p", "40p"]
    x_pct = [1, 10, 40]
    tag_labels = ["1%\n(244)", "10%\n(2,456)", "40%\n(9,827)"]
    e09_map = {(r["method"], r["tag"]): r for r in e09["aggregate"]}

    methods_a = [
        ("moment_logit", "原始功率流 (P)", "#666666", "s", ":", "#FFFFFF"),
        ("diff_logit", "一阶差分流 (ΔP)", "#444444", "^", "-.", "#FFFFFF"),
        ("moment_diff_logit", "双流拼接 (P + ΔP)", "#222222", "o", "--", "#222222"),
        ("stack_moment_diff", "双流 + DYN 概率堆叠", "#000000", "D", "-", "#000000"),
    ]
    for m_key, m_label, color, marker, ls, mfc in methods_a:
        means = [e09_map[(m_key, t)]["mean"] for t in tags]
        sds = [e09_map[(m_key, t)]["sample_sd"] for t in tags]
        ax1.errorbar(
            x_pct,
            means,
            yerr=sds,
            marker=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle=ls,
            color=color,
            linewidth=2.1 if m_key in ("moment_diff_logit", "stack_moment_diff") else 1.6,
            markersize=6.2,
            capsize=3,
            label=m_label,
        )
        for xi, val, sd_val in zip(x_pct, means, sds):
            if m_key == "stack_moment_diff":
                dx, dy, ha, va = (1.0, sd_val + 0.65, "center", "bottom")
            elif m_key == "moment_diff_logit":
                if xi == 1:
                    dx, dy, ha, va = (1.16, 0.2, "left", "center")
                else:
                    dx, dy, ha, va = (0.86, 0.35, "right", "bottom")
            elif m_key == "diff_logit":
                dx, dy, ha, va = (1.0, -sd_val - 0.75, "center", "top")
            else:
                dx, dy, ha, va = (1.0, -sd_val - 0.75, "center", "top")
            ax1.text(
                xi * dx,
                val + dy,
                f"{val:.2f}",
                ha=ha,
                va=va,
                fontsize=7.6,
                color="#111111" if m_key in ("moment_diff_logit", "stack_moment_diff") else "#333333",
                fontweight="bold" if m_key in ("moment_diff_logit", "stack_moment_diff") else "normal",
            )

    ax1.set_xscale("log")
    ax1.set_xticks(x_pct, tag_labels)
    ax1.minorticks_off()
    ax1.set_xlim(0.7, 58)
    ax1.set_ylim(65, 100)
    ax1.set_xlabel("训练标注预算（样本数，对数坐标）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax1, "(a) 不同输入流与 DYN 堆叠的样本效率曲线", y=-0.24)
    ax1.legend(loc="lower right", frameon=True, fontsize=7.8)
    ax1.grid(True, alpha=0.25)

    # Panel (b): Pretrained (solid + filled markers) vs Random init (dashed + hollow markers)
    e16_map = {(r["tag"], r["initialization"], r["stream"]): r for r in [*e16["aggregate"], *e16b["aggregate"]]}
    tags_b = ["1p", "5p", "10p", "20p", "40p"]
    x_b = [1, 5, 10, 20, 40]
    stream_specs = [
        ("dual", "pretrained", "预训练 双流 (P+ΔP)", "#000000", "o", "-", "#000000"),
        ("dual", "random", "随机初始化 双流 (P+ΔP)", "#222222", "o", "--", "#FFFFFF"),
        ("diff", "pretrained", "预训练 差分流 (ΔP)", "#333333", "^", "-", "#333333"),
        ("diff", "random", "随机初始化 差分流 (ΔP)", "#555555", "^", "--", "#FFFFFF"),
        ("orig", "pretrained", "预训练 原序列流 (P)", "#555555", "s", "-.", "#555555"),
        ("orig", "random", "随机初始化 原序列流 (P)", "#777777", "s", ":", "#FFFFFF"),
    ]
    for stream, init, label, color, marker, ls, mfc in stream_specs:
        means = [e16_map[(t, init, stream)]["mean"] for t in tags_b]
        sds = [e16_map[(t, init, stream)]["sample_sd"] for t in tags_b]
        ax2.errorbar(
            x_b,
            means,
            yerr=sds,
            marker=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle=ls,
            color=color,
            linewidth=2.0 if init == "pretrained" else 1.4,
            markersize=5.8 if init == "pretrained" else 5.2,
            capsize=2.5,
            label=label,
        )

    # Highlight pretraining net gain on dual stream at 1%, 10%, 40%
    for xi, t in [(1, "1p"), (10, "10p"), (40, "40p")]:
        vp = e16_map[(t, "pretrained", "dual")]["mean"]
        vr = e16_map[(t, "random", "dual")]["mean"]
        ax2.annotate(
            f"双流 +{vp - vr:.2f}pp",
            xy=(xi, vp),
            xytext=(xi, vp + 2.6),
            ha="center",
            fontsize=7.4,
            color="#000000",
            fontweight="bold",
        )

    ax2.set_xscale("log")
    ax2.set_xticks(x_b, [f"{v}%" for v in x_b])
    ax2.minorticks_off()
    ax2.set_xlim(0.75, 55)
    ax2.set_ylim(42, 99)
    ax2.set_xlabel("训练标注比例（固定 C = 1.0 同架构对照，对数坐标）")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax2, "(b) 预训练权重与随机初始化的样本效率曲线", y=-0.24)
    ax2.legend(loc="lower right", frameon=True, fontsize=7.2, ncol=2)
    ax2.grid(True, alpha=0.25)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig5_dual_stream_and_pretraining")


def plot_fig4_dyn_feature_ablation() -> None:
    setup_chinese_journal_style()
    e08 = load_json(
        "artifacts/imports/e08_strong_features_20261020/e08_strong_features_20261020_v1/analysis/summary.json"
    )
    agg = {(r["method"], r["tag"]): r for r in e08["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

    # Panel (a): F15 vs DYN-60 curves across available budgets (monochrome)
    methods = [
        ("dyn_hgb", "60维动力学特征 (DYN) + HGB", "#000000", "o", "-", "#000000", ["1p", "5p", "10p", "40p", "100p"]),
        ("dyn_svm", "60维动力学特征 (DYN) + RBF-SVM", "#333333", "s", "-.", "#FFFFFF", ["1p", "10p", "40p"]),
        ("f15_hgb", "15维基础统计 (F15) + HGB", "#444444", "^", "--", "#444444", ["1p", "5p", "10p", "40p", "100p"]),
        ("f15_svm", "15维基础统计 (F15) + RBF-SVM", "#666666", "v", ":", "#FFFFFF", ["1p", "10p", "40p"]),
    ]
    for m_key, m_label, color, marker, ls, mfc, m_tags in methods:
        x_m = [int(t[:-1]) for t in m_tags]
        means = [agg[(m_key, t)]["macro_f1_mean"] for t in m_tags]
        sds = [agg[(m_key, t)]["macro_f1_sd"] for t in m_tags]
        ax1.errorbar(
            x_m,
            means,
            yerr=sds,
            marker=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle=ls,
            color=color,
            linewidth=2.2 if m_key == "dyn_hgb" else 1.6,
            markersize=6.0,
            capsize=3,
            label=m_label,
        )
        for xi, val, sd_val in zip(x_m, means, sds):
            if m_key in ("dyn_hgb", "f15_hgb") and xi in (1, 10, 40, 100):
                if m_key == "dyn_hgb":
                    dx, dy, ha, va = (1.0, sd_val + 0.45, "center", "bottom")
                elif xi == 1:
                    dx, dy, ha, va = (1.14, 0.25, "left", "bottom")
                elif xi == 10:
                    dx, dy, ha, va = (0.88, 0.35, "right", "bottom")
                elif xi == 40:
                    dx, dy, ha, va = (1.12, 0.15, "left", "bottom")
                else:
                    dx, dy, ha, va = (1.0, -sd_val - 0.55, "center", "top")
                ax1.text(
                    xi * dx,
                    val + dy,
                    f"{val:.2f}",
                    ha=ha,
                    va=va,
                    fontsize=7.5,
                    fontweight="bold" if m_key == "dyn_hgb" else "normal",
                    color="#000000" if m_key == "dyn_hgb" else "#333333",
                )

    x_pct = [1, 5, 10, 40, 100]
    ax1.set_xscale("log")
    ax1.set_xticks(x_pct, [f"{v}%" for v in x_pct])
    ax1.minorticks_off()
    ax1.set_xlim(0.75, 135)
    ax1.set_ylim(72, 98.5)
    ax1.set_xlabel("训练标注比例（对数坐标）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax1, "(a) F15 与 DYN 特征在不同标注预算下的性能曲线", y=-0.24)
    ax1.legend(loc="lower right", frameon=True, fontsize=7.8)
    ax1.grid(True, alpha=0.25)

    # Panel (b): Group ablation profile at 40% labels (Cleveland dot-and-line profile, monochrome)
    grp_map = {r["group"]: r for r in e08["group_ablation_40p"]}
    group_order = [
        ("roughness", "局部粗糙度与转折 (12维)"),
        ("trend", "分段趋势与斜率 (18维)"),
        ("amplitude", "原尺度幅值与分位 (17维)"),
        ("spectral", "FFT频谱与自相关 (7维)"),
        ("shape", "形态回撤与高阶矩 (4维)"),
        ("length", "有效观测长度占比 (2维)"),
    ]
    y = np.arange(len(group_order))
    only_vals = [grp_map[g]["only_group_f1"] for g, _ in group_order]
    drop_vals = [grp_map[g]["drop_group_f1"] for g, _ in group_order]
    deltas = [grp_map[g]["drop_delta_pp"] for g, _ in group_order]

    ax2.axvline(95.003, color="#222222", linestyle=":", linewidth=1.5, label="完整 60 维 DYN 基线 (95.00%)")
    for yi, ov, dv in zip(y, only_vals, drop_vals):
        ax2.plot([ov, dv], [yi, yi], color="#BDBDBD", linewidth=2.2, zorder=1)

    ax2.plot(
        only_vals,
        y,
        marker="o",
        markerfacecolor="#FFFFFF",
        markeredgecolor="#333333",
        markeredgewidth=1.6,
        linestyle="--",
        color="#333333",
        linewidth=1.6,
        markersize=6.5,
        label="仅保留该组特征 (Keep-One)",
        zorder=3,
    )
    ax2.plot(
        drop_vals,
        y,
        marker="D",
        markerfacecolor="#000000",
        markeredgecolor="#000000",
        linestyle="-",
        color="#000000",
        linewidth=1.8,
        markersize=6.0,
        label="剔除该组特征 (Drop-One)",
        zorder=3,
    )

    for yi, ov, dv, d_pp in zip(y, only_vals, drop_vals, deltas):
        if yi == 0:
            ax2.text(ov - 1.2, yi + 0.25, f"{ov:.2f}%", va="top", ha="right", fontsize=7.5, color="#333333", fontweight="bold")
            ax2.text(dv + 1.0, yi - 0.12, f"{dv:.2f}% ({d_pp:+.2f}pp)", va="bottom", ha="left", fontsize=7.5, color="#000000", fontweight="bold")
        elif yi == 5:
            ax2.text(ov + 1.5, yi - 0.16, f"{ov:.2f}%", va="bottom", ha="left", fontsize=7.5, color="#333333")
            ax2.text(dv + 0.8, yi - 0.12, f"{dv:.2f}%", va="bottom", ha="left", fontsize=7.5, color="#000000")
        else:
            ax2.text(
                ov - 1.2,
                yi,
                f"{ov:.2f}%",
                va="center",
                ha="right",
                fontsize=7.5,
                color="#333333",
            )
            delta_txt = f" ({d_pp:+.2f}pp)" if abs(d_pp) >= 0.1 else ""
            ax2.text(
                dv + 0.8,
                yi,
                f"{dv:.2f}%{delta_txt}",
                va="center",
                ha="left",
                fontsize=7.5,
                color="#000000",
                fontweight="bold" if d_pp < -0.3 else "normal",
            )

    ax2.set_yticks(y, [label for _, label in group_order], fontsize=8.4)
    ax2.invert_yaxis()
    ax2.set_xlim(35, 108)
    ax2.set_ylim(5.6, -0.65)
    ax2.set_xlabel("40% 标注预算验证集 Macro-F1 (%)")
    panel_caption(ax2, "(b) DYN 特征分组消融轮廓（40% 标注）", y=-0.24)
    ax2.legend(loc="upper left", frameon=True, fontsize=7.4)
    ax2.grid(True, alpha=0.25)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig6_dyn_feature_ablation")


def plot_fig5_stacking_ablation_and_low_label() -> None:
    setup_chinese_journal_style()
    e12b = load_json(
        "artifacts/imports/e12b_fusion_ablation_corrected_20261020/e12b_fusion_ablation_corrected/summary.json"
    )
    per_seed = {(r["tag"], r["method"]): np.asarray(r["per_seed"]) for r in e12b["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [1.05, 1.0]})

    # Panel (a): seed-paired net gains (forest-plot / errorbar profile; seeds 42/44/45/46 in identical order)
    pairs = [
        ("tcn_meta", "tcn", "同型元学习器：TCN+meta − TCN"),
        ("full", "tcn_meta", "完整融合 − TCN+meta"),
        ("tcn+dyn", "tcn", "加入 DYN：TCN+DYN − TCN"),
        ("tcn+md", "tcn", "加入 MOMENT/ΔP：TCN+MD − TCN"),
        ("dyn+md", "dyn", "加入 MOMENT/ΔP：DYN+MD − DYN"),
        ("full", "dyn+md", "TCN 边际：Full − (DYN+MD)"),
        ("full", "tcn+md", "DYN 边际：Full − (TCN+MD)"),
        ("full", "tcn+dyn", "MOMENT/ΔP 边际：Full − (TCN+DYN)"),
    ]
    y = np.arange(len(pairs))
    h = 0.16
    for k, (tag, color, marker, mfc, ls, label) in enumerate(
        [
            ("10p", "#000000", "o", "#000000", "-", "10% 标注 (2,456)"),
            ("40p", "#444444", "s", "#FFFFFF", "--", "40% 标注 (9,827)"),
        ]
    ):
        diffs = [per_seed[(tag, l)] - per_seed[(tag, r)] for l, r, _ in pairs]
        means = [float(d.mean()) for d in diffs]
        sds = [float(d.std(ddof=1)) for d in diffs]
        pos = [int((d > 0).sum()) for d in diffs]
        offs = y - h if k == 0 else y + h
        ax1.errorbar(
            means,
            offs,
            xerr=sds,
            fmt=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle=ls,
            color=color,
            linewidth=1.5,
            markersize=5.8,
            capsize=2.8,
            label=label,
        )
        for yy, m, sd_val, n_pos in zip(offs, means, sds, pos):
            ax1.text(
                m + sd_val + 0.12,
                yy,
                f"{m:+.2f} ({n_pos}/4)",
                va="center",
                fontsize=7.3,
                color=color,
                fontweight="bold" if k == 0 and m > 0.7 else "normal",
            )
    ax1.axvline(0, color="#333333", linestyle=":", linewidth=1.1)
    ax1.set_yticks(y, [p[2] for p in pairs], fontsize=8.2)
    ax1.invert_yaxis()
    ax1.set_xlim(-0.4, 7.8)
    ax1.set_xlabel("配对 Macro-F1 差值 (pp，均值 ± 种子间 SD)")
    ax1.legend(loc="lower right", frameon=True, fontsize=8.0)
    ax1.grid(True, alpha=0.25)
    panel_caption(ax1, "(a) 支路贡献的逐种子配对差值曲线", y=-0.16)

    # Panel (b): unified-protocol label-budget curves across 1%..100% (E18 + E18-100, monochrome)
    e18 = load_json("artifacts/imports/e18_low_label_20261020/e18_low_label/e18_low_label_20261020_v1/summary.json")
    e18_100 = load_json(
        "artifacts/imports/e18_low_label_100p_20261020/e18_low_label_100p/e18_low_label_100p_20261020_v1/summary.json"
    )
    agg18 = {(r["tag"], r["method"]): r for r in [*e18["aggregate"], *e18_100["aggregate"]]}
    tags = ["1p", "5p", "10p", "20p", "40p", "100p"]
    x_pct = [1, 5, 10, 20, 40, 100]
    series = [
        ("moment_logit", "MOMENT 原序列", "#666666", "s", ":", "#FFFFFF"),
        ("moment_diff_logit", "MOMENT 双流（P+ΔP）", "#444444", "D", "--", "#FFFFFF"),
        ("f15_hgb", "F15 + HGB", "#777777", "v", ":", "#777777"),
        ("dyn_hgb", "DYN + HGB", "#333333", "^", "-.", "#333333"),
        ("tcn", "Masked TCN", "#111111", "o", "--", "#FFFFFF"),
        ("tcn_dyn", "TCN + DYN", "#222222", "P", "-.", "#222222"),
        ("dyn_moment_diff", "DYN + MOMENT/ΔP", "#444444", "X", "-", "#444444"),
        ("full", "完整融合（本文）", "#000000", "*", "-", "#000000"),
    ]
    for key, label, color, marker, ls, mfc in series:
        m = [agg18[(t, key)]["mean"] for t in tags]
        sd = [agg18[(t, key)]["sample_sd"] for t in tags]
        ax2.errorbar(
            x_pct,
            m,
            yerr=sd,
            label=label,
            color=color,
            marker=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.3,
            linestyle=ls,
            capsize=2.5,
            linewidth=2.3 if key == "full" else 1.5,
            markersize=7.8 if key == "full" else 5.4,
        )
    ax2.set_xscale("log")
    ax2.set_xticks(x_pct, [f"{v}%" for v in x_pct])
    ax2.minorticks_off()
    ax2.set_xlim(0.75, 130)
    ax2.set_ylim(50, 100)
    ax2.set_xlabel("训练标注比例（对数坐标）")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    ax2.legend(loc="lower right", frameon=True, fontsize=7.5, ncol=2)
    ax2.grid(True, which="major", alpha=0.25)
    panel_caption(ax2, "(b) 统一协议下 1%～100% 标注比例曲线", y=-0.16)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig7_stacking_ablation_and_low_label")


def plot_fig6_robustness_and_latency() -> None:
    setup_chinese_journal_style()
    e13 = load_json(
        "artifacts/imports/e13_robustness_20261020/e13_robustness/e13b_robustness_20261020_v1/summary.json"
    )
    agg13 = {(r["tag"], r["method"], r["perturbation"]): r for r in e13["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # Panel (a): Noise sensitivity at 40% label budget (monochrome)
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
        ("full", "三路异构堆叠 (Full Stack)", "#000000", "o", "-", "#000000"),
        ("tcn", "TCN 单分支", "#222222", "s", "--", "#FFFFFF"),
        ("tcn_dyn", "TCN + DYN 双路堆叠", "#333333", "^", "-.", "#333333"),
        ("md_logit", "双流 MOMENT + ΔP", "#444444", "D", "--", "#444444"),
        ("dyn_hgb", "60维动态特征 (DYN + HGB)", "#666666", "v", ":", "#FFFFFF"),
    ]
    for m_key, label, color, marker, ls, mfc in series:
        means = [agg13[("40p", m_key, p)]["mean"] for p in noise_levels]
        sds = [agg13[("40p", m_key, p)]["sample_sd"] for p in noise_levels]
        ax1.errorbar(
            x,
            means,
            yerr=sds,
            label=label,
            color=color,
            marker=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle=ls,
            capsize=3,
            linewidth=2.1 if m_key == "full" else 1.6,
            markersize=5.8,
        )

    ax1.set_xticks(x, noise_labels, fontsize=8.4)
    ax1.set_ylim(15, 103)
    ax1.set_xlabel("有效观测区间加性高斯噪声强度（全局 RMS 百分比 / 原尺度单位）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax1, "(a) 加性高斯噪声敏感性（40% 标注）", y=-0.27)
    ax1.legend(loc="lower left", frameon=True, fontsize=8.0)
    ax1.grid(True, alpha=0.25)

    # Panel (b): Accuracy vs Latency trade-off on RTX 4080 (batch=32 & batch=1, monochrome)
    points = [
        ("DYN + HGB", 1.990, 11.430, 95.00, "#555555", "v"),
        ("TCN 单分支", 0.039, 0.882, 97.53, "#222222", "s"),
        ("TCN + DYN", 2.092, 12.595, 97.93, "#333333", "^"),
        ("三路异构堆叠 (Full Stack)", 11.128, 46.103, 98.09, "#000000", "o"),
    ]
    # Connect Pareto frontier at batch=32 (TCN -> TCN+DYN -> Full Stack)
    ax2.plot(
        [0.039, 2.092, 11.128],
        [97.53, 97.93, 98.09],
        color="#666666",
        linestyle="-.",
        linewidth=1.3,
        zorder=2,
        label="批大小 32 帕累托前沿",
    )
    for name, lat32, lat1, f1, color, marker in points:
        ax2.scatter(lat32, f1, color=color, marker=marker, s=90, zorder=4, label=f"{name} (bs=32)")
        ax2.scatter(lat1, f1, edgecolors=color, marker=marker, s=60, facecolors="#FFFFFF", linewidths=1.8, zorder=4)
        ax2.plot([lat32, lat1], [f1, f1], color=color, linestyle="--", linewidth=1.1, alpha=0.7)
        offset_y = -24 if name == "TCN + DYN" else (8 if "Full" in name else 6)
        ax2.annotate(
            f"{name}\n({f1:.2f}%, {lat32:.2f} ms @bs=32)",
            xy=(lat32, f1),
            xytext=(6, offset_y),
            textcoords="offset points",
            fontsize=8.0,
            color="#000000" if "Full" in name else "#222222",
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
    save_both(fig, "fig8_robustness_and_latency")


def plot_fig7_tcn_and_calibration() -> None:
    setup_chinese_journal_style()
    e05c = load_json(
        "artifacts/imports/e05c_tcn_replication_20261019/terminal_artifacts/summary.json"
    )
    n2 = load_json(
        "artifacts/imports/n2_tcn_low_label_20261020/n2_tcn_low_label_20261020_v1/summary.json"
    )
    e18_deep = load_json(
        "artifacts/analysis/e18_deep_analysis_semantics_v2/summary.json"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel (a): E05C + N2 TCN architecture & masking curves across label budgets (monochrome)
    agg05 = {(r["condition"], r["fraction"]): r for r in e05c["aggregate"]}
    agg_n2 = {r["fraction"]: r for r in n2["aggregate"]}

    x_10_100 = [10, 40, 100]
    fracs_10_100 = [0.1, 0.4, 1.0]
    x_full = [1, 5, 10, 40, 100]
    masked_means = [
        agg_n2[0.01]["macro_f1_mean"],
        agg_n2[0.05]["macro_f1_mean"],
        *(agg05[("legacy_masked50", f)]["macro_f1_mean_percent"] for f in fracs_10_100),
    ]
    masked_sds = [
        agg_n2[0.01]["macro_f1_sd"],
        agg_n2[0.05]["macro_f1_sd"],
        *(agg05[("legacy_masked50", f)]["macro_f1_sample_sd_percent"] for f in fracs_10_100),
    ]
    unmasked_means = [agg05[("legacy_unmasked50", f)]["macro_f1_mean_percent"] for f in fracs_10_100]
    unmasked_sds = [agg05[("legacy_unmasked50", f)]["macro_f1_sample_sd_percent"] for f in fracs_10_100]
    res_means = [agg05[("current_fixed50", f)]["macro_f1_mean_percent"] for f in fracs_10_100]
    res_sds = [agg05[("current_fixed50", f)]["macro_f1_sample_sd_percent"] for f in fracs_10_100]

    ax1.errorbar(
        x_full,
        masked_means,
        yerr=masked_sds,
        marker="o",
        markerfacecolor="#000000",
        markeredgecolor="#000000",
        linestyle="-",
        color="#000000",
        linewidth=2.2,
        markersize=6.0,
        capsize=3,
        label="6 层因果 TCN（RF=127，掩码池化）",
    )
    ax1.errorbar(
        x_10_100,
        unmasked_means,
        yerr=unmasked_sds,
        marker="^",
        markerfacecolor="#FFFFFF",
        markeredgecolor="#333333",
        markeredgewidth=1.4,
        linestyle="--",
        color="#333333",
        linewidth=1.7,
        markersize=5.8,
        capsize=3,
        label="6 层因果 TCN（RF=127，无掩码池化）",
    )
    ax1.errorbar(
        x_10_100,
        res_means,
        yerr=res_sds,
        marker="s",
        markerfacecolor="#666666",
        markeredgecolor="#666666",
        linestyle=":",
        color="#666666",
        linewidth=1.8,
        markersize=5.5,
        capsize=3,
        label="4 块残差 TCN（RF=61，掩码池化）",
    )

    for xi, val in zip(x_full, masked_means):
        if xi == 1:
            ax1.text(xi * 1.15, val - 2.0, f"{val:.2f}", ha="left", va="top", fontsize=7.6, color="#000000", fontweight="bold")
        elif xi == 5:
            ax1.text(xi * 0.88, val + 2.2, f"{val:.2f}", ha="right", va="bottom", fontsize=7.6, color="#000000", fontweight="bold")
        else:
            ax1.text(xi, val + 2.0, f"{val:.2f}", ha="center", va="bottom", fontsize=7.6, color="#000000", fontweight="bold")
    for xi, val in zip(x_10_100, res_means):
        ax1.text(xi, val - 2.6, f"{val:.2f}", ha="center", va="top", fontsize=7.5, color="#444444")

    ax1.text(
        0.03,
        0.96,
        "补零长度 1024 → 1536 时：\n"
        "无掩码池化：最大 logit 漂移 418～531，预测改变 100/33 条\n"
        "掩码池化：logit 漂移严格为 0",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=7.4,
        color="#111111",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F5F5F5", "edgecolor": "#666666", "alpha": 0.94},
    )
    ax1.set_xscale("log")
    ax1.set_xticks(x_full, ["1%", "5%", "10%", "40%", "100%"])
    ax1.minorticks_off()
    ax1.set_xlim(0.75, 135)
    ax1.set_ylim(38, 112)
    ax1.set_xlabel("训练标注比例（对数坐标）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    panel_caption(ax1, "(a) TCN 结构复原与掩码池化的样本效率曲线", y=-0.24)
    ax1.legend(loc="lower right", frameon=True, fontsize=7.6)
    ax1.grid(True, alpha=0.25)

    # Panel (b): Per-seed training-tertile length-stratified curves (corrected semantics v2, monochrome)
    strata_labels = ["短序列组\n(L ≤ 266)", "中等序列组\n(266 < L ≤ 399)", "长序列组\n(L > 399)"]
    x2 = np.arange(3)
    s_specs = [
        ("40p", "full", "40% 完整融合", "#000000", "o", "-", "#000000"),
        ("40p", "tcn", "40% Masked TCN", "#333333", "s", "--", "#FFFFFF"),
        ("10p", "full", "10% 完整融合", "#222222", "D", "-.", "#222222"),
        ("10p", "tcn", "10% Masked TCN", "#555555", "^", ":", "#FFFFFF"),
    ]
    for t_key, m_key, s_label, color, marker, ls, mfc in s_specs:
        rec = e18_deep["length_strata"][t_key][m_key]
        means = rec["mean"]
        sds = rec["sd"]
        ax2.errorbar(
            x2,
            means,
            yerr=sds,
            marker=marker,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle=ls,
            color=color,
            linewidth=2.1 if m_key == "full" else 1.6,
            markersize=6.0,
            capsize=3,
            label=s_label,
        )
        for xi, val, sd_val in zip(x2, means, sds):
            if t_key == "40p" and m_key == "full":
                dy, va = 0.65, "bottom"
            elif t_key == "40p" and m_key == "tcn":
                dy, va = -0.75, "top"
            elif t_key == "10p" and m_key == "full":
                dy, va = -1.25 if xi < 2 else 0.75, ("top" if xi < 2 else "bottom")
            else:
                dy, va = -sd_val - 0.45, "top"
            ax2.text(
                xi,
                val + dy,
                f"{val:.2f}",
                ha="center",
                va=va,
                fontsize=7.4,
                fontweight="bold" if m_key == "full" else "normal",
                color=color,
            )

    long10 = e18_deep["length_strata"]["10p"]["full"]["mean"][2] - e18_deep["length_strata"]["10p"]["tcn"]["mean"][2]
    long40 = e18_deep["length_strata"]["40p"]["full"]["mean"][2] - e18_deep["length_strata"]["40p"]["tcn"]["mean"][2]
    ax2.annotate(
        f"长序列组相对 TCN：\n10% 标注 {long10:+.2f} pp\n40% 标注 {long40:+.2f} pp",
        xy=(2.0, e18_deep["length_strata"]["10p"]["full"]["mean"][2]),
        xytext=(1.05, 84.2),
        arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.2, "connectionstyle": "arc3,rad=-0.15"},
        fontsize=7.5,
        fontweight="bold",
        color="#111111",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F5F5F5", "edgecolor": "#555555", "alpha": 0.94},
    )
    ax2.text(
        0.03,
        0.25,
        "温度缩放（元学习器 OOF 拟合）：\n"
        "10% 堆叠 τ=0.853，ECE 3.52% → 0.94%\n"
        "40% 堆叠 τ=1.001，ECE 0.56%",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=7.4,
        color="#111111",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#FAFAFA", "edgecolor": "#777777", "alpha": 0.94},
    )

    ax2.set_xticks(x2, strata_labels)
    ax2.set_xlim(-0.25, 2.3)
    ax2.set_ylim(77.5, 101)
    ax2.set_xlabel("有效序列长度分组（各 seed 训练集三分位点）")
    ax2.set_ylabel("验证集分组 Macro-F1 (%)")
    panel_caption(ax2, "(b) 按序列长度分层的性能退化曲线", y=-0.24)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.8)
    ax2.grid(True, alpha=0.25)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig4_tcn_and_length_strata")


def plot_fig1_dataset_overview() -> None:
    """Training-split only: class-wise example curves and valid-length ECDF curves (monochrome)."""
    import csv

    setup_chinese_journal_style()
    split_candidates = [
        ROOT / "artifacts/improvements/e01/e01_20261017_v2_pilot_seed42/split_manifest.json",
        ROOT / "artifacts/imports/e01_validation_20261017/runs/e01_20261017_v2_pilot_seed42/split_manifest.json",
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
            values = [float(v) for v in row["charging_powers_str"].replace('"', "").split(",") if v.strip()]
            by_class[str(row["InsertedColumn"]).strip()].append(values)

    names = LABEL_NAMES_ZH
    fig = plt.figure(figsize=(11.6, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], hspace=0.56, wspace=0.28)

    # Panels (a)-(c): three random examples per class (fixed RNG, monochrome with distinct line styles)
    rng = np.random.default_rng(42)
    sample_styles = [
        ("#000000", "-", "样例 1"),
        ("#444444", "--", "样例 2"),
        ("#777777", "-.", "样例 3"),
    ]
    for col, cls in enumerate(("0", "1", "2")):
        ax = fig.add_subplot(gs[0, col])
        seqs = by_class[cls]
        chosen = rng.choice(len(seqs), size=3, replace=False)
        for idx, (sc, sls, slabel) in zip(chosen, sample_styles):
            ax.plot(np.arange(len(seqs[idx])), seqs[idx], color=sc, linestyle=sls, linewidth=1.25, label=slabel)
        ax.set_xlabel("有效采样点序号")
        if col == 0:
            ax.set_ylabel("充电功率（原始记录单位）")
            ax.legend(loc="upper right", frameon=True, fontsize=7.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        panel_caption(ax, f"({'abc'[col]}) {names[cls]}（类别 {cls}）样例", y=-0.3)

    # Panel (d): smooth empirical cumulative distribution curves (ECDF) per class (monochrome + markers)
    ax = fig.add_subplot(gs[1, :])
    class_styles = {
        "0": ("#000000", "-", "o", "#000000"),
        "1": ("#333333", "--", "s", "#FFFFFF"),
        "2": ("#666666", "-.", "^", "#666666"),
    }
    for cls in ("0", "1", "2"):
        lengths = np.sort(np.asarray([len(s) for s in by_class[cls]]))
        ecdf = np.arange(1, len(lengths) + 1) / len(lengths) * 100.0
        c_col, c_ls, c_mk, c_mfc = class_styles[cls]
        med = int(np.median(lengths))
        step = max(1, len(lengths) // 10)
        ax.plot(
            lengths,
            ecdf,
            linewidth=2.0,
            color=c_col,
            linestyle=c_ls,
            marker=c_mk,
            markevery=step,
            markersize=5.2,
            markerfacecolor=c_mfc,
            markeredgecolor=c_col,
            markeredgewidth=1.3,
            label=f"{names[cls]}（类别 {cls}，n={len(lengths):,}，中位数 {med}）",
        )
        ax.axvline(med, color=c_col, linestyle=":", linewidth=1.1, alpha=0.8)

    ax.axhline(50.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.text(32, 52.5, "中位累积概率 (50%)", fontsize=7.8, color="#333333")
    ax.set_xlim(0, 830)
    ax.set_ylim(0, 103)
    ax.set_xlabel("有效序列长度 L（采样点）")
    ax.set_ylabel("经验累积概率 (%)")
    ax.legend(loc="lower right", frameon=True, fontsize=8.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_caption(ax, "(d) 训练集各类有效长度经验累积分布曲线 (ECDF)", y=-0.3)
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
