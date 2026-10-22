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

    # Background group containers
    draw_box(
        ax, 0.21, 0.65, 0.51, 0.31, "", fc="#FFF7ED", ec="#D55E00", lw=1.4, radius=0.02
    )
    ax.text(
        0.225,
        0.935,
        "分支 I：双流时序大模型表征分支（创新一、创新二、创新四）",
        fontsize=9.8,
        fontweight="bold",
        color="#B45309",
        va="top",
    )

    draw_box(
        ax, 0.21, 0.34, 0.51, 0.27, "", fc="#F0FDF4", ec="#009E73", lw=1.4, radius=0.02
    )
    ax.text(
        0.225,
        0.585,
        "分支 II：原尺度物理动态特征旁路（创新三：补偿 RevIN 幅值擦除）",
        fontsize=9.8,
        fontweight="bold",
        color="#047857",
        va="top",
    )

    draw_box(
        ax, 0.21, 0.04, 0.51, 0.26, "", fc="#EFF6FF", ec="#0072B2", lw=1.4, radius=0.02
    )
    ax.text(
        0.225,
        0.278,
        "分支 III：局部高分辨率因果膨胀卷积分支（掩码不变性恢复）",
        fontsize=9.8,
        fontweight="bold",
        color="#1D4ED8",
        va="top",
    )

    # Input box
    draw_box(
        ax,
        0.015,
        0.34,
        0.155,
        0.32,
        "变长充电功率序列\n"
        "X ∈ R^L (18 ≤ L ≤ 1024)\n"
        "────────────────\n"
        "同步生成二值掩码\n"
        "M ∈ {0, 1}^1024\n"
        "(有效观测=1, 补零=0)",
        fc="#F1F5F9",
        ec="#334155",
        lw=1.5,
        fontsize=9.0,
        fontweight="bold",
    )

    # Branch I internal boxes
    draw_box(
        ax,
        0.225,
        0.78,
        0.145,
        0.11,
        "原始功率流 P\nZ-score + 掩码对齐",
        fc="#FFFFFF",
        ec="#D55E00",
        fontsize=8.5,
    )
    draw_box(
        ax,
        0.225,
        0.665,
        0.145,
        0.10,
        "一阶差分流 ΔP\n瞬态跳变与斜率捕获",
        fc="#FFFFFF",
        ec="#D55E00",
        fontsize=8.5,
    )
    draw_box(
        ax,
        0.39,
        0.67,
        0.155,
        0.22,
        "冻结 MOMENT-1-large\n(24层 Transformer, Patch=8)\n──────────────\n创新一：掩码感知池化\n仅聚合 M_patch=1 块\n消除右侧补零稀释",
        fc="#FFEDD5",
        ec="#C2410C",
        fontsize=8.2,
        fontweight="bold",
    )
    draw_box(
        ax,
        0.565,
        0.67,
        0.14,
        0.22,
        "双流拼接表征\n[e_orig || e_diff] (2048维)\n───────────\nOOF 逻辑回归 / SVM\n或 创新四：扇入增益\nγ=√(1039/B) 融合头",
        fc="#FFFFFF",
        ec="#C2410C",
        fontsize=8.0,
    )

    # Branch II internal boxes
    draw_box(
        ax,
        0.225,
        0.365,
        0.295,
        0.18,
        "60 维原尺度功率动态特征提取 (DYN)\n"
        "仅在 M_t=1 真实观测区间计算，保留绝对幅值：\n"
        "• 局部粗糙度 (12)  • 分段趋势斜率 (18)  • 幅值分位数 (17)\n"
        "• FFT频谱与自相关 (7)  • 形态极值 (4)  • 序列长度 (2)",
        fc="#DCFCE7",
        ec="#059669",
        fontsize=8.1,
    )
    draw_box(
        ax,
        0.545,
        0.365,
        0.16,
        0.18,
        "直方图梯度提升树\n(DYN-HGB)\n──────────\n非线性特征交互建模\n输出折外概率 p^DYN",
        fc="#FFFFFF",
        ec="#059669",
        fontsize=8.4,
        fontweight="bold",
    )

    # Branch III internal boxes
    draw_box(
        ax,
        0.225,
        0.065,
        0.295,
        0.175,
        "恢复版 6 层因果膨胀卷积网络 (Masked TCN)\n"
        "膨胀率 d ∈ {1,2,4,8,16,32}, 128 通道, 卷积核 k=3\n"
        "SpatialDropout1D + 掩码均值池化 (补零不变性)",
        fc="#DBEAFE",
        ec="#1D4ED8",
        fontsize=8.0,
    )
    draw_box(
        ax,
        0.545,
        0.065,
        0.16,
        0.175,
        "时域局部形态分类器\n(5 折训练验证早停)\n──────────\n捕获高频微秒/秒级扰动\n输出折外概率 p^TCN",
        fc="#FFFFFF",
        ec="#1D4ED8",
        fontsize=8.4,
        fontweight="bold",
    )

    # Stacking & Calibration box (Innovation 5)
    draw_box(
        ax,
        0.755,
        0.18,
        0.23,
        0.66,
        "创新五：三路异构 5 折 OOF\n概率堆叠与温度缩放校准\n"
        "══════════════════\n"
        "折外对数几率拼接特征：\n"
        "z_i = [log p^TCN, log p^DYN,\n"
        "       log p^(MOMENT/ΔP)] ∈ R^9\n"
        "──────────────────\n"
        "正则化多项式逻辑回归元学习器\n"
        "+ 温度缩放校准 (T > 0)\n"
        "──────────────────\n"
        "三分类安全诊断输出：\n"
        "① 正常充电曲线 (Class 0)\n"
        "② 充电设备故障 (Class 1)\n"
        "③ 动力电池异常 (Class 2)\n"
        "(40%标注 Macro-F1: 98.09%\n"
        " 电池异常召回率: 97.38%)",
        fc="#FEF2F2",
        ec="#B91C1C",
        lw=1.6,
        fontsize=8.4,
        fontweight="bold",
        text_color="#7F1D1D",
    )

    # Arrows from Input to 3 branches
    draw_arrow(ax, (0.17, 0.56), (0.225, 0.835), color="#D55E00", lw=1.5)
    draw_arrow(ax, (0.17, 0.53), (0.225, 0.715), color="#D55E00", lw=1.5)
    draw_arrow(ax, (0.17, 0.48), (0.225, 0.455), color="#009E73", lw=1.5)
    draw_arrow(ax, (0.17, 0.42), (0.225, 0.155), color="#0072B2", lw=1.5)

    # Arrows inside Branch I
    draw_arrow(ax, (0.37, 0.835), (0.39, 0.81), color="#C2410C")
    draw_arrow(ax, (0.37, 0.715), (0.39, 0.74), color="#C2410C")
    draw_arrow(ax, (0.545, 0.78), (0.565, 0.78), color="#C2410C")

    # Arrows inside Branch II & III
    draw_arrow(ax, (0.52, 0.455), (0.545, 0.455), color="#059669")
    draw_arrow(ax, (0.52, 0.155), (0.545, 0.155), color="#1D4ED8")

    # Arrows to Stacking
    draw_arrow(ax, (0.705, 0.78), (0.755, 0.64), color="#B91C1C", lw=1.6)
    draw_arrow(ax, (0.705, 0.455), (0.755, 0.50), color="#B91C1C", lw=1.6)
    draw_arrow(ax, (0.705, 0.155), (0.755, 0.36), color="#B91C1C", lw=1.6)

    save_both(fig, "fig1_overall_framework")


def plot_fig2_pooling_and_head_stabilization() -> None:
    setup_chinese_journal_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # Panel (a): Pooling strategy comparison across classifiers (from audited E01/E02/M04)
    categories = [
        "补零全局均值\n+ Linear 探针",
        "末端块 (Last)\n+ Linear 探针",
        "掩码感知均值\n+ Linear 探针",
        "补零全局均值\n+ RBF-SVM",
        "掩码分位数池化\n+ RBF-SVM",
        "掩码感知均值\n+ RBF-SVM",
    ]
    values = [69.46, 51.31, 77.41, 83.21, 83.39, 85.17]
    colors = ["#94A3B8", "#CBD5E1", "#0072B2", "#94A3B8", "#56B4E9", "#D55E00"]

    x = np.arange(len(categories))
    bars = ax1.bar(x, values, width=0.58, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        is_best = val in (77.41, 85.17)
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.8,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if is_best else "normal",
            color="#D55E00" if val == 85.17 else ("#0072B2" if val == 77.41 else "#333333"),
        )

    ax1.annotate(
        "消除补零稀释\n+7.95 pp",
        xy=(2, 77.41),
        xytext=(0.75, 81.5),
        arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.2},
        fontsize=8.3,
        fontweight="bold",
        color="#0072B2",
        ha="center",
    )
    ax1.set_xticks(x, categories, fontsize=8.2)
    ax1.set_ylim(45, 92)
    ax1.set_ylabel("Macro-F1 (%)")
    ax1.set_title("(a) 变长序列不同池化策略对冻结大模型表征的影响", pad=10)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.grid(False, axis="x")

    # Panel (b): E03D convergence trajectories (Original scale vs Fan-in gain compensation)
    root_e03d = ROOT / "artifacts/imports/e03d_head_validation_20261018"
    curve_specs = [
        ("fusion", "fanin_gain", "融合分支（扇入增益补偿 γ=1.00）", "#D55E00", "-"),
        ("fusion", "legacy_fresh", "融合分支（未缩放原始输入）", "#D55E00", "--"),
        ("statistics", "fanin_gain", "统计分支（扇入增益补偿 γ=8.32）", "#009E73", "-"),
        ("statistics", "legacy_fresh", "统计分支（未缩放原始输入）", "#009E73", "--"),
        ("moment", "fanin_gain", "MOMENT分支（扇入增益补偿 γ=1.01）", "#0072B2", "-"),
        ("moment", "legacy_fresh", "MOMENT分支（未缩放原始输入）", "#0072B2", "--"),
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
    ax2.text(33, 72.8, "早期预算边界 (30轮)\n融合头 +6.58 pp", fontsize=8.0, color="#475569")
    ax2.set_xlim(1, 200)
    ax2.set_ylim(71, 93.5)
    ax2.set_xlabel("训练轮次 (Epoch)")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    ax2.set_title("(b) 有效扇入增益补偿与固定 AMP 缩放的收敛轨迹对比", pad=10)
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
    ax1.set_title("(a) 一阶差分双流表征与动态特征堆叠在不同预算下的增益", pad=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.2)
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
    ax2.set_title("(b) 预训练时序大模型权重与随机初始化编码器的归因对比", pad=10)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.2)
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
    ax1.set_title("(a) 60维原尺度动态特征 (DYN) 与 15维统计特征性能对比", pad=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.0)
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
                val + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%{delta_txt}",
                va="center",
                ha="left",
                fontsize=7.5,
                color="#D55E00",
                fontweight="bold" if d_pp < -0.3 else "normal",
            )

    ax2.set_yticks(y, [label for _, label in group_order], fontsize=8.4)
    ax2.invert_yaxis()
    ax2.set_xlim(35, 108)
    ax2.set_xlabel("40% 标注预算验证集 Macro-F1 (%)")
    ax2.set_title("(b) 60维动态特征六大物理子集单组保留与单组剔除消融", pad=24)
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
    agg12b = {(r["tag"], r["method"]): r for r in e12b["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7))

    # Panel (a): 8-way fusion ablation at 10% and 40% budgets
    methods = [
        ("tcn", "TCN 单分支 (固定50轮)"),
        ("tcn_meta", "TCN + 元分类器控制"),
        ("md", "MOMENT/ΔP 双流 (C=1.0)"),
        ("dyn", "DYN-60 + HGB 单分支"),
        ("tcn+md", "TCN + MOMENT/ΔP"),
        ("dyn+md", "DYN + MOMENT/ΔP"),
        ("tcn+dyn", "TCN + DYN"),
        ("full", "三路异构堆叠 (Full Stack)"),
    ]
    y = np.arange(len(methods))
    h = 0.36
    m10 = [agg12b[("10p", m)]["mean"] for m, _ in methods]
    s10 = [agg12b[("10p", m)]["sample_sd"] for m, _ in methods]
    m40 = [agg12b[("40p", m)]["mean"] for m, _ in methods]
    s40 = [agg12b[("40p", m)]["sample_sd"] for m, _ in methods]

    b10 = ax1.barh(
        y + h / 2,
        m10,
        xerr=s10,
        height=h,
        capsize=2.5,
        label="10% 标注预算 (2,456 样本)",
        color="#56B4E9",
        edgecolor="white",
        alpha=0.92,
    )
    b40 = ax1.barh(
        y - h / 2,
        m40,
        xerr=s40,
        height=h,
        capsize=2.5,
        label="40% 标注预算 (9,827 样本)",
        color="#D55E00",
        edgecolor="white",
        alpha=0.92,
    )

    for bar, val, sd_val in zip(b10, m10, s10):
        ax1.text(
            val + sd_val + 0.18,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#0369A1",
            fontweight="bold" if val == max(m10) else "normal",
        )
    for bar, val, sd_val in zip(b40, m40, s40):
        ax1.text(
            val + sd_val + 0.18,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#B45309",
            fontweight="bold" if val == max(m40) else "normal",
        )

    ax1.set_yticks(y, [label for _, label in methods], fontsize=8.4)
    ax1.invert_yaxis()
    ax1.set_xlim(86.0, 100.8)
    ax1.set_xlabel("验证集 Macro-F1 (%)")
    ax1.set_title("(a) 三路异构概率堆叠 8 路正交消融对比", pad=24)
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        columnspacing=1.2,
    )
    ax1.grid(True, axis="x", alpha=0.25)
    ax1.grid(False, axis="y")

    # Panel (b): Multi-budget sample efficiency curve (1%, 5%, 10%, 20%, 40%)
    curve_data = load_json("artifacts/analysis/legacy_tcn_moment_curve_20261021/summary.json")
    e08 = load_json(
        "artifacts/imports/e08_strong_features_20261020/e08_strong_features_20261020_v1/analysis/summary.json"
    )
    e09 = load_json(
        "artifacts/imports/e09_analysis_corrected_20261020/analysis_four_seed_corrected/summary.json"
    )

    c_map = {(r["model"], r["fraction"]): r for r in curve_data["aggregate"]}
    e08_map = {(r["method"], r["fraction"]): r for r in e08["aggregate"]}
    e09_map = {(r["method"], r["tag"]): r for r in e09["aggregate"]}

    fractions = [0.01, 0.05, 0.10, 0.20, 0.40]
    x_pct = [1, 5, 10, 20, 40]

    # 1. Frozen MOMENT + RBF-SVM (1, 5, 10, 20, 40)
    mom_m = [c_map[("moment", f)]["mean_percent"] for f in fractions]
    mom_s = [c_map[("moment", f)]["sample_sd_percent"] for f in fractions]
    ax2.errorbar(
        x_pct, mom_m, yerr=mom_s, label="冻结单流 MOMENT + RBF-SVM", color="#7F7F7F", marker="s", capsize=3
    )

    # 2. Restored Legacy Masked TCN (1, 5, 10, NaN at 20, 40)
    tcn_m = [c_map[("tcn", f)]["mean_percent"] if ("tcn", f) in c_map else np.nan for f in fractions]
    tcn_s = [c_map[("tcn", f)]["sample_sd_percent"] if ("tcn", f) in c_map else np.nan for f in fractions]
    ax2.errorbar(
        x_pct, tcn_m, yerr=tcn_s, label="恢复版掩码 TCN (50轮验证优选)", color="#0072B2", marker="o", capsize=3
    )

    # 3. DYN + HGB (1, 5, 10, 40)
    dyn_x = [1, 5, 10, 40]
    dyn_m = [e08_map[("dyn_hgb", f)]["macro_f1_mean"] for f in (0.01, 0.05, 0.10, 0.40)]
    dyn_s = [e08_map[("dyn_hgb", f)]["macro_f1_sd"] for f in (0.01, 0.05, 0.10, 0.40)]
    ax2.errorbar(
        dyn_x, dyn_m, yerr=dyn_s, label="60维动态特征 (DYN + HGB)", color="#009E73", marker="^", capsize=3
    )

    # 4. Dual-stream MOMENT/ΔP + DYN stack (1, 10, 40)
    md_x = [1, 10, 40]
    md_m = [e09_map[("stack_moment_diff", t)]["mean"] for t in ("1p", "10p", "40p")]
    md_s = [e09_map[("stack_moment_diff", t)]["sample_sd"] for t in ("1p", "10p", "40p")]
    ax2.errorbar(
        md_x,
        md_m,
        yerr=md_s,
        label="双流 MOMENT/ΔP + DYN 堆叠",
        color="#CC79A7",
        marker="D",
        linestyle="--",
        capsize=3,
    )

    # 5. Full 3-Way Stack (10, 40)
    full_x = [10, 40]
    full_m = [agg12b[("10p", "full")]["mean"], agg12b[("40p", "full")]["mean"]]
    full_s = [agg12b[("10p", "full")]["sample_sd"], agg12b[("40p", "full")]["sample_sd"]]
    ax2.errorbar(
        full_x,
        full_m,
        yerr=full_s,
        label="三路异构堆叠系统 (Full Stack)",
        color="#D55E00",
        marker="*",
        markersize=8.5,
        linewidth=2.3,
        capsize=3.5,
    )

    # Key annotations on Panel (b)
    ax2.annotate(
        "1% 极低预算:\n双流+DYN 堆叠 88.72%\n(较 TCN +33.21 pp)",
        xy=(1, 88.72),
        xytext=(2.3, 60.5),
        arrowprops={"arrowstyle": "->", "color": "#CC79A7", "lw": 1.2, "connectionstyle": "arc3,rad=-0.15"},
        fontsize=7.6,
        color="#831843",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FDF2F8", "edgecolor": "#FBCFE8", "alpha": 0.94},
    )
    ax2.annotate(
        "Full Stack: 98.09%",
        xy=(40, 98.09),
        xytext=(15.5, 99.2),
        arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 1.2},
        fontsize=8.2,
        color="#D55E00",
        fontweight="bold",
    )

    ax2.set_xscale("log")
    ax2.set_xticks(x_pct, [f"{v}%" for v in x_pct])
    ax2.set_xlim(0.8, 52)
    ax2.set_ylim(40, 103)
    ax2.set_xlabel("训练标注预算比例（对数坐标）")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    ax2.set_title("(b) 1%～40% 标注预算跨度下各模型样本效率演进曲线", pad=10)
    ax2.legend(loc="lower right", frameon=True, fontsize=7.8)
    ax2.grid(True, which="major", alpha=0.25)
    ax2.grid(False, which="minor")

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
    ax1.set_title("(a) 40% 标注预算下加性高斯测量噪声敏感度分析", pad=10)
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
    ax2.set_title("(b) RTX 4080 平台分类精度与在线推理时延帕累托权衡", pad=10)
    ax2.grid(True, which="both", alpha=0.25)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_both(fig, "fig6_robustness_and_latency")


if __name__ == "__main__":
    plot_fig1_overall_framework()
    plot_fig2_pooling_and_head_stabilization()
    plot_fig3_dual_stream_and_pretraining()
    plot_fig4_dyn_feature_ablation()
    plot_fig5_stacking_ablation_and_low_label()
    plot_fig6_robustness_and_latency()
