"""生成中文本科毕业论文插图（300 DPI PNG + 矢量 PDF）到 docs/thesis/figures/。

基于 `scripts/generate_system_figure.py` 与 `scripts/generate_journal_figures_en.py`，
保持相同的经审计数据源与排版规范，将全部坐标轴、图例、方框标签改为规范中文，
字体优先采用 Times New Roman + SimSun（宋体），满足毕业论文图表要求。

运行命令（在项目根目录）：
    uv run --no-sync python scripts/generate_thesis_figures_zh.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_journal_figures import (  # noqa: E402
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
from tcn_moment.label_semantics import LABEL_NAMES_ZH  # noqa: E402

THESIS_FIG_DIR = ROOT / "docs" / "thesis" / "figures"
ASSET_PATH = ROOT / "docs" / "figures" / "assets" / "ebike_charging_schematic.jpg"
ASSET_CROP = (60, 110, 1140, 785)

FILL = "#F2F2F2"
EDGE = "#7A7A7A"
ENERGY = "#9A9A9A"

SAMPLE_CURVE = np.array(
    [
        262, 264, 264, 265, 266, 266, 267, 267, 268, 268, 269, 270, 271, 272,
        274, 275, 277, 280, 285, 287, 241, 205, 179, 159, 143, 130, 119, 109,
        100, 93, 87, 82, 77, 72, 69, 68, 67, 67, 66, 65,
    ],
    dtype=float,
)  # fmt: skip


def _register_fonts() -> None:
    candidates = [
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/simsun.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
    ]
    for fp in candidates:
        if fp.exists():
            fm.fontManager.addfont(str(fp))


def setup_thesis_zh_style() -> None:
    """配置中文毕业论文图表样式：英文 Times New Roman、中文宋体/黑体。"""
    _register_fonts()
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimSun",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "font.serif": ["SimSun", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 9.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.2,
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


def tidy_zh(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_zh(fig: plt.Figure, stem: str) -> None:
    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS_FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(THESIS_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved {THESIS_FIG_DIR / stem}.pdf and .png")
    plt.close(fig)


def log_budget_axis_zh(ax: plt.Axes, ticks: list[int], xlim: tuple[float, float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{v}%" for v in ticks])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=7.8, pad=4.0)
    ax.set_xlim(*xlim)
    ax.set_xlabel("训练标注比例（对数坐标）")


# --------------------------------------------------------------------------- #
# 图 2-1：训练集充电功率曲线样例与有效长度分布
# --------------------------------------------------------------------------- #
def plot_fig2_1_dataset_overview() -> None:
    setup_thesis_zh_style()
    split_candidates = [
        ROOT / "artifacts/improvements/e01/e01_20261017_v2_pilot_seed42/split_manifest.json",
        ROOT / "artifacts/imports/e01_validation_20261017/runs/e01_20261017_v2_pilot_seed42/split_manifest.json",
    ]
    split_path = next((p for p in split_candidates if p.exists()), None)
    if split_path is None:
        raise FileNotFoundError("需要 seed42 划分清单文件")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split["protocol"]["random_state"] != 42:
        raise ValueError("需要 seed42 划分清单")
    train_ids = set(split["splits"]["train"])
    if len(train_ids) != 24569:
        raise ValueError("训练集样本数异常")

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

    names = {"0": "正常充电", "1": "电池异常", "2": "充电器故障"}
    fig = plt.figure(figsize=(FIG_W, 4.35))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.55, wspace=0.30)

    rng = np.random.default_rng(42)
    sample_styles = [(INK, "-", "样例 1"), (MID, "--", "样例 2"), (LIGHT, "-.", "样例 3")]
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
        ax.set_title(f"({'abcd'[col]}) {names[cls]}", loc="left", pad=4.0)
        if col == 0:
            ax.set_ylabel("充电功率（记录单位）")
        tidy_zh(ax)
        if col == 2:
            ax.legend(loc="upper right", fontsize=7.0)

    ax = fig.add_subplot(gs[1, :])
    class_styles = {"0": (INK, "-", "o"), "1": (MID, "--", "s"), "2": (LIGHT, "-.", "^")}
    for cls in ("0", "1", "2"):
        lengths = np.sort(np.asarray([len(s) for s in by_class[cls]]))
        ecdf = np.arange(1, len(lengths) + 1) / len(lengths) * 100.0
        c_col, c_ls, c_mk = class_styles[cls]
        med = float(np.median(lengths))
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
            label=f"{names[cls]}（中位数 {med:g}）",
        )
        ax.axvline(med, color=c_col, linestyle=":", linewidth=0.8, alpha=0.8)

    ax.axhline(50.0, color=LIGHT, linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 830)
    ax.set_ylim(0, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("有效序列长度 $L$（采样点）")
    ax.set_ylabel("经验累积概率 (%)")
    ax.legend(loc="lower right", ncol=1)
    ax.set_title("(d) 各类有效序列长度经验累积分布", loc="left", pad=5.0)
    tidy_zh(ax)
    save_zh(fig, "fig2_1_dataset_overview")


# --------------------------------------------------------------------------- #
# 图 3-1：非侵入式电动自行车充电安全监测系统架构
# --------------------------------------------------------------------------- #
def plot_fig3_1_system_architecture() -> None:
    setup_thesis_zh_style()
    if not ASSET_PATH.exists():
        raise FileNotFoundError(f"缺少示意图资源: {ASSET_PATH}")

    w, h = 6.00, 3.62
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, w)
    ax.set_ylim(0.0, h)
    ax.axis("off")

    def _box(x0, y0, x1, y1, *, fc="#FFFFFF", ec=INK, lw=0.8, ls="-", r=0.04, z=2):
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor=fc,
                edgecolor=ec,
                linewidth=lw,
                linestyle=ls,
                zorder=z,
            )
        )

    def _arrow(p0, p1, *, color=INK, lw=0.9, scale=7.0, ls="-", style="-|>", z=5):
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle=style,
                mutation_scale=scale,
                linewidth=lw,
                linestyle=ls,
                color=color,
                shrinkA=0,
                shrinkB=0,
                joinstyle="miter",
                capstyle="butt",
                zorder=z,
            )
        )

    def _energy_arrow(x0, x1, y, label):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y),
                (x1, y),
                arrowstyle="simple,head_width=0.75,head_length=0.55,tail_width=0.32",
                mutation_scale=9.0,
                facecolor=ENERGY,
                edgecolor="none",
                shrinkA=0,
                shrinkB=0,
                zorder=4,
            )
        )
        ax.text((x0 + x1) / 2, y + 0.085, label, ha="center", va="bottom", fontsize=6.2, color=MID)

    def _marker(xy, num, *, r=0.062):
        ax.add_patch(Circle(xy, r, facecolor=INK, edgecolor="#FFFFFF", linewidth=0.8, zorder=8))
        ax.text(
            xy[0],
            xy[1] - 0.003,
            num,
            ha="center",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color="#FFFFFF",
            zorder=9,
        )

    def _layer_title(x, y, title, sub):
        t = ax.text(x, y, title, ha="left", va="center", fontsize=8.0, fontweight="bold", color=INK)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        x_end = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
        ax.text(x_end + 0.08, y, sub, ha="left", va="center", fontsize=6.8, color=MID)

    def _load_schematic(path: Path) -> np.ndarray:
        img = Image.open(path).convert("L").crop(ASSET_CROP)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        arr = np.asarray(img, dtype=np.uint8).copy()
        arr[arr >= 244] = 255
        return np.stack([arr, arr, arr], axis=-1)

    # 上层：物理感知层
    a_y0, a_y1 = 1.98, 3.58
    _box(0.03, a_y0, w - 0.03, a_y1, fc="#FFFFFF", ec=EDGE, lw=0.8, r=0.06, z=1)
    _layer_title(0.13, a_y1 - 0.15, "物理感知层", "现场充电硬件拓扑与单通道采样")

    blk_y0, blk_y1 = 2.80, 3.20
    blk_cy = (blk_y0 + blk_y1) / 2
    blocks = [
        (0.13, 1.01, "1", "充电桩", "交流插座"),
        (1.27, 2.15, "2", "外置充电器", "AC/DC 变换器"),
        (2.41, 3.29, "3", "锂离子电池组", "电动自行车车载"),
    ]
    for x0, x1, num, title, sub in blocks:
        _box(x0, blk_y0, x1, blk_y1, fc="#FFFFFF", ec=INK, lw=0.85, r=0.035, z=3)
        ax.text((x0 + x1) / 2, blk_cy + 0.075, title, ha="center", va="center", fontsize=6.8, fontweight="bold", color=INK, zorder=6)
        ax.text((x0 + x1) / 2, blk_cy - 0.085, sub, ha="center", va="center", fontsize=6.0, color=MID, zorder=6)
        _marker((x0 + 0.005, blk_y1 - 0.005), num, r=0.058)
    _energy_arrow(1.01, 1.27, blk_cy, "AC")
    _energy_arrow(2.15, 2.41, blk_cy, "DC")

    # 功率计量模块
    m_xy = (0.57, 2.53)
    m_r = 0.105
    ax.plot([m_xy[0], m_xy[0]], [blk_y0, m_xy[1] + m_r], color=INK, lw=0.8, zorder=4)
    ax.add_patch(Circle(m_xy, m_r, facecolor=FILL, edgecolor=INK, linewidth=1.0, zorder=6))
    ax.text(m_xy[0], m_xy[1], "W", ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK, zorder=7)
    ax.text(m_xy[0] + m_r + 0.06, m_xy[1] + 0.055, "桩端有功功率计量模块", ha="left", va="center", fontsize=6.6, fontweight="bold", color=INK)
    ax.text(m_xy[0] + m_r + 0.06, m_xy[1] - 0.085, "单通道采集，无需 BMS 通讯或硬件改造", ha="left", va="center", fontsize=6.0, color=MID)

    # 图例
    lg_y = 2.17
    ax.add_patch(
        FancyArrowPatch(
            (1.22, lg_y),
            (1.50, lg_y),
            arrowstyle="simple,head_width=0.75,head_length=0.55,tail_width=0.32",
            mutation_scale=9.0,
            facecolor=ENERGY,
            edgecolor="none",
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )
    ax.text(1.56, lg_y, "能量流", ha="left", va="center", fontsize=6.2, color=MID)
    _arrow((2.26, lg_y), (2.54, lg_y), lw=0.9, ls=(0, (3.0, 1.6)), scale=6.5)
    ax.text(2.60, lg_y, "数据流", ha="left", va="center", fontsize=6.2, color=MID)

    # 右侧充电场景示意图
    img = _load_schematic(ASSET_PATH)
    ih, iw = img.shape[:2]
    img_x1 = w - 0.12
    img_y0, img_y1 = a_y0 + 0.05, a_y1 - 0.05
    img_x0 = img_x1 - (img_y1 - img_y0) * iw / ih
    ax.imshow(img, extent=(img_x0, img_x1, img_y0, img_y1), aspect="auto", zorder=2, interpolation="lanczos")

    def img_pt(px: float, py: float) -> tuple[float, float]:
        u = (px - ASSET_CROP[0]) / iw
        v = 1.0 - (py - ASSET_CROP[1]) / ih
        return img_x0 + u * (img_x1 - img_x0), img_y0 + v * (img_y1 - img_y0)

    for num, mk, tg in (
        ("1", (975, 170), (1045, 290)),
        ("2", (560, 745), (668, 745)),
        ("3", (540, 300), (650, 445)),
    ):
        p_mk, p_tg = img_pt(*mk), img_pt(*tg)
        ax.plot([p_mk[0], p_tg[0]], [p_mk[1], p_tg[1]], color=INK, lw=0.7, zorder=7)
        ax.scatter([p_tg[0]], [p_tg[1]], s=6, color=INK, zorder=7, linewidths=0)
        _marker(p_mk, num)
    ax.text(img_x0 + 0.02, img_y1 - 0.04, "充电场景示意图", ha="left", va="top", fontsize=6.0, color=LIGHT)

    # 下层：信息分析层
    b_y0, b_y1 = 0.03, 1.80
    _box(0.03, b_y0, w - 0.03, b_y1, fc="#FAFAFA", ec=EDGE, lw=0.8, ls=(0, (4.0, 2.0)), r=0.06, z=1)
    _layer_title(0.74, b_y1 - 0.15, "信息分析层", "监测平台（云端服务器或边缘网关）")

    blk_top, blk_bot = 1.55, 0.13
    main_y = 0.80

    # 数据上行箭头
    s1_x0, s1_x1 = 0.13, 1.75
    up_x = m_xy[0]
    _arrow((up_x, m_xy[1] - m_r), (up_x, blk_top), lw=0.9, ls=(0, (3.0, 1.6)), scale=6.5)
    ax.text(up_x + 0.06, 1.89, r"$P(t)$", ha="left", va="center", fontsize=6.8, color=INK)

    # 模块 1：序列构建与掩码生成
    _box(s1_x0, blk_bot, s1_x1, blk_top, fc="#FFFFFF", ec=INK, lw=0.85, r=0.04, z=2)
    s1_cx = (s1_x0 + s1_x1) / 2
    ax.text(s1_cx, blk_top - 0.10, "序列清洗与掩码构建", ha="center", va="center", fontsize=6.9, fontweight="bold", color=INK)

    n_valid, n_total = len(SAMPLE_CURVE), 56
    ax_c = fig.add_axes((0.36 / w, 0.78 / h, 1.30 / w, 0.56 / h))
    t_valid = np.arange(n_valid)
    ax_c.fill_between([n_valid - 0.5, n_total], -20, 330, color="#E6E6E6", lw=0, zorder=1)
    ax_c.plot(t_valid, SAMPLE_CURVE, color=INK, lw=1.15, zorder=3)
    ax_c.plot([n_valid - 1, n_total], [0, 0], color=MID, lw=0.9, ls=(0, (2.5, 1.5)), zorder=2)
    ax_c.text(1.5, 300, r"$\mathbf{x}_i$", fontsize=6.8, color=INK, va="center")
    ax_c.text((n_valid + n_total) / 2, 150, "右侧\n补零", fontsize=6.0, color=MID, ha="center", va="center", linespacing=1.05)
    ax_c.set_xlim(0, n_total)
    ax_c.set_ylim(-20, 340)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.grid(False)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_c.spines[side].set_linewidth(0.6)

    ax_m = fig.add_axes((0.36 / w, 0.64 / h, 1.30 / w, 0.10 / h))
    ax_m.add_patch(Rectangle((0, 0), n_valid - 0.5, 1, facecolor="#555555", edgecolor="none"))
    ax_m.add_patch(Rectangle((n_valid - 0.5, 0), n_total - n_valid + 0.5, 1, facecolor="#FFFFFF", edgecolor="none"))
    ax_m.add_patch(Rectangle((0, 0), n_total, 1, facecolor="none", edgecolor=INK, lw=0.6))
    ax_m.text((n_valid - 0.5) / 2, 0.5, "1", ha="center", va="center", fontsize=5.8, color="#FFFFFF", fontweight="bold")
    ax_m.text((n_valid + n_total) / 2, 0.5, "0", ha="center", va="center", fontsize=5.8, color=INK, fontweight="bold")
    ax_m.set_xlim(0, n_total)
    ax_m.set_ylim(0, 1)
    ax_m.set_yticks([])
    ax_m.set_xticks([0.5, n_valid - 0.5, n_total - 0.5])
    ax_m.set_xticklabels(["1", r"$L_i$", r"$T$"], fontsize=6.2)
    ax_m.tick_params(axis="x", length=1.8, width=0.6, pad=1.0)
    ax_m.grid(False)
    for sp in ax_m.spines.values():
        sp.set_visible(False)
    ax.text(0.31, 0.69, r"$\mathbf{m}_i$", ha="right", va="center", fontsize=6.8, color=INK)

    ax.text(s1_cx, 0.37, r"清洗去重（$L_i \geq 18$）", ha="center", va="center", fontsize=6.2, color=MID)
    ax.text(s1_cx, 0.22, r"右侧补齐至 $T=1024$ + 掩码", ha="center", va="center", fontsize=6.2, color=MID)

    # 模块 2：三分支异构分类器
    s2_x0, s2_x1 = 2.03, 4.13
    _arrow((s1_x1, main_y), (s2_x0, main_y))
    ax.text((s1_x1 + s2_x0) / 2, main_y + 0.07, r"$\mathbf{x}_i,\mathbf{m}_i$", ha="center", va="bottom", fontsize=6.3, color=INK)

    _box(s2_x0, blk_bot, s2_x1, blk_top, fc="#FFFFFF", ec=INK, lw=0.85, r=0.04, z=2)
    s2_cx = (s2_x0 + s2_x1) / 2
    ax.text(s2_cx, blk_top - 0.10, "三分支异构分类器", ha="center", va="center", fontsize=6.9, fontweight="bold", color=INK)

    bus_x = s2_x0 + 0.11
    br_x0, br_x1 = s2_x0 + 0.20, s2_x0 + 1.32
    st_x0, st_x1 = s2_x1 - 0.54, s2_x1 - 0.09
    br_h, br_gap = 0.33, 0.07
    br_top = blk_top - 0.22
    branches = [
        ("冻结 MOMENT 双流", r"$P,\Delta P$ 掩码池化", r"$\mathbf{p}^{\mathrm{MD}}$"),
        ("60维 DYN + HGB", "原尺度动力学特征", r"$\mathbf{p}^{\mathrm{DYN}}$"),
        ("掩码 TCN", "因果膨胀卷积网络", r"$\mathbf{p}^{\mathrm{TCN}}$"),
    ]
    br_cys = []
    for k, (title, sub, prob) in enumerate(branches):
        y1 = br_top - k * (br_h + br_gap)
        y0 = y1 - br_h
        cy = (y0 + y1) / 2
        br_cys.append(cy)
        _box(br_x0, y0, br_x1, y1, fc=FILL, ec=MID, lw=0.7, r=0.03, z=3)
        ax.text((br_x0 + br_x1) / 2, cy + 0.068, title, ha="center", va="center", fontsize=6.3, fontweight="bold", color=INK, zorder=6)
        ax.text((br_x0 + br_x1) / 2, cy - 0.075, sub, ha="center", va="center", fontsize=5.8, color=MID, zorder=6)
        _arrow((bus_x, cy), (br_x0, cy), lw=0.75, scale=5.5)
        _arrow((br_x1, cy), (st_x0, cy), lw=0.75, scale=5.5)
        ax.text((br_x1 + st_x0) / 2, cy + 0.035, prob, ha="center", va="bottom", fontsize=6.0, color=INK)
    ax.plot([s2_x0, bus_x], [main_y, main_y], color=INK, lw=0.9, zorder=5)
    ax.plot([bus_x, bus_x], [br_cys[-1], br_cys[0]], color=INK, lw=0.9, zorder=5)

    st_y1 = br_top
    st_y0 = br_top - 3 * br_h - 2 * br_gap
    _box(st_x0, st_y0, st_x1, st_y1, fc=FILL, ec=INK, lw=1.0, r=0.03, z=3)
    st_cx = (st_x0 + st_x1) / 2
    st_cy = (st_y0 + st_y1) / 2
    ax.text(st_cx, st_cy + 0.21, "五折", ha="center", va="center", fontsize=6.2, color=INK, zorder=6)
    ax.text(st_cx, st_cy + 0.07, "OOF", ha="center", va="center", fontsize=6.2, color=INK, zorder=6)
    ax.text(st_cx, st_cy - 0.07, "逻辑回归", ha="center", va="center", fontsize=6.1, fontweight="bold", color=INK, zorder=6)
    ax.text(st_cx, st_cy - 0.21, "堆叠融合", ha="center", va="center", fontsize=6.1, fontweight="bold", color=INK, zorder=6)

    # 模块 3：状态判别与平台处置
    s3_x0, s3_x1 = 4.41, w - 0.13
    _arrow((s2_x1, main_y), (s3_x0, main_y))
    ax.text((s2_x1 + s3_x0) / 2, main_y + 0.07, r"$\hat{y}_i$", ha="center", va="bottom", fontsize=6.6, color=INK)

    _box(s3_x0, blk_bot, s3_x1, blk_top, fc="#FFFFFF", ec=INK, lw=0.85, r=0.04, z=2)
    s3_cx = (s3_x0 + s3_x1) / 2
    ax.text(s3_cx, blk_top - 0.10, "状态判别与平台处置", ha="center", va="center", fontsize=6.9, fontweight="bold", color=INK)

    outcomes = [
        ("0", "正常充电", "持续供电，常规日志", "#FFFFFF", MID, 0.7, "-"),
        ("1", "电池异常", "断电保护，立即告警", FILL, INK, 1.3, "-"),
        ("2", "充电器故障", "异常预警，运维派单", "#FFFFFF", INK, 0.8, (0, (3.0, 1.5))),
    ]
    oc_x0, oc_x1 = s3_x0 + 0.08, s3_x1 - 0.08
    for k, (cls, title, act, fc, ec, lw, ls) in enumerate(outcomes):
        y1 = br_top - k * (br_h + br_gap)
        y0 = y1 - br_h
        cy = (y0 + y1) / 2
        _box(oc_x0, y0, oc_x1, y1, fc=fc, ec=ec, lw=lw, ls=ls, r=0.03, z=3)
        tag_x = oc_x0 + 0.10
        ax.add_patch(Rectangle((tag_x - 0.065, cy - 0.065), 0.13, 0.13, facecolor=INK, edgecolor="none", zorder=5))
        ax.text(tag_x, cy - 0.002, cls, ha="center", va="center", fontsize=6.2, fontweight="bold", color="#FFFFFF", zorder=6)
        tx = tag_x + 0.11
        ax.text(tx, cy + 0.068, title, ha="left", va="center", fontsize=6.4, fontweight="bold", color=INK, zorder=6)
        ax.text(tx, cy - 0.075, act, ha="left", va="center", fontsize=5.8, color=MID, zorder=6)

    save_zh(fig, "fig3_1_system_architecture")


# --------------------------------------------------------------------------- #
# 图 3-2：基于冻结时序大模型的三分支异构融合总体框架
# --------------------------------------------------------------------------- #
def plot_fig3_2_overall_framework() -> None:
    setup_thesis_zh_style()
    w, h = 6.2, 3.32
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    edge = "#3A3A3A"
    lw_box = 1.25

    def box(x0: float, y0: float, x1: float, y1: float, text: str, size: float = 9.5) -> None:
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white", edgecolor=edge, linewidth=lw_box)
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
            linespacing=1.2,
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
                linewidth=1.4,
            )
        )

    def arrow(p0: tuple[float, float], p1: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.25,
                color=INK,
                shrinkA=0,
                shrinkB=0,
            )
        )

    def prob(x: float, y: float, sup: str) -> None:
        ax.text(x, y, rf"$\boldsymbol{{p}}^{{\mathrm{{{sup}}}}}$", ha="left", va="center", fontsize=12.5, color=INK)

    yI, yII, yIII = (2.58, 3.28), (1.57, 2.27), (0.54, 1.24)
    for y0, y1 in (yI, yII, yIII):
        container(y0, y1)
    cI, cIII = sum(yI) / 2, sum(yIII) / 2

    box(0.03, 1.62, 0.81, 2.24, "")
    ax.text(0.42, 2.04, "输入序列", ha="center", va="center", fontsize=11.0, fontweight="bold", color=INK)
    ax.text(0.42, 1.78, r"$\boldsymbol{x}_i\quad\boldsymbol{m}_i$", ha="center", va="center", fontsize=12.0, color=INK)
    ax.plot([0.81, 0.97], [1.93, 1.93], color=INK, lw=1.25)
    ax.plot([0.97, 0.97], [cIII, cI], color=INK, lw=1.25)
    for yc in (cI, 1.93, cIII):
        arrow((0.97, yc), (1.20, yc))

    # 分支 I
    ax.text(1.43, 3.07, r"$P$", ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(1.40, 2.79, r"$\Delta P$", ha="center", va="center", fontsize=12.5, color=INK)
    arrow((1.53, 3.07), (1.77, 3.07))
    arrow((1.55, 2.79), (1.77, 2.79))
    box(1.77, 2.67, 2.74, 3.19, "冻结\nMOMENT", size=9.2)
    arrow((2.74, 2.93), (2.90, 2.93))
    box(2.90, 2.67, 3.92, 3.19, "掩码感知\n分块池化", size=9.2)
    arrow((3.92, 2.93), (4.09, 2.93))
    prob(4.12, 2.95, "MD")

    # 分支 II
    box(1.65, 1.66, 2.68, 2.18, "60维原尺度\n动力学特征", size=9.0)
    arrow((2.68, 1.92), (2.85, 1.92))
    box(2.85, 1.66, 3.65, 2.18, "HGB\n分类器", size=9.2)
    arrow((3.65, 1.92), (3.86, 1.92))
    prob(3.89, 1.94, "DYN")

    # 分支 III
    box(2.25, 0.62, 3.24, 1.16, "六层掩码\nTCN", size=9.5)
    arrow((3.24, 0.89), (3.44, 0.89))
    prob(3.48, 0.91, "TCN")

    # 概率堆叠总线
    xb = 4.85
    ax.plot([4.695, xb], [cI, cI], color=INK, lw=1.25)
    ax.plot([4.695, xb], [cIII, cIII], color=INK, lw=1.25)
    ax.plot([xb, xb], [cIII, cI], color=INK, lw=1.25)
    arrow((4.695, 1.92), (5.07, 1.92))

    box(5.07, 1.66, 6.15, 2.18, "五折 OOF\n交叉拟合", size=9.2)
    arrow((5.61, 1.66), (5.61, 1.42))
    box(5.07, 0.88, 6.15, 1.42, "逻辑回归\n元学习器", size=9.2)
    arrow((5.61, 0.88), (5.61, 0.64))
    box(5.07, 0.04, 6.15, 0.64, "三分类输出", size=9.8)

    save_zh(fig, "fig3_2_overall_framework")


# --------------------------------------------------------------------------- #
# 图 4-1：池化方式修正与下游分类头输入尺度优化效果
# --------------------------------------------------------------------------- #
def plot_fig4_1_pooling_and_head_stabilization() -> None:
    setup_thesis_zh_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H + 0.1))

    stages = ["默认\n池化", "掩码\n池化", "解冻\n最后2层", "RBF-SVM\n分类头", "全量\n微调"]
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
    ax1.tick_params(axis="x", labelsize=7.2)
    ax1.set_xlim(-0.55, 4.55)
    ax1.set_ylim(64, 100)
    ax1.set_xlabel("适配阶段（全量训练标注）")
    ax1.set_ylabel("测试集 Macro-F1 (%)")
    ax1.grid(axis="x", visible=False)
    handles = [
        Line2D([], [], color=INK, marker="o", linewidth=1.6, markersize=4.6, label="MOMENT-1-large"),
        Line2D([], [], color=LIGHT, linestyle="--", linewidth=1.0, label="TCN 基线 95.99"),
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
    ax2.set_xlabel("训练轮次（Epoch）")
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    branch_handles = [
        Line2D([], [], color=c, marker=m, linestyle="none", markersize=4.2, label=lab)
        for c, m, lab in ((INK, "o", "融合输入"), (MID, "s", "统计特征"), (LIGHT, "^", "MOMENT"))
    ]
    style_handles = [
        Line2D([], [], color=INK, linestyle="-", linewidth=1.3, label="含尺度补偿"),
        Line2D([], [], color=INK, linestyle="--", linewidth=1.0, label="无尺度补偿"),
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
        tidy_zh(ax)
    fig.tight_layout(w_pad=1.8)
    save_zh(fig, "fig4_1_pooling_and_head_stabilization")


# --------------------------------------------------------------------------- #
# 图 4-2：TCN 结构配置、池化方式对比及按序列长度分层的分类性能
# --------------------------------------------------------------------------- #
def plot_fig4_2_tcn_and_length_strata() -> None:
    setup_thesis_zh_style()
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
        (x_full, masked_means, masked_sds, "6 层 TCN，掩码池化", INK, "o", "-", INK, 1.6),
        (
            x_10_100,
            [agg05[("legacy_unmasked50", f)]["macro_f1_mean_percent"] for f in fracs],
            [agg05[("legacy_unmasked50", f)]["macro_f1_sample_sd_percent"] for f in fracs],
            "6 层 TCN，均值池化",
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
            "4 块残差 TCN，掩码池化",
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
    log_budget_axis_zh(ax1, x_full, (0.75, 135))
    ax1.set_ylim(34, 106)
    ax1.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    ax1.legend(loc="lower right", fontsize=6.8, handlelength=2.0, borderaxespad=0.2)
    panel_label(ax1, "(a)")

    strata = ["短序列\n($L\\leq266$)", "中等序列\n(266–399)", "长序列\n($L>399$)"]
    x2 = np.arange(3)
    specs = [
        ("40p", "full", "40% 完整融合", INK, "o", "-"),
        ("40p", "tcn", "40% 掩码 TCN", INK, "s", "--"),
        ("10p", "full", "10% 完整融合", MID, "D", "-"),
        ("10p", "tcn", "10% 掩码 TCN", MID, "^", "--"),
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
    ax2.set_xlabel("有效序列长度分组")
    ax2.set_ylabel("分组 Macro-F1 (%)")
    ax2.legend(
        loc="lower left",
        ncol=2,
        columnspacing=0.8,
        fontsize=6.8,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
    )
    ax2.grid(axis="x", visible=False)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy_zh(ax)
    fig.tight_layout(w_pad=1.6)
    save_zh(fig, "fig4_2_tcn_and_length_strata")


# --------------------------------------------------------------------------- #
# 图 4-3：一阶差分双流表征效果与同架构预训练权重归因对比
# --------------------------------------------------------------------------- #
def plot_fig4_3_dual_stream_and_pretraining() -> None:
    setup_thesis_zh_style()
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
        ("moment_logit", "原始序列流 $P$", LIGHT, "s", ":"),
        ("diff_logit", "一阶差分流 $\\Delta P$", MID, "^", "--"),
        ("moment_diff_logit", "双流拼接 $P+\\Delta P$", INK, "o", "-"),
        ("stack_moment_diff", "双流 + DYN 堆叠", INK, "D", "-"),
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
    log_budget_axis_zh(ax1, x_pct, (0.75, 55))
    ax1.set_ylim(65, 100)
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    ax1.legend(loc="lower right", fontsize=7.0)
    panel_label(ax1, "(a)")

    e16_map = {
        (r["tag"], r["initialization"], r["stream"]): r
        for r in [*e16["aggregate"], *e16b["aggregate"]]
    }
    tags_b = ["1p", "5p", "10p", "20p", "40p"]
    x_b = [1, 5, 10, 20, 40]
    stream_specs = [
        ("dual", "o", "双流 $P+\\Delta P$", INK),
        ("diff", "^", "差分 $\\Delta P$", MID),
        ("orig", "s", "原始 $P$", LIGHT),
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
    log_budget_axis_zh(ax2, x_b, (0.75, 55))
    ax2.set_ylim(42, 106)
    ax2.set_yticks([50, 60, 70, 80, 90])
    ax2.set_ylabel("验证集 Macro-F1 (%)")
    init_handles = [
        Line2D([], [], color=INK, linestyle="-", label="预训练权重"),
        Line2D([], [], color=INK, linestyle="--", label="随机初始化"),
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
        tidy_zh(ax)
    fig.tight_layout(w_pad=1.8)
    save_zh(fig, "fig4_3_dual_stream_and_pretraining")


# --------------------------------------------------------------------------- #
# 图 4-4：原尺度动力学特征（DYN）跨预算性能与特征组分贡献消融
# --------------------------------------------------------------------------- #
def plot_fig4_4_dyn_feature_ablation() -> None:
    setup_thesis_zh_style()
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
    log_budget_axis_zh(ax1, [1, 5, 10, 40, 100], (0.75, 135))
    ax1.set_ylim(72, 101)
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    ax1.legend(loc="lower right", fontsize=7.0)
    panel_label(ax1, "(a)")

    grp = {r["group"]: r for r in e08["group_ablation_40p"]}
    order = [
        ("roughness", "粗糙度与转折点"),
        ("trend", "分段趋势"),
        ("amplitude", "幅值与分位数"),
        ("spectral", "频谱与自相关"),
        ("shape", "形状特征"),
        ("length", "长度特征"),
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
    ax2.tick_params(axis="y", labelsize=7.6)
    ax2.set_xlim(35, 103)
    ax2.set_ylim(5.6, -1.2)
    ax2.set_xlabel("40% 标注下验证集 Macro-F1 (%)")
    ax2.legend(loc="upper left", fontsize=7.0)
    ax2.grid(axis="y", visible=False)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy_zh(ax)
    fig.tight_layout(w_pad=1.4)
    save_zh(fig, "fig4_4_dyn_feature_ablation")


# --------------------------------------------------------------------------- #
# 图 4-5：分支互补性配对消融与统一协议下跨标注预算性能曲线
# --------------------------------------------------------------------------- #
def plot_fig4_5_stacking_ablation_and_low_label() -> None:
    setup_thesis_zh_style()
    e12b = load_json(
        "artifacts/imports/e12b_fusion_ablation_corrected_20261020/e12b_fusion_ablation_corrected/summary.json"
    )
    per_seed = {(r["tag"], r["method"]): np.asarray(r["per_seed"]) for r in e12b["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_W, PANEL_H + 0.75), gridspec_kw={"width_ratios": [0.85, 1.15]}
    )

    pairs = [
        ("tcn_meta", "tcn", "TCN + 元学习器\n\u2212 TCN"),
        ("full", "tcn_meta", "完整融合\n\u2212 TCN + 元学习器"),
        ("tcn+dyn", "tcn", "TCN + DYN \u2212 TCN"),
        ("tcn+md", "tcn", "TCN + MOMENT/\u0394P\n\u2212 TCN"),
        ("dyn+md", "dyn", "DYN + MOMENT/\u0394P\n\u2212 DYN"),
        ("full", "dyn+md", "TCN 边际贡献"),
        ("full", "tcn+md", "DYN 边际贡献"),
        ("full", "tcn+dyn", "MOMENT/\u0394P 边际贡献"),
    ]
    y = np.arange(len(pairs))
    off = 0.17
    for k, (tag, color, marker, mfc, ls, label) in enumerate(
        [("10p", INK, "o", INK, "-", "10% 标注"), ("40p", MID, "s", "white", "--", "40% 标注")]
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
    ax1.set_xlabel("逐种子配对 Macro-F1 差值（百分点）")
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
        ("moment_logit", "MOMENT 原始序列流", LIGHT, "s", ":", "white"),
        ("moment_diff_logit", "MOMENT 双流拼接", MID, "D", "--", "white"),
        ("dyn_hgb", "DYN + HGB", MID, "^", "-.", MID),
        ("tcn", "掩码 TCN", INK, "o", "--", "white"),
        ("dyn_moment_diff", "DYN + MOMENT/\u0394P", MID, "X", "-", MID),
        ("full", "三分支完整融合", INK, "*", "-", INK),
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
    log_budget_axis_zh(ax2, x_pct, (0.8, 125))
    ax2.tick_params(axis="x", labelsize=7.2)
    ax2.set_ylim(48, 101)
    ax2.set_ylabel("验证集 Macro-F1 (%)")
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
        tidy_zh(ax)
    fig.tight_layout(w_pad=1.2)
    save_zh(fig, "fig4_5_stacking_ablation_and_low_label")


# --------------------------------------------------------------------------- #
# 图 5-1：40% 标注预算下模型的加性高斯噪声敏感性与精度—推理时延折中关系
# --------------------------------------------------------------------------- #
def plot_fig5_1_robustness_and_latency() -> None:
    setup_thesis_zh_style()
    e13 = load_json(
        "artifacts/imports/e13_robustness_20261020/e13_robustness/e13b_robustness_20261020_v1/summary.json"
    )
    agg13 = {(r["tag"], r["method"], r["perturbation"]): r for r in e13["aggregate"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, PANEL_H + 0.1))

    levels = ["clean", "noise_0p1pct", "noise_0p25pct", "noise_0p5pct", "noise_1pct"]
    x = np.arange(len(levels))
    series = [
        ("full", "三分支完整融合", INK, "o", "-"),
        ("tcn", "掩码 TCN", INK, "s", "--"),
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
    ax1.set_xlabel("加性高斯噪声强度（% RMS）")
    ax1.set_ylabel("验证集 Macro-F1 (%)")
    ax1.legend(loc="lower left", ncol=2, columnspacing=0.8, fontsize=6.7, handlelength=1.7)
    ax1.grid(axis="x", visible=False)
    panel_label(ax1, "(a)")

    points = [
        ("DYN+HGB", 1.990, 11.430, 95.00, LIGHT, "v"),
        ("TCN", 0.039, 0.882, 97.53, INK, "s"),
        ("TCN+DYN", 2.092, 12.595, 97.93, MID, "^"),
        ("完整融合", 11.128, 46.103, 98.09, INK, "o"),
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
        "完整融合": ("完整融合\n98.09", 7, -6, "left", "top"),
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
            fontweight="bold" if name == "完整融合" else "normal",
            color=INK if name != "DYN+HGB" else LIGHT,
        )
    ax2.set_xscale("log")
    ax2.grid(False, which="minor")
    ax2.grid(True, which="major")
    ax2.set_xlim(0.015, 200)
    ax2.set_ylim(93.9, 99.0)
    ax2.set_yticks([94.5, 95.5, 96.5, 97.5, 98.5])
    ax2.set_xlabel("单条记录推理时延（ms，对数坐标）")
    ax2.set_ylabel("40% 标注下验证集 Macro-F1 (%)")
    handles = [
        Line2D([], [], color=INK, marker="s", linestyle="none", markersize=4.2, label="批大小 = 32"),
        Line2D(
            [],
            [],
            color=INK,
            marker="s",
            linestyle="none",
            markersize=4.2,
            markerfacecolor="white",
            label="批大小 = 1",
        ),
    ]
    ax2.legend(handles=handles, loc="lower right", fontsize=7.0, handlelength=1.0)
    panel_label(ax2, "(b)")

    for ax in (ax1, ax2):
        tidy_zh(ax)
    fig.tight_layout(w_pad=1.8)
    save_zh(fig, "fig5_1_robustness_and_latency")


if __name__ == "__main__":
    plot_fig2_1_dataset_overview()
    plot_fig3_1_system_architecture()
    plot_fig3_2_overall_framework()
    plot_fig4_1_pooling_and_head_stabilization()
    plot_fig4_2_tcn_and_length_strata()
    plot_fig4_3_dual_stream_and_pretraining()
    plot_fig4_4_dyn_feature_ablation()
    plot_fig4_5_stacking_ablation_and_low_label()
    plot_fig5_1_robustness_and_latency()
