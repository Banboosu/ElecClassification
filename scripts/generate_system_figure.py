"""Generate the system architecture diagram (vector PDF + 300-dpi PNG).

Follows the two-container layout and crisp serif style of the journal
framework diagram while embedding the illustrative e-bike charging scene inside
the Charging site container:

* Charging site: e-bike charging schematic alongside the four-stage hardware
  stack (Charging pile -> Power meter -> Charger -> Li-ion battery).
* Monitoring platform: Cleaning, padding & mask -> Three-branch classifier ->
  three-class status output (Normal / Battery anomaly / Charger fault).

Run from the project root:
    uv run --no-sync python scripts/generate_system_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_journal_figures import INK  # noqa: E402
from generate_journal_figures_en import OUT_DIRS, setup_english_journal_style  # noqa: E402

ASSET_PATH = ROOT / "docs" / "figures" / "assets" / "ebike_charging_schematic.jpg"
ZH_FIG_DIRS = [ROOT / "docs" / "figures" / "zh"]
THESIS_ZH_FIG_DIR = ROOT / "docs" / "thesis" / "figures"
BOX_EDGE = "#3A3A3A"
BOX_LW = 1.3
CONTAINER_LW = 1.6


def _register_zh_fonts() -> None:
    """Register Chinese UI/serif fonts on native Windows or WSL."""
    for base in (Path(r"C:\Windows\Fonts"), Path("/mnt/c/Windows/Fonts")):
        if not base.exists():
            continue
        for fname in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "times.ttf", "timesbd.ttf"):
            fp = base / fname
            if fp.exists():
                fm.fontManager.addfont(str(fp))
        break


def _container(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.14",
            facecolor="white",
            edgecolor=INK,
            linewidth=CONTAINER_LW,
            zorder=1,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        y1 - 0.25,
        title,
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color=INK,
        zorder=4,
    )


def _box(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str,
    *,
    fontsize: float = 10.5,
) -> None:
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="white",
            edgecolor=BOX_EDGE,
            linewidth=BOX_LW,
            zorder=3,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
        linespacing=1.15,
        zorder=5,
    )


def _arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=10.0,
            linewidth=BOX_LW,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def _load_schematic(path: Path) -> np.ndarray:
    """Load the e-bike charging schematic cropped tightly on a pure-white background."""
    img = Image.open(path).convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.08)
    arr = np.asarray(img, dtype=np.uint8).copy()
    arr[arr >= 244] = 255
    mask = arr < 245
    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        pad = 8
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(arr.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(arr.shape[1], int(xs.max()) + pad + 1)
        arr = arr[y0:y1, x0:x1]
    return np.stack([arr, arr, arr], axis=-1)


_LANG_SPECS = {
    "en": {
        "site_title": "Charging site",
        "plat_title": "Monitoring platform",
        "title_fs": 15.5,
        "hw_bounds_x": (2.76, 4.24),
        "hw_labels": ["Li-ion battery", "Charger", "Power meter", "Charging pile"],
        "hw_fs": 13.0,
        "c1_x": (4.93, 6.09),
        "c2_x": (6.77, 8.09),
        "pre_label": "Cleaning,\npadding\n& mask",
        "pre_fs": 12.8,
        "cls_label": "Three-branch\nclassifier",
        "cls_fs": 12.8,
        "split_x": 8.21,
        "out_x": (8.36, 9.38),
        "out_labels": ["Normal", "Battery\nanomaly", "Charger\nfault"],
        "out_fs": 12.8,
    },
    "zh": {
        "site_title": "充电现场",
        "plat_title": "监测平台",
        "title_fs": 14.8,
        "hw_bounds_x": (2.74, 4.26),
        "hw_labels": ["锂离子电池组", "外置充电器", "功率计量模块", "充电桩"],
        "hw_fs": 12.0,
        "c1_x": (4.92, 6.12),
        "c2_x": (6.72, 8.04),
        "pre_label": "序列清洗、\n补齐与掩码",
        "pre_fs": 11.8,
        "cls_label": "三分支\n异构分类器",
        "cls_fs": 12.0,
        "split_x": 8.17,
        "out_x": (8.31, 9.41),
        "out_labels": ["正常充电", "电池异常", "充电器故障"],
        "out_fs": 11.8,
    },
}


def _configure_style(lang: str) -> None:
    setup_english_journal_style()
    if lang == "zh":
        _register_zh_fonts()
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "DejaVu Sans"],
                "mathtext.fontset": "stix",
                "axes.unicode_minus": False,
                "pdf.fonttype": 42,
            }
        )
    else:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "mathtext.fontset": "stix",
            }
        )


def plot_fig_system_architecture(lang: str = "en") -> None:
    """Render system architecture diagram in English or Chinese."""
    if lang not in _LANG_SPECS:
        raise ValueError(f"Unsupported language: {lang}")
    spec = _LANG_SPECS[lang]
    _configure_style(lang)

    if not ASSET_PATH.exists():
        raise FileNotFoundError(f"Missing e-bike schematic asset: {ASSET_PATH}")

    w, h = 9.56, 3.32
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, w)
    ax.set_ylim(0.0, h)
    ax.axis("off")

    # Outer containers
    site_x0, site_x1 = 0.04, 4.38
    plat_x0, plat_x1 = 4.80, 9.52
    cont_y0, cont_y1 = 0.05, 3.27
    _container(ax, site_x0, cont_y0, site_x1, cont_y1, spec["site_title"])
    _container(ax, plat_x0, cont_y0, plat_x1, cont_y1, spec["plat_title"])

    # ----------------------------------------------------------------------- #
    # Charging site: e-bike schematic (left) + hardware stack (right)
    # ----------------------------------------------------------------------- #
    img = _load_schematic(ASSET_PATH)
    ih, iw = img.shape[:2]
    img_x0, img_x1 = 0.14, 2.64 if lang == "zh" else 2.66
    img_h = (img_x1 - img_x0) * ih / iw
    img_cy = 1.44
    img_y0, img_y1 = img_cy - img_h / 2, img_cy + img_h / 2
    ax.imshow(
        img,
        extent=(img_x0, img_x1, img_y0, img_y1),
        aspect="auto",
        zorder=2,
        interpolation="lanczos",
    )

    hw_x0, hw_x1 = spec["hw_bounds_x"]
    hw_cx = (hw_x0 + hw_x1) / 2
    hw_h = 0.48
    hw_gap = 0.20
    hw_step = hw_h + hw_gap
    hw_bot = 0.20
    hw_labels = spec["hw_labels"]
    hw_bounds: list[tuple[float, float]] = []
    for k, label in enumerate(hw_labels):
        y0 = hw_bot + k * hw_step
        y1 = y0 + hw_h
        hw_bounds.append((y0, y1))
        _box(ax, hw_x0, y0, hw_x1, y1, label, fontsize=spec["hw_fs"])

    for k in range(len(hw_labels) - 1, 0, -1):
        _arrow(ax, (hw_cx, hw_bounds[k][0]), (hw_cx, hw_bounds[k - 1][1]))

    # Common center line aligned with "Power meter" (index 2)
    main_y = sum(hw_bounds[2]) / 2

    # ----------------------------------------------------------------------- #
    # Monitoring platform: preprocessing -> classifier -> 3-class outcomes
    # ----------------------------------------------------------------------- #
    c1_x0, c1_x1 = spec["c1_x"]
    c2_x0, c2_x1 = spec["c2_x"]
    proc_h = 0.88
    proc_y0, proc_y1 = main_y - proc_h / 2, main_y + proc_h / 2

    # Cross-container data uplink arrow P
    _arrow(ax, (hw_x1, main_y), (c1_x0, main_y))
    ax.text(
        (site_x1 + plat_x0) / 2,
        main_y + 0.09,
        r"$\mathbfit{P}$",
        ha="center",
        va="bottom",
        fontsize=15.0,
        color=INK,
    )

    _box(ax, c1_x0, proc_y0, c1_x1, proc_y1, spec["pre_label"], fontsize=spec["pre_fs"])
    _arrow(ax, (c1_x1, main_y), (c2_x0, main_y))
    ax.text(
        (c1_x1 + c2_x0) / 2,
        main_y + 0.09,
        r"$\mathbfit{x}_{\mathbfit{i}},\,\mathbfit{m}_{\mathbfit{i}}$",
        ha="center",
        va="bottom",
        fontsize=14.5,
        color=INK,
    )
    _box(ax, c2_x0, proc_y0, c2_x1, proc_y1, spec["cls_label"], fontsize=spec["cls_fs"])

    # Splitter bus to the three outcome boxes
    out_x0, out_x1 = spec["out_x"]
    out_h = 0.52
    out_dy = 0.66
    out_ys = [main_y + out_dy, main_y, main_y - out_dy]

    x_split = spec["split_x"]
    ax.plot([c2_x1, x_split], [main_y, main_y], color=INK, lw=BOX_LW, zorder=4)
    ax.plot([x_split, x_split], [out_ys[-1], out_ys[0]], color=INK, lw=BOX_LW, zorder=4)
    for cy, label in zip(out_ys, spec["out_labels"]):
        _arrow(ax, (x_split, cy), (out_x0, cy))
        _box(ax, out_x0, cy - out_h / 2, out_x1, cy + out_h / 2, label, fontsize=spec["out_fs"])

    if lang == "zh":
        for d in ZH_FIG_DIRS:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / "fig_system_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
            fig.savefig(d / "fig_system_architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved {d / 'fig_system_architecture.pdf'} and .png")
        THESIS_ZH_FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(THESIS_ZH_FIG_DIR / "fig3_1_system_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(THESIS_ZH_FIG_DIR / "fig3_1_system_architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {THESIS_ZH_FIG_DIR / 'fig3_1_system_architecture.pdf'} and .png")
    else:
        for d in OUT_DIRS:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / "fig_system_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
            fig.savefig(d / "fig_system_architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
            print(f"Saved {d / 'fig_system_architecture.pdf'} and .png")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate system architecture figure(s).")
    parser.add_argument("--lang", choices=["en", "zh", "all"], default="all")
    args = parser.parse_args()
    for _lang in (["en", "zh"] if args.lang == "all" else [args.lang]):
        plot_fig_system_architecture(_lang)
