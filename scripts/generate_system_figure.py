"""Generate the system architecture diagram (vector PDF + 300-dpi PNG).

Two-layer layout sized to the IOP text width (153 mm), so the printed font
sizes are close to the nominal sizes used below:

* Physical sensing layer: energy path charging pile -> inline charger ->
  Li-ion battery pack, with the single-channel outlet power meter marked
  explicitly, next to an illustrative schematic of the charging scene whose
  numbered markers match the topology blocks.
* Cyber analysis layer (server or edge gateway): sequence construction
  (cleaning, right-padding, validity mask) -> compact three-branch classifier
  (details in the framework figure) -> three-class status and platform action.

Run from the project root:
    uv run --no-sync python scripts/generate_system_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_journal_figures import INK, LIGHT, MID  # noqa: E402
from generate_journal_figures_en import OUT_DIRS, setup_english_journal_style  # noqa: E402

ASSET_PATH = ROOT / "docs" / "figures" / "assets" / "ebike_charging_schematic.jpg"
# Crop window (left, top, right, bottom) in source pixels: removes empty margins.
ASSET_CROP = (60, 110, 1140, 785)

FILL = "#F2F2F2"  # light panel / emphasis fill
EDGE = "#7A7A7A"  # panel outline
ENERGY = "#9A9A9A"  # thick energy-flow arrows

# Representative normal charging curve downsampled from training record C4A7GVPA44KF8QUU (L=237).
SAMPLE_CURVE = np.array(
    [
        262, 264, 264, 265, 266, 266, 267, 267, 268, 268, 269, 270, 271, 272,
        274, 275, 277, 280, 285, 287, 241, 205, 179, 159, 143, 130, 119, 109,
        100, 93, 87, 82, 77, 72, 69, 68, 67, 67, 66, 65,
    ],
    dtype=float,
)  # fmt: skip


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _box(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    fc: str = "#FFFFFF",
    ec: str = INK,
    lw: float = 0.8,
    ls: str = "-",
    r: float = 0.04,
    z: float = 2,
) -> None:
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


def _arrow(
    ax: plt.Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 0.9,
    scale: float = 7.0,
    ls: str = "-",
    style: str = "-|>",
    z: float = 5,
) -> None:
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


def _energy_arrow(ax: plt.Axes, x0: float, x1: float, y: float, label: str) -> None:
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
    ax.text((x0 + x1) / 2, y + 0.085, label, ha="center", va="bottom", fontsize=6.0, color=MID)


def _marker(ax: plt.Axes, xy: tuple[float, float], num: str, *, r: float = 0.062) -> None:
    """Circled index number used to link schematic parts to topology blocks."""
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


def _layer_title(ax: plt.Axes, x: float, y: float, title: str, sub: str) -> None:
    t = ax.text(x, y, title, ha="left", va="center", fontsize=7.8, fontweight="bold", color=INK)
    fig = ax.figure
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    x_end = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x_end + 0.06, y, sub, ha="left", va="center", fontsize=6.8, color=MID)


def _load_schematic(path: Path) -> np.ndarray:
    """Cropped, high-contrast grayscale schematic on a pure-white background."""
    img = Image.open(path).convert("L").crop(ASSET_CROP)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    arr = np.asarray(img, dtype=np.uint8).copy()
    arr[arr >= 244] = 255
    return np.stack([arr, arr, arr], axis=-1)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_fig_system_architecture() -> None:
    setup_english_journal_style()
    if not ASSET_PATH.exists():
        raise FileNotFoundError(f"Missing e-bike schematic asset: {ASSET_PATH}")

    w, h = 6.00, 3.62
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, w)
    ax.set_ylim(0.0, h)
    ax.axis("off")

    # ======================================================================= #
    # Layer A: physical sensing
    # ======================================================================= #
    a_y0, a_y1 = 1.98, 3.58
    _box(ax, 0.03, a_y0, w - 0.03, a_y1, fc="#FFFFFF", ec=EDGE, lw=0.8, r=0.06, z=1)
    _layer_title(ax, 0.13, a_y1 - 0.15, "Physical sensing layer", "on-site charging hardware")

    # ---- Energy-path topology: pile -> charger -> battery ------------------ #
    blk_y0, blk_y1 = 2.80, 3.20
    blk_cy = (blk_y0 + blk_y1) / 2
    blocks = [
        (0.13, 1.01, "1", "Charging pile", "AC outlet"),
        (1.27, 2.15, "2", "Inline charger", "AC/DC converter"),
        (2.41, 3.29, "3", "Li-ion battery", "pack on e-bike"),
    ]
    for x0, x1, num, title, sub in blocks:
        _box(ax, x0, blk_y0, x1, blk_y1, fc="#FFFFFF", ec=INK, lw=0.85, r=0.035, z=3)
        ax.text(
            (x0 + x1) / 2,
            blk_cy + 0.075,
            title,
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            color=INK,
            zorder=6,
        )
        ax.text(
            (x0 + x1) / 2, blk_cy - 0.085, sub, ha="center", va="center", fontsize=6.2, color=MID, zorder=6
        )
        _marker(ax, (x0 + 0.005, blk_y1 - 0.005), num, r=0.058)
    _energy_arrow(ax, 1.01, 1.27, blk_cy, "AC")
    _energy_arrow(ax, 2.15, 2.41, blk_cy, "DC")

    # ---- Single-channel power meter at the pile outlet ---------------------- #
    m_xy = (0.57, 2.53)
    m_r = 0.105
    ax.plot([m_xy[0], m_xy[0]], [blk_y0, m_xy[1] + m_r], color=INK, lw=0.8, zorder=4)
    ax.add_patch(Circle(m_xy, m_r, facecolor=FILL, edgecolor=INK, linewidth=1.0, zorder=6))
    ax.text(m_xy[0], m_xy[1], "W", ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK, zorder=7)
    ax.text(
        m_xy[0] + m_r + 0.06,
        m_xy[1] + 0.055,
        "Outlet active-power meter",
        ha="left",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        m_xy[0] + m_r + 0.06,
        m_xy[1] - 0.085,
        "single channel, no BMS link or retrofit",
        ha="left",
        va="center",
        fontsize=6.2,
        color=MID,
    )

    # ---- Legend: energy vs data flow ---------------------------------------- #
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
    ax.text(1.56, lg_y, "Energy flow", ha="left", va="center", fontsize=6.2, color=MID)
    _arrow(ax, (2.26, lg_y), (2.54, lg_y), lw=0.9, ls=(0, (3.0, 1.6)), scale=6.5)
    ax.text(2.60, lg_y, "Data flow", ha="left", va="center", fontsize=6.2, color=MID)

    # ---- Illustrative schematic with matching markers ----------------------- #
    img = _load_schematic(ASSET_PATH)
    ih, iw = img.shape[:2]
    img_x1 = w - 0.12
    img_y0, img_y1 = a_y0 + 0.05, a_y1 - 0.05
    img_x0 = img_x1 - (img_y1 - img_y0) * iw / ih
    ax.imshow(img, extent=(img_x0, img_x1, img_y0, img_y1), aspect="auto", zorder=2, interpolation="lanczos")

    def img_pt(px: float, py: float) -> tuple[float, float]:
        """Map source-image pixel coordinates to axes data coordinates."""
        u = (px - ASSET_CROP[0]) / iw
        v = 1.0 - (py - ASSET_CROP[1]) / ih
        return img_x0 + u * (img_x1 - img_x0), img_y0 + v * (img_y1 - img_y0)

    # (marker position, target position) in source pixels.
    for num, mk, tg in (
        ("1", (975, 170), (1045, 290)),  # pile outlet
        ("2", (560, 745), (668, 745)),  # inline charger brick
        ("3", (540, 300), (650, 445)),  # battery pack
    ):
        p_mk, p_tg = img_pt(*mk), img_pt(*tg)
        ax.plot([p_mk[0], p_tg[0]], [p_mk[1], p_tg[1]], color=INK, lw=0.7, zorder=7)
        ax.scatter([p_tg[0]], [p_tg[1]], s=6, color=INK, zorder=7, linewidths=0)
        _marker(ax, p_mk, num)
    ax.text(
        img_x0 + 0.02,
        img_y1 - 0.04,
        "Illustrative scene",
        ha="left",
        va="top",
        fontsize=5.9,
        fontstyle="italic",
        color=LIGHT,
    )

    # ======================================================================= #
    # Layer B: cyber analysis (server or edge gateway)
    # ======================================================================= #
    b_y0, b_y1 = 0.03, 1.80
    _box(ax, 0.03, b_y0, w - 0.03, b_y1, fc="#FAFAFA", ec=EDGE, lw=0.8, ls=(0, (4.0, 2.0)), r=0.06, z=1)
    _layer_title(ax, 0.74, b_y1 - 0.15, "Cyber analysis layer", "monitoring platform (server or edge gateway)")

    blk_top, blk_bot = 1.55, 0.13
    main_y = 0.80  # common height of the inter-module data flow

    # ---- Data uplink from meter into the platform --------------------------- #
    s1_x0, s1_x1 = 0.13, 1.75
    up_x = m_xy[0]
    _arrow(ax, (up_x, m_xy[1] - m_r), (up_x, blk_top), lw=0.9, ls=(0, (3.0, 1.6)), scale=6.5)
    ax.text(up_x + 0.06, 1.89, r"$P(t)$", ha="left", va="center", fontsize=6.8, color=INK)

    # ---- Module 1: sequence construction ------------------------------------ #
    _box(ax, s1_x0, blk_bot, s1_x1, blk_top, fc="#FFFFFF", ec=INK, lw=0.85, r=0.04, z=2)
    s1_cx = (s1_x0 + s1_x1) / 2
    ax.text(s1_cx, blk_top - 0.10, "Sequence construction", ha="center", va="center", fontsize=6.9,
            fontweight="bold", color=INK)  # fmt: skip

    n_valid, n_total = len(SAMPLE_CURVE), 56
    ax_c = fig.add_axes((0.36 / w, 0.78 / h, 1.30 / w, 0.56 / h))
    t_valid = np.arange(n_valid)
    ax_c.fill_between([n_valid - 0.5, n_total], -20, 330, color="#E6E6E6", lw=0, zorder=1)
    ax_c.plot(t_valid, SAMPLE_CURVE, color=INK, lw=1.15, zorder=3)
    ax_c.plot([n_valid - 1, n_total], [0, 0], color=MID, lw=0.9, ls=(0, (2.5, 1.5)), zorder=2)
    ax_c.text(1.5, 300, r"$\mathbf{x}_i$", fontsize=6.8, color=INK, va="center")
    ax_c.text((n_valid + n_total) / 2, 150, "zero\npad", fontsize=6.0, color=MID, ha="center", va="center",
              linespacing=1.0)  # fmt: skip
    ax_c.set_xlim(0, n_total)
    ax_c.set_ylim(-20, 340)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.grid(False)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_c.spines[side].set_linewidth(0.6)

    # Mask strip m_i under the curve.
    ax_m = fig.add_axes((0.36 / w, 0.64 / h, 1.30 / w, 0.10 / h))
    ax_m.add_patch(Rectangle((0, 0), n_valid - 0.5, 1, facecolor="#555555", edgecolor="none"))
    ax_m.add_patch(Rectangle((n_valid - 0.5, 0), n_total - n_valid + 0.5, 1, facecolor="#FFFFFF", edgecolor="none"))
    ax_m.add_patch(Rectangle((0, 0), n_total, 1, facecolor="none", edgecolor=INK, lw=0.6))
    ax_m.text((n_valid - 0.5) / 2, 0.5, "1", ha="center", va="center", fontsize=5.8, color="#FFFFFF",
              fontweight="bold")  # fmt: skip
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

    ax.text(s1_cx, 0.37, r"Clean ($L_i \geq 18$, de-dup.)", ha="center", va="center", fontsize=6.3, color=MID)
    ax.text(s1_cx, 0.22, r"Right-pad to $T=1024$ + mask", ha="center", va="center", fontsize=6.3, color=MID)

    # ---- Module 2: three-branch classifier ---------------------------------- #
    s2_x0, s2_x1 = 2.03, 4.13
    _arrow(ax, (s1_x1, main_y), (s2_x0, main_y))
    ax.text((s1_x1 + s2_x0) / 2, main_y + 0.07, r"$\mathbf{x}_i,\mathbf{m}_i$", ha="center", va="bottom",
            fontsize=6.3, color=INK)  # fmt: skip

    _box(ax, s2_x0, blk_bot, s2_x1, blk_top, fc="#FFFFFF", ec=INK, lw=0.85, r=0.04, z=2)
    s2_cx = (s2_x0 + s2_x1) / 2
    ax.text(s2_cx, blk_top - 0.10, "Three-branch classifier", ha="center", va="center",
            fontsize=6.9, fontweight="bold", color=INK)  # fmt: skip

    bus_x = s2_x0 + 0.11
    br_x0, br_x1 = s2_x0 + 0.20, s2_x0 + 1.32
    st_x0, st_x1 = s2_x1 - 0.54, s2_x1 - 0.09
    br_h, br_gap = 0.33, 0.07
    br_top = blk_top - 0.22
    branches = [
        ("Frozen MOMENT", r"$P$, $\Delta P$, mask pooling", r"$\mathbf{p}^{\mathrm{MD}}$"),
        ("60-D DYN + HGB", "raw-scale statistics", r"$\mathbf{p}^{\mathrm{DYN}}$"),
        ("Masked TCN", "causal dilated conv.", r"$\mathbf{p}^{\mathrm{TCN}}$"),
    ]
    br_cys = []
    for k, (title, sub, prob) in enumerate(branches):
        y1 = br_top - k * (br_h + br_gap)
        y0 = y1 - br_h
        cy = (y0 + y1) / 2
        br_cys.append(cy)
        _box(ax, br_x0, y0, br_x1, y1, fc=FILL, ec=MID, lw=0.7, r=0.03, z=3)
        ax.text((br_x0 + br_x1) / 2, cy + 0.068, title, ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=INK, zorder=6)  # fmt: skip
        ax.text((br_x0 + br_x1) / 2, cy - 0.075, sub, ha="center", va="center", fontsize=6.0, color=MID, zorder=6)
        _arrow(ax, (bus_x, cy), (br_x0, cy), lw=0.75, scale=5.5)
        _arrow(ax, (br_x1, cy), (st_x0, cy), lw=0.75, scale=5.5)
        ax.text((br_x1 + st_x0) / 2, cy + 0.035, prob, ha="center", va="bottom", fontsize=6.0, color=INK)
    ax.plot([s2_x0, bus_x], [main_y, main_y], color=INK, lw=0.9, zorder=5)
    ax.plot([bus_x, bus_x], [br_cys[-1], br_cys[0]], color=INK, lw=0.9, zorder=5)

    st_y1 = br_top
    st_y0 = br_top - 3 * br_h - 2 * br_gap
    _box(ax, st_x0, st_y0, st_x1, st_y1, fc=FILL, ec=INK, lw=1.0, r=0.03, z=3)
    st_cx = (st_x0 + st_x1) / 2
    st_cy = (st_y0 + st_y1) / 2
    ax.text(st_cx, st_cy + 0.20, "5-fold", ha="center", va="center", fontsize=6.3, color=INK, zorder=6)
    ax.text(st_cx, st_cy + 0.07, "OOF", ha="center", va="center", fontsize=6.3, color=INK, zorder=6)
    ax.text(st_cx, st_cy - 0.07, "logistic", ha="center", va="center", fontsize=6.3, fontweight="bold",
            color=INK, zorder=6)  # fmt: skip
    ax.text(st_cx, st_cy - 0.20, "stacking", ha="center", va="center", fontsize=6.3, fontweight="bold",
            color=INK, zorder=6)  # fmt: skip

    # ---- Module 3: status and platform action ------------------------------- #
    s3_x0, s3_x1 = 4.41, w - 0.13
    _arrow(ax, (s2_x1, main_y), (s3_x0, main_y))
    ax.text((s2_x1 + s3_x0) / 2, main_y + 0.07, r"$\hat{y}_i$", ha="center", va="bottom", fontsize=6.6, color=INK)

    _box(ax, s3_x0, blk_bot, s3_x1, blk_top, fc="#FFFFFF", ec=INK, lw=0.85, r=0.04, z=2)
    s3_cx = (s3_x0 + s3_x1) / 2
    ax.text(s3_cx, blk_top - 0.10, "Status and action", ha="center", va="center", fontsize=6.9,
            fontweight="bold", color=INK)  # fmt: skip

    outcomes = [
        ("0", "Normal charging", "continue, routine log", "#FFFFFF", MID, 0.7, "-"),
        ("1", "Battery anomaly", "power cut-off, alarm", FILL, INK, 1.3, "-"),
        ("2", "Charger fault", "warning, O&M dispatch", "#FFFFFF", INK, 0.8, (0, (3.0, 1.5))),
    ]
    oc_x0, oc_x1 = s3_x0 + 0.08, s3_x1 - 0.08
    for k, (cls, title, act, fc, ec, lw, ls) in enumerate(outcomes):
        y1 = br_top - k * (br_h + br_gap)
        y0 = y1 - br_h
        cy = (y0 + y1) / 2
        _box(ax, oc_x0, y0, oc_x1, y1, fc=fc, ec=ec, lw=lw, ls=ls, r=0.03, z=3)
        tag_x = oc_x0 + 0.10
        ax.add_patch(Rectangle((tag_x - 0.065, cy - 0.065), 0.13, 0.13, facecolor=INK, edgecolor="none", zorder=5))
        ax.text(tag_x, cy - 0.002, cls, ha="center", va="center", fontsize=6.2, fontweight="bold",
                color="#FFFFFF", zorder=6)  # fmt: skip
        tx = tag_x + 0.11
        ax.text(tx, cy + 0.068, title, ha="left", va="center", fontsize=6.5, fontweight="bold", color=INK, zorder=6)
        ax.text(tx, cy - 0.075, act, ha="left", va="center", fontsize=6.0, color=MID, zorder=6)

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_system_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(d / "fig_system_architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {d / 'fig_system_architecture.pdf'} and .png")
    plt.close(fig)


if __name__ == "__main__":
    plot_fig_system_architecture()
