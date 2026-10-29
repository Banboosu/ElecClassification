"""Plot corrected class-1 battery / class-2 charger validation operating points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcn_moment.label_semantics import SEMANTICS_VERSION
from tcn_moment.paper_style import apply_paper_style

METHODS = {
    "dyn_hgb": ("DYN+HGB", "#0072B2"),
    "tcn": ("TCN", "#009E73"),
    "tcn_dyn": ("TCN+DYN", "#56B4E9"),
    "full": ("Full", "#000000"),
    "multirocket": ("MultiRocket", "#B2182B"),
}


def followup_plots(result: dict, output: Path, e10_path: Path) -> None:
    fine = {(r["tag"], r["method"]): r for r in result["aggregate"] if r["suite"] == "e13_fine"}
    if fine:
        levels = ("clean", "noise_0p1pct", "noise_0p25pct", "noise_0p5pct", "noise_1pct")
        x = [0., .1, .25, .5, 1.]
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
        for method, name, color in (("dyn_hgb", "DYN+HGB", "#0072B2"),
                                    ("md_logit", "MOMENT dual", "#D55E00"),
                                    ("tcn", "TCN", "#009E73"), ("full", "Full", "#000000")):
            for axis, cls, metric in zip(axes, ("battery", "battery", "charger"), ("recall", "fpr", "fpr"), strict=True):
                values = [100 * fine[("40p", f"{level}/{method}")][cls][metric]["mean"] for level in levels]
                axis.plot(x, values, marker="o", linewidth=1.5, color=color, label=name)
        for axis, title in zip(axes, ("Battery (1): recall", "Battery (1): FPR", "Charger (2): FPR"), strict=True):
            axis.set_xlabel("Noise SD / training RMS (%)")
            axis.set_ylabel("Rate (%)")
            axis.set_title(title, fontsize=9)
            axis.grid(alpha=.25)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle("40% training labels; clean-trained; perturbed validation; four-seed means", fontsize=9)
        fig.tight_layout()
        for suffix in ("svg", "png"):
            fig.savefig(output / f"e13_class_risks_v2.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    if e10_path.exists():
        e10 = json.loads(e10_path.read_text(encoding="utf-8"))
        if e10["battery_label"] != "1" or not e10["descriptive_only"]:
            raise ValueError("E10 must be the corrected descriptive class-1 analysis")
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
        for tag, color in (("10p", "#D55E00"), ("40p", "#000000")):
            values = e10["aggregate"][tag]["alert_curve"]
            for axis, metric in zip(axes, ("recall", "fpr"), strict=True):
                axis.plot([100*r["alert_rate"] for r in values], [100*r[metric] for r in values],
                          marker="o", color=color, label=f"{tag[:-1]}% training labels")
        for axis, ylabel in zip(axes, ("Battery recall (%)", "Battery FPR (%)"), strict=True):
            axis.set_xlabel("Top-score alert fraction (%)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.25)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle("E10: battery label 1; descriptive validation alarm curves only", fontsize=10)
        fig.tight_layout()
        for suffix in ("svg", "png"):
            fig.savefig(output / f"e10_battery_alert_v2.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/analysis/label_semantics_correction_20261029_v2/summary.json"))
    parser.add_argument("--e10", type=Path, default=Path("artifacts/analysis/e10_battery_semantics_v2_20261029/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/label_semantics_v2"))
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    if result["semantics_version"] != SEMANTICS_VERSION or result["battery_label"] != "1" or result["charger_label"] != "2":
        raise ValueError("Only the audited corrected semantic mapping may be plotted")
    lookup = {(r["tag"], r["method"]): r for r in result["aggregate"] if r["suite"] in ("e18", "e18_100p")}
    tags = ("1p", "5p", "10p", "20p", "40p", "100p")
    x = [int(tag[:-1]) for tag in tags]
    apply_paper_style(plt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cls, title in (("battery", "Battery (raw label 1)"), ("charger", "Charger fault (raw label 2)")):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
        for method, (name, color) in METHODS.items():
            for axis, metric in zip(axes, ("recall", "fpr"), strict=True):
                values = [100 * lookup[(tag, method)][cls][metric]["mean"] for tag in tags]
                axis.plot(x, values, marker="o", linewidth=1.6, color=color, label=name)
        for axis, ylabel in zip(axes, ("Recall (%)", "False positive rate (%)"), strict=True):
            axis.set_xscale("log")
            axis.set_xticks(x, [f"{value}%" for value in x])
            axis.set_xlabel("Training-label budget")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.25)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{title}: validation argmax, four-seed means", fontsize=10)
        fig.tight_layout()
        for suffix in ("svg", "png"):
            fig.savefig(args.output_dir / f"e18_{cls}_recall_fpr_v2.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    followup_plots(result, args.output_dir, args.e10)
    print("Wrote corrected battery/charger figures:", args.output_dir)


if __name__ == "__main__":
    main()
