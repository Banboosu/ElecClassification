"""Plot audited E05B validation histories; no model loading or inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/figures/e05b_tcn_legacy_pilot_20261019.svg"))
    args = parser.parse_args()
    summary = json.loads((args.analysis / "summary.json").read_text())
    if summary["audit_status"] != "passed" or summary["test_evaluated"] or summary["conditions"] != 8:
        raise ValueError("Require the completed validation-only audit")
    styles = {
        "current_early7": ("Current, patience 7", "#111111", ":"),
        "current_fixed50": ("Current, fixed 50", "#277da8", "-"),
        "legacy_masked50": ("Legacy structure, masked", "#e68613", "-"),
        "legacy_unmasked50": ("Legacy structure, unmasked", "#2c8a55", "-"),
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, fraction in zip(axes, (.1, 1.0)):
        rows = [r for r in summary["per_condition"] if r["fraction"] == fraction]
        for row in rows:
            condition = row["condition"]
            history = json.loads((args.analysis / f"fraction{round(fraction*100):03d}_{condition}/history.json").read_text())
            label, color, linestyle = styles[condition]
            zorder = 4 if condition == "current_early7" else 2
            axis.plot([h["epoch"] for h in history], [100*h["macro_f1"] for h in history],
                      color=color, linestyle=linestyle, linewidth=1.6, label=label, zorder=zorder)
            axis.scatter([row["selected_epoch"]], [100*row["selected_macro_f1"]],
                         color=color, s=32, marker="o", edgecolor="white", linewidth=.5, zorder=5)
        axis.set_title(f"{fraction*100:g}% training labels (seed 43)")
        axis.set_xlabel("Epoch")
        axis.set_xlim(1, 50)
        axis.set_ylim(0, 100)
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation Macro-F1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=4, frameon=False)
    fig.text(.5, .015, "Dots: selected checkpoints. Reused validation / one discovery seed; no new test evaluation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
