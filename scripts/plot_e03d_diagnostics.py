"""Posthoc descriptive curves; not used for model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/improvements/e03d_head_diagnostic_stable")
    parser.add_argument("--output", default="docs/figures/e03d_head_diagnostic_20261018.svg")
    args = parser.parse_args()
    root = Path(args.root)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, branch in zip(axes, ("statistics", "moment", "fusion"), strict=True):
        for mode, label, color in (
            ("legacy_fresh", "Original scale", "#777777"),
            ("fanin_gain", "Fixed input gain", "#0072B2"),
        ):
            history = json.loads(
                (
                    root / f"e03d_head_diagnostic_20261018_v2_{branch}_{mode}/history.json"
                ).read_text()
            )
            ax.plot(
                [r["epoch"] for r in history],
                [r["validation_macro_f1"] * 100 for r in history],
                label=label,
                color=color,
                linewidth=1,
            )
        ax.axvline(30, color="#999999", linestyle=":", linewidth=0.8)
        if branch == "fusion":
            ax.axhline(
                91.14590740737443,
                color="#D55E00",
                linestyle="--",
                linewidth=1,
                label="Parent fusion SVM",
            )
        ax.set_title(branch.capitalize())
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.set_xlim(1, 200)
    axes[0].set_ylabel("Validation Macro-F1 (%)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "E03D: frozen cached heads, seed43, 40% labels (exploratory validation)", fontsize=10
    )
    fig.tight_layout()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
