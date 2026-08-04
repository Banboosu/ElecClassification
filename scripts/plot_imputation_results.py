from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "moment_zero_shot": "MOMENT zero-shot",
    "linear": "Linear interpolation",
    "forward_fill": "Forward fill",
    "mean": "Visible-value mean",
}
METHOD_COLORS = {
    "moment_zero_shot": "#d95f02",
    "linear": "#1b9e77",
    "forward_fill": "#377eb8",
    "mean": "#777777",
}
PATTERN_LABELS = {
    "random_patches": "Random patch missingness",
    "contiguous_block": "Contiguous block missingness",
}


def plot_summary(input_dir: Path, output_stem: Path) -> None:
    summary = pd.read_csv(input_dir / "summary.csv")
    methods = tuple(METHOD_LABELS)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, pattern in zip(axes, PATTERN_LABELS, strict=True):
        selected_pattern = summary[summary["pattern"] == pattern]
        for method in methods:
            selected = selected_pattern[selected_pattern["method"] == method].sort_values(
                "mask_rate"
            )
            axis.errorbar(
                selected["mask_rate"] * 100,
                selected["macro_nrmse_mean"],
                yerr=selected["macro_nrmse_sample_std"],
                marker="o",
                linewidth=2,
                capsize=3,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        axis.set_title(PATTERN_LABELS[pattern])
        axis.set_xlabel("Masked valid points (%)")
        axis.set_xticks([10, 25, 40, 60])
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Macro-NRMSE (lower is better)")
    axes[1].legend(frameon=True, fontsize=9)
    figure.suptitle("Zero-shot imputation on charging-power sequences (5 mask seeds)")
    figure.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")


def _hidden_prediction(values: np.ndarray, observation_mask: np.ndarray) -> np.ndarray:
    prediction = values.astype(np.float64, copy=True)
    prediction[observation_mask.astype(bool)] = np.nan
    return prediction


def plot_examples(input_dir: Path, output_stem: Path, example_index: int) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.5))
    for axis, pattern in zip(axes, PATTERN_LABELS, strict=True):
        archive = np.load(input_dir / f"examples_{pattern}_rate0p6.npz")
        length = int(archive["lengths"][example_index])
        sample_id = str(archive["sample_ids"][example_index])
        positions = np.arange(length)
        raw = archive["raw_values"][example_index, :length]
        observed = archive["observation_mask"][example_index, :length].astype(bool)
        axis.plot(positions, raw, color="#222222", alpha=0.5, linewidth=1.3, label="Ground truth")
        axis.scatter(
            positions[observed],
            raw[observed],
            s=8,
            color="#222222",
            label="Observed points",
            zorder=4,
        )
        for method in ("moment_zero_shot", "linear", "forward_fill"):
            prediction = _hidden_prediction(
                archive[method][example_index, :length],
                observed,
            )
            axis.plot(
                positions,
                prediction,
                linewidth=1.8,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        axis.set_title(f"{PATTERN_LABELS[pattern]} — sample {sample_id}, 60% target rate")
        axis.set_ylabel("Charging power")
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Time index")
    axes[0].legend(frameon=True, ncol=2, fontsize=9)
    figure.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot zero-shot imputation results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/imports/moment_imputation_zero_shot_thesis_v2"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("docs/figures/moment_imputation_macro_nrmse_20260803"),
    )
    parser.add_argument(
        "--examples-output",
        type=Path,
        default=Path("docs/figures/moment_imputation_examples_20260803"),
    )
    parser.add_argument("--example-index", type=int, default=0)
    args = parser.parse_args()
    plot_summary(args.input, args.summary_output)
    plot_examples(args.input, args.examples_output, args.example_index)
    print(f"Saved {args.summary_output.with_suffix('.png')}")
    print(f"Saved {args.examples_output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
