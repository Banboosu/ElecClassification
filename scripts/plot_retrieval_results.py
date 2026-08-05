from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from tcn_moment.config import load_config
from tcn_moment.data import load_dataset
from tcn_moment.paper_style import PAPER_COLORS, apply_paper_style

METHODS = ("moment", "raw_resampled", "statistical")
METHOD_LABELS = {
    "moment": "MOMENT",
    "raw_resampled": "Raw shape",
    "statistical": "Statistical",
    "length_only": "Length only",
}
COLORS = {
    "moment": PAPER_COLORS["moment_rbf"],
    "raw_resampled": PAPER_COLORS["raw"],
    "statistical": PAPER_COLORS["statistical"],
    "length_only": PAPER_COLORS["neutral"],
}


def _mean_std(
    frame: pd.DataFrame,
    condition: str,
    method: str,
    metric: str,
) -> tuple[float, float]:
    values = frame[
        (frame["condition"] == condition) & (frame["method"] == method)
    ][metric].to_numpy(dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def plot_metrics(run_dir: Path, output_stem: Path) -> None:
    frame = pd.read_csv(run_dir / "condition_metrics.csv")
    length_frame = pd.read_csv(run_dir / "length_only_condition_metrics.csv")
    chance = 1.0 / 3.0
    figure, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)

    clean_methods = (*METHODS, "length_only")
    clean_values = []
    clean_errors = []
    for method in clean_methods:
        if method == "length_only":
            values = length_frame["macro_precision_at_10"].to_numpy()
            clean_values.append(float(values.mean()))
            clean_errors.append(float(values.std(ddof=1)))
        else:
            mean, std = _mean_std(frame, "clean", method, "macro_precision_at_10")
            clean_values.append(mean)
            clean_errors.append(std)
    x = np.arange(len(clean_methods))
    axes[0, 0].bar(
        x,
        np.asarray(clean_values) * 100,
        yerr=np.asarray(clean_errors) * 100,
        color=[COLORS[method] for method in clean_methods],
        capsize=4,
    )
    axes[0, 0].axhline(chance * 100, color="black", linestyle="--", linewidth=1.2)
    axes[0, 0].set_xticks(x, [METHOD_LABELS[method] for method in clean_methods])
    axes[0, 0].set_ylabel("Macro-Precision@10 (%)")
    axes[0, 0].set_title("A. Clean-query semantic retrieval")
    axes[0, 0].set_ylim(25, 75)
    for position, value in zip(x, clean_values, strict=True):
        axes[0, 0].text(position, value * 100 + 1.0, f"{value * 100:.1f}", ha="center")

    class_x = np.arange(3)
    width = 0.18
    class_methods = (*METHODS, "length_only")
    for offset, method in enumerate(class_methods):
        if method == "length_only":
            values = [
                length_frame[f"class_{label}_precision_at_10"].mean()
                for label in range(3)
            ]
            errors = [
                length_frame[f"class_{label}_precision_at_10"].std(ddof=1)
                for label in range(3)
            ]
        else:
            selected = frame[
                (frame["condition"] == "clean") & (frame["method"] == method)
            ]
            values = [float(selected[f"class_{label}_precision_at_10"].iloc[0]) for label in range(3)]
            errors = [0.0, 0.0, 0.0]
        axes[0, 1].bar(
            class_x + (offset - 1.5) * width,
            np.asarray(values) * 100,
            width,
            yerr=np.asarray(errors) * 100,
            color=COLORS[method],
            label=METHOD_LABELS[method],
            capsize=3,
        )
    axes[0, 1].axhline(chance * 100, color="black", linestyle="--", linewidth=1.2)
    axes[0, 1].set_xticks(class_x, ["Class 0", "Class 1", "Class 2"])
    axes[0, 1].set_ylabel("Precision@10 within class (%)")
    axes[0, 1].set_title("B. Clean-query class-wise retrieval")
    axes[0, 1].legend(ncol=2, fontsize=9)

    conditions = ("random_patches_rate0.4", "contiguous_block_rate0.4")
    condition_labels = ("Random patches", "Contiguous block")
    condition_x = np.arange(len(conditions))
    width = 0.24
    for offset, method in enumerate(METHODS):
        values, errors = zip(
            *[
                _mean_std(frame, condition, method, "macro_precision_at_10")
                for condition in conditions
            ],
            strict=True,
        )
        axes[1, 0].bar(
            condition_x + (offset - 1) * width,
            np.asarray(values) * 100,
            width,
            yerr=np.asarray(errors) * 100,
            color=COLORS[method],
            label=METHOD_LABELS[method],
            capsize=4,
        )
    axes[1, 0].axhline(chance * 100, color="black", linestyle="--", linewidth=1.2)
    axes[1, 0].set_xticks(condition_x, condition_labels)
    axes[1, 0].set_ylabel("Macro-Precision@10 (%)")
    axes[1, 0].set_title("C. Semantic retrieval with 40% masked queries")
    axes[1, 0].legend()
    axes[1, 0].set_ylim(30, 72)

    for offset, method in enumerate(METHODS):
        values, errors = zip(
            *[
                _mean_std(frame, condition, method, "clean_neighbor_overlap_at_10")
                for condition in conditions
            ],
            strict=True,
        )
        axes[1, 1].bar(
            condition_x + (offset - 1) * width,
            np.asarray(values) * 100,
            width,
            yerr=np.asarray(errors) * 100,
            color=COLORS[method],
            label=METHOD_LABELS[method],
            capsize=4,
        )
    axes[1, 1].set_xticks(condition_x, condition_labels)
    axes[1, 1].set_ylabel("Clean Top-10 neighbour overlap (%)")
    axes[1, 1].set_title("D. Neighbour-set stability after 40% masking")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Frozen representation retrieval on charging-power sequences",
        fontsize=17,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    figure.savefig(output_stem.with_suffix(".svg"))
    plt.close(figure)


def _resample_zscore(values: np.ndarray, mask: np.ndarray, target_length: int = 256) -> np.ndarray:
    sequence = values[mask.astype(bool)].astype(np.float64)
    std = float(sequence.std())
    normalized = (sequence - sequence.mean()) / std if std > 1e-8 else np.zeros_like(sequence)
    return np.interp(
        np.linspace(0.0, 1.0, target_length),
        np.linspace(0.0, 1.0, len(normalized)),
        normalized,
    )


def plot_examples(run_dir: Path, config_path: str, output_stem: Path) -> None:
    config = load_config(config_path)
    bundle = load_dataset(config.data)
    examples = pd.read_csv(run_dir / "example_neighbors.csv")
    examples = examples[examples["condition"] == "clean"]
    query_rows = examples[["query_id", "query_label"]].drop_duplicates()
    selected_queries = [
        str(query_rows[query_rows["query_label"].astype(str) == str(label)]["query_id"].iloc[0])
        for label in range(3)
    ]
    train_index = {str(sample_id): row for row, sample_id in enumerate(bundle.ids_train)}
    test_index = {str(sample_id): row for row, sample_id in enumerate(bundle.ids_test)}

    figure, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
    method_order = ("moment", "raw_resampled", "statistical")
    time_axis = np.linspace(0.0, 1.0, 256)
    for row, method in enumerate(method_order):
        for column, query_id in enumerate(selected_queries):
            axis = axes[row, column]
            query_index = test_index[query_id]
            query_curve = _resample_zscore(
                bundle.x_test[query_index], bundle.mask_test[query_index]
            )
            axis.plot(time_axis, query_curve, color="black", linewidth=2.4, zorder=5)
            selected = examples[
                (examples["method"] == method) & (examples["query_id"].astype(str) == query_id)
            ].sort_values("rank").head(3)
            same_count = 0
            for _, neighbor in selected.iterrows():
                neighbor_index = train_index[str(neighbor["neighbor_id"])]
                curve = _resample_zscore(
                    bundle.x_train[neighbor_index], bundle.mask_train[neighbor_index]
                )
                same_label = bool(neighbor["same_label"])
                same_count += int(same_label)
                axis.plot(
                    time_axis,
                    curve,
                    color="#009E73" if same_label else "#CC3311",
                    alpha=0.72,
                    linewidth=1.3,
                )
            axis.set_title(
                f"{METHOD_LABELS[method]} — query class {column}\n"
                f"same-class neighbours: {same_count}/3"
            )
            axis.grid(alpha=0.2)
            if row == 2:
                axis.set_xlabel("Normalized time")
            if column == 0:
                axis.set_ylabel("Per-sequence z-score")
    legend = [
        Line2D([0], [0], color="black", linewidth=2.4, label="Query"),
        Line2D([0], [0], color="#009E73", linewidth=1.5, label="Same-class neighbour"),
        Line2D([0], [0], color="#CC3311", linewidth=1.5, label="Different-class neighbour"),
    ]
    figure.legend(handles=legend, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("Deterministic clean-query retrieval examples", fontsize=17, y=0.965)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    figure.savefig(output_stem.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retrieval experiment results.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "artifacts/moment_retrieval/moment_retrieval_zero_shot_thesis_v1"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/moment_retrieval_zero_shot.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    apply_paper_style(plt)
    plot_metrics(
        args.run_dir,
        args.output_dir / "moment_retrieval_metrics_20260803",
    )
    plot_examples(
        args.run_dir,
        args.config,
        args.output_dir / "moment_retrieval_examples_20260803",
    )
    print(f"Saved retrieval figures to {args.output_dir}")


if __name__ == "__main__":
    main()
