from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcn_moment.paper_style import PAPER_COLORS, apply_paper_style

MODEL_LABELS = {
    "TCN": "TCN",
    "MOMENT_FULL_FINETUNE": "MOMENT full fine-tune",
    "MOMENT_RBF_SVM": "MOMENT + RBF-SVM",
    "MOMENT_RBF_SVM_FEW_SHOT": "MOMENT + RBF-SVM",
}

MODEL_COLORS = {
    "TCN": PAPER_COLORS["tcn"],
    "MOMENT_FULL_FINETUNE": PAPER_COLORS["moment_full"],
    "MOMENT_RBF_SVM": PAPER_COLORS["moment_rbf"],
    "MOMENT_RBF_SVM_FEW_SHOT": PAPER_COLORS["moment_rbf"],
}

T_CRITICAL_95_DF4 = 2.7764451051977987


def _mean_std(values: pd.Series) -> tuple[float, float]:
    numeric = values.astype(float).to_numpy()
    return float(numeric.mean()), float(numeric.std(ddof=1))


def _paired_comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    comparisons: list[dict[str, float | str | int]] = []
    metrics = (
        "test_battery_recall",
        "test_battery_precision",
        "test_false_positive_rate",
        "test_f2",
        "test_average_precision",
        "test_roc_auc",
    )

    def compare(
        first: pd.DataFrame,
        second: pd.DataFrame,
        *,
        comparison: str,
        fraction: float,
        operating_point: str,
    ) -> None:
        merged = first.merge(second, on="seed", suffixes=("_first", "_second"))
        for metric in metrics:
            differences = (
                merged[f"{metric}_first"].astype(float)
                - merged[f"{metric}_second"].astype(float)
            ).to_numpy()
            mean = float(differences.mean())
            std = float(differences.std(ddof=1))
            half_width = T_CRITICAL_95_DF4 * std / np.sqrt(len(differences))
            comparisons.append(
                {
                    "comparison": comparison,
                    "train_fraction": fraction,
                    "operating_point": operating_point,
                    "metric": metric,
                    "n": len(differences),
                    "mean_difference": mean,
                    "sample_std_difference": std,
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                }
            )

    for operating_point in ("argmax", "max_f2", "recall_0.95", "recall_0.99"):
        selected = rows[rows["operating_point"] == operating_point]
        for fraction in (0.01, 0.05, 0.1, 0.2, 0.4):
            first = selected[
                (selected["model"] == "MOMENT_RBF_SVM_FEW_SHOT")
                & np.isclose(selected["train_fraction"].astype(float), fraction)
            ]
            second = selected[
                (selected["model"] == "TCN")
                & np.isclose(selected["train_fraction"].astype(float), fraction)
            ]
            compare(
                first,
                second,
                comparison="MOMENT_RBF_SVM_FEW_SHOT-minus-TCN",
                fraction=fraction,
                operating_point=operating_point,
            )

    selected = rows[
        (rows["operating_point"].isin(("argmax", "max_f2", "recall_0.95", "recall_0.99")))
        & np.isclose(rows["train_fraction"].astype(float), 1.0)
    ]
    for operating_point in selected["operating_point"].unique():
        point = selected[selected["operating_point"] == operating_point]
        first = point[point["model"] == "TCN"]
        second = point[point["model"] == "MOMENT_FULL_FINETUNE"]
        compare(
            first,
            second,
            comparison="TCN-minus-MOMENT_FULL_FINETUNE",
            fraction=1.0,
            operating_point=str(operating_point),
        )
    return pd.DataFrame(comparisons)


def _few_shot_summary(rows: pd.DataFrame) -> pd.DataFrame:
    selected = rows[
        (rows["operating_point"] == "argmax")
        & (rows["model"].isin(("TCN", "MOMENT_RBF_SVM_FEW_SHOT")))
        & (rows["train_fraction"].astype(float) < 1)
    ].copy()
    metrics = [
        "test_battery_recall",
        "test_battery_precision",
        "test_false_positive_rate",
        "test_f2",
        "test_average_precision",
        "test_roc_auc",
        "test_false_negatives",
        "test_false_positives",
    ]
    aggregate = selected.groupby(["model", "train_fraction"])[metrics].agg(
        ["count", "mean", "std", "min", "max"]
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def _full_label_summary(rows: pd.DataFrame) -> pd.DataFrame:
    selected = rows[
        np.isclose(rows["train_fraction"].astype(float), 1.0)
        & rows["model"].isin(("TCN", "MOMENT_FULL_FINETUNE", "MOMENT_RBF_SVM"))
    ].copy()
    metrics = [
        "test_battery_recall",
        "test_battery_precision",
        "test_miss_rate",
        "test_false_positive_rate",
        "test_f2",
        "test_average_precision",
        "test_roc_auc",
        "test_false_negatives",
        "test_false_positives",
    ]
    aggregate = selected.groupby(["model", "operating_point"])[metrics].agg(
        ["count", "mean", "std", "min", "max"]
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def _plot_few_shot(rows: pd.DataFrame, output_stem: Path) -> None:
    selected = rows[
        (rows["operating_point"] == "argmax")
        & (rows["model"].isin(("TCN", "MOMENT_RBF_SVM_FEW_SHOT")))
        & (rows["train_fraction"].astype(float) < 1)
    ].copy()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_specs = (
        ("test_battery_recall", "Battery recall (%)"),
        ("test_false_positive_rate", "False-positive rate (%)"),
        ("test_average_precision", "Battery PR-AUC (%)"),
    )
    for model in ("TCN", "MOMENT_RBF_SVM_FEW_SHOT"):
        model_rows = selected[selected["model"] == model]
        fractions = sorted(model_rows["train_fraction"].astype(float).unique())
        for axis, (metric, title) in zip(axes, metric_specs, strict=True):
            means = []
            stds = []
            for fraction in fractions:
                values = model_rows[
                    np.isclose(model_rows["train_fraction"].astype(float), fraction)
                ][metric]
                mean, std = _mean_std(values)
                means.append(100 * mean)
                stds.append(100 * std)
            axis.errorbar(
                np.asarray(fractions) * 100,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=3,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
            axis.set_title(title)
            axis.set_xlabel("Labelled training data (%)")
            axis.grid(alpha=0.25)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 100)
    axes[2].set_ylim(0, 100)
    axes[0].legend(loc="lower right")
    figure.suptitle("Battery-abnormality detection under limited labels (argmax)")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_full_threshold_tradeoff(rows: pd.DataFrame, output_stem: Path) -> None:
    selected = rows[
        np.isclose(rows["train_fraction"].astype(float), 1.0)
        & rows["model"].isin(("TCN", "MOMENT_FULL_FINETUNE", "MOMENT_RBF_SVM"))
    ].copy()
    operating_points = ("argmax", "max_f2", "recall_0.95", "recall_0.98", "recall_0.99")
    figure, axis = plt.subplots(figsize=(8, 6))
    for model in ("TCN", "MOMENT_FULL_FINETUNE", "MOMENT_RBF_SVM"):
        means_x = []
        means_y = []
        errors_x = []
        errors_y = []
        for point in operating_points:
            values = selected[
                (selected["model"] == model)
                & (selected["operating_point"] == point)
            ]
            mean_x, std_x = _mean_std(values["test_false_positive_rate"])
            mean_y, std_y = _mean_std(values["test_battery_recall"])
            means_x.append(100 * mean_x)
            means_y.append(100 * mean_y)
            errors_x.append(100 * std_x)
            errors_y.append(100 * std_y)
        axis.errorbar(
            means_x,
            means_y,
            xerr=errors_x,
            yerr=errors_y,
            marker="o",
            linewidth=2,
            capsize=3,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        for x, y, point in zip(means_x, means_y, operating_points, strict=True):
            axis.annotate(
                point.replace("recall_", "R"),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("Test false-positive rate (%)")
    axis.set_ylabel("Test battery recall (%)")
    axis.set_title("Full-label battery recall–false-alarm trade-off")
    axis.set_xlim(left=0)
    axis.set_ylim(75, 100.5)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze battery safety evaluation results.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/battery_safety_thesis_v1"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    apply_paper_style(plt)

    input_dir: Path = args.input_dir
    figure_dir: Path = args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(input_dir / "per_seed_metrics.csv")
    if len(rows) != 325:
        raise ValueError(f"Expected 325 per-seed operating-point rows, got {len(rows)}.")
    if rows["run_name"].nunique() != 45:
        raise ValueError("Expected 45 unique completed source runs.")

    few_shot = _few_shot_summary(rows)
    full_label = _full_label_summary(rows)
    paired = _paired_comparisons(rows)
    few_shot.to_csv(input_dir / "few_shot_battery_summary.csv", index=False)
    full_label.to_csv(input_dir / "full_label_battery_summary.csv", index=False)
    paired.to_csv(input_dir / "paired_comparisons.csv", index=False)

    _plot_few_shot(rows, figure_dir / "battery_safety_few_shot_20260804")
    _plot_full_threshold_tradeoff(
        rows,
        figure_dir / "battery_safety_threshold_tradeoff_20260804",
    )

    metadata = json.loads((input_dir / "metrics.json").read_text(encoding="utf-8"))
    analysis = {
        "protocol_version": 1,
        "source_protocol": metadata,
        "validated_per_seed_rows": len(rows),
        "validated_source_runs": int(rows["run_name"].nunique()),
        "full_label_summary_csv": str(input_dir / "full_label_battery_summary.csv"),
        "few_shot_summary_csv": str(input_dir / "few_shot_battery_summary.csv"),
        "paired_comparisons_csv": str(input_dir / "paired_comparisons.csv"),
        "figures": [
            str(figure_dir / "battery_safety_few_shot_20260804.png"),
            str(figure_dir / "battery_safety_threshold_tradeoff_20260804.png"),
        ],
    }
    (input_dir / "derived_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
