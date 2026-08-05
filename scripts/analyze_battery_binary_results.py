from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcn_moment.paper_style import PAPER_COLORS, apply_paper_style

FRACTIONS = (0.01, 0.05, 0.1, 1.0)
MODEL_SPECS = (
    ("TCN", "TCN", PAPER_COLORS["tcn"], "-"),
    ("MOMENT_MULTICLASS", "MOMENT three-class", PAPER_COLORS["moment_rbf"], "--"),
    ("BATTERY_BINARY_MOMENT_RBF_SVM", "MOMENT binary", PAPER_COLORS["moment_full"], "-"),
    (
        "BATTERY_BINARY_LOGISTIC_REGRESSION",
        "Logistic regression",
        PAPER_COLORS["logistic_regression"],
        ":",
    ),
    (
        "BATTERY_BINARY_RANDOM_FOREST",
        "Random forest",
        PAPER_COLORS["random_forest"],
        "-",
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def combined_recall95_rows(
    baseline: pd.DataFrame,
    binary: pd.DataFrame,
) -> pd.DataFrame:
    old = baseline.loc[
        (baseline["operating_point"] == "recall_0.95")
        & baseline["model"].isin(("TCN", "MOMENT_RBF_SVM", "MOMENT_RBF_SVM_FEW_SHOT"))
        & baseline["train_fraction"].astype(float).isin(FRACTIONS)
    ].copy()
    old["comparison_model"] = old["model"].replace(
        {
            "MOMENT_RBF_SVM": "MOMENT_MULTICLASS",
            "MOMENT_RBF_SVM_FEW_SHOT": "MOMENT_MULTICLASS",
        }
    )
    new = binary.loc[
        (binary["operating_point"] == "recall_0.95")
        & binary["train_fraction"].astype(float).isin(FRACTIONS)
    ].copy()
    new["comparison_model"] = new["model"]
    columns = [
        "comparison_model",
        "seed",
        "train_fraction",
        "test_average_precision",
        "test_battery_recall",
        "test_battery_precision",
        "test_false_positive_rate",
        "test_f2",
        "test_false_negatives",
        "test_false_positives",
    ]
    return pd.concat([old[columns], new[columns]], ignore_index=True)


def aggregate_combined(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "test_average_precision",
        "test_battery_recall",
        "test_battery_precision",
        "test_false_positive_rate",
        "test_f2",
        "test_false_negatives",
        "test_false_positives",
    ]
    aggregate = rows.groupby(["comparison_model", "train_fraction"])[metrics].agg(
        ["count", "mean", "std", "min", "max"]
    )
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def _plot_combined(rows: pd.DataFrame, output_stem: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(FRACTIONS))
    for model, label, color, line_style in MODEL_SPECS:
        selected = rows.loc[rows["comparison_model"] == model]
        ap_means = []
        ap_stds = []
        fpr_means = []
        fpr_stds = []
        for fraction in FRACTIONS:
            values = selected.loc[
                np.isclose(selected["train_fraction"].astype(float), fraction)
            ]
            ap_means.append(100 * values["test_average_precision"].astype(float).mean())
            ap_stds.append(100 * values["test_average_precision"].astype(float).std(ddof=1))
            fpr_means.append(
                100 * values["test_false_positive_rate"].astype(float).mean()
            )
            fpr_stds.append(
                100 * values["test_false_positive_rate"].astype(float).std(ddof=1)
            )
        axes[0].errorbar(
            x,
            ap_means,
            yerr=ap_stds,
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
            color=color,
            linestyle=line_style,
        )
        axes[1].errorbar(
            x,
            fpr_means,
            yerr=fpr_stds,
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
            color=color,
            linestyle=line_style,
        )
    labels = ["1%", "5%", "10%", "100%"]
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Labelled training data")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test battery PR-AUC (%)")
    axes[0].set_title("Ranking quality")
    axes[0].set_ylim(65, 101)
    axes[1].set_ylabel("Test false-positive rate (%)")
    axes[1].set_title("FPR at validation-targeted 95% recall")
    axes[1].set_ylim(0, 100)
    axes[0].legend(fontsize=8, loc="lower right")
    figure.suptitle("Dedicated battery detection: five-seed comparison")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def statistical_feature_importance(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        metrics = _read_json(run_dir / "metrics.json")
        seed = int(metrics["seed"])
        fraction = float(metrics["data"]["train_subset"]["requested_fraction"])
        feature_names = list(metrics["feature_names"])

        forest = joblib.load(run_dir / "battery_random_forest.joblib")
        for feature, importance in zip(
            feature_names,
            forest.feature_importances_,
            strict=True,
        ):
            rows.append(
                {
                    "estimator": "random_forest",
                    "seed": seed,
                    "train_fraction": fraction,
                    "feature": feature,
                    "signed_value": float(importance),
                    "normalized_importance": float(importance),
                }
            )

        pipeline = joblib.load(run_dir / "battery_logistic_regression.joblib")
        coefficients = pipeline.named_steps["logisticregression"].coef_[0]
        scale = float(np.abs(coefficients).sum())
        for feature, coefficient in zip(feature_names, coefficients, strict=True):
            rows.append(
                {
                    "estimator": "logistic_regression",
                    "seed": seed,
                    "train_fraction": fraction,
                    "feature": feature,
                    "signed_value": float(coefficient),
                    "normalized_importance": float(abs(coefficient) / scale),
                }
            )
    return pd.DataFrame(rows)


def aggregate_feature_importance(rows: pd.DataFrame) -> pd.DataFrame:
    aggregate = rows.groupby(["estimator", "train_fraction", "feature"])[
        ["signed_value", "normalized_importance"]
    ].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate["rank"] = aggregate.groupby(["estimator", "train_fraction"])[
        "normalized_importance_mean"
    ].rank(method="first", ascending=False)
    return aggregate.sort_values(["estimator", "train_fraction", "rank"])


def _plot_feature_importance(aggregate: pd.DataFrame, output_stem: Path) -> None:
    feature_order = (
        aggregate.groupby("feature")["normalized_importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for axis, estimator, title in zip(
        axes,
        ("logistic_regression", "random_forest"),
        ("Logistic regression", "Random forest"),
        strict=True,
    ):
        selected = aggregate.loc[aggregate["estimator"] == estimator]
        matrix = np.asarray(
            [
                [
                    selected.loc[
                        np.isclose(selected["train_fraction"].astype(float), fraction)
                        & (selected["feature"] == feature),
                        "normalized_importance_mean",
                    ].iloc[0]
                    for fraction in FRACTIONS
                ]
                for feature in feature_order
            ]
        )
        image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
        axis.set_title(title)
        axis.set_xticks(range(len(FRACTIONS)), ["1%", "5%", "10%", "100%"])
        axis.set_xlabel("Labelled training data")
        axis.set_yticks(range(len(feature_order)), feature_order)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Statistical feature")
    figure.suptitle("Normalized feature importance for battery-vs-rest baselines")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze formal battery binary results.")
    parser.add_argument(
        "--formal-dir",
        type=Path,
        default=Path("artifacts/battery_binary_analysis/formal_five_seed"),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("artifacts/battery_safety_thesis_v1"),
    )
    parser.add_argument("--stats-run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    apply_paper_style(plt)

    formal_dir: Path = args.formal_dir
    figure_dir: Path = args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(args.baseline_dir / "per_seed_metrics.csv")
    binary = pd.read_csv(formal_dir / "per_seed_metrics.csv")
    combined = combined_recall95_rows(baseline, binary)
    combined_aggregate = aggregate_combined(combined)
    combined.to_csv(formal_dir / "combined_recall95_per_seed.csv", index=False)
    combined_aggregate.to_csv(
        formal_dir / "combined_recall95_aggregate.csv",
        index=False,
    )
    _plot_combined(combined, figure_dir / "battery_binary_comparison_20260804")

    importance = statistical_feature_importance(args.stats_run_dirs)
    importance_aggregate = aggregate_feature_importance(importance)
    importance.to_csv(formal_dir / "feature_importance_per_seed.csv", index=False)
    importance_aggregate.to_csv(
        formal_dir / "feature_importance_aggregate.csv",
        index=False,
    )
    _plot_feature_importance(
        importance_aggregate,
        figure_dir / "battery_binary_feature_importance_20260804",
    )

    top_features = {}
    for (estimator, fraction), rows in importance_aggregate.groupby(
        ["estimator", "train_fraction"]
    ):
        top_features[f"{estimator}::{float(fraction):g}"] = rows.nsmallest(5, "rank")[
            ["feature", "normalized_importance_mean", "signed_value_mean"]
        ].to_dict(orient="records")
    analysis = {
        "protocol_version": 1,
        "combined_model_count": int(combined["comparison_model"].nunique()),
        "combined_per_seed_rows": len(combined),
        "statistical_source_runs": len(args.stats_run_dirs),
        "top_features": top_features,
        "figures": [
            str(figure_dir / "battery_binary_comparison_20260804.png"),
            str(figure_dir / "battery_binary_feature_importance_20260804.png"),
        ],
    }
    (formal_dir / "derived_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
