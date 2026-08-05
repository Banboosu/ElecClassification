from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tcn_moment.io_utils import atomic_write_json


LOW_LABEL_FRACTIONS = (0.01, 0.05, 0.1)
FULL_LABEL_FRACTION = 1.0
AP_IMPROVEMENT_GATE = 0.01
FPR_REDUCTION_GATE = 0.02
STATISTICAL_AP_MARGIN = 0.02


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _validation_row(
    *,
    payload: dict[str, Any],
    evaluation: dict[str, Any],
    run_dir: Path,
    family: str,
    variant: str,
    train_fraction: float,
    selected_positive_weight: float,
    selected_c: float | None = None,
) -> dict[str, Any]:
    ranking = evaluation["ranking"]["validation"]
    recall_point = evaluation["operating_points"]["recall_0.95"]["validation"]
    return {
        "run_name": payload["run_name"],
        "run_dir": str(run_dir),
        "source_model": payload["model"],
        "family": family,
        "variant": variant,
        "seed": int(payload["seed"]),
        "train_fraction": float(train_fraction),
        "selected_positive_weight": float(selected_positive_weight),
        "selected_c": selected_c,
        "validation_average_precision": float(ranking["average_precision"]),
        "validation_roc_auc": float(ranking["roc_auc"]),
        "validation_recall_at_target_95": float(recall_point["battery_recall"]),
        "validation_fpr_at_target_95": float(recall_point["false_positive_rate"]),
        "validation_precision_at_target_95": float(
            recall_point["battery_precision"]
        ),
        "validation_f2_at_target_95": float(recall_point["f2"]),
    }


def validation_rows_from_binary_metrics(
    payload: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Extract validation-only rows without accessing any test result fields."""
    model = str(payload["model"])
    if model == "BATTERY_BINARY_TCN":
        return [
            _validation_row(
                payload=payload,
                evaluation=payload,
                run_dir=run_dir,
                family="TCN",
                variant="binary_tcn",
                train_fraction=payload["data"]["train_subset"]["requested_fraction"],
                selected_positive_weight=payload["selected_positive_weight"],
            )
        ]
    if model == "BATTERY_BINARY_STATISTICAL":
        fraction = payload["data"]["train_subset"]["requested_fraction"]
        return [
            _validation_row(
                payload=payload,
                evaluation=evaluation,
                run_dir=run_dir,
                family="STATISTICAL",
                variant=estimator_name,
                train_fraction=fraction,
                selected_positive_weight=evaluation["selected_positive_weight"],
            )
            for estimator_name, evaluation in payload["results"].items()
        ]
    if model == "BATTERY_BINARY_MOMENT_RBF_SVM":
        rows = []
        for evaluation in payload["fractions"]:
            class_weight = evaluation["best_params"]["class_weight"]
            rows.append(
                _validation_row(
                    payload=payload,
                    evaluation=evaluation,
                    run_dir=run_dir,
                    family="MOMENT_RBF_SVM",
                    variant="binary_moment_rbf_svm",
                    train_fraction=evaluation["train_fraction"],
                    selected_positive_weight=class_weight.get("1", class_weight.get(1)),
                    selected_c=float(evaluation["best_params"]["C"]),
                )
            )
        return rows
    raise ValueError(f"Unsupported battery binary model: {model}")


def baseline_validation_rows(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    selected = frame.loc[
        (frame["seed"].astype(int) == seed)
        & (frame["operating_point"] == "recall_0.95")
        & frame["model"].isin(
            ["TCN", "MOMENT_RBF_SVM", "MOMENT_RBF_SVM_FEW_SHOT"]
        )
    ].copy()
    selected["family"] = selected["model"].replace(
        {
            "MOMENT_RBF_SVM_FEW_SHOT": "MOMENT_RBF_SVM",
        }
    )
    columns = [
        "run_name",
        "model",
        "family",
        "seed",
        "train_fraction",
        "validation_average_precision",
        "validation_battery_recall",
        "validation_false_positive_rate",
    ]
    selected = selected[columns].rename(
        columns={
            "run_name": "baseline_run_name",
            "model": "baseline_model",
            "validation_battery_recall": "baseline_validation_recall_at_target_95",
            "validation_false_positive_rate": "baseline_validation_fpr_at_target_95",
        }
    )
    duplicated = selected.duplicated(["family", "seed", "train_fraction"], keep=False)
    if duplicated.any():
        values = selected.loc[duplicated, ["family", "seed", "train_fraction"]]
        raise ValueError(f"Duplicate baseline validation rows:\n{values}")
    return selected.reset_index(drop=True)


def compare_validation_rows(
    binary: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    comparisons = []
    for row in binary.to_dict(orient="records"):
        fraction = float(row["train_fraction"])
        family = str(row["family"])
        if family == "STATISTICAL":
            candidates = baseline.loc[baseline["train_fraction"] == fraction]
            if candidates.empty:
                continue
            reference = candidates.sort_values(
                "validation_average_precision", ascending=False
            ).iloc[0]
        else:
            candidates = baseline.loc[
                (baseline["family"] == family)
                & (baseline["train_fraction"] == fraction)
            ]
            if len(candidates) != 1:
                continue
            reference = candidates.iloc[0]
        comparisons.append(
            {
                **row,
                "baseline_run_name": reference["baseline_run_name"],
                "baseline_model": reference["baseline_model"],
                "baseline_validation_average_precision": float(
                    reference["validation_average_precision"]
                ),
                "baseline_validation_recall_at_target_95": float(
                    reference["baseline_validation_recall_at_target_95"]
                ),
                "baseline_validation_fpr_at_target_95": float(
                    reference["baseline_validation_fpr_at_target_95"]
                ),
                "average_precision_delta": float(
                    row["validation_average_precision"]
                    - reference["validation_average_precision"]
                ),
                "fpr_delta_at_target_95": float(
                    row["validation_fpr_at_target_95"]
                    - reference["baseline_validation_fpr_at_target_95"]
                ),
            }
        )
    return pd.DataFrame(comparisons)


def _full_or_two_low_label_passes(values: dict[float, bool]) -> bool:
    full_pass = bool(values.get(FULL_LABEL_FRACTION, False))
    low_label_count = sum(bool(values.get(fraction, False)) for fraction in LOW_LABEL_FRACTIONS)
    return full_pass or low_label_count >= 2


def gate_decisions(comparisons: pd.DataFrame) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for family in ("TCN", "MOMENT_RBF_SVM"):
        family_rows = comparisons.loc[comparisons["family"] == family]
        ap_passes = {
            float(row.train_fraction): bool(
                row.average_precision_delta >= AP_IMPROVEMENT_GATE
            )
            for row in family_rows.itertuples()
        }
        fpr_passes = {
            float(row.train_fraction): bool(
                row.validation_recall_at_target_95 >= 0.95
                and row.fpr_delta_at_target_95 <= -FPR_REDUCTION_GATE
            )
            for row in family_rows.itertuples()
        }
        ap_gate = _full_or_two_low_label_passes(ap_passes)
        fpr_gate = _full_or_two_low_label_passes(fpr_passes)
        decisions[family] = {
            "expand_to_five_seeds": ap_gate or fpr_gate,
            "average_precision_gate_passed": ap_gate,
            "fpr_gate_passed": fpr_gate,
            "average_precision_pass_by_fraction": ap_passes,
            "fpr_pass_by_fraction": fpr_passes,
        }

    statistical = comparisons.loc[comparisons["family"] == "STATISTICAL"]
    for variant, rows in statistical.groupby("variant"):
        close_passes = {
            float(row.train_fraction): bool(
                row.average_precision_delta >= -STATISTICAL_AP_MARGIN
            )
            for row in rows.itertuples()
        }
        decisions[f"STATISTICAL::{variant}"] = {
            "expand_to_five_seeds": _full_or_two_low_label_passes(close_passes),
            "within_two_ap_points_by_fraction": close_passes,
        }
    return decisions


def _markdown_report(
    comparisons: pd.DataFrame,
    decisions: dict[str, Any],
    seed: int,
) -> str:
    lines = [
        f"# 电池专用二分类 seed {seed} 验证集门控",
        "",
        "> 本报告只提取 validation 指标；test 字段未被访问或用于门控。",
        "",
        "| 模型 | 标签比例 | Val PR-AUC | 对照 PR-AUC | 差值 | Val FPR@95R | 对照 FPR | 差值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = comparisons.sort_values(["family", "variant", "train_fraction"])
    for row in ordered.itertuples():
        label = row.variant if row.family == "STATISTICAL" else row.family
        lines.append(
            f"| {label} | {row.train_fraction:.0%} | "
            f"{row.validation_average_precision:.2%} | "
            f"{row.baseline_validation_average_precision:.2%} | "
            f"{row.average_precision_delta:+.2%} | "
            f"{row.validation_fpr_at_target_95:.2%} | "
            f"{row.baseline_validation_fpr_at_target_95:.2%} | "
            f"{row.fpr_delta_at_target_95:+.2%} |"
        )
    lines.extend(["", "## 门控决定", ""])
    for family, decision in decisions.items():
        word = "扩展到五随机种子" if decision["expand_to_five_seeds"] else "不扩展"
        lines.append(f"- `{family}`：{word}。")
    lines.extend(
        [
            "",
            "门控阈值在实验运行前已写入实验记录；本报告不包含任何 test 数值。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    binary_run_dirs: list[Path],
    baseline_csv: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for run_dir in binary_run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing completed metrics: {metrics_path}")
        payload = _read_json(metrics_path)
        if int(payload["seed"]) != seed:
            raise ValueError(f"Unexpected seed in {metrics_path}: {payload['seed']}")
        rows.extend(validation_rows_from_binary_metrics(payload, run_dir))
    binary = pd.DataFrame(rows)
    baseline = baseline_validation_rows(pd.read_csv(baseline_csv), seed)
    comparisons = compare_validation_rows(binary, baseline)
    decisions = gate_decisions(comparisons)

    output_dir.mkdir(parents=True, exist_ok=True)
    binary.to_csv(output_dir / "binary_validation_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "validation_comparisons.csv", index=False)
    result = {
        "seed": seed,
        "selection_split": "validation",
        "test_fields_accessed": False,
        "test_metrics_used_for_gate": False,
        "gate": {
            "average_precision_improvement": AP_IMPROVEMENT_GATE,
            "fpr_reduction_at_0.95_recall": FPR_REDUCTION_GATE,
            "statistical_average_precision_margin": STATISTICAL_AP_MARGIN,
            "full_or_at_least_two_low_label_fractions": True,
        },
        "decisions": decisions,
        "binary_run_dirs": [str(path) for path in binary_run_dirs],
        "baseline_csv": str(baseline_csv),
    }
    atomic_write_json(output_dir / "gate_decision.json", result)
    (output_dir / "validation_gate.md").write_text(
        _markdown_report(comparisons, decisions, seed),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the pre-registered validation-only gate to battery binary runs."
    )
    parser.add_argument("--binary-run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("artifacts/battery_safety_thesis_v1/per_seed_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/battery_binary_analysis/pilot_seed42"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(
        binary_run_dirs=args.binary_run_dirs,
        baseline_csv=args.baseline_csv,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(f"Saved validation-only gate: {result['decisions']}")


if __name__ == "__main__":
    main()
