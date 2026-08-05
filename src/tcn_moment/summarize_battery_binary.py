from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcn_moment.evaluate_battery_safety import _aggregate_rows, _flat_rows
from tcn_moment.io_utils import atomic_write_json


T_CRITICAL_95_DF4 = 2.7764451051977987
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_FRACTIONS = (0.01, 0.05, 0.1, 1.0)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def results_from_binary_metrics(
    payload: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    common = {
        "run_name": payload["run_name"],
        "seed": int(payload["seed"]),
        "run_dir": str(run_dir),
    }
    model = str(payload["model"])
    if model == "BATTERY_BINARY_STATISTICAL":
        fraction = float(payload["data"]["train_subset"]["requested_fraction"])
        return [
            {
                **common,
                "model": f"BATTERY_BINARY_{name.upper()}",
                "variant": name,
                "train_fraction": fraction,
                "selected_positive_weight": float(values["selected_positive_weight"]),
                "selected_c": None,
                "ranking": values["ranking"],
                "operating_points": values["operating_points"],
            }
            for name, values in payload["results"].items()
        ]
    if model == "BATTERY_BINARY_MOMENT_RBF_SVM":
        results = []
        for values in payload["fractions"]:
            class_weight = values["best_params"]["class_weight"]
            results.append(
                {
                    **common,
                    "model": model,
                    "variant": "binary_moment_rbf_svm",
                    "train_fraction": float(values["train_fraction"]),
                    "selected_positive_weight": float(
                        class_weight.get("1", class_weight.get(1))
                    ),
                    "selected_c": float(values["best_params"]["C"]),
                    "ranking": values["ranking"],
                    "operating_points": values["operating_points"],
                }
            )
        return results
    raise ValueError(f"Formal summary does not accept model {model!r} from {run_dir}.")


def _attach_selection_metadata(
    rows: pd.DataFrame,
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    keys = ["run_name", "model", "variant", "seed", "train_fraction"]
    metadata = pd.DataFrame(
        [
            {
                **{key: result[key] for key in keys},
                "run_dir": result["run_dir"],
                "selected_positive_weight": result["selected_positive_weight"],
                "selected_c": result["selected_c"],
            }
            for result in results
        ]
    )
    return rows.merge(metadata, on=keys, how="left", validate="many_to_one")


def validate_five_seed_completeness(results: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(
        [
            {
                "model": result["model"],
                "train_fraction": result["train_fraction"],
                "seed": result["seed"],
            }
            for result in results
        ]
    ).drop_duplicates()
    expected = set(EXPECTED_SEEDS)
    problems = []
    for (model, fraction), group in frame.groupby(["model", "train_fraction"]):
        seeds = set(group["seed"].astype(int))
        if seeds != expected:
            problems.append(f"{model} fraction={fraction:g}: seeds={sorted(seeds)}")
    expected_groups = {
        (model, fraction)
        for model in (
            "BATTERY_BINARY_LOGISTIC_REGRESSION",
            "BATTERY_BINARY_RANDOM_FOREST",
            "BATTERY_BINARY_MOMENT_RBF_SVM",
        )
        for fraction in EXPECTED_FRACTIONS
    }
    actual_groups = {
        (str(row.model), float(row.train_fraction))
        for row in frame[["model", "train_fraction"]].drop_duplicates().itertuples()
    }
    missing = expected_groups - actual_groups
    if missing:
        problems.append(f"missing model/fraction groups: {sorted(missing)}")
    if problems:
        raise ValueError("Incomplete five-seed results: " + "; ".join(problems))


def validate_gate(gate: dict[str, Any]) -> None:
    decisions = gate["decisions"]
    required_true = (
        "MOMENT_RBF_SVM",
        "STATISTICAL::logistic_regression",
        "STATISTICAL::random_forest",
    )
    for name in required_true:
        if not decisions[name]["expand_to_five_seeds"]:
            raise ValueError(f"Formal summary was not authorized by validation gate: {name}")
    if decisions["TCN"]["expand_to_five_seeds"]:
        raise ValueError("Gate unexpectedly authorized TCN; update the formal protocol first.")
    if gate.get("test_metrics_used_for_gate") is not False:
        raise ValueError("Gate metadata does not certify test isolation.")


def _baseline_model_for_moment(fraction: float) -> str:
    return "MOMENT_RBF_SVM" if np.isclose(fraction, 1.0) else "MOMENT_RBF_SVM_FEW_SHOT"


def paired_comparisons(binary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "test_battery_recall",
        "test_battery_precision",
        "test_false_positive_rate",
        "test_f2",
        "test_average_precision",
        "test_roc_auc",
        "test_false_negatives",
        "test_false_positives",
    )
    rows: list[dict[str, Any]] = []
    groups = binary.groupby(["model", "train_fraction", "operating_point"])
    for (model, fraction, operating_point), binary_group in groups:
        fraction = float(fraction)
        if model == "BATTERY_BINARY_MOMENT_RBF_SVM":
            references = (_baseline_model_for_moment(fraction),)
        else:
            references = ("TCN", _baseline_model_for_moment(fraction))
        for reference_model in references:
            reference = baseline.loc[
                (baseline["model"] == reference_model)
                & np.isclose(baseline["train_fraction"].astype(float), fraction)
                & (baseline["operating_point"] == operating_point)
            ]
            merged = binary_group.merge(
                reference,
                on="seed",
                suffixes=("_binary", "_baseline"),
                validate="one_to_one",
            )
            if set(merged["seed"].astype(int)) != set(EXPECTED_SEEDS):
                raise ValueError(
                    f"Incomplete paired comparison: {model} vs {reference_model}, "
                    f"fraction={fraction:g}, operating_point={operating_point}"
                )
            for metric in metrics:
                differences = (
                    merged[f"{metric}_binary"].astype(float)
                    - merged[f"{metric}_baseline"].astype(float)
                ).to_numpy()
                mean = float(differences.mean())
                std = float(differences.std(ddof=1))
                half_width = T_CRITICAL_95_DF4 * std / np.sqrt(len(differences))
                rows.append(
                    {
                        "binary_model": model,
                        "baseline_model": reference_model,
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
    return pd.DataFrame(rows)


def _markdown_summary(aggregate: pd.DataFrame) -> str:
    selected = aggregate.loc[
        aggregate["operating_point"].isin(("max_f2", "recall_0.95"))
    ].sort_values(["model", "train_fraction", "operating_point"])
    lines = [
        "# 电池异常专用二分类五随机种子汇总",
        "",
        "| 模型 | 标签比例 | 运行点 | Test PR-AUC | Recall | Precision | FPR | F2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples():
        lines.append(
            f"| {row.model} | {row.train_fraction:.0%} | {row.operating_point} | "
            f"{row.test_average_precision_mean:.2%} ± {row.test_average_precision_std:.2%} | "
            f"{row.test_battery_recall_mean:.2%} ± {row.test_battery_recall_std:.2%} | "
            f"{row.test_battery_precision_mean:.2%} ± {row.test_battery_precision_std:.2%} | "
            f"{row.test_false_positive_rate_mean:.2%} ± "
            f"{row.test_false_positive_rate_std:.2%} | "
            f"{row.test_f2_mean:.2%} ± {row.test_f2_std:.2%} |"
        )
    lines.extend(
        [
            "",
            "所有阈值均在各 seed 的 validation split 上选择，然后原样应用于 test。",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(
    *,
    run_dirs: list[Path],
    baseline_csv: Path,
    gate_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    gate = _read_json(gate_path)
    validate_gate(gate)
    results = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing completed metrics: {metrics_path}")
        results.extend(results_from_binary_metrics(_read_json(metrics_path), run_dir))
    validate_five_seed_completeness(results)

    per_seed = _attach_selection_metadata(_flat_rows(results), results)
    aggregate = _aggregate_rows(per_seed)
    seed_counts = (
        per_seed.groupby(["model", "train_fraction", "operating_point"])["seed"]
        .nunique()
        .rename("n_seeds")
        .reset_index()
    )
    aggregate = aggregate.merge(
        seed_counts,
        on=["model", "train_fraction", "operating_point"],
        validate="one_to_one",
    )
    baseline = pd.read_csv(baseline_csv)
    paired = paired_comparisons(per_seed, baseline)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_comparisons.csv", index=False)
    (output_dir / "summary.md").write_text(
        _markdown_summary(aggregate),
        encoding="utf-8",
    )
    metadata = {
        "protocol_version": 1,
        "source_run_count": len(run_dirs),
        "model_fraction_seed_results": len(results),
        "per_seed_operating_point_rows": len(per_seed),
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_fractions": list(EXPECTED_FRACTIONS),
        "gate_decision": str(gate_path),
        "baseline_csv": str(baseline_csv),
        "outputs": {
            "per_seed_metrics": str(output_dir / "per_seed_metrics.csv"),
            "aggregate_metrics": str(output_dir / "aggregate_metrics.csv"),
            "paired_comparisons": str(output_dir / "paired_comparisons.csv"),
            "markdown": str(output_dir / "summary.md"),
        },
    }
    atomic_write_json(output_dir / "metrics.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize gate-approved five-seed battery binary experiments."
    )
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("artifacts/battery_safety_thesis_v1/per_seed_metrics.csv"),
    )
    parser.add_argument(
        "--gate-decision",
        type=Path,
        default=Path("artifacts/battery_binary_analysis/pilot_seed42/gate_decision.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/battery_binary_analysis/formal_five_seed"),
    )
    args = parser.parse_args()
    metadata = summarize(
        run_dirs=args.run_dirs,
        baseline_csv=args.baseline_csv,
        gate_path=args.gate_decision,
        output_dir=args.output_dir,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
