"""Audit and summarize E01 validation results. Never load test predictions/labels."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import t

from tcn_moment.data import train_subset_sha256
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.train_feature_fusion import BASELINE_NAMES, MODEL_NAMES, validation_gate

DATASET_SHA256 = "5615a96a7894caed5d14463c77167af8098bdc1e1ebf32a33a89a12c3c5cf6e6"
FRACTIONS = (0.01, 0.05, 0.1, 0.2, 0.4)
SEEDS = (42, 43, 44, 45, 46)


def metrics_from_confusion(matrix: list[list[int]]) -> dict[str, float]:
    size = len(matrix)
    supports = [sum(row) for row in matrix]
    total = sum(supports)
    if total <= 0 or any(support == 0 for support in supports):
        raise ValueError("Validation confusion matrix must contain all classes.")
    precisions, recalls, f1s = [], [], []
    for label in range(size):
        tp = matrix[label][label]
        predicted = sum(row[label] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / supports[label]
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    result = {
        "accuracy": sum(matrix[i][i] for i in range(size)) / total,
        "balanced_accuracy": statistics.mean(recalls),
    }
    for average in ("macro", "weighted"):
        for name, values in (("precision", precisions), ("recall", recalls), ("f1", f1s)):
            result[f"{average}_{name}"] = (
                statistics.mean(values)
                if average == "macro"
                else sum(value * support for value, support in zip(values, supports, strict=True))
                / total
            )
    return result


def paired_summary(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("Paired uncertainty requires at least two seed pairs.")
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    half_width = float(t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values))
    return {
        "n": len(values),
        "difference_pp": mean,
        "ci95_lower_pp": mean - half_width,
        "ci95_upper_pp": mean + half_width,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_run(run: Path, cache_root: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status = json.loads((run / "status.json").read_text())
    plan = json.loads((run / "experiment_plan.json").read_text())
    manifest = json.loads((run / "split_manifest.json").read_text())
    payload = json.loads((run / "metrics.json").read_text())
    seed = plan["seed"]
    if status["status"] != "completed" or payload["test_evaluated"] or plan["evaluate_test"]:
        raise ValueError(f"Incomplete or non-validation-only run: {run}")
    if seed not in SEEDS or plan["fractions"] != list(FRACTIONS):
        raise ValueError(f"Unexpected seed/fractions: {run}")
    expected_models = list(MODEL_NAMES) if seed == 42 else list(BASELINE_NAMES)
    if plan["models"] != expected_models:
        raise ValueError(f"Model expansion differs from the frozen gate: {run}")
    if manifest["protocol"]["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("Dataset hash mismatch.")
    split_ids = {name: np.asarray(ids, dtype=str) for name, ids in manifest["splits"].items()}
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    if any(len(split_sets[name]) != len(ids) for name, ids in split_ids.items()):
        raise ValueError("Duplicate IDs in split manifest.")
    if (
        split_sets["train"] & split_sets["validation"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["validation"] & split_sets["test"]
    ):
        raise ValueError("Overlapping sample IDs in data split.")
    if {name: len(ids) for name, ids in split_ids.items()} != {
        "train": 24569,
        "validation": 3510,
        "test": 7020,
    }:
        raise ValueError("Unexpected split sizes.")
    for name, digest in plan["source_hashes"].items():
        if sha256_file(run / "source" / name) != digest:
            raise ValueError(f"Source snapshot checksum mismatch: {run}/{name}")
    if list(run.rglob("test_predictions.csv")):
        raise ValueError("Validation run contains test predictions.")
    previous_ids: set[str] = set()
    subset_hashes = {}
    for fraction in FRACTIONS:
        tag = f"{fraction * 100:g}".replace(".", "p")
        ids = np.load(run / f"train_ids_{tag}.npy", allow_pickle=False).astype(str)
        record = json.loads((run / f"train_subset_{tag}.json").read_text())
        selected = set(ids)
        expected_counts = [
            max(1, math.floor(manifest["class_counts"]["train"][str(k)] * fraction))
            for k in range(3)
        ]
        if (
            len(selected) != len(ids)
            or not selected <= split_sets["train"]
            or not previous_ids <= selected
            or len(ids) != sum(expected_counts)
            or record["class_counts"] != expected_counts
        ):
            raise ValueError(f"Invalid/non-nested training subset: {run}, {fraction}")
        subset_hashes[fraction] = train_subset_sha256(ids)
        if record["sample_ids_sha256"] != subset_hashes[fraction]:
            raise ValueError("Training subset hash mismatch.")
        folds = np.load(run / f"cv_folds_{tag}.npy", allow_pickle=False)
        counts = np.bincount(folds, minlength=5)
        if len(folds) != len(ids) or set(folds) != set(range(5)) or counts.max() - counts.min() > 1:
            raise ValueError("Invalid CV fold assignments.")
        previous_ids = selected
    if cache_root is not None:
        cache = cache_root / f"seed{seed}"
        cache_manifest = json.loads((cache / "manifest.json").read_text())
        if set(cache_manifest["splits"]) != {"train", "validation"}:
            raise ValueError("Feature cache contains unexpected/test embeddings.")
        if cache_manifest["signature"]["source_hashes"] != plan["source_hashes"]:
            raise ValueError("Feature/source protocol mismatch.")
        if cache_manifest["signature"]["split_sha256"] != sha256_file(run / "split_manifest.json"):
            raise ValueError("Cached feature split mismatch.")
        for split in ("train", "validation"):
            path = cache / f"{split}.npz"
            if sha256_file(path) != cache_manifest["splits"][split]["sha256"]:
                raise ValueError("Feature cache corruption.")
            with np.load(path, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["sample_ids"], split_ids[split])
                if (
                    arrays["features"].shape != (len(split_ids[split]), 1024)
                    or not np.isfinite(arrays["features"]).all()
                ):
                    raise ValueError("Invalid cached feature values/shape.")
    rows = payload["results"]
    if {(r["model"], r["train_fraction"]) for r in rows} != {
        (name, fraction) for name in expected_models for fraction in FRACTIONS
    } or len(rows) != len(expected_models) * len(FRACTIONS):
        raise ValueError("Missing or duplicate experimental conditions.")
    labels_by_id: dict[str, int] = {}
    prediction_count, maximum_error = 0, 0.0
    for row in rows:
        if (
            row["seed"] != seed
            or row["test_evaluated"]
            or "test_metrics" in row
            or row["dataset_sha256"] != DATASET_SHA256
            or row["train_ids_sha256"] != subset_hashes[row["train_fraction"]]
        ):
            raise ValueError("Condition protocol mismatch/test metrics found.")
        tag = f"{row['train_fraction'] * 100:g}".replace(".", "p")
        condition = run / f"{row['model']}_fraction_{tag}"
        if json.loads((condition / "metrics.json").read_text()) != row:
            raise ValueError("Condition/root metrics disagree.")
        if not (condition / "classifier.joblib").is_file():
            raise ValueError("Missing trained classifier.")
        # Only load this project's own trusted training artifacts.
        fitted = joblib.load(condition / "classifier.joblib")
        if (
            fitted.classes_.tolist() != [0, 1, 2]
            or fitted.n_features_in_ != row["feature_dimension"]
        ):
            raise ValueError("Saved classifier class/feature metadata mismatch.")
        parameters = fitted.get_params(deep=True)
        if any(parameters[key] != value for key, value in row["best_params"].items()):
            raise ValueError("Saved classifier parameters differ from selected CV parameters.")
        del fitted
        with (condition / "validation_predictions.csv").open(newline="") as handle:
            predictions = list(csv.DictReader(handle))
        if [item["sample_id"] for item in predictions] != split_ids["validation"].tolist():
            raise ValueError("Validation prediction IDs/order mismatch.")
        confusion = [[0] * 3 for _ in range(3)]
        for item in predictions:
            true, predicted = int(item["true_index"]), int(item["predicted_index"])
            if (
                true not in range(3)
                or predicted not in range(3)
                or item["true_label"] != str(true)
                or item["predicted_label"] != str(predicted)
            ):
                raise ValueError("Invalid class encoding.")
            sid = item["sample_id"]
            if labels_by_id.setdefault(sid, true) != true:
                raise ValueError("Methods disagree on validation ground truth.")
            confusion[true][predicted] += 1
        if confusion != row["validation_metrics"]["confusion_matrix"]:
            raise ValueError("Saved confusion matrix does not match predictions.")
        for metric, actual in metrics_from_confusion(confusion).items():
            error = abs(actual - row["validation_metrics"][metric])
            maximum_error = max(maximum_error, error)
            if error > 1e-12:
                raise ValueError(f"Saved metric does not match predictions: {metric}")
        with (condition / "cv_results.csv").open(newline="") as handle:
            cv_rows = list(csv.DictReader(handle))
        selected = [
            candidate
            for candidate in cv_rows
            if ast.literal_eval(candidate["params"]) == row["best_params"]
        ]
        if (
            len(selected) != 1
            or abs(float(selected[0]["mean_test_score"]) - row["cv_score"]) > 1e-12
        ):
            raise ValueError("Selected parameters do not match their recorded CV score.")
        expected_candidates = (
            9 if row["model"] == "moment_paper_svm" else (4 if row["model"] == "stats_rf" else 15)
        )
        if len(cv_rows) != expected_candidates:
            raise ValueError("CV candidate count changed.")
        for candidate in cv_rows:
            scores = [float(candidate[f"split{fold}_test_score"]) for fold in range(5)]
            if not all(math.isfinite(value) for value in scores):
                raise ValueError("Non-finite CV score.")
            if abs(statistics.mean(scores) - float(candidate["mean_test_score"])) > 1e-12:
                raise ValueError("CV mean does not match its five fold scores.")
        if (
            not math.isfinite(row["cv_score"])
            or not math.isfinite(row["fit_seconds"])
            or abs(max(float(item["mean_test_score"]) for item in cv_rows) - row["cv_score"])
            > 1e-12
        ):
            raise ValueError("Selected CV score differs from saved CV grid.")
        prediction_count += len(predictions)
    return rows, {
        "seed": seed,
        "run": str(run),
        "conditions": len(rows),
        "validation_predictions": prediction_count,
        "max_metric_recomputation_error": maximum_error,
        "duration_seconds": status["duration_seconds"],
        "status": "passed",
    }


def analyze(args: argparse.Namespace) -> None:
    root = Path(args.run_root)
    runs = sorted(root.glob(f"{args.suite}_*seed*"))
    if len(runs) != 5:
        raise ValueError("Expected exactly five E01 runs.")
    all_rows, checks = [], []
    for run in runs:
        rows, audit = audit_run(run, Path(args.cache_root) if args.cache_root else None)
        all_rows.extend(rows)
        checks.append(audit)
    if {check["seed"] for check in checks} != set(SEEDS):
        raise ValueError("Missing or duplicate seeds.")
    pilot = [row for row in all_rows if row["seed"] == 42]
    gate = validation_gate(pilot)
    plan = json.loads(Path(args.expansion_plan).read_text())
    if plan["test_evaluated"] or any(
        plan[key] != gate[key] for key in ("candidates", "models_to_expand", "rule")
    ):
        raise ValueError("Frozen expansion plan does not match recomputed validation gate.")
    aggregates = []
    lookup = {(row["model"], row["train_fraction"], row["seed"]): row for row in all_rows}
    for fraction in FRACTIONS:
        for model in BASELINE_NAMES:
            conditions = [lookup[(model, fraction, seed)] for seed in SEEDS]
            record = {"model": model, "train_fraction": fraction, "n_seeds": len(conditions)}
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                values = [row["validation_metrics"][metric] * 100 for row in conditions]
                record[metric + "_mean_percent"] = statistics.mean(values)
                record[metric + "_sample_std_pp"] = statistics.stdev(values)
            aggregates.append(record)
    pairs = []
    for fraction in FRACTIONS:
        for first, second in (
            ("stats_svm", "moment_paper_svm"),
            ("stats_rf", "moment_paper_svm"),
            ("moment_tuned_svm", "moment_paper_svm"),
        ):
            deltas = [
                (
                    lookup[(first, fraction, seed)]["validation_metrics"]["macro_f1"]
                    - lookup[(second, fraction, seed)]["validation_metrics"]["macro_f1"]
                )
                * 100
                for seed in SEEDS
            ]
            pairs.append(
                {
                    "first": first,
                    "second": second,
                    "train_fraction": fraction,
                    **paired_summary(deltas),
                }
            )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "aggregate_validation_metrics.csv", aggregates)
    write_csv(output / "paired_validation_comparisons.csv", pairs)
    write_csv(
        output / "per_seed_validation_metrics.csv",
        [
            {
                "seed": row["seed"],
                "model": row["model"],
                "train_fraction": row["train_fraction"],
                "train_samples": row["train_samples"],
                "accuracy": row["validation_metrics"]["accuracy"],
                "macro_f1": row["validation_metrics"]["macro_f1"],
                "fit_seconds": row["fit_seconds"],
            }
            for row in all_rows
        ],
    )
    summary = {
        "status": "passed",
        "recorded_at": datetime.now(UTC).isoformat(),
        "test_evaluated": False,
        "total_conditions": len(all_rows),
        "run_audits": checks,
        "aggregates": aggregates,
        "paired_comparisons": pairs,
        "validation_gate": gate,
        "pilot_results": pilot,
        "limitations": [
            "Validation only, not final test results.",
            "Paired seed intervals are descriptive: random splits overlap; n=5; no multiplicity correction.",
            "Fusion conditions have one pilot seed only; do not compare them as five-seed results.",
            "Pilot timing mixes slow/fast CV and interrupted execution; do not pool it for speed claims.",
        ],
    }
    atomic_write_json(output / "summary.json", summary)
    table = [
        "# E01 validation audit",
        "",
        f"Audit: PASSED; {len(all_rows)} conditions; test not evaluated.",
        "",
        "Validation Macro-F1 (%), mean ± sample SD over five seeds:",
        "",
        "| Training labels | Stats RF | Stats SVM | MOMENT paper SVM | MOMENT tuned SVM |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fraction in FRACTIONS:
        values = [
            next(
                row
                for row in aggregates
                if row["model"] == model and row["train_fraction"] == fraction
            )
            for model in BASELINE_NAMES
        ]
        table.append(
            f"| {fraction:.0%} | "
            + " | ".join(
                f"{row['macro_f1_mean_percent']:.2f} ± {row['macro_f1_sample_std_pp']:.2f}"
                for row in values
            )
            + " |"
        )
    table.extend(
        [
            "",
            "Both fusion candidates failed the preregistered low-label validation expansion gate.",
            "No test evaluation was performed. See summary.json for audit checks and limitations.",
        ]
    )
    (output / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="artifacts/improvements/e01")
    parser.add_argument("--suite", default="e01_20261017_v2")
    parser.add_argument("--cache-root", default="artifacts/improvements/e01_20261017_v2_features")
    parser.add_argument(
        "--expansion-plan", default="artifacts/improvements/e01_20261017_v2_expansion_plan.json"
    )
    parser.add_argument("--output-dir", default="artifacts/analysis/e01_validation_20261017")
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
