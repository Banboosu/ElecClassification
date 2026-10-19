"""Stdlib-only independent E06 audit of saved outputs. No model inference."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from audit_e05c_saved_outputs import average_precision, digest, metrics, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--startup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis
    frozen = read(root / "frozen_protocol.json")
    p = frozen["protocol"]
    if digest(args.startup) != digest(root / "frozen_protocol.json"):
        raise ValueError("Original pre-outcome freeze changed")
    if (p["seeds"] != [42, 44, 45, 46] or p["conditions"] != ["tcn_only", "tcn_moment", "tcn_random"]
        or p["train_fraction"] != 0.1 or p["test_evaluated"]):
        raise ValueError("Scope mismatch")
    for name, want in read(root / "output_manifest.json").items():
        if digest(root / name) != want:
            raise ValueError(f"Saved output changed: {name}")
    if read(root / "suite_status.json")["status"] != "completed":
        raise ValueError("Suite incomplete")
    summary = read(root / "summary.json")
    if summary["audit_status"] != "passed" or summary["test_evaluated"]:
        raise ValueError("Summary not audited/validation-only")
    raw = Path(p["dataset"])
    if digest(raw) != p["dataset_sha256"]:
        raise ValueError("Raw dataset changed")
    with raw.open(newline="", encoding="utf-8") as stream:
        truth = {row["charging_station_id"]: row["InsertedColumn"]
                 for row in csv.DictReader(stream, delimiter=" ")}
    parent_protocol = read(root / "source_snapshot" / p["parent_protocol"])
    maximum_error = maximum_ap_error = 0.0
    count = epochs = 0
    scores = {}

    def check_csv(file, ids, expected=None):
        nonlocal maximum_error, maximum_ap_error, count
        with file.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [r["sample_id"] for r in rows] != ids:
            raise ValueError("Saved prediction IDs/order mismatch")
        cm = [[0] * 3 for _ in range(3)]
        for row in rows:
            y, pred = int(row["y_true"]), int(row["y_pred"])
            if truth[row["sample_id"]] != row["y_true"]:
                raise ValueError("Ground truth differs from original raw CSV")
            values = [float(row[f"p{i}"]) for i in range(3)]
            if (not all(math.isfinite(x) and 0 <= x <= 1 for x in values)
                or abs(sum(values) - 1) > 2e-7 or values.index(max(values)) != pred):
                raise ValueError("Invalid probabilities or argmax")
            cm[y][pred] += 1
        calculated = metrics(cm)
        ap = average_precision(rows)
        if expected:
            if expected["confusion_matrix"] != cm:
                raise ValueError("Confusion matrix mismatch")
            for key, value in calculated.items():
                if key not in expected:
                    continue
                errors = ([abs(a - b) for a, b in zip(value, expected[key])]
                          if isinstance(value, list) else [abs(value - expected[key])])
                maximum_error = max(maximum_error, *errors)
            maximum_ap_error = max(maximum_ap_error, abs(ap - expected["battery_AP"]))
            maximum_error = max(maximum_error, abs(cm[2][2] / sum(cm[2]) - expected["battery_recall"]),
                                abs((cm[0][2] + cm[1][2]) / (sum(cm[0]) + sum(cm[1])) - expected["battery_FPR"]))
        count += len(rows)
        return calculated["macro_f1"]

    for seed in p["seeds"]:
        split_name = parent_protocol["split_manifests"][str(seed)]
        split_file = root / "input_snapshot" / split_name
        if digest(split_file) != frozen["file_sha256"][split_name]:
            raise ValueError("Original split snapshot changed")
        ids = read(split_file)["splits"]["validation"]
        directory = root / f"seed{seed}"
        provenance = read(directory / "feature_provenance.json")
        if (provenance["tcn_restore_max_probability_drift"] > 1e-5
            or provenance["tcn_restore_argmax_disagreements"] != 0
            or provenance["test_evaluated"]):
            raise ValueError("Parent restoration gate failed")
        original = check_csv(directory / "original_tcn_validation.csv", ids)
        original_row = next(row for row in summary["rows"]
                            if row["seed"] == seed and row["condition"] == "original_tcn")
        maximum_error = max(maximum_error, abs(original - original_row["metrics"]["macro_f1"]))
        scores[seed, "original_tcn"] = original
        hashes = []
        for condition in p["conditions"]:
            out = directory / condition
            result = read(out / "result.json")
            history = read(out / "history.json")
            if (result["seed"] != seed or result["condition"] != condition
                or result["protocol_sha256"] != frozen["protocol_sha256"] or result["test_evaluated"]
                or result["epochs"] != 100 or result["optimizer_steps"] != 100 * math.ceil(2456 / 128)):
                raise ValueError("Condition/budget/protocol mismatch")
            for name, expected in result["output_sha256"].items():
                if digest(out / name) != expected:
                    raise ValueError("Condition output hash mismatch")
            if [row["epoch"] for row in history] != list(range(101)):
                raise ValueError("Incomplete head history")
            if any(not math.isfinite(row["validation_macro_f1"]) for row in history):
                raise ValueError("Nonfinite validation history")
            if any(not math.isfinite(row["train_loss"]) for row in history[1:]):
                raise ValueError("Nonfinite training loss")
            selected = max(history, key=lambda row: row["validation_macro_f1"])
            if selected["epoch"] != result["selected_epoch"]:
                raise ValueError("Best checkpoint/first tie/epoch0 rule mismatch")
            maximum_error = max(maximum_error, abs(result["epoch0_metrics"]["macro_f1"] - original))
            scores[seed, condition] = check_csv(out / "validation_selected.csv", ids, result["selected_metrics"])
            check_csv(out / "validation_endpoint.csv", ids, result["endpoint_metrics"])
            maximum_error = max(maximum_error, abs(selected["validation_macro_f1"] - scores[seed, condition]),
                                abs(history[-1]["validation_macro_f1"] - result["endpoint_metrics"]["macro_f1"]))
            hashes.append(result["initial_head_state_sha256"])
            epochs += 100
        if len(set(hashes)) != 1:
            raise ValueError("Paired head initial state mismatch")
    for condition, row in summary["aggregate"].items():
        values = [scores[seed, condition] * 100 for seed in p["seeds"]]
        maximum_error = max(maximum_error, abs(statistics.mean(values) - row["macro_f1_mean_percent"]),
                            abs(statistics.stdev(values) - row["macro_f1_sd_percent"]))
    rules = []
    for reference, threshold in (("tcn_only", 0.5), ("original_tcn", 0.5), ("tcn_random", 0.25)):
        values = [(scores[s, "tcn_moment"] - scores[s, reference]) * 100 for s in p["seeds"]]
        row = summary["comparisons"][reference]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        half = 3.182446305284263 * sd / 2
        maximum_error = max(maximum_error, abs(mean - row["mean_pp"]),
                            abs(mean - half - row["descriptive_95pct_interval_pp"][0]),
                            abs(mean + half - row["descriptive_95pct_interval_pp"][1]),
                            *(abs(a - b) for a, b in zip(values, row["per_seed_pp"])))
        passed = mean >= threshold and sum(x > 0 for x in values) >= 3
        if row["passed"] != passed or row["positive_seeds"] != sum(x > 0 for x in values):
            raise ValueError("Fixed success rule mismatch")
        rules.append(passed)
    if (summary["all_success_rules_passed"] != all(rules) or epochs != 1200
        or count != 98280 or maximum_error > 1e-12 or maximum_ap_error > 1e-12):
        raise ValueError("Incomplete/inconsistent outputs")
    proof = {"status": "passed", "head_training_epochs_audited": epochs,
             "prediction_rows_audited": count, "maximum_independent_metric_error": maximum_error,
             "maximum_independent_AP_error": maximum_ap_error, "test_evaluated": False,
             "inference_performed": False, "pre_outcome_local_freeze_bytes_verified": True,
             "all_original_success_rules_passed": all(rules),
             "summary_sha256": digest(root / "summary.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
