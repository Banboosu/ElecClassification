"""Independent stdlib-only E07 saved-output audit; no training or inference."""
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
    if digest(root / "frozen_protocol.json") != digest(args.startup):
        raise ValueError("Original pre-outcome freeze differs")
    if (p["seeds"] != [42, 44, 45, 46] or p["test_evaluated"]
        or p["maximum_training_conditions"] != 8 or p["maximum_additional_epochs_total"] != 240):
        raise ValueError("Frozen compact scope mismatch")
    if read(root / "suite_status.json")["status"] != "completed":
        raise ValueError("Suite incomplete")
    for name, expected in read(root / "output_manifest.json").items():
        if digest(root / name) != expected:
            raise ValueError(f"Saved output changed: {name}")
    summary = read(root / "summary.json")
    if summary["test_evaluated"] or summary["audit_status"] != "passed":
        raise ValueError("Incomplete summary")
    raw = Path(p["dataset"])
    if digest(raw) != p["dataset_sha256"]:
        raise ValueError("Raw dataset changed")
    with raw.open(newline="", encoding="utf-8") as stream:
        truth = {r["charging_station_id"]: r["InsertedColumn"] for r in csv.DictReader(stream, delimiter=" ")}
    tcn = read(root / "source_snapshot" / p["tcn_protocol"])
    maximum_error = maximum_ap_error = 0.
    count = epochs = 0
    scores = {}
    for seed in p["seeds"]:
        split_name = tcn["split_manifests"][str(seed)]
        # E03E references use the same verified original E01 split files.
        split = root / "input_snapshot" / split_name
        if digest(split) != frozen["file_sha256"][split_name]:
            raise ValueError("Original split snapshot differs")
        ids = read(split)["splits"]["validation"]
        gate = read(root / "technical_gates" / f"seed{seed}/result.json")
        if gate["status"] != "passed" or gate["within_pair_probability_max_error"] != 0:
            raise ValueError("Technical gate failed")
        initial_rows = []
        head_hashes = []
        for condition in ("frozen_fusion", "lora8_fusion"):
            directory = root / f"seed{seed}" / condition
            payload = read(directory / "metrics.json")
            history = read(directory / "history.json")
            if (payload["seed"] != seed or payload["condition"]["name"] != condition
                or payload["protocol_sha256"] != frozen["protocol_sha256"] or payload["test_evaluated"]
                or not payload["all_frozen_parameters_unchanged"]
                or read(directory / "status.json")["status"] != "completed"):
                raise ValueError("Condition provenance mismatch")
            for filename, key in (("model_best.pt", "model_best_sha256"), ("checkpoint_last.pt", "checkpoint_last_sha256")):
                if digest(directory / filename) != payload[key]:
                    raise ValueError("Learned-state hash mismatch")
            if ([r["epoch"] for r in history] != list(range(1, 31))
                or any(r["grad_scaler_scale"] != 1024 or r["skipped_optimizer_steps"] or r["early_stop"] for r in history)
                or any(not all(math.isfinite(r[k]) for k in ("training_loss", "validation_loss", "validation_macro_f1")) for r in history)):
                raise ValueError("History/budget/precision mismatch")
            if condition == "lora8_fusion":
                if payload["parameters"]["adapter_parameters"] != 786432 or any(not r["adapter_B_gradient_norm_first_window"] > 0 for r in history):
                    raise ValueError("Adapter gradient/count invalid")
            elif payload["parameters"]["adapter_parameters"] or any(r["adapter_B_gradient_norm_first_window"] is not None for r in history):
                raise ValueError("Frozen adapter role invalid")
            head_hashes.append(payload["initial_head_sha256"])
            initial = read(directory / "initial_metrics.json")
            values = [initial["macro_f1"], *[r["validation_macro_f1"] for r in history]]
            best, best_epoch = values[0], 0
            for i, value in enumerate(values[1:], 1):
                if value > best + 1e-12:
                    best, best_epoch = value, i
            if best_epoch != payload["best_epoch"] or abs(best - payload["validation_metrics"]["macro_f1"]) > 1e-12:
                raise ValueError("Epoch0-inclusive selection mismatch")
            for filename, expected in (("initial_predictions.csv", initial), ("validation_predictions.csv", payload["validation_metrics"]),
                                       ("endpoint_predictions.csv", payload["endpoint_validation_metrics"])):
                with (directory / filename).open(newline="") as stream:
                    raw_rows = list(csv.DictReader(stream))
                if [r["sample_id"] for r in raw_rows] != ids or any(truth[r["sample_id"]] != r["true_index"] for r in raw_rows):
                    raise ValueError("Saved ordered IDs or raw truth differ")
                cm = [[0] * 3 for _ in range(3)]
                rows = []
                for r in raw_rows:
                    y, pred = int(r["true_index"]), int(r["predicted_index"])
                    prob = [float(r[f"probability_{i}"]) for i in range(3)]
                    if (not all(math.isfinite(x) and 0 <= x <= 1 for x in prob)
                        or abs(sum(prob) - 1) > 1e-5 or prob.index(max(prob)) != pred):
                        raise ValueError("Invalid probabilities")
                    cm[y][pred] += 1
                    rows.append({"y_true": str(y), "p2": str(prob[2])})
                if cm != expected["confusion_matrix"]:
                    raise ValueError("Confusion matrix mismatch")
                calculated = metrics(cm)
                for key, value in calculated.items():
                    if key in expected and not isinstance(value, list):
                        maximum_error = max(maximum_error, abs(value - expected[key]))
                summary_row = next(r for r in summary["per_condition"] if r["seed"] == seed and r["condition"] == condition)
                if filename == "validation_predictions.csv":
                    scores[seed, condition] = calculated["macro_f1"]
                    selected = summary_row["selected"]
                    maximum_error = max(maximum_error, abs(best - calculated["macro_f1"]),
                                        abs(selected["macro_f1"] - calculated["macro_f1"]),
                                        abs(cm[2][2] / sum(cm[2]) - selected["battery_recall"]),
                                        abs((cm[0][2] + cm[1][2]) / (sum(cm[0]) + sum(cm[1])) - selected["battery_FPR"]))
                    maximum_ap_error = max(maximum_ap_error, abs(average_precision(rows) - selected["battery_AP"]))
                if filename == "initial_predictions.csv":
                    initial_rows.append(raw_rows)
                if filename == "endpoint_predictions.csv":
                    maximum_error = max(maximum_error, abs(values[-1] - calculated["macro_f1"]))
                count += len(raw_rows)
            epochs += len(history)
        if len(set(head_hashes)) != 1 or initial_rows[0] != initial_rows[1]:
            raise ValueError("Paired initial head/probabilities mismatch")
    for condition, row in summary["aggregate"].items():
        values = [scores[seed, condition] * 100 for seed in p["seeds"]]
        maximum_error = max(maximum_error, abs(statistics.mean(values) - row["macro_f1_mean_percent"]),
                            abs(statistics.stdev(values) - row["macro_f1_sample_sd_percent"]))
    comparisons = {"continued_frozen_rule": [], "strongest_nonadapted_rule": []}
    for seed in p["seeds"]:
        references = frozen["references"][str(seed)]
        strongest = max(scores[seed, "frozen_fusion"], *(references[name]["macro_f1"] for name in p["success_rules"]["strongest_nonadapted_set"] if name != "continued_frozen_fusion"))
        comparisons["continued_frozen_rule"].append(100 * (scores[seed, "lora8_fusion"] - scores[seed, "frozen_fusion"]))
        comparisons["strongest_nonadapted_rule"].append(100 * (scores[seed, "lora8_fusion"] - strongest))
    rules = []
    for key, values in comparisons.items():
        row = summary[key]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        half = 3.182446305284263 * sd / 2
        maximum_error = max(maximum_error, abs(mean - row["mean_pp"]),
                            abs(mean - half - row["descriptive_ci95_pp"][0]), abs(mean + half - row["descriptive_ci95_pp"][1]))
        passed = mean >= .5 and sum(v > 0 for v in values) >= 3
        if row["rule_passed"] != passed or row["positive_seeds"] != sum(v > 0 for v in values):
            raise ValueError("Fixed gate mismatch")
        rules.append(passed)
    if (epochs != 240 or count != 84240 or maximum_error > 1e-12 or maximum_ap_error > 1e-12
        or summary["replication_rule_passed"] != all(rules)):
        raise ValueError("Incomplete/inconsistent saved results")
    proof = {"status": "passed", "epochs_audited": epochs, "prediction_rows_audited": count,
             "maximum_independent_metric_error": maximum_error, "maximum_independent_AP_error": maximum_ap_error,
             "test_evaluated": False, "inference_performed": False, "original_local_freeze_bytes_verified": True,
             "replication_rule_passed": all(rules), "summary_sha256": digest(root / "summary.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
