"""Independent stdlib-only CSV/confusion/AP audit of completed E03T outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def average_precision(rows):
    ranked = sorted((float(r["probability_2"]), int(r["true_index"]) == 2) for r in rows)
    ranked.reverse()
    positives = sum(y for _, y in ranked)
    tp = seen = i = 0
    value = 0.0
    while i < len(ranked):
        j = i
        added = 0
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            added += ranked[j][1]
            j += 1
        tp += added
        seen += j - i
        value += added / positives * tp / seen
        i = j
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.analysis
    summary = json.loads((root / "summary.json").read_text())
    f = json.loads((root / "frozen_protocol.json").read_text())
    if summary["audit_status"] != "passed" or summary["conditions"] != 28:
        raise ValueError("Missing completed primary audit.")
    with (args.project / "data/raw/最新多.csv").open(newline="", encoding="utf-8-sig") as handle:
        labels = {
            r["charging_station_id"]: r["InsertedColumn"]
            for r in csv.DictReader(handle, delimiter=" ")
        }
    maximum_metric_error = maximum_ap_error = 0.0
    total = ap_conditions = 0
    for seed in f["protocol"]["seeds"]:
        parent = Path(f["selected_models"][str(seed)]["fusion_svm"]["directory"]).parent
        ids = json.loads((args.project / parent / "split_manifest.json").read_text())["splits"][
            "test"
        ]
        for method in f["protocol"]["methods"]:
            out = root / f"seed{seed}" / method
            with (out / "test_predictions.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if [r["sample_id"] for r in rows] != ids or len(set(ids)) != 7020:
                raise ValueError("Test ID/order mismatch.")
            matrix = [[0] * 3 for _ in range(3)]
            for row in rows:
                truth = int(row["true_index"])
                pred = int(row["predicted_index"])
                if str(truth) != labels[row["sample_id"]] or not 0 <= pred < 3:
                    raise ValueError("Original CSV truth or predicted class mismatch.")
                matrix[truth][pred] += 1
            m = json.loads((out / "metrics.json").read_text())["test_metrics"]
            if matrix != m["confusion_matrix"]:
                raise ValueError("Confusion matrix mismatch.")
            accuracy = sum(matrix[i][i] for i in range(3)) / len(rows)
            f1 = (
                sum(
                    2 * matrix[i][i] / (sum(matrix[i]) + sum(matrix[j][i] for j in range(3)))
                    for i in range(3)
                )
                / 3
            )
            error = max(abs(accuracy - m["accuracy"]), abs(f1 - m["macro_f1"]))
            maximum_metric_error = max(maximum_metric_error, error)
            if error > 1e-12:
                raise ValueError("Independent confusion-derived metric mismatch.")
            if "probability_2" in rows[0]:
                ap_conditions += 1
                error = abs(average_precision(rows) - m["battery_average_precision"])
                maximum_ap_error = max(maximum_ap_error, error)
                if error > 1e-12:
                    raise ValueError("Independent grouped-tie average precision mismatch.")
            total += len(rows)
    if total != 196560 or ap_conditions != 16:
        raise ValueError("Incomplete fixed conditions/AP coverage.")
    report = {
        "status": "passed",
        "audited_prediction_rows": total,
        "conditions": 28,
        "probability_conditions_with_independent_AP": ap_conditions,
        "maximum_confusion_derived_metric_abs_error": maximum_metric_error,
        "maximum_grouped_tie_AP_abs_error": maximum_ap_error,
        "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
        "method": "stdlib CSV parser, direct 3x3 confusion equations, grouped-tie AP; original raw CSV truth and split manifests",
        "inference_performed": False,
        "selection_performed": False,
    }
    (root / "independent_prediction_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
