"""Verify E07 fixed-reference copies and selected class metrics from saved CSVs only."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from audit_e05c_saved_outputs import average_precision, digest, metrics, read


def audit(base):
    root = base / "terminal_artifacts"
    frozen, summary = read(root / "frozen_protocol.json"), read(root / "summary.json")
    p = frozen["protocol"]
    manifest = read(base / "fixed_reference_manifest.json")
    copied = base / "fixed_references"
    actual = {str(f.relative_to(copied)) for f in copied.rglob("*") if f.is_file()}
    if actual != set(manifest["files"]) or len(actual) != 32:
        raise ValueError("Incomplete fixed reference set")
    for name, record in manifest["files"].items():
        file = copied / name
        if (file.stat().st_size != record["bytes"] or digest(file) != record["sha256"]
            or record["sha256"] != frozen["file_sha256"][name]):
            raise ValueError("Fixed reference hash/size mismatch")
    dataset = Path(p["dataset"])
    if digest(dataset) != p["dataset_sha256"]:
        raise ValueError("Raw dataset changed")
    with dataset.open(newline="", encoding="utf-8") as stream:
        truth = {r["charging_station_id"]: r["InsertedColumn"] for r in csv.DictReader(stream, delimiter=" ")}
    max_error = max_ap_error = 0.

    def check(file, ids, expected):
        nonlocal max_error, max_ap_error
        with file.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [r["sample_id"] for r in rows] != ids:
            raise ValueError("Reference IDs/order mismatch")
        ykey = "true_index" if "true_index" in rows[0] else "y_true"
        pkey = "predicted_index" if "predicted_index" in rows[0] else "y_pred"
        probability_keys = ([f"probability_{i}" for i in range(3)] if "probability_0" in rows[0]
                            else [f"p{i}" for i in range(3)] if "p0" in rows[0] else None)
        cm = [[0] * 3 for _ in range(3)]
        ap_rows = []
        for row in rows:
            if truth[row["sample_id"]] != row[ykey]:
                raise ValueError("Raw truth mismatch")
            y, prediction = int(row[ykey]), int(row[pkey])
            if y not in (0, 1, 2) or prediction not in (0, 1, 2):
                raise ValueError("Invalid class")
            cm[y][prediction] += 1
            if probability_keys:
                prob = [float(row[k]) for k in probability_keys]
                if (not all(math.isfinite(v) and 0 <= v <= 1 for v in prob)
                    or abs(sum(prob) - 1) > 1e-5 or prob.index(max(prob)) != prediction):
                    raise ValueError("Invalid reference probabilities")
                ap_rows.append({"y_true": str(y), "p2": str(prob[2])})
        if cm != expected["confusion_matrix"]:
            raise ValueError("Reference confusion mismatch")
        calculated = {**metrics(cm), "battery_recall": cm[2][2] / sum(cm[2]),
                      "battery_FPR": (cm[0][2] + cm[1][2]) / (sum(cm[0]) + sum(cm[1]))}
        for key, value in calculated.items():
            if isinstance(value, list):
                max_error = max(max_error, *(abs(a - b) for a, b in zip(value, expected[key])))
            else:
                max_error = max(max_error, abs(value - expected[key]))
        if ap_rows:
            max_ap_error = max(max_ap_error, abs(average_precision(ap_rows) - expected["battery_AP"]))
        return len(rows)

    count = selected_count = 0
    for seed in p["seeds"]:
        ids = read(root / "prepared" / f"seed{seed}/split_manifest.json")["splits"]["validation"]
        for name, filename in manifest["paths"][str(seed)].items():
            expected = frozen["references"][str(seed)][name]
            if expected != summary["references"][str(seed)][name]:
                raise ValueError("Frozen reference summary changed")
            count += check(copied / filename, ids, expected)
        for c in ("frozen_fusion", "lora8_fusion"):
            row = next(r for r in summary["per_condition"] if r["seed"] == seed and r["condition"] == c)
            selected_count += check(root / f"seed{seed}" / c / "validation_predictions.csv", ids, row["selected"])
    if count != 112320 or selected_count != 28080 or max_error > 1e-12 or max_ap_error > 1e-12:
        raise ValueError("Incomplete/inconsistent reference or selected-class audit")
    return {"status": "passed", "fixed_reference_files": 32, "fixed_reference_bytes": manifest["total_bytes"],
            "fixed_reference_prediction_rows": count, "selected_class_metric_rows_rechecked": selected_count,
            "reference_sha256_mismatches": 0, "maximum_independent_metric_error": max_error,
            "maximum_independent_AP_error": max_ap_error, "test_evaluated": False, "inference_performed": False,
            "summary_sha256": digest(root / "summary.json"), "audit_source_sha256": digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserved-root", type=Path, required=True)
    args = parser.parse_args()
    proof = audit(args.preserved_root)
    (args.preserved_root / "independent_reference_audit.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
