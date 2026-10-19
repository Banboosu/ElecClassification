"""Independently check E07 startup snapshots without model inference."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from audit_e05c_saved_outputs import digest, metrics, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup", type=Path, required=True)
    args = parser.parse_args()
    base = args.startup
    snapshot = base / "runtime_snapshot"
    frozen = read(base / "frozen_protocol.json")
    p = frozen["protocol"]
    if digest(base / "frozen_protocol.json") != digest(snapshot / "frozen_protocol.json"):
        raise ValueError("Pre-launch frozen bytes differ")
    for folder in ("source_snapshot", "input_snapshot"):
        for file in (snapshot / folder).rglob("*"):
            if file.is_file():
                key = str(file.relative_to(snapshot / folder))
                if digest(file) != frozen["file_sha256"][key]:
                    raise ValueError("Frozen source/input snapshot changed: " + key)
    raw = Path(p["dataset"])
    if digest(raw) != p["dataset_sha256"]:
        raise ValueError("Dataset changed")
    with raw.open(newline="", encoding="utf-8") as stream:
        truth = {r["charging_station_id"]: r["InsertedColumn"] for r in csv.DictReader(stream, delimiter=" ")}
    max_error = 0.
    rows_audited = 0
    per_seed = []
    for seed in p["seeds"]:
        manifest = read(snapshot / f"prepared/seed{seed}/split_manifest.json")
        ids = manifest["splits"]["validation"]
        parent_file = base / f"parent_predictions/seed{seed}.csv"
        remote_parent = f"artifacts/improvements/e03e_gain_replication/e03e_gain_replication_20261018_v1_seed{seed}_fusion/validation_predictions.csv"
        if digest(parent_file) != frozen["file_sha256"][remote_parent]:
            raise ValueError("Original parent predictions changed")
        with parent_file.open(newline="") as stream:
            old = list(csv.DictReader(stream))
        if [r["sample_id"] for r in old] != ids or any(truth[r["sample_id"]] != r["true_index"] for r in old):
            raise ValueError("Original parent IDs/truth differ")
        old_cm = [[0] * 3 for _ in range(3)]
        for row in old:
            old_cm[int(row["true_index"])][int(row["predicted_index"])] += 1
        max_error = max(max_error, abs(metrics(old_cm)["macro_f1"] - frozen["references"][str(seed)]["parent_fusion_nn"]["macro_f1"]))
        gate_dir = snapshot / f"technical_gates/seed{seed}"
        gate = read(gate_dir / "result.json")
        for filename, want in gate["output_sha256"].items():
            if digest(gate_dir / filename) != want:
                raise ValueError("Gate output changed")
        pair = []
        differences = []
        for i, condition in enumerate(("frozen_fusion", "lora8_fusion")):
            with (gate_dir / f"{condition}_initial_predictions.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if [r["sample_id"] for r in rows] != ids:
                raise ValueError("Gate IDs/order differ")
            cm = [[0] * 3 for _ in range(3)]
            for row in rows:
                y, pred = int(row["true_index"]), int(row["predicted_index"])
                values = [float(row[f"probability_{j}"]) for j in range(3)]
                if (truth[row["sample_id"]] != row["true_index"]
                    or not all(math.isfinite(x) and 0 <= x <= 1 for x in values)
                    or abs(sum(values) - 1) > 1e-5 or values.index(max(values)) != pred):
                    raise ValueError("Invalid initial truth/probabilities")
                cm[y][pred] += 1
            calculated = metrics(cm)
            expected = read(gate_dir / f"{condition}_initial_metrics.json")
            for key, value in calculated.items():
                if key in expected and not isinstance(value, list):
                    max_error = max(max_error, abs(value - expected[key]))
            disagreement = sum(a["predicted_index"] != b["predicted_index"] for a, b in zip(old, rows))
            drift = abs(calculated["macro_f1"] - frozen["references"][str(seed)]["parent_fusion_nn"]["macro_f1"]) * 100
            if disagreement > 1 or drift > .05 + 1e-12:
                raise ValueError("Fixed migration tolerance exceeded")
            max_error = max(max_error, abs(drift - gate["migration"][i]["macro_f1_absolute_difference_pp"]))
            if disagreement != gate["migration"][i]["prediction_disagreements"]:
                raise ValueError("Migration disagreement count mismatch")
            pair.append(rows)
            differences.append(disagreement)
            rows_audited += len(rows)
        if pair[0] != pair[1]:
            raise ValueError("Treatments do not have identical starting predictions")
        per_seed.append({"seed": seed, "parent_argmax_disagreements_each_treatment": differences,
                         "treatment_initial_probabilities_identical": True})
    if max_error > 1e-12 or rows_audited != 28080:
        raise ValueError("Incomplete/inconsistent startup outputs")
    proof = {"status": "passed", "initial_prediction_rows_audited": rows_audited,
             "maximum_independent_metric_error": max_error, "per_seed": per_seed,
             "test_evaluated": False, "inference_performed": False,
             "training_completed": False, "prelaunch_freeze_and_input_snapshots_verified": True}
    (base / "independent_startup_audit.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
