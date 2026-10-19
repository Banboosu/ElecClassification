"""Independently check preserved E03F predictions/history/weights; no inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score(file):
    with file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids, truth, probs = [], [], []
    cm = [[0] * 3 for _ in range(3)]
    for row in rows:
        y, pred = int(row["true_index"]), int(row["predicted_index"])
        p = [float(row[f"probability_{i}"]) for i in range(3)]
        if not (0 <= y < 3 and 0 <= pred < 3 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                and abs(sum(p) - 1) <= 1e-5 and p.index(max(p)) == pred):
            raise ValueError("Invalid prediction")
        ids.append(row["sample_id"])
        truth.append(y)
        probs.append(p)
        cm[y][pred] += 1
    if len(rows) != 3510 or len(set(ids)) != 3510:
        raise ValueError("Incomplete/duplicated validation rows")
    f1 = statistics.mean(2 * cm[i][i] / (sum(cm[i]) + sum(cm[j][i] for j in range(3)))
                         if sum(cm[i]) + sum(cm[j][i] for j in range(3)) else 0 for i in range(3))
    return ids, truth, probs, cm, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts
    suite = "e03f_strong_head_lora_20261019_v2"
    analysis = root / f"{suite}_analysis"
    summary = json.loads((analysis / "summary.json").read_text())
    frozen = json.loads((analysis / "frozen_protocol.json").read_text())
    if summary["audit_status"] != "passed" or summary["test_evaluated"]:
        raise ValueError("Require completed validation-only audit")
    reference, heads, initial, rows, max_error, scores = None, {}, {}, 0, 0.0, {}
    metrics = {}
    for case in frozen["protocol"]["conditions"]:
        d = root / f"{suite}_{case['name']}"
        m = json.loads((d / "metrics.json").read_text())
        h = json.loads((d / "history.json").read_text())
        init = json.loads((d / "initial_metrics.json").read_text())
        if (m["test_evaluated"] or m["condition"] != case
            or m["protocol_sha256"] != frozen["protocol_sha256"] or not m["all_frozen_parameters_unchanged"]
            or [r["epoch"] for r in h] != list(range(1, 31))):
            raise ValueError("Incomplete/mismatched condition")
        if any(r["grad_scaler_scale"] != 1024 or r["skipped_optimizer_steps"] or r["early_stop"] for r in h):
            raise ValueError("Numerical/training budget changed")
        for name, key in (("model_best.pt", "model_best_sha256"), ("checkpoint_last.pt", "checkpoint_last_sha256")):
            if digest(d / name) != m[key]:
                raise ValueError("Model bytes changed")
        for filename, expected in (("initial_predictions.csv", init),
                                    ("validation_predictions.csv", m["validation_metrics"]),
                                    ("endpoint_predictions.csv", m["endpoint_validation_metrics"])):
            ids, truth, probs, cm, f1 = score(d / filename)
            if reference is None:
                reference = (ids, truth)
            if (ids, truth) != reference or cm != expected["confusion_matrix"]:
                raise ValueError("Ordered IDs/truth/confusion mismatch")
            max_error = max(max_error, abs(f1 - expected["macro_f1"]))
            rows += len(ids)
            if filename == "initial_predictions.csv":
                if case["branch"] in initial and initial[case["branch"]] != probs:
                    raise ValueError("Within-branch initial probabilities differ")
                initial[case["branch"]] = probs
        values = [init["macro_f1"]] + [r["validation_macro_f1"] for r in h]
        if values.index(max(values)) != m["best_epoch"] or abs(max(values) - m["validation_metrics"]["macro_f1"]) > 1e-12:
            raise ValueError("Epoch-zero-inclusive selection mismatch")
        if abs(h[-1]["validation_macro_f1"] - m["endpoint_validation_metrics"]["macro_f1"]) > 1e-12:
            raise ValueError("Endpoint mismatch")
        heads.setdefault(case["branch"], set()).add(m["initial_head_sha256"])
        scores[case["name"]] = m["validation_metrics"]["macro_f1"]
        metrics[case["name"]] = m
    if any(len(v) != 1 for v in heads.values()) or max_error > 1e-12 or rows != 42120:
        raise ValueError("Audit failed")
    strongest = max([scores["frozen_fusion"], scores["frozen_moment"]] + list(frozen["reference_macro_f1"].values()))
    deltas = {"over_continued_frozen_fusion_pp": 100*(scores["lora8_fusion"]-scores["frozen_fusion"]),
              "over_strongest_nonadapted_pp": 100*(scores["lora8_fusion"]-strongest),
              "over_rank_matched_moment_pp": 100*(scores["lora8_fusion"]-scores["lora8_moment"])}
    for key, value in deltas.items():
        if abs(summary[key] - value) > 1e-12:
            raise ValueError("Frozen gate delta mismatch")
    gate = deltas["over_continued_frozen_fusion_pp"] >= .5 and deltas["over_strongest_nonadapted_pp"] >= .5 and deltas["over_rank_matched_moment_pp"] >= .25
    if gate != summary["pilot_gate_passed"]:
        raise ValueError("Gate mismatch")
    report = {"status": "passed", "prediction_rows_audited": rows, "epochs_audited": 120,
              "maximum_independent_f1_error": max_error, "pilot_gate_passed": gate,
              "summary_sha256": digest(analysis / "summary.json"), "test_evaluated": False,
              "inference_performed": False, **deltas}
    out = root.parent / "independent_output_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
