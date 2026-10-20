"""Aggregation and audit helpers for E08 (validation-only dynamics-feature study)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

try:  # scipy is a scikit-learn dependency; keep a fixed fallback for n=4.
    from scipy.stats import t as student_t
except ImportError:  # pragma: no cover
    student_t = None


GATE_SPECS: dict[str, dict[str, Any]] = {
    "G1": {"left": "dyn_hgb", "right": "f15_hgb", "tag": "40p"},
    "G2": {"left": "dyn_hgb", "right": "f15_hgb", "tag": "10p"},
    "G3": {"left": "stack", "right": "dyn_hgb", "tag": "40p"},
    "G4": {"left": "dyn_svm", "right": "f15_svm", "tag": "40p"},
}


def paired_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    """Mean/SD/positive count and a descriptive paired t interval over seeds."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("paired_summary requires a non-empty one-dimensional sequence.")
    n = len(array)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else 0.0
    half_width = None
    if n > 1 and sd > 0.0:
        critical = 3.182446 if student_t is None else float(student_t.ppf(0.975, n - 1))
        half_width = critical * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "positive_seeds": int(np.sum(array > 0.0)),
        "per_seed": array.tolist(),
        "descriptive_ci95": (
            None if half_width is None else [mean - half_width, mean + half_width]
        ),
    }


def _condition_map(seed_results: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
    mapping: dict[tuple[int, str, str], dict[str, Any]] = {}
    for result in seed_results:
        seed = int(result["seed"])
        for record in result["conditions"]:
            key = (seed, record["method"], record["tag"])
            if key in mapping:
                raise ValueError(f"Duplicate condition: {key}")
            mapping[key] = record
    return mapping


def aggregate_conditions(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate validation metrics by method and label budget across seeds."""
    mapping = _condition_map(seed_results)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (_, method, tag), record in mapping.items():
        groups.setdefault((method, tag), []).append(record)
    rows: list[dict[str, Any]] = []
    for (method, tag), records in sorted(groups.items()):
        f1_values = [record["validation"]["macro_f1"] * 100.0 for record in records]
        accuracy = [record["validation"]["accuracy"] * 100.0 for record in records]
        battery_recall = [record["validation"]["battery"]["recall"] * 100.0 for record in records]
        battery_fpr = [
            record["validation"]["battery"]["false_positive_rate"] * 100.0 for record in records
        ]
        battery_ap = [
            record["validation"]["battery"]["average_precision"] * 100.0 for record in records
        ]
        seconds = [record["fit_seconds"] for record in records]
        rows.append(
            {
                "method": method,
                "tag": tag,
                "fraction": records[0]["fraction"],
                "n_seeds": len(records),
                "n_train": records[0]["n_train"],
                "macro_f1_mean": float(np.mean(f1_values)),
                "macro_f1_sd": float(np.std(f1_values, ddof=1)) if len(records) > 1 else 0.0,
                "accuracy_mean": float(np.mean(accuracy)),
                "battery_recall_mean": float(np.mean(battery_recall)),
                "battery_fpr_mean": float(np.mean(battery_fpr)),
                "battery_ap_mean": float(np.mean(battery_ap)),
                "fit_seconds_median": float(np.median(seconds)),
                "per_seed_macro_f1": f1_values,
            }
        )
    return rows


def paired_differences(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute paired validation differences for every method pair available per seed."""
    mapping = _condition_map(seed_results)
    methods = sorted({method for (_, method, _) in mapping})
    tags = sorted({tag for (_, _, tag) in mapping})
    rows: list[dict[str, Any]] = []
    for left in methods:
        for right in methods:
            if left == right:
                continue
            for tag in tags:
                differences: list[float] = []
                for seed in sorted({item[0] for item in mapping}):
                    left_record = mapping.get((seed, left, tag))
                    right_record = mapping.get((seed, right, tag))
                    if left_record is None or right_record is None:
                        continue
                    differences.append(
                        (left_record["validation"]["macro_f1"] - right_record["validation"]["macro_f1"])
                        * 100.0
                    )
                if not differences:
                    continue
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "tag": tag,
                        **paired_summary(differences),
                    }
                )
    return rows


def evaluate_gates(
    protocol: dict[str, Any],
    paired: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the frozen E08 gates; a gate passes only if both conditions hold."""
    by_id = {record["id"]: record for record in protocol["primary_gates"]}
    results: list[dict[str, Any]] = []
    for gate_id, spec in GATE_SPECS.items():
        requirement = by_id.get(gate_id)
        if requirement is None:
            raise ValueError(f"Frozen protocol is missing gate {gate_id}.")
        match = next(
            (
                record
                for record in paired
                if record["left"] == spec["left"]
                and record["right"] == spec["right"]
                and record["tag"] == spec["tag"]
            ),
            None,
        )
        if match is None:
            results.append({"id": gate_id, "status": "not_evaluated", "reason": "missing conditions"})
            continue
        mean_passes = match["mean"] >= float(requirement["mean_gain_pp"])
        positive_passes = match["positive_seeds"] >= int(requirement["positive_seeds"])
        results.append(
            {
                "id": gate_id,
                "comparison": f"{spec['left']} - {spec['right']} at {spec['tag']}",
                "mean_gain_pp": match["mean"],
                "required_mean_pp": requirement["mean_gain_pp"],
                "positive_seeds": match["positive_seeds"],
                "required_positive_seeds": requirement["positive_seeds"],
                "status": "passed" if mean_passes and positive_passes else "failed",
            }
        )
    return results


def audit_condition(
    *,
    predictions_path: Path,
    metrics_path: Path,
    expected_ids: set[str],
) -> dict[str, Any]:
    """Recompute macro-F1 from saved predictions and check identity/row integrity."""
    with predictions_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty predictions: {predictions_path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Duplicate sample IDs in {predictions_path}")
    if set(sample_ids) != expected_ids:
        raise ValueError(f"Prediction IDs do not match validation IDs: {predictions_path}")
    if any(row["split"] != "validation" for row in rows):
        raise ValueError(f"Non-validation rows found: {predictions_path}")
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)
    recomputed = float(f1_score(y_true, y_pred, average="macro"))
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    recorded = float(metrics["validation"]["macro_f1"])
    return {
        "predictions": str(predictions_path),
        "rows": len(rows),
        "recomputed_macro_f1": recomputed,
        "recorded_macro_f1": recorded,
        "absolute_error": abs(recomputed - recorded),
    }
