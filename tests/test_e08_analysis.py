from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tcn_moment.e08_analysis import (
    audit_condition,
    evaluate_gates,
    paired_summary,
)


def _paired(left: str, right: str, tag: str, values: list[float]) -> dict:
    return {"left": left, "right": right, "tag": tag, **paired_summary(values)}


class E08AnalysisTests(unittest.TestCase):
    def test_paired_summary(self):
        summary = paired_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["mean"], 2.5)
        self.assertAlmostEqual(summary["sample_sd"], 1.2909944487358056)
        self.assertEqual(summary["positive_seeds"], 4)
        self.assertLess(summary["descriptive_ci95"][0], summary["mean"])
        self.assertGreater(summary["descriptive_ci95"][1], summary["mean"])

    def test_gates_require_both_mean_and_positive_count(self):
        protocol = {
            "primary_gates": [
                {"id": "G1", "mean_gain_pp": 2.0, "positive_seeds": 3},
                {"id": "G2", "mean_gain_pp": 2.0, "positive_seeds": 3},
                {"id": "G3", "mean_gain_pp": 0.3, "positive_seeds": 3},
                {"id": "G4", "mean_gain_pp": 1.0, "positive_seeds": 3},
            ]
        }
        paired = [
            _paired("dyn_hgb", "f15_hgb", "40p", [3.0, 3.0, 3.0, -1.0]),
            _paired("dyn_hgb", "f15_hgb", "10p", [1.0, 1.0, 1.0, 1.0]),
            _paired("stack", "dyn_hgb", "40p", [0.4, 0.4, 0.4, 0.4]),
            _paired("dyn_svm", "f15_svm", "40p", [2.0, 2.0, 2.0, 2.0]),
        ]
        gates = {gate["id"]: gate for gate in evaluate_gates(protocol, paired)}
        self.assertEqual(gates["G1"]["status"], "passed")
        self.assertEqual(gates["G2"]["status"], "failed")  # mean is +1, below +2
        self.assertEqual(gates["G3"]["status"], "passed")
        self.assertEqual(gates["G4"]["status"], "passed")

    def test_audit_condition_recomputes_and_rejects_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            metrics = root / "metrics.json"
            rows = [
                {
                    "sample_id": "a",
                    "split": "validation",
                    "y_true": 0,
                    "y_pred": 0,
                    "score_0": 0.9,
                    "score_1": 0.05,
                    "score_2": 0.05,
                    "score_battery": 0.05,
                },
                {
                    "sample_id": "b",
                    "split": "validation",
                    "y_true": 2,
                    "y_pred": 2,
                    "score_0": 0.05,
                    "score_1": 0.05,
                    "score_2": 0.9,
                    "score_battery": 0.9,
                },
            ]
            with predictions.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            metrics.write_text(
                json.dumps({"validation": {"macro_f1": 1.0}}), encoding="utf-8"
            )
            record = audit_condition(
                predictions_path=predictions,
                metrics_path=metrics,
                expected_ids={"a", "b"},
            )
            self.assertEqual(record["rows"], 2)
            self.assertAlmostEqual(record["recomputed_macro_f1"], 1.0)
            rows[0]["split"] = "test"
            with predictions.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                audit_condition(
                    predictions_path=predictions,
                    metrics_path=metrics,
                    expected_ids={"a", "b"},
                )


if __name__ == "__main__":
    unittest.main()
