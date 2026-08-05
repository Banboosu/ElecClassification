from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tcn_moment.summarize_battery_binary import (
    paired_comparisons,
    results_from_binary_metrics,
    validate_gate,
)


class BatteryBinarySummaryTests(unittest.TestCase):
    def test_statistical_payload_becomes_two_model_results(self) -> None:
        evaluation = {
            "selected_positive_weight": 2.0,
            "ranking": {"validation": {}, "test": {}},
            "operating_points": {},
        }
        payload = {
            "model": "BATTERY_BINARY_STATISTICAL",
            "run_name": "run",
            "seed": 42,
            "data": {"train_subset": {"requested_fraction": 0.05}},
            "results": {
                "logistic_regression": evaluation,
                "random_forest": evaluation,
            },
        }
        results = results_from_binary_metrics(payload, Path("run"))
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {result["model"] for result in results},
            {
                "BATTERY_BINARY_LOGISTIC_REGRESSION",
                "BATTERY_BINARY_RANDOM_FOREST",
            },
        )

    def test_gate_must_authorize_only_selected_families(self) -> None:
        gate = {
            "test_metrics_used_for_gate": False,
            "decisions": {
                "TCN": {"expand_to_five_seeds": False},
                "MOMENT_RBF_SVM": {"expand_to_five_seeds": True},
                "STATISTICAL::logistic_regression": {
                    "expand_to_five_seeds": True
                },
                "STATISTICAL::random_forest": {"expand_to_five_seeds": True},
            },
        }
        validate_gate(gate)
        gate["decisions"]["TCN"]["expand_to_five_seeds"] = True
        with self.assertRaises(ValueError):
            validate_gate(gate)

    def test_paired_comparison_reports_binary_minus_baseline(self) -> None:
        seeds = [42, 43, 44, 45, 46]
        binary = pd.DataFrame(
            {
                "model": "BATTERY_BINARY_MOMENT_RBF_SVM",
                "train_fraction": 1.0,
                "operating_point": "max_f2",
                "seed": seeds,
                "test_battery_recall": np.full(5, 0.9),
                "test_battery_precision": np.full(5, 0.8),
                "test_false_positive_rate": np.full(5, 0.2),
                "test_f2": np.full(5, 0.88),
                "test_average_precision": np.full(5, 0.92),
                "test_roc_auc": np.full(5, 0.95),
                "test_false_negatives": np.full(5, 10),
                "test_false_positives": np.full(5, 20),
            }
        )
        baseline = binary.copy()
        baseline["model"] = "MOMENT_RBF_SVM"
        baseline["test_average_precision"] = 0.90
        comparison = paired_comparisons(binary, baseline)
        row = comparison.loc[comparison["metric"] == "test_average_precision"].iloc[0]
        self.assertAlmostEqual(row["mean_difference"], 0.02)
        self.assertAlmostEqual(row["ci95_low"], 0.02)
        self.assertAlmostEqual(row["ci95_high"], 0.02)


if __name__ == "__main__":
    unittest.main()
