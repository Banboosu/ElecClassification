from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from tcn_moment.analyze_battery_binary import (
    compare_validation_rows,
    gate_decisions,
    validation_rows_from_binary_metrics,
)


def evaluation(average_precision: float, fpr: float) -> dict:
    return {
        "ranking": {
            "validation": {
                "average_precision": average_precision,
                "roc_auc": 0.9,
            },
            "test": {"must_not_be_read": object()},
        },
        "operating_points": {
            "recall_0.95": {
                "validation": {
                    "battery_recall": 0.95,
                    "false_positive_rate": fpr,
                    "battery_precision": 0.8,
                    "f2": 0.9,
                },
                "test": {"must_not_be_read": object()},
            }
        },
    }


class BatteryBinaryAnalysisTests(unittest.TestCase):
    def test_tcn_parser_reads_validation_only(self) -> None:
        payload = {
            "model": "BATTERY_BINARY_TCN",
            "run_name": "run",
            "seed": 42,
            "data": {"train_subset": {"requested_fraction": 0.01}},
            "selected_positive_weight": 4.0,
            **evaluation(0.8, 0.2),
        }
        row = validation_rows_from_binary_metrics(payload, Path("run"))[0]
        self.assertEqual(row["validation_average_precision"], 0.8)
        self.assertEqual(row["validation_fpr_at_target_95"], 0.2)

    def test_gate_requires_full_or_two_low_label_passes(self) -> None:
        rows = []
        for fraction, ap_delta in ((0.01, 0.011), (0.05, 0.012), (0.1, 0.0), (1.0, 0.0)):
            rows.append(
                {
                    "family": "TCN",
                    "variant": "binary_tcn",
                    "train_fraction": fraction,
                    "average_precision_delta": ap_delta,
                    "validation_recall_at_target_95": 0.95,
                    "fpr_delta_at_target_95": 0.0,
                }
            )
        decision = gate_decisions(pd.DataFrame(rows))["TCN"]
        self.assertTrue(decision["average_precision_gate_passed"])
        self.assertTrue(decision["expand_to_five_seeds"])

    def test_statistical_model_uses_best_existing_baseline(self) -> None:
        binary = pd.DataFrame(
            [
                {
                    "family": "STATISTICAL",
                    "variant": "logistic_regression",
                    "train_fraction": 0.01,
                    "validation_average_precision": 0.80,
                    "validation_fpr_at_target_95": 0.30,
                }
            ]
        )
        baseline = pd.DataFrame(
            [
                {
                    "family": "TCN",
                    "train_fraction": 0.01,
                    "baseline_run_name": "tcn",
                    "baseline_model": "TCN",
                    "validation_average_precision": 0.75,
                    "baseline_validation_recall_at_target_95": 0.95,
                    "baseline_validation_fpr_at_target_95": 0.4,
                },
                {
                    "family": "MOMENT_RBF_SVM",
                    "train_fraction": 0.01,
                    "baseline_run_name": "moment",
                    "baseline_model": "MOMENT_RBF_SVM_FEW_SHOT",
                    "validation_average_precision": 0.78,
                    "baseline_validation_recall_at_target_95": 0.95,
                    "baseline_validation_fpr_at_target_95": 0.35,
                },
            ]
        )
        row = compare_validation_rows(binary, baseline).iloc[0]
        self.assertEqual(row["baseline_model"], "MOMENT_RBF_SVM_FEW_SHOT")
        self.assertAlmostEqual(row["average_precision_delta"], 0.02)


if __name__ == "__main__":
    unittest.main()
