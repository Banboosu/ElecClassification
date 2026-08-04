from __future__ import annotations

import unittest

import numpy as np

from tcn_moment.evaluate_battery_safety import (
    binary_metrics_from_predictions,
    evaluate_operating_points,
    threshold_for_max_fbeta,
    threshold_for_target_recall,
)


class BatterySafetyMetricTests(unittest.TestCase):
    def test_binary_metrics_report_misses_and_false_alarms(self) -> None:
        labels = np.asarray([2, 2, 2, 0, 0, 1, 1])
        predicted_positive = np.asarray([True, True, False, True, False, False, False])

        metrics = binary_metrics_from_predictions(labels, predicted_positive, 2)

        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 3)
        self.assertAlmostEqual(metrics["battery_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["battery_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 4)

    def test_target_recall_uses_highest_feasible_validation_threshold(self) -> None:
        labels = np.asarray([2, 2, 2, 2, 0, 0, 0])
        scores = np.asarray([0.9, 0.8, 0.7, 0.1, 0.85, 0.65, 0.05])

        threshold = threshold_for_target_recall(labels, scores, 2, 0.75)
        metrics = binary_metrics_from_predictions(labels, scores >= threshold, 2)

        self.assertEqual(threshold, 0.7)
        self.assertEqual(metrics["battery_recall"], 0.75)
        self.assertEqual(metrics["false_positives"], 1)

    def test_max_f2_threshold_prefers_recall_over_precision(self) -> None:
        labels = np.asarray([2, 2, 0, 0])
        scores = np.asarray([0.9, 0.4, 0.8, 0.1])

        threshold = threshold_for_max_fbeta(labels, scores, 2, beta=2)

        self.assertEqual(threshold, 0.4)

    def test_test_threshold_is_selected_from_validation_only(self) -> None:
        result = evaluate_operating_points(
            validation_labels=np.asarray([2, 2, 0, 0]),
            validation_predictions=np.asarray([2, 0, 0, 0]),
            validation_scores=np.asarray([0.9, 0.6, 0.8, 0.1]),
            test_labels=np.asarray([2, 2, 0, 0]),
            test_predictions=np.asarray([2, 2, 0, 0]),
            test_scores=np.asarray([0.7, 0.5, 0.65, 0.2]),
            critical_index=2,
            target_recalls=(1.0,),
        )

        operating_point = result["operating_points"]["recall_1"]
        self.assertEqual(operating_point["threshold"], 0.6)
        self.assertEqual(operating_point["validation"]["battery_recall"], 1.0)
        self.assertEqual(operating_point["test"]["battery_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
