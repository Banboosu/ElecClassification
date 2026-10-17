from __future__ import annotations

import unittest

from scripts.analyze_e01_validation import metrics_from_confusion, paired_summary
from tcn_moment.metrics import classification_metrics


class E01ValidationAnalysisTests(unittest.TestCase):
    def test_independent_metric_recomputation_matches_common_metrics(self):
        matrix = [[8, 2, 0], [1, 9, 0], [3, 7, 0]]
        true, predicted = [], []
        for i, row in enumerate(matrix):
            for j, count in enumerate(row):
                true.extend([i] * count)
                predicted.extend([j] * count)
        expected = classification_metrics(true, predicted, ["0", "1", "2"])
        for key, value in metrics_from_confusion(matrix).items():
            self.assertAlmostEqual(value, expected[key], places=12)

    def test_paired_summary_reports_difference_in_supplied_percentage_points(self):
        result = paired_summary([1, 2, 3, 4, 5])
        self.assertEqual(result["n"], 5)
        self.assertEqual(result["difference_pp"], 3)
        self.assertLess(result["ci95_lower_pp"], 3)
        self.assertGreater(result["ci95_upper_pp"], 3)

    def test_missing_class_and_single_seed_uncertainty_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics_from_confusion([[2, 0], [0, 0]])
        with self.assertRaises(ValueError):
            paired_summary([2])


if __name__ == "__main__":
    unittest.main()
