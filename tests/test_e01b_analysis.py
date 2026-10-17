from __future__ import annotations

import unittest
from pathlib import Path

from tcn_moment.analyze_e01b import DEFAULT_PROTOCOL, budget_rule, read_protocol


class E01BAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.protocol = read_protocol(Path(DEFAULT_PROTOCOL))
        self.rule = self.protocol["success_rule"]

    def test_rule_requires_mean_gain_and_three_positive_replications(self):
        self.assertTrue(budget_rule([2, 2, 1, -1], self.rule)["rule_passed"])
        self.assertFalse(budget_rule([4, 4, -1, -1], self.rule)["rule_passed"])
        self.assertFalse(budget_rule([0.5, 0.5, 0.5, 0.5], self.rule)["rule_passed"])

    def test_discovery_seed_is_excluded_and_test_stays_locked(self):
        self.assertNotIn(42, self.protocol["replication_seeds"])
        self.assertEqual(self.protocol["fractions"], [0.2, 0.4])
        self.assertEqual(self.protocol["models"], ["fusion_stats_svm"])
        self.assertFalse(self.protocol["test_evaluated"])

    def test_missing_replications_and_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            budget_rule([1, 2, 3], self.rule)
        with self.assertRaises(ValueError):
            budget_rule([1, 2, 3, float("nan")], self.rule)


if __name__ == "__main__":
    unittest.main()
