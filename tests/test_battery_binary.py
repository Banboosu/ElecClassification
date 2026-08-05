from __future__ import annotations

import unittest

import numpy as np

from tcn_moment.config import load_config
from tcn_moment.train_battery_binary import (
    binary_svm_cv_results,
    binary_targets,
    build_binary_svm_search,
    select_best_candidate,
)


class BatteryBinaryTests(unittest.TestCase):
    def test_binary_targets_map_only_critical_class_to_one(self) -> None:
        labels = np.asarray([0, 1, 2, 2, 1])
        np.testing.assert_array_equal(binary_targets(labels, 2), [0, 0, 1, 1, 0])

    def test_candidate_selection_uses_validation_ap_and_lower_weight_tie_break(self) -> None:
        selected = select_best_candidate(
            [
                {"positive_weight": 4.0, "best_validation_average_precision": 0.8},
                {"positive_weight": 1.0, "best_validation_average_precision": 0.8},
                {"positive_weight": 2.0, "best_validation_average_precision": 0.7},
            ]
        )
        self.assertEqual(selected["positive_weight"], 1.0)

    def test_binary_svm_search_uses_training_only_average_precision(self) -> None:
        config = load_config("configs/experiments/battery_binary/moment_svm.yaml")
        search = build_binary_svm_search(config, (1.0, 2.0, 4.0))

        self.assertEqual(search.scoring, "average_precision")
        self.assertEqual(search.cv, 5)
        self.assertEqual(search.param_grid["C"], [1.0, 10.0, 100.0, 1000.0, 10000.0])
        self.assertEqual(
            search.param_grid["class_weight"],
            [{0: 1.0, 1: 1.0}, {0: 1.0, 1: 2.0}, {0: 1.0, 1: 4.0}],
        )

    def test_binary_svm_cv_results_names_average_precision_and_weight(self) -> None:
        class Search:
            cv_results_ = {
                "params": [{"C": 10.0, "class_weight": {0: 1.0, 1: 4.0}}],
                "mean_test_score": np.asarray([0.91]),
                "std_test_score": np.asarray([0.02]),
                "rank_test_score": np.asarray([1]),
                "mean_fit_time": np.asarray([1.2]),
                "mean_score_time": np.asarray([0.3]),
            }

        row = binary_svm_cv_results(Search())[0]
        self.assertEqual(row["positive_weight"], 4.0)
        self.assertEqual(row["mean_validation_average_precision"], 0.91)


if __name__ == "__main__":
    unittest.main()
