from __future__ import annotations

import unittest

import numpy as np

from tcn_moment.config import load_config
from tcn_moment.train_moment_svm import (
    build_paper_svm_search,
    select_paper_training_subset,
)


class MomentSVMTests(unittest.TestCase):
    def test_paper_subset_is_reproducible_and_stratified(self) -> None:
        features = np.arange(360, dtype=np.float32).reshape(120, 3)
        labels = np.repeat(np.arange(3), 40)
        sample_ids = np.asarray([f"sample-{index}" for index in range(120)])

        first = select_paper_training_subset(features, labels, sample_ids, 60)
        second = select_paper_training_subset(features, labels, sample_ids, 60)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        np.testing.assert_array_equal(np.bincount(first[1]), [20, 20, 20])

    def test_svm_search_matches_official_c_grid(self) -> None:
        config = load_config("configs/experiments/moment_svm_rbf.yaml")
        search = build_paper_svm_search(config)

        self.assertEqual(search.param_grid["C"], list(config.svm.c_values))
        self.assertEqual(search.estimator.kernel, "rbf")
        self.assertEqual(search.estimator.gamma, "scale")
        self.assertEqual(search.cv, 5)
        self.assertEqual(search.scoring, "accuracy")


if __name__ == "__main__":
    unittest.main()
