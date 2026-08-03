from __future__ import annotations

import unittest

import numpy as np

from tcn_moment.config import load_config
from tcn_moment.data import stratified_train_subset_indices


class FewShotSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.repeat(np.arange(3), 100)
        self.sample_ids = np.asarray(
            [f"class-{label}-sample-{index:03d}" for label in range(3) for index in range(100)]
        )

    def test_subsets_are_reproducible_stratified_and_nested(self) -> None:
        small = stratified_train_subset_indices(
            self.labels,
            self.sample_ids,
            0.05,
            random_state=42,
        )
        repeated = stratified_train_subset_indices(
            self.labels,
            self.sample_ids,
            0.05,
            random_state=42,
        )
        large = stratified_train_subset_indices(
            self.labels,
            self.sample_ids,
            0.20,
            random_state=42,
        )

        np.testing.assert_array_equal(small, repeated)
        np.testing.assert_array_equal(np.bincount(self.labels[small]), [5, 5, 5])
        np.testing.assert_array_equal(np.bincount(self.labels[large]), [20, 20, 20])
        self.assertTrue(set(self.sample_ids[small]).issubset(set(self.sample_ids[large])))

    def test_selected_ids_do_not_depend_on_input_row_order(self) -> None:
        expected = stratified_train_subset_indices(
            self.labels,
            self.sample_ids,
            0.10,
            random_state=43,
        )
        permutation = np.random.default_rng(9).permutation(len(self.labels))
        actual = stratified_train_subset_indices(
            self.labels[permutation],
            self.sample_ids[permutation],
            0.10,
            random_state=43,
        )

        self.assertEqual(
            set(self.sample_ids[expected]),
            set(self.sample_ids[permutation][actual]),
        )

    def test_full_fraction_keeps_all_rows(self) -> None:
        indices = stratified_train_subset_indices(
            self.labels,
            self.sample_ids,
            1.0,
            random_state=42,
        )
        np.testing.assert_array_equal(indices, np.arange(len(self.labels)))

    def test_few_shot_configs_share_the_declared_budgets(self) -> None:
        moment = load_config("configs/experiments/few_shot/moment_svm.yaml")
        self.assertEqual(
            moment.svm.few_shot_fractions,
            (0.01, 0.05, 0.1, 0.2, 0.4),
        )
        for tag, fraction in (
            ("01", 0.01),
            ("05", 0.05),
            ("10", 0.10),
            ("20", 0.20),
            ("40", 0.40),
        ):
            tcn = load_config(
                f"configs/experiments/few_shot/tcn_{tag}_percent.yaml"
            )
            self.assertEqual(tcn.data.train_fraction, fraction)
            self.assertEqual(tcn.data.normalize, "none")


if __name__ == "__main__":
    unittest.main()
