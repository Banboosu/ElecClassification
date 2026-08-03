from __future__ import annotations

import unittest

import numpy as np

from tcn_moment.config import load_config
from tcn_moment.evaluate_imputation import (
    ErrorAccumulator,
    _centered_block,
    baseline_prediction,
    generate_observation_mask,
)


class ImputationMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input_mask = np.asarray(
            [
                [1] * 40,
                [1] * 32 + [0] * 8,
            ],
            dtype=np.uint8,
        )
        self.lengths = np.asarray([40, 32])
        self.sample_ids = np.asarray(["sample-a", "sample-b"])

    def _mask(self, rate: float, pattern: str) -> np.ndarray:
        return generate_observation_mask(
            self.input_mask,
            self.lengths,
            self.sample_ids,
            mask_rate=rate,
            pattern=pattern,
            mask_seed=42,
            patch_len=8,
            min_complete_patches=3,
        )

    def test_random_patch_masks_are_reproducible_aligned_and_nested(self) -> None:
        small = self._mask(0.25, "random_patches")
        repeated = self._mask(0.25, "random_patches")
        large = self._mask(0.60, "random_patches")

        np.testing.assert_array_equal(small, repeated)
        self.assertTrue(
            set(np.flatnonzero(small[0] == 0)).issubset(
                set(np.flatnonzero(large[0] == 0))
            )
        )
        self.assertEqual(int((small[0] == 0).sum()), 8)
        self.assertEqual(int((large[0] == 0).sum()), 24)
        self.assertTrue(np.all(small[1, 32:] == 0))
        for patch_start in range(0, 40, 8):
            self.assertIn(int(small[0, patch_start : patch_start + 8].sum()), {0, 8})

    def test_contiguous_masked_patches_form_one_block(self) -> None:
        observation = self._mask(0.60, "contiguous_block")
        for row, valid_length in enumerate(self.lengths):
            complete_length = (int(valid_length) // 8) * 8
            hidden_patches = [
                patch
                for patch in range(complete_length // 8)
                if observation[row, patch * 8 : (patch + 1) * 8].sum() == 0
            ]
            self.assertEqual(
                hidden_patches,
                list(range(hidden_patches[0], hidden_patches[-1] + 1)),
            )

    def test_centered_blocks_are_nested_for_increasing_sizes(self) -> None:
        for total in range(3, 20):
            for center in range(total):
                previous: set[int] = set()
                for size in range(1, total):
                    current = set(_centered_block(center, size, total).tolist())
                    self.assertTrue(previous.issubset(current))
                    previous = current


class ImputationBaselineTests(unittest.TestCase):
    def test_baselines_use_visible_values_only(self) -> None:
        values = np.asarray([0.0, 1.0, 999.0, 3.0, 4.0])
        visible = np.asarray([True, True, False, True, True])

        self.assertEqual(baseline_prediction(values, visible, "mean")[2], 2.0)
        self.assertEqual(baseline_prediction(values, visible, "forward_fill")[2], 1.0)
        self.assertEqual(baseline_prediction(values, visible, "linear")[2], 2.0)
        self.assertAlmostEqual(baseline_prediction(values, visible, "pchip")[2], 2.0)

    def test_error_accumulator_reports_raw_and_normalized_metrics(self) -> None:
        accumulator = ErrorAccumulator()
        accumulator.update(
            target=np.asarray([1.0, 3.0]),
            prediction=np.asarray([2.0, 1.0]),
            full_valid_sequence=np.asarray([0.0, 1.0, 2.0, 3.0]),
        )
        result = accumulator.result()

        self.assertAlmostEqual(float(result["mae"]), 1.5)
        self.assertAlmostEqual(float(result["rmse"]), np.sqrt(2.5))
        self.assertEqual(result["evaluated_points"], 2)
        self.assertEqual(result["evaluated_sequences"], 1)

    def test_imputation_config_matches_thesis_protocol(self) -> None:
        config = load_config("configs/experiments/moment_imputation_zero_shot.yaml")
        self.assertEqual(config.data.normalize, "none")
        self.assertEqual(config.imputation.mask_rates, (0.1, 0.25, 0.4, 0.6))
        self.assertEqual(
            config.imputation.mask_patterns,
            ("random_patches", "contiguous_block"),
        )
        self.assertEqual(config.imputation.mask_seeds, (42, 43, 44, 45, 46))


if __name__ == "__main__":
    unittest.main()
