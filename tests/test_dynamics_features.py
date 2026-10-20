from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from tcn_moment.dynamics_features import (
    BASE_FEATURE_COUNT,
    DYNAMICS_FEATURE_COUNT,
    DYNAMICS_FEATURE_NAMES,
    FEATURE_GROUPS,
    extract_dynamics_features,
    group_indices,
    select_feature_set,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _padded(values: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((1, length), dtype=np.float32)
    mask = np.zeros((1, length), dtype=bool)
    x[0, : len(values)] = values
    mask[0, : len(values)] = True
    return x, mask


class DynamicsFeaturesTests(unittest.TestCase):
    def test_contract_is_unique_and_partitioned(self):
        self.assertEqual(len(DYNAMICS_FEATURE_NAMES), DYNAMICS_FEATURE_COUNT)
        self.assertEqual(len(set(DYNAMICS_FEATURE_NAMES)), DYNAMICS_FEATURE_COUNT)
        grouped = [name for names in FEATURE_GROUPS.values() for name in names]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), set(DYNAMICS_FEATURE_NAMES))
        self.assertEqual(BASE_FEATURE_COUNT, 15)

    def test_padding_and_max_length_do_not_change_valid_observations(self):
        rng = np.random.default_rng(0)
        values = np.cumsum(rng.normal(0, 10, 200)) + 300
        short_x, short_mask = _padded(values, 1024)
        long_x, long_mask = _padded(values, 1280)
        short = extract_dynamics_features(short_x, short_mask, max_length=1024)
        long = extract_dynamics_features(long_x, long_mask, max_length=1280)
        # The two length descriptors intentionally depend on the padding budget.
        np.testing.assert_allclose(short[0, 2:15], long[0, 2:15])
        np.testing.assert_allclose(short[0, 16:], long[0, 16:])
        self.assertAlmostEqual(float(short[0, 0]), 200.0)
        self.assertAlmostEqual(float(long[0, 15]), 200.0 / 1280.0)

    def test_deterministic_and_finite(self):
        rng = np.random.default_rng(1)
        x = np.zeros((5, 64), dtype=np.float32)
        mask = np.zeros((5, 64), dtype=bool)
        for index in range(5):
            length = 18 + index * 9
            x[index, :length] = rng.normal(0, 5, length) + 100
            mask[index, :length] = True
        first = extract_dynamics_features(x, mask)
        second = extract_dynamics_features(x, mask)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())

    def test_constant_and_short_series_are_finite(self):
        for length in (2, 18, 50):
            values = np.full(length, 4.0, dtype=np.float32)
            x, mask = _padded(values, 64)
            features = extract_dynamics_features(x, mask)
            self.assertTrue(np.isfinite(features).all())

    def test_base_slice_matches_reference_extractor(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required to import the reference baseline module")
        from tcn_moment.train_baselines import extract_statistical_features

        rng = np.random.default_rng(3)
        x = np.zeros((4, 32), dtype=np.float32)
        mask = np.zeros((4, 32), dtype=bool)
        for index in range(4):
            length = 18 + index * 3
            x[index, :length] = rng.normal(0, 2, length)
            mask[index, :length] = True
        dynamics = extract_dynamics_features(x, mask)
        np.testing.assert_allclose(dynamics[:, :BASE_FEATURE_COUNT], extract_statistical_features(x, mask))

    def test_feature_set_selection_and_group_indices(self):
        features = np.arange(60, dtype=np.float32).reshape(1, -1)
        self.assertEqual(select_feature_set(features, "f15").shape, (1, 15))
        self.assertEqual(select_feature_set(features, "dyn").shape, (1, 60))
        with self.assertRaises(ValueError):
            select_feature_set(features, "unknown")
        for group in FEATURE_GROUPS:
            indices = group_indices(group)
            self.assertEqual(len(indices), len(FEATURE_GROUPS[group]))
            self.assertEqual(len(set(indices)), len(indices))
            self.assertEqual([DYNAMICS_FEATURE_NAMES[index] for index in indices], list(FEATURE_GROUPS[group]))
        with self.assertRaises(ValueError):
            group_indices("unknown")

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            extract_dynamics_features(np.zeros((2, 5)), np.zeros((2, 4), dtype=bool))
        with self.assertRaises(ValueError):
            extract_dynamics_features(np.zeros((1, 20)), np.ones((1, 20), dtype=bool), max_length=0)


if __name__ == "__main__":
    unittest.main()
