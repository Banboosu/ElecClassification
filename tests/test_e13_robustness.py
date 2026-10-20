from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tcn_moment.run_e13_robustness import (
    _diff_pad,
    _load,
    _normalize_pad,
    _perturb,
    synthetic_check,
)


class E13RobustnessTests(unittest.TestCase):
    def test_protocol_is_validation_only(self):
        protocol = _load(Path("configs/experiments/improvements/e13_robustness_protocol.json"))
        self.assertEqual(protocol["experiment_id"], "E13")
        self.assertFalse(protocol["test_evaluated"])
        self.assertFalse(protocol["automatic_expansion"])
        self.assertFalse(protocol["training_on_perturbed"])
        self.assertIn("clean", protocol["perturbations"])

    def test_noise_is_deterministic_and_clipped(self):
        x = np.zeros((1, 6), dtype=np.float32)
        x[0, :4] = [0.0, 1.0, 2.0, 3.0]
        mask = np.zeros((1, 6), dtype=bool)
        mask[0, :4] = True
        spec = {"kind": "noise", "sigma_fraction": 0.05}
        first = _perturb(x, mask, spec, np.random.default_rng(3), 50.0, 0.0)
        second = _perturb(x, mask, spec, np.random.default_rng(3), 50.0, 0.0)
        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(first >= 0))
        np.testing.assert_allclose(x[0, :4], [0.0, 1.0, 2.0, 3.0])

    def test_quantization_uses_absolute_step(self):
        x = np.zeros((1, 5), dtype=np.float32)
        x[0, :4] = [0.0, 1.0, 2.0, 3.0]
        mask = np.zeros((1, 5), dtype=bool)
        mask[0, :4] = True
        out = _perturb(x, mask, {"kind": "quantization", "step_w": 2.0}, np.random.default_rng(1), 10.0, 0.0)
        np.testing.assert_allclose(out[0, :4], [0.0, 0.0, 2.0, 4.0])

    def test_normalization_and_difference_use_valid_values_only(self):
        normalized = _normalize_pad(np.asarray([1.0, 2.0, 3.0]), 8)
        self.assertAlmostEqual(float(normalized[:3].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[:3].std()), 1.0, places=6)
        self.assertTrue(np.all(normalized[3:] == 0))
        differences, mask = _diff_pad(np.asarray([1.0, 2.0, 4.0, 8.0]), 8)
        self.assertEqual(int(mask.sum()), 3)
        self.assertTrue(np.all(differences[3:] == 0))
        self.assertTrue(np.all(np.diff(differences[:3]) > 0))

    def test_synthetic_check(self):
        result = synthetic_check()
        self.assertTrue(result["noise_deterministic"])
        self.assertEqual(result["quantization"], [0.0, 0.0, 2.0, 4.0, 4.0])


if __name__ == "__main__":
    unittest.main()
