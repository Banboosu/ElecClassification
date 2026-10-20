from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from tcn_moment.run_e16_random_attribution import _load, _normalize_original, synthetic_check


class E16RandomAttributionTests(unittest.TestCase):
    def test_protocol_is_validation_only(self):
        protocol = _load(Path("configs/experiments/improvements/e16_random_attribution_protocol.json"))
        self.assertEqual(protocol["experiment_id"], "E16")
        self.assertFalse(protocol["test_evaluated"])
        self.assertFalse(protocol["automatic_expansion"])
        self.assertEqual(tuple(protocol["initializations"]), ("pretrained", "random"))
        self.assertEqual(tuple(protocol["streams"]), ("orig", "diff", "dual"))

    def test_original_normalization_uses_valid_values(self):
        x = np.zeros((2, 8), dtype=np.float32)
        mask = np.zeros((2, 8), dtype=bool)
        x[0, :4] = [1, 2, 3, 4]
        x[1, :4] = [5, 5, 5, 5]
        mask[0, :4] = True
        mask[1, :4] = True
        out = _normalize_original(x, mask, 8)
        self.assertAlmostEqual(float(out[0, :4].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(out[0, :4].std()), 1.0, places=6)
        self.assertTrue(np.all(out[1] == 0))
        self.assertTrue(np.all(out[0, 4:] == 0))

    def test_synthetic_check(self):
        result = synthetic_check()
        self.assertGreater(result["diff_nonzero"], 0)
        self.assertGreater(result["diff_mask"], 0)


if __name__ == "__main__":
    unittest.main()
