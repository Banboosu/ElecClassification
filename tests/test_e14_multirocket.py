from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tcn_moment.run_e14_multirocket import (
    _interp_z,
    _load,
    _pad_raw,
    _softmax,
    synthetic_check,
)


class E14MultiRocketTests(unittest.TestCase):
    def test_protocol_is_validation_only(self):
        protocol = _load(Path("configs/experiments/improvements/e14_multirocket_fast_protocol.json"))
        self.assertEqual(protocol["experiment_id"], "E14")
        self.assertFalse(protocol["test_evaluated"])
        self.assertFalse(protocol["automatic_expansion"])
        self.assertEqual(set(protocol["variants"]), {"interp_z"})
        self.assertEqual(protocol["num_features"], 2000)
        self.assertEqual(protocol["ridge_alphas"], [100.0, 1000.0])
        self.assertEqual(protocol["ridge_cv"], 3)

    def test_preprocessing_variants(self):
        x = np.zeros((2, 8), dtype=np.float32)
        mask = np.zeros((2, 8), dtype=bool)
        x[0, :4] = [1, 2, 3, 4]
        x[1, :4] = [5, 5, 5, 5]
        mask[0, :4] = True
        mask[1, :4] = True
        interp = _interp_z(x, mask, 8)
        pad = _pad_raw(x, mask, 8)
        self.assertEqual(interp.shape, (2, 8))
        self.assertEqual(pad.shape, (2, 8))
        np.testing.assert_allclose(pad[0, :4], [1, 2, 3, 4])
        self.assertTrue(np.all(pad[0, 4:] == 0))
        self.assertTrue(np.allclose(interp[1], 0.0))

    def test_softmax_is_normalized(self):
        scores = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        probabilities = _softmax(scores)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.all(probabilities >= 0))

    def test_synthetic_check(self):
        result = synthetic_check()
        self.assertEqual(result["interp_shape"], [3, 16])
        self.assertEqual(result["pad_shape"], [3, 16])


if __name__ == "__main__":
    unittest.main()
