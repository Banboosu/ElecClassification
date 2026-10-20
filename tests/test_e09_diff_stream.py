from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tcn_moment.run_e09_diff_stream import _diff_arrays, _gate, _load


class E09PilotTests(unittest.TestCase):
    def test_diff_stream_is_masked_and_standardized(self):
        x = np.zeros((2, 8), dtype=np.float32)
        mask = np.zeros((2, 8), dtype=bool)
        x[0, :5] = [1.0, 2.0, 4.0, 8.0, 16.0]
        mask[0, :5] = True
        x[1, :6] = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        mask[1, :6] = True
        out, out_mask = _diff_arrays(x, mask)
        self.assertAlmostEqual(float(out[0, :4].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(out[0, :4].std()), 1.0, places=5)
        self.assertTrue(np.all(np.diff(out[0, :4]) > 0))
        self.assertEqual(int(out_mask[0].sum()), 4)
        self.assertEqual(int(out_mask[1].sum()), 5)
        np.testing.assert_allclose(out[1, :5], 0.0)

    def test_protocol_scope(self):
        protocol = _load(Path("configs/experiments/improvements/e09_diff_stream_pilot_protocol.json"))
        self.assertEqual(protocol["seed"], 42)
        self.assertEqual(protocol["fractions"], [0.1, 0.4])
        self.assertFalse(protocol["test_evaluated"])

    def test_gate_requires_both_checks(self):
        def row(method, tag, value):
            return {"method": method, "tag": tag, "validation": {"macro_f1": value}}

        conditions = [
            row("moment_logit", "10p", 0.80),
            row("moment_diff_logit", "10p", 0.82),
            row("stack_moment", "10p", 0.93),
            row("stack_moment_diff", "10p", 0.935),
            row("stack_three", "10p", 0.936),
            row("moment_logit", "40p", 0.84),
            row("moment_diff_logit", "40p", 0.86),
            row("stack_moment", "40p", 0.95),
            row("stack_moment_diff", "40p", 0.955),
            row("stack_three", "40p", 0.956),
        ]
        gate = _gate(conditions)
        self.assertTrue(gate["passed"])
        conditions[7]["validation"]["macro_f1"] = 0.96
        self.assertFalse(_gate(conditions)["passed"])


if __name__ == "__main__":
    unittest.main()
