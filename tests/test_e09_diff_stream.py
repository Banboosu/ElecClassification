from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tcn_moment.run_e09_diff_stream import _diff_arrays, _gate, _load, _stack_three


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


    def test_stack_three_oof_blocks_do_not_overlap(self):
        recorded = []

        class DummyModel:
            def __init__(self, kind):
                self.kind = kind

            def fit(self, x, y):
                return self

            def predict_proba(self, x):
                if self.kind == "hgb":
                    out = np.zeros((len(x), 3))
                    out[:, 2] = 1.0
                    return out
                return np.eye(3)[x[:, 0].astype(int)]

            def predict(self, x):
                return self.predict_proba(x).argmax(1)

        class RecordingMeta(DummyModel):
            def __init__(self, *args, **kwargs):
                super().__init__("meta")

            def fit(self, x, y):
                recorded.append(np.asarray(x))
                return self

        x_train = np.zeros((15, 2), dtype=np.float32)
        y_train = np.repeat(np.arange(3), 5)
        moment_train = np.zeros((15, 2), dtype=np.float32)
        diff_train = np.ones((15, 2), dtype=np.float32)
        x_val = np.zeros((3, 2), dtype=np.float32)
        moment_val = np.zeros((3, 2), dtype=np.float32)
        diff_val = np.ones((3, 2), dtype=np.float32)
        factory = lambda kind: (lambda params, seed: DummyModel(kind))
        with mock.patch("tcn_moment.run_e08._hgb_factory", factory("hgb")), \
             mock.patch("tcn_moment.run_e08._logit_factory", factory("logit")), \
             mock.patch("tcn_moment.run_e08._cv_select", return_value=({"C": 1.0}, [])), \
             mock.patch("sklearn.linear_model.LogisticRegression", RecordingMeta):
            _stack_three(
                x_train, y_train, x_val,
                moment_train, moment_val, diff_train, diff_val,
                hgb_params={}, seed=42,
            )
        self.assertEqual(len(recorded), 1)
        oof = recorded[0]
        self.assertEqual(oof.shape, (15, 9))
        np.testing.assert_allclose(oof[:, :3], np.tile([0, 0, 1], (15, 1)))
        np.testing.assert_allclose(oof[:, 3:6], np.tile([1, 0, 0], (15, 1)))
        np.testing.assert_allclose(oof[:, 6:9], np.tile([0, 1, 0], (15, 1)))


if __name__ == "__main__":
    unittest.main()
