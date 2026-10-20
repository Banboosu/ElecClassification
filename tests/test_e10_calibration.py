from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_e10_calibration_safety.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_e10_calibration_safety", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class E10CalibrationTests(unittest.TestCase):
    def test_outer_meta_oof_uses_heldout_fold(self):
        module = _load_module()
        rng = np.random.default_rng(7)
        y = np.repeat(np.arange(3), 10)
        folds = np.tile(np.arange(5), 6)
        oof = rng.normal(size=(30, 6))
        stack = module._outer_meta_oof(oof, y, folds, 1.0, 42)
        mask = folds == 0
        expected = LogisticRegression(C=1.0, max_iter=5000, random_state=42).fit(
            oof[~mask], y[~mask]
        ).predict_proba(oof[mask])
        np.testing.assert_allclose(stack[mask], expected, rtol=0, atol=1e-12)
        self.assertTrue(np.allclose(stack.sum(1), 1.0))

    def test_operating_points_report_prevalence_and_direction(self):
        module = _load_module()
        y = np.asarray([2] * 20 + [0] * 80)
        scores = np.concatenate([np.linspace(0.9, 0.5, 20), np.linspace(0.8, 0.0, 80)])
        result = module._operating_points(y, scores)
        self.assertAlmostEqual(result["prevalence"], 0.2)
        self.assertGreaterEqual(
            result["fixed_fpr"][-1]["recall"], result["fixed_fpr"][0]["recall"]
        )
        self.assertGreaterEqual(
            result["fixed_recall"][-1]["fpr"], result["fixed_recall"][0]["fpr"]
        )


if __name__ == "__main__":
    unittest.main()
