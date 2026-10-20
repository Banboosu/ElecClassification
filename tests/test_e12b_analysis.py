from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_e12b_fusion_ablation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_e12b_fusion_ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class E12bAnalysisTests(unittest.TestCase):
    def test_tcn_meta_control_is_present(self):
        module = _load_module()
        self.assertIn("tcn_meta", module.METHODS)
        self.assertIn("tcn", module.METHODS)

    def test_meta_uses_selected_c_for_fit(self):
        module = _load_module()
        rng = np.random.default_rng(11)
        y = np.repeat(np.arange(3), 10)
        folds = np.tile(np.arange(5), 6)
        oof = rng.normal(size=(30, 3))
        validation = rng.normal(size=(9, 3))
        probabilities, selected, _ = module._meta(oof, y, validation, seed=42)
        expected = LogisticRegression(
            C=selected["C"], max_iter=5000, random_state=42
        ).fit(oof, y).predict_proba(validation)
        np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
