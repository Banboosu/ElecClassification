from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_e12b_fusion_ablation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_e12b_fusion_ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class E12bTests(unittest.TestCase):
    def test_align_reorders_by_reference_ids(self):
        module = _load_module()
        probs = np.asarray([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
        aligned = module._align(["b", "a"], ["a", "b"], probs)
        np.testing.assert_allclose(aligned, probs[::-1])

    def test_align_rejects_different_id_sets(self):
        module = _load_module()
        with self.assertRaises(ValueError):
            module._align(["a", "b"], ["a", "c"], np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
