from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_e09_expansion.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_e09_expansion", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class E09AnalysisTests(unittest.TestCase):
    def test_fraction_tags_are_distinct(self):
        module = _load_module()
        self.assertEqual(module._fraction_tag(0.01), "1p")
        self.assertEqual(module._fraction_tag(0.05), "5p")
        self.assertEqual(module._fraction_tag(0.1), "10p")
        self.assertEqual(module._fraction_tag(0.4), "40p")
        self.assertIsNone(module._fraction_tag(0.2))


if __name__ == "__main__":
    unittest.main()
