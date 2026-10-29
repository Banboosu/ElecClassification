"""Never execute user-rejected legacy code, or use it as a live dependency."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rejected_legacy_audit", ROOT / "scripts/audit_rejected_legacy_sources.py")
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
REJECTED = ["legacy/ceshi.py", "legacy/画曲线测试.py"]


class RejectedLegacySourceTests(unittest.TestCase):
    def test_references_detect_import_and_dynamic_file_path(self):
        for text in ("import ceshi", "from legacy import ceshi", "import legacy.ceshi",
                     'importlib.util.spec_from_file_location("x", "legacy/ceshi.py")',
                     'exec(open("legacy/画曲线测试.py").read())'):
            self.assertTrue(AUDIT.python_references(text, REJECTED), text)
        self.assertFalse(AUDIT.python_references('Path("legacy/改进TCN.py")', REJECTED))

    def test_audit_reads_but_does_not_change_or_import_rejected_sources(self):
        import sys
        before = {name: AUDIT.sha(ROOT / name) for name in REJECTED}
        modules_before = set(sys.modules)
        result = AUDIT.audit(ROOT, include_artifacts=False)
        self.assertEqual(before, {name: AUDIT.sha(ROOT / name) for name in REJECTED})
        self.assertFalse(result["legacy_executed"])
        self.assertFalse(result["models_run"])
        self.assertFalse(result["test_scored"])
        self.assertFalse(set(sys.modules).difference(modules_before).intersection({"ceshi", "legacy.ceshi", "画曲线测试"}))
        self.assertEqual(result["active_named_references"], [])
        self.assertEqual(result["distinctive_api_or_output_hits"], [])
        self.assertEqual(result["exact_nontrivial_function_copies"], [])
        self.assertEqual(result["exact_top_level_plot_blocks"], [])

    def test_other_plot_draft_is_a_similarity_lead_not_an_independent_source(self):
        rejected = (ROOT / "legacy/画曲线测试.py").read_text(encoding="utf-8")
        related = (ROOT / "legacy/画曲线.py").read_text(encoding="utf-8")
        signatures = {fingerprint for _, fingerprint in AUDIT.plot_blocks(rejected)}
        self.assertTrue(any(fp in signatures for _, fp in AUDIT.plot_blocks(related)))


if __name__ == "__main__":
    unittest.main()
