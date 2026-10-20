"""Synthetic statistics and audit-proof guards; no experimental training/data needed."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    spec = importlib.util.spec_from_file_location("e07_report", SCRIPTS / "report_e07_completion.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)


class CompletionReportTests(unittest.TestCase):
    def test_sample_standard_deviation_not_population(self):
        row = report.stats([1., 2., 3., 4.])
        self.assertEqual(row["mean"], 2.5)
        self.assertAlmostEqual(row["sample_sd"], (5 / 3) ** .5)
        self.assertEqual(row["per_seed"], [1., 2., 3., 4.])

    def test_direction_is_strictly_positive_and_ci_is_descriptive(self):
        row = report.paired([-1., 0., 1., 2.])
        self.assertEqual(row["positive_seeds"], 2)
        self.assertEqual(row["mean"], .5)
        self.assertLess(row["descriptive_ci95"][0], 0)
        self.assertAlmostEqual(sum(row["descriptive_ci95"]) / 2, .5)
        self.assertNotIn("rule_passed", row)

    def test_rejects_incomplete_mismatched_or_test_audit_proofs(self):
        names = ("independent_output_audit.json", "independent_reference_audit.json", "checkpoint_audit.json")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "terminal_artifacts").mkdir()
            file = base / "terminal_artifacts/summary.json"
            file.write_text("{}")
            good = {"status": "passed", "summary_sha256": report.digest(file), "test_evaluated": False}
            for target in names:
                for key, value in (("status", "failed"), ("summary_sha256", "changed"), ("test_evaluated", True)):
                    for name in names:
                        (base / name).write_text(json.dumps(good))
                    (base / target).write_text(json.dumps({**good, key: value}))
                    with self.subTest(target=target, key=key), self.assertRaisesRegex(ValueError, "audit proofs"):
                        report.analyze(base)

    def test_rejects_incomplete_preservation_even_if_metric_audit_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "terminal_artifacts").mkdir()
            file = base / "terminal_artifacts/summary.json"
            file.write_text("{}")
            good = {"status": "passed", "summary_sha256": report.digest(file), "test_evaluated": False}
            for name in ("independent_output_audit.json", "independent_reference_audit.json", "checkpoint_audit.json"):
                (base / name).write_text(json.dumps(good))
            for status, mismatches in (("pending", 0), ("completed_verified", 1)):
                (base / "preservation_verified.json").write_text(json.dumps({"status": status, "sha256_mismatches": mismatches}))
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, "preservation"):
                    report.analyze(base)


if __name__ == "__main__":
    unittest.main()
