from __future__ import annotations

import csv
import importlib.util
import statistics
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plot_legacy_tcn_moment_low_label.py"
SPEC = importlib.util.spec_from_file_location("legacy_tcn_moment_plot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LegacyTCNMomentPlotTests(unittest.TestCase):
    def records(self):
        # Deliberately reverse one method: pairing must use seed, not row position.
        return [
            {"model": model, "fraction": 0.01, "seed": seed,
             "macro_f1_percent": float(seed) + (10 if model == "moment" else 0)}
            for model in ("moment", "tcn")
            for seed in (MODULE.SEEDS if model == "moment" else reversed(MODULE.SEEDS))
        ]

    def test_seed_pairing_sample_sd_and_no_imputed_points(self):
        aggregate, paired = MODULE.summarize(self.records())
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired[0]["mean_pp"], 10)
        self.assertEqual(paired[0]["moment_wins"], 4)
        self.assertAlmostEqual(aggregate[0]["sample_sd_percent"], statistics.stdev(MODULE.SEEDS))
        self.assertNotIn(0.2, [row["fraction"] for row in aggregate])

    def test_mismatched_seed_population_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same four seeds"):
            MODULE.summarize(self.records()[:-1])
        rows = self.records()
        rows[0]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "same four seeds"):
            MODULE.summarize(rows)

    def test_duplicate_seed_is_rejected(self):
        rows = self.records()
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            MODULE.summarize(rows + [rows[0]])

    def test_validation_flag_is_required(self):
        MODULE.require_validation({"test_evaluated": False})
        for value in ({}, {"test_evaluated": True}, {"test_evaluated": None}):
            with self.assertRaises(ValueError):
                MODULE.require_validation(value)

    def test_prediction_audit_checks_truth_order_and_f1(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["sample_id", "true_index", "predicted_index",
                                 "true_label", "predicted_label"])
                writer.writerows([("a", 0, 0, 0, 0), ("b", 1, 1, 1, 1), ("c", 2, 2, 2, 2)])
            truth = {"a": 0, "b": 1, "c": 2}
            self.assertEqual(MODULE.score_predictions(path, ["a", "b", "c"], truth), (1.0, 3))
            with self.assertRaisesRegex(ValueError, "validation IDs"):
                MODULE.score_predictions(path, ["b", "a", "c"], truth)
            with self.assertRaisesRegex(ValueError, "raw truth"):
                MODULE.score_predictions(path, ["a", "b", "c"], {**truth, "c": 0})


if __name__ == "__main__":
    unittest.main()
