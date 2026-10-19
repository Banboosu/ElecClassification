import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tcn_moment import final_test as final


class FinalTestTests(unittest.TestCase):
    def setUp(self):
        self.p = final.read_protocol(Path(final.DEFAULT_PROTOCOL))

    def test_fixed_scope_and_no_fitting(self):
        self.assertEqual(self.p["seeds"], [42, 44, 45, 46])
        self.assertEqual(len(self.p["methods"]) * len(self.p["seeds"]), 28)
        self.assertFalse(self.p["training_allowed"])
        self.assertFalse(self.p["tuning_allowed"])
        self.assertFalse(self.p["automatic_shutdown_allowed"])

    def test_changed_scope_and_gates_rejected(self):
        for key, value in [
            ("seeds", [42, 43, 44, 45, 46]),
            ("training_allowed", True),
            ("train_fraction", 1.0),
            ("methods", ["fusion_nn"]),
        ]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                p = copy.deepcopy(self.p)
                p[key] = value
                f = Path(tmp) / "p.json"
                f.write_text(json.dumps(p))
                with self.assertRaises(ValueError):
                    final.read_protocol(f)
        with tempfile.TemporaryDirectory() as tmp:
            p = copy.deepcopy(self.p)
            p["success_rules"]["primary_minimum_mean_difference_pp"] = 0
            f = Path(tmp) / "p.json"
            f.write_text(json.dumps(p))
            with self.assertRaises(ValueError):
                final.read_protocol(f)

    def test_decision_needs_all_seeds_and_strict_direction(self):
        self.assertTrue(final.decision([0.5, 0.5, 0.5, 0.5], 0.5)["passed"])
        self.assertFalse(final.decision([1, 1, 0, 0], 0.5)["passed"])
        self.assertFalse(final.decision([0.49, 0.49, 0.49, 0.49], 0.5)["passed"])
        with self.assertRaises(ValueError):
            final.decision([1, 1, 1], 0.5)
        with self.assertRaises(ValueError):
            final.decision([1, 1, 1, np.nan], 0.5)

    def test_probabilities_cannot_be_fabricated_or_misaligned(self):
        good = np.eye(3)
        final.validate_probabilities(good, 3)
        for bad in [good[:, :2], good * 2, np.full((3, 3), np.nan)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                final.validate_probabilities(bad, 3)
        with self.assertRaises(ValueError):
            final.result_metrics(np.arange(3), np.array([1, 0, 2]), good, ["0", "1", "2"])

    def test_argmax_battery_metrics_and_optional_ap(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 2, 1, 1, 2, 0])
        m = final.result_metrics(truth, pred, None, ["0", "1", "2"])
        self.assertEqual(m["battery_argmax_recall"], 0.5)
        self.assertEqual(m["battery_argmax_fpr"], 0.25)
        self.assertIsNone(m["battery_average_precision"])
        m = final.result_metrics(truth, truth, np.eye(3)[truth], ["0", "1", "2"])
        self.assertEqual(m["battery_average_precision"], 1)

    def test_invalid_class_predictions_rejected(self):
        with self.assertRaises(ValueError):
            final.result_metrics([0, 1, 2], [0, 1, 5], None, ["0", "1", "2"])

    def test_nested_report_audit_catches_class_report_tampering(self):
        m = final.result_metrics([0, 1, 2], [0, 1, 2], None, ["0", "1", "2"])
        final.assert_metrics_equal(m, copy.deepcopy(m))
        altered = copy.deepcopy(m)
        altered["classification_report"]["2"]["recall"] = 0.9
        with self.assertRaises(ValueError):
            final.assert_metrics_equal(m, altered)
        altered = copy.deepcopy(m)
        altered["macro_f1"] = float("nan")
        with self.assertRaises(ValueError):
            final.assert_metrics_equal(m, altered)

    def test_baseline_inference_has_no_fit_and_correct_branch_input(self):
        class Pipeline:
            classes_ = np.arange(3)

            def fit(self, *args):
                raise AssertionError("No fitting allowed")

            def predict(self, x):
                self.last_shape = x.shape
                return np.zeros(len(x), dtype=int)

            def predict_proba(self, x):
                return np.tile([1.0, 0.0, 0.0], (len(x), 1))

        model = Pipeline()
        features = np.zeros((6, 1024))
        stats = np.zeros((6, 15))
        final.pipeline_predict(model, "fusion_svm", features, stats)
        self.assertEqual(model.last_shape, (6, 1039))
        final.pipeline_predict(model, "matched_statistics_svm", features, stats)
        self.assertEqual(model.last_shape, (6, 15))
        _, probs = final.pipeline_predict(model, "statistics_rf", features, stats)
        self.assertEqual(probs.shape, (6, 3))
        model.classes_ = np.array([2, 1, 0])
        with self.assertRaises(ValueError):
            final.pipeline_predict(model, "statistics_svm", features, stats)

    def test_preflight_global_barrier_blocks_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = copy.deepcopy(self.p)
            p["output_dir"] = tmp
            root = final.directory(p)
            root.mkdir()
            (root / "preflight.json").write_text(json.dumps({"status": "failed"}))
            with patch.object(
                final, "encode", side_effect=AssertionError("test must not be encoded")
            ) as encoder:
                with self.assertRaises(ValueError):
                    final.make_test_cache(p, {"protocol_sha256": "test"}, 42)
                encoder.assert_not_called()

    def test_parity_reports_hardware_drift_without_selection(self):
        old = pd.DataFrame(
            {
                "predicted_index": [0, 1, 2],
                "probability_0": [1.0, 0.0, 0.0],
                "probability_1": [0.0, 1.0, 0.0],
                "probability_2": [0.0, 0.0, 1.0],
            }
        )
        r = final.parity(old, np.arange(3), np.eye(3))
        self.assertEqual(r["prediction_agreement"], 1)
        self.assertEqual(r["max_probability_abs_error"], 0)
        with self.assertRaises(ValueError):
            final.parity(old, [0, 1])


if __name__ == "__main__":
    unittest.main()
