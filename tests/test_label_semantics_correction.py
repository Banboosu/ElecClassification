"""Regression checks for the user-confirmed 1=battery, 2=charger mapping."""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tcn_moment.label_semantics import (
    BATTERY_LABEL, CHARGER_LABEL, LABEL_NAMES_ZH, class_index,
)

SPEC = importlib.util.spec_from_file_location(
    "correct_label_semantics", Path(__file__).resolve().parents[1] / "scripts/correct_label_semantics.py"
)
assert SPEC and SPEC.loader
correction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(correction)


class LabelSemanticsCorrectionTests(unittest.TestCase):
    def test_user_confirmed_mapping_and_nonstandard_encoder_order(self):
        self.assertEqual(LABEL_NAMES_ZH, {"0": "正常", "1": "电池异常", "2": "充电器故障"})
        self.assertEqual(BATTERY_LABEL, "1")
        self.assertEqual(CHARGER_LABEL, "2")
        self.assertEqual(class_index(["2", "0", "1"], BATTERY_LABEL), 2)
        with self.assertRaises(ValueError):
            class_index(["0", "2"], BATTERY_LABEL)
        with self.assertRaises(ValueError):
            class_index(["0", "1", "1"], BATTERY_LABEL)

    def test_battery_and_charger_not_interchangeable(self):
        y = [0, 1, 2, 2, 0, 1]
        scores = [[.9, .05, .05], [.1, .8, .1], [.1, .1, .8],
                  [.7, .1, .2], [.1, .1, .8], [.1, .7, .2]]
        pred = [0, 1, 2, 0, 2, 1]
        result = correction.evaluate(y, pred, scores)
        self.assertEqual(result["battery"]["recall"], 1.)
        self.assertEqual(result["battery"]["fpr"], 0.)
        self.assertEqual(result["charger"]["recall"], .5)
        self.assertEqual(result["charger"]["fpr"], .25)
        self.assertAlmostEqual(result["macro_f1"], 2/3)

    def test_global_label_permutation_preserves_global_metrics(self):
        y, pred = [0, 1, 2, 1, 2], [1, 1, 2, 0, 1]
        scores = [[.1, .8, .1], [.2, .7, .1], [.1, .1, .8], [.8, .1, .1], [.1, .7, .2]]
        first = correction.evaluate(y, pred, scores)
        swap = {0: 0, 1: 2, 2: 1}
        second = correction.evaluate(
            [swap[c] for c in y], [swap[c] for c in pred],
            [[row[0], row[2], row[1]] for row in scores],
        )
        self.assertEqual(first["macro_f1"], second["macro_f1"])
        self.assertEqual(first["accuracy"], second["accuracy"])

    def test_average_precision_groups_ties(self):
        self.assertEqual(correction.average_precision([1, 0, 1, 2], [.5]*4, 1), .5)
        self.assertEqual(correction.average_precision([1, 0], [1., 0.], 1), 1.)
        self.assertEqual(correction.average_precision([1, 0], [0., 1.], 1), .5)
        self.assertEqual(correction.average_precision([0, 2], [.9, .1], 1), 0.)

    def test_validation_only_reader_checks_ids_truth_scores_and_argmax(self):
        fields = ["sample_id", "split", "y_true", "y_pred", "score_0", "score_1", "score_2", "score_battery"]
        good = [["001", "validation", 0, 0, .8, .1, .1, .1],
                ["002", "validation", 1, 1, .1, .8, .1, .1]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            def save(rows):
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                    writer.writerows(rows)
            save(good)
            y, pred, scores = correction.read_validation(path, ["001", "002"], {"001": 0, "002": 1})
            self.assertEqual(y, pred)
            # Historical score_battery=score_2 is intentionally ignored.
            self.assertEqual(correction.evaluate(y, pred, scores)["battery"]["average_precision"], 1.)
            bad_cases = [good[::-1], [good[0], good[0]]]
            for col, value in ((1, "test"), (2, 2), (3, 2), (4, "nan"), (4, -.1), (4, .2)):
                rows = [r.copy() for r in good]
                rows[0][col] = value
                bad_cases.append(rows)
            for rows in bad_cases:
                save(rows)
                with self.assertRaises(ValueError):
                    correction.read_validation(path, ["001", "002"], {"001": 0, "002": 1})

    def test_raw_svc_decision_scores_are_not_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            path.write_text("sample_id,split,y_true,y_pred,score_0,score_1,score_2\n"
                            "001,validation,1,2,-1,2,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                correction.read_validation(path, ["001"], {"001": 1})
            y, pred, scores = correction.read_validation(path, ["001"], {"001": 1}, score_type="decision_function")
            self.assertEqual(pred, [2])  # SVC votes are not forced to score argmax.
            self.assertEqual(scores, [[-1., 2., 1.]])
            self.assertEqual(y, [1])

    def test_expanded_scope_preserves_failed_and_repaired_versions(self):
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root / "configs/experiments/improvements/label_semantics_correction_expanded_20261029.json").read_text())
        self.assertEqual(sum(s["expected_conditions"] for s in protocol["inputs"]), 1496)
        sources = {s["name"]: s for s in protocol["inputs"]}
        self.assertEqual(sources["e09"]["exclude_regex"], "/stack_three/")
        self.assertEqual(sources["e09_stackfix"]["expected_conditions"], 12)
        self.assertNotEqual(sources["e13_stress"]["include_regex"], sources["e13_fine"]["include_regex"])
        self.assertFalse(protocol["test_evaluated"])
        self.assertFalse(protocol["retrain"])

    def test_current_battery_clis_require_label_one_and_explicit_test_permission(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("evaluate_battery_safety.py", "train_battery_binary.py", "evaluate_error_analysis.py"):
            text = (root / "src/tcn_moment" / name).read_text()
            tree = ast.parse(text)
            arguments = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
            critical = next(node for node in arguments if node.args and isinstance(node.args[0], ast.Constant)
                            and node.args[0].value == "--critical-label")
            default = next(k.value for k in critical.keywords if k.arg == "default")
            self.assertIsInstance(default, ast.Name)
            self.assertEqual(default.id, "BATTERY_LABEL")
            self.assertIn("--allow-test-evaluation", text)
            self.assertIn("if not args.allow_test_evaluation:", text)

    def test_class_one_descriptive_working_points_and_quantiles(self):
        path = Path(__file__).resolve().parents[1] / "scripts/reassess_e10_battery_semantics.py"
        spec = importlib.util.spec_from_file_location("e10_battery_reassessment", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(module)
        self.assertEqual(module.quantile([0., 1., 2., 3.], .5), 1.5)
        self.assertEqual(module.quantile([1., 1., 1.], .9), 1.)
        y = [1] * 20 + [2] * 80
        scores = [.9] * 20 + [.1] * 80
        result = module.working_points(y, scores)
        self.assertEqual(result["prevalence"], .2)
        self.assertAlmostEqual(result["perfect_ranking_fpr_at_50pct_alert"], .375)
        self.assertEqual(result["alert_curve"][-1]["recall"], 1.)
        self.assertEqual(result["alert_curve"][-1]["fpr"], .375)
        with self.assertRaises(ValueError):
            module.working_points([2, 2], [.1, .9])

    def test_missing_or_duplicate_seeds_not_silently_averaged(self):
        with self.assertRaises(ValueError):
            correction.aggregate([{"suite": "e18", "tag": "40p", "method": "full", "seed": 42}], [42, 44, 45, 46])


if __name__ == "__main__":
    unittest.main()
