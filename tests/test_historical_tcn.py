import copy
import unittest

from tcn_moment import analyze_historical_tcn as audit


def rows(values):
    return [
        {"sample_id": str(i), "true_index": str(y), "predicted_index": str(p)}
        for i, (y, p) in enumerate(values)
    ]


class HistoricalTCNTests(unittest.TestCase):
    def test_confusion_metrics(self):
        result = audit.metrics(rows([(0, 0), (0, 2), (1, 1), (1, 1), (2, 2), (2, 0)]))
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["battery_recall"], 0.5)
        self.assertEqual(result["battery_fpr"], 0.25)

    def test_invalid_or_missing_classes_rejected(self):
        for value in ([], rows([(0, 0)]), rows([(0, 4), (1, 1), (2, 2)])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.metrics(value)

    def test_explicit_order_ids_and_truth_required(self):
        original = rows([(0, 0), (1, 1), (2, 2)])
        audit.match_rows(original, original)
        altered = copy.deepcopy(original)
        altered[0]["true_index"] = "1"
        for wrong in (altered, list(reversed(original)), original[:2]):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                audit.match_rows(wrong, original)

    def test_tied_average_precision(self):
        values = rows([(0, 0), (1, 1), (2, 2)])
        for r in values:
            r["battery_score"] = ".5"
        self.assertAlmostEqual(audit.average_precision(values, "battery_score"), 1 / 3)
        values[-1]["battery_score"] = ".9"
        self.assertEqual(audit.average_precision(values, "battery_score"), 1)

    def test_archived_metrics_cannot_drift(self):
        m = audit.metrics(rows([(0, 0), (1, 1), (2, 2)]))
        self.assertEqual(audit.compare_metrics(m, m), 0)
        altered = copy.deepcopy(m)
        altered["macro_f1"] = 0.99
        with self.assertRaises(ValueError):
            audit.compare_metrics(m, altered)

    def test_scope_is_retrospective_and_no_inference(self):
        p = audit.read(audit.DEFAULT_PROTOCOL)
        audit.validate_protocol(p)
        p["model_inference_allowed"] = True
        with self.assertRaises(ValueError):
            audit.validate_protocol(p)
        with self.assertRaises(ValueError):
            audit.describe([1, 2, 3])
        self.assertEqual(audit.describe([1, 1, 1, 1])["mean"], 1)


if __name__ == "__main__":
    unittest.main()
