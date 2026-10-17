from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tcn_moment.analyze_e01c import DEFAULT_PROTOCOL, evaluate_differences, read_protocol
from tcn_moment.train_e01c import condition_arrays, forbid_pretrained_loading


class E01CControlsTests(unittest.TestCase):
    def setUp(self):
        self.protocol = read_protocol(Path(DEFAULT_PROTOCOL))

    def test_scope_has_sixteen_controls_and_no_test_evaluation(self):
        p = self.protocol
        self.assertEqual(len(p["seeds"]) * len(p["fractions"]) * len(p["conditions"]), 16)
        self.assertNotIn(42, p["seeds"])
        self.assertFalse(p["test_evaluated"])
        self.assertEqual(p["fixed_search"]["gamma_tokens"], ["reference_scale", 0.05, 5.0])

    def test_two_attribution_claims_keep_separate_practical_thresholds(self):
        values = [0.6, 0.6, 0.6, 0.6]
        self.assertTrue(
            evaluate_differences(values, self.protocol, "pretrained_vs_random")["rule_passed"]
        )
        self.assertFalse(
            evaluate_differences(values, self.protocol, "beyond_statistics")["rule_passed"]
        )
        with self.assertRaises(ValueError):
            evaluate_differences([1, 2], self.protocol, "beyond_statistics")

    def test_random_condition_cannot_accidentally_use_pretrained_columns(self):
        pretrained = np.ones((3, 1024), dtype=np.float32)
        random = np.zeros_like(pretrained)
        statistics = np.arange(45, dtype=np.float32).reshape(3, 15)
        reference, controls = condition_arrays(pretrained, random, statistics)
        np.testing.assert_array_equal(reference[:, :1024], pretrained)
        np.testing.assert_array_equal(controls["random_fusion_matched_kernel"][:, :1024], random)
        np.testing.assert_array_equal(
            controls["random_fusion_matched_kernel"][:, 1024:], statistics
        )
        np.testing.assert_array_equal(controls["statistics_matched_kernel"], statistics)
        with self.assertRaises(ValueError):
            condition_arrays(pretrained, random[:, :-1], statistics)

    def test_random_extraction_guard_blocks_all_standard_weight_loaders(self):
        from momentfm import MOMENTPipeline
        from transformers import T5EncoderModel, T5Model

        original = MOMENTPipeline.from_pretrained.__func__
        with forbid_pretrained_loading():
            for target in (MOMENTPipeline, T5Model, T5EncoderModel):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    target.from_pretrained("must-not-download")
        self.assertIs(MOMENTPipeline.from_pretrained.__func__, original)


if __name__ == "__main__":
    unittest.main()
