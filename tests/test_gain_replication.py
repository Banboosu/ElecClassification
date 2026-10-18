from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_moment.data import prepare_sequence, sequence_mask
from tcn_moment.replicate_gain_heads import (
    DEFAULT_PROTOCOL,
    decision,
    fusion_run,
    legacy_run,
    matched_run,
    new_head,
    normalize_masked_rows,
    read_protocol,
)
from tcn_moment.run_e03d_stable import original_head_digest


class GainReplicationTests(unittest.TestCase):
    def setUp(self):
        self.p = read_protocol(Path(DEFAULT_PROTOCOL))

    def test_pilot_excluded_and_twelve_heads(self):
        self.assertNotIn(43, self.p["replication_seeds"])
        self.assertEqual(len(self.p["replication_seeds"]) * len(self.p["branches"]), 12)
        self.assertFalse(self.p["success_rules"]["automatically_evaluate_test"])

    def test_masked_normalization_exactly_matches_original_preprocessing(self):
        for sequence in ([2, 4, 3], [5, 5, 5, 5], list(range(20))):
            raw = prepare_sequence(sequence, 8, "none")[None]
            mask = sequence_mask(len(sequence), 8)[None]
            actual = normalize_masked_rows(raw, mask)[0]
            np.testing.assert_array_equal(actual, prepare_sequence(sequence, 8, "zscore"))
        bad = np.array([[1, 0, 1, 0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            normalize_masked_rows(bad, bad)

    def test_seed_specific_parameters_same_across_branches_and_different_across_seeds(self):
        norm = {"mean": [0] * 15, "scale": [1] * 15}
        digests = {original_head_digest(new_head(self.p, 44, b, norm)) for b in self.p["branches"]}
        self.assertEqual(len(digests), 1)
        self.assertNotEqual(
            next(iter(digests)), original_head_digest(new_head(self.p, 45, "fusion", norm))
        )
        self.assertIsInstance(new_head(self.p, 42, "fusion", norm).input_gain, torch.Tensor)

    def test_discovery_seed42_uses_correct_parent_and_new_matched_control(self):
        self.assertIn("v2_pilot_seed42", str(fusion_run(42)))
        self.assertIn("v2_expansion_seed44", str(legacy_run(44)))
        self.assertIn("e01b_high_label", str(fusion_run(44)))
        self.assertIn("matched_statistics_seed42", str(matched_run(self.p, 42)))
        self.assertIn("statistics_matched_kernel_fraction_40", str(matched_run(self.p, 44)))

    def test_frozen_mean_and_direction_rules(self):
        self.assertTrue(decision([0.8, 0.8, 0.8, -0.1], self.p, 0.5)["rule_passed"])
        self.assertFalse(decision([0.49] * 4, self.p, 0.5)["rule_passed"])
        self.assertFalse(decision([2, 2, -0.1, -0.1], self.p, 0.5)["rule_passed"])
        with self.assertRaises(ValueError):
            decision([1, 1, 1], self.p, 0.5)


if __name__ == "__main__":
    unittest.main()
