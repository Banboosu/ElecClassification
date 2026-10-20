from __future__ import annotations

import unittest
from pathlib import Path

from tcn_moment import run_n2_tcn_low_label as n2


class N2ProtocolTests(unittest.TestCase):
    def test_protocol_is_validation_only(self):
        protocol = n2._load_protocol(Path("configs/experiments/improvements/n2_tcn_low_label_protocol.json"))
        self.assertEqual(protocol["seeds"], [42, 44, 45, 46])
        self.assertEqual(protocol["train_fractions"], [0.01, 0.05])
        self.assertEqual(protocol["condition"], "legacy_masked50")
        self.assertFalse(protocol["test_evaluated"])
        self.assertFalse(protocol["automatic_expansion"])

    def test_selection_rule_matches_e05b(self):
        history = [
            {"epoch": 1, "macro_f1": 0.60},
            {"epoch": 2, "macro_f1": 0.601},
            {"epoch": 3, "macro_f1": 0.6005},
            {"epoch": 4, "macro_f1": 0.602},
        ]
        epoch, score = n2._selected_epoch(history, 0.0001)
        self.assertEqual(epoch, 4)
        self.assertAlmostEqual(score, 0.602)

    def test_seed_layout_matches_e05c(self):
        protocol = n2._load_protocol(Path("configs/experiments/improvements/n2_tcn_low_label_protocol.json"))
        cp = n2._seed_protocol(protocol, 44)
        self.assertEqual(cp["seed"], 44)
        self.assertTrue(cp["suite"] == "seed44")
        self.assertTrue(cp["output_dir"].endswith("n2_tcn_low_label_20261020_v1"))
        self.assertEqual(cp["split_manifest"], protocol["split_manifests"]["44"])

    def test_case_names(self):
        self.assertEqual(n2._case(0.01), "fraction001_legacy_masked50")
        self.assertEqual(n2._case(0.05), "fraction005_legacy_masked50")


if __name__ == "__main__":
    unittest.main()
