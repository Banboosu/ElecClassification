from __future__ import annotations

import unittest
from pathlib import Path

from tcn_moment import run_n2b_lr_sweep as n2b


class N2bTests(unittest.TestCase):
    def test_protocol_scope(self):
        protocol = n2b._load(Path("configs/experiments/improvements/n2b_lr_sweep_protocol.json"))
        self.assertEqual(protocol["learning_rates"], [0.0003, 0.001, 0.003])
        self.assertEqual(protocol["train_fractions"], [0.01, 0.05])
        self.assertFalse(protocol["test_evaluated"])

    def test_case_layout(self):
        root = Path("suite")
        path = n2b._case_dir(root, 0.0003, 44, 0.01)
        self.assertEqual(path, Path("suite/lr0.0003/seed44/fraction001_legacy_masked50"))
        other = n2b._case_dir(root, 0.003, 46, 0.05)
        self.assertEqual(other, Path("suite/lr0.003/seed46/fraction005_legacy_masked50"))


if __name__ == "__main__":
    unittest.main()
