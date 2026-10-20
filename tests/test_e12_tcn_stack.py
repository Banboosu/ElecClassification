from __future__ import annotations

import unittest
from pathlib import Path

from tcn_moment import run_e12_tcn_stack as e12


class E12Tests(unittest.TestCase):
    def test_protocol_scope(self):
        protocol = e12._load(Path("configs/experiments/improvements/e12_tcn_stack_protocol.json"))
        self.assertEqual(protocol["seeds"], [42, 44, 45, 46])
        self.assertEqual(protocol["fractions"], [0.4])
        self.assertEqual(protocol["folds"], 5)
        self.assertFalse(protocol["test_evaluated"])

    def test_seed_protocol_layout(self):
        protocol = e12._load(Path("configs/experiments/improvements/e12_tcn_stack_protocol.json"))
        cp = e12._seed_protocol(protocol, 44)
        self.assertEqual(cp["seed"], 44)
        self.assertEqual(cp["split_manifest"], protocol["split_manifests"]["44"])


if __name__ == "__main__":
    unittest.main()
