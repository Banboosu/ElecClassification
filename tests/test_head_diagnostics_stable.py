from __future__ import annotations

import unittest
from pathlib import Path

import torch

from tcn_moment.head_diagnostics_stable import FixedHeadGradScaler
from tcn_moment.run_e03d_stable import DEFAULT_PROTOCOL, cases, read_protocol


class StableHeadDiagnosticTests(unittest.TestCase):
    def test_all_six_restart_fresh_and_keep_budgets_and_gates(self):
        p = read_protocol(Path(DEFAULT_PROTOCOL))
        self.assertEqual(len(cases(p)), 6)
        self.assertEqual(p["modes"], ["legacy_fresh", "fanin_gain"])
        self.assertEqual(p["budgets"], [30, 200])
        self.assertEqual(p["candidate_rules"]["minimum_gain_over_strongest_reference_pp"], 0.5)
        self.assertIn("Fixed1024", p["training"]["loss_scale_policy"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA fixed-scale integration")
    def test_scale_stays_fixed_and_rejects_old_dynamic_scale(self):
        scaler = FixedHeadGradScaler("cuda", init_scale=9999, growth_interval=1)
        p = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
        optimizer = torch.optim.SGD([p], lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            scaler.scale(p.square().sum()).backward()
            scaler.step(optimizer)
            scaler.update()
            self.assertEqual(scaler.get_scale(), 1024)
        state = scaler.state_dict()
        restored = FixedHeadGradScaler("cuda")
        restored.load_state_dict(state)
        self.assertEqual(restored.get_scale(), 1024)
        state["scale"] = 32768
        with self.assertRaises(ValueError):
            restored.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
