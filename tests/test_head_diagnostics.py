from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_moment.head_diagnostics import (
    GainFusionHead,
    add_gain_to_state,
    clone_parent_checkpoint,
    fan_in_gain,
    prefix_fixed_schedule,
    training_hooks,
)
from tcn_moment.moment_peft import FusionHead, PeftClassifier, warmup_cosine
from tcn_moment.run_e03d import DEFAULT_PROTOCOL, cases, judge, read_protocol


class HeadDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_gain_one_is_identical_to_parent_for_every_branch(self):
        for branch in ("statistics", "moment", "fusion"):
            original = FusionHead(
                [0] * 3, [1] * 3, branch=branch, moment_dim=4, stats_dim=3, hidden_dim=8, dropout=0
            )
            new = GainFusionHead(
                [0] * 3, [1] * 3, branch=branch, moment_dim=4, stats_dim=3, hidden_dim=8, dropout=0
            )
            new.load_state_dict({**original.state_dict(), "input_gain": torch.tensor(1.0)})
            x, s = torch.randn(5, 4), torch.randn(5, 3)
            torch.testing.assert_close(original(x, s), new(x, s), rtol=0, atol=0)

    def test_gain_preserves_relative_branch_balance_and_is_serialized(self):
        self.assertAlmostEqual(fan_in_gain("fusion"), math.sqrt(1039 / 2))
        self.assertAlmostEqual(fan_in_gain("moment"), math.sqrt(1039))
        head = GainFusionHead(
            [0] * 3,
            [1] * 3,
            branch="fusion",
            moment_dim=4,
            stats_dim=3,
            hidden_dim=8,
            dropout=0,
            input_gain=3,
        )
        base = GainFusionHead(
            [0] * 3, [1] * 3, branch="fusion", moment_dim=4, stats_dim=3, hidden_dim=8, dropout=0
        )
        base.load_state_dict({**head.state_dict(), "input_gain": torch.tensor(1.0)})
        captured = []
        hook = head.network[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        )
        x, s = torch.randn(6, 4), torch.randn(6, 3)
        head(x, s)
        hook.remove()
        other = []
        hook = base.network[0].register_forward_pre_hook(lambda module, args: other.append(args[0]))
        base(x, s)
        hook.remove()
        torch.testing.assert_close(captured[0], other[0] * 3)
        self.assertIn("input_gain", head.state_dict())

    def test_schedule_prefix_exact_and_extension_constant(self):
        for step in range(30):
            value = prefix_fixed_schedule(step, 200, 20, 0.1, prefix_steps=30, prefix_warmup=3)
            self.assertEqual(value, warmup_cosine(step, 30, 3, 0.1))
        for step in (30, 31, 199):
            self.assertEqual(
                prefix_fixed_schedule(step, 200, 20, 0.1, prefix_steps=30, prefix_warmup=3), 0.1
            )

    def test_parent_checkpoint_is_not_mutated_and_adapted_state_rejected(self):
        state = {"classifier": {"weight": torch.ones(2)}, "adapters": {}}
        parent = {
            "epoch": 30,
            "history": [{"early_stop": False} for _ in range(30)],
            "metadata": {"old": True},
            "learned_state": state,
            "best_learned_state": copy.deepcopy(state),
            "optimizer": {"x": torch.ones(1)},
        }
        copied = clone_parent_checkpoint(parent, {"new": True})
        copied["learned_state"]["classifier"]["weight"].zero_()
        self.assertTrue(parent["learned_state"]["classifier"]["weight"].eq(1).all())
        self.assertNotIn("input_gain", parent["learned_state"]["classifier"])
        with self.assertRaises(ValueError):
            add_gain_to_state({"classifier": {}, "adapters": {"a": torch.ones(1)}}, 1)

    def test_scope_and_no_relabeling_of_parent_lora_failure(self):
        p = read_protocol(Path(DEFAULT_PROTOCOL))
        self.assertEqual(len(cases(p)), 6)
        self.assertEqual(p["budgets"], [30, 200])
        cells = {}
        for branch in p["branches"]:
            for mode in p["modes"]:
                for budget in p["budgets"]:
                    cells[(branch, mode, budget)] = {"score": 0.88, "endpoint": 0.88}
        cells[("fusion", "fanin_gain", 200)] = {"score": 0.90, "endpoint": 0.89}
        result = judge(p, cells, {"strong_svm": 0.9115})
        self.assertFalse(any(c["eligible_for_proposed_replication"] for c in result["candidates"]))
        self.assertFalse(result["automatic_lora_started"])

    @unittest.skipUnless(torch.cuda.is_available(), "GPU cached-head training integration")
    def test_hooks_capture_prefix_and_restore_parent_functions(self):
        from tcn_moment import train_e03
        from tcn_moment.analyze_e03 import DEFAULT_PROTOCOL as PARENT_PROTOCOL
        from tcn_moment.analyze_e03 import read_protocol as read_parent

        p = read_parent(Path(PARENT_PROTOCOL))
        p["training"].update(maximum_epochs=2, minimum_epochs=2)
        rng = np.random.default_rng(11)
        train = train_e03.gpu_dataset(
            rng.normal(size=(25, 1024)), rng.normal(size=(25, 15)), np.arange(25) % 3
        )
        val = train_e03.gpu_dataset(
            rng.normal(size=(15, 1024)), rng.normal(size=(15, 15)), np.arange(15) % 3
        )
        model = PeftClassifier(
            None, GainFusionHead([0] * 15, [1] * 15, branch="fusion", input_gain=2)
        ).cuda()
        original_schedule, original_save = train_e03.warmup_cosine, train_e03.atomic_torch_save
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with training_hooks(train_e03, root, steps_per_epoch=1, prefix_epochs=1):
                train_e03.train(
                    model, train, val, p, root, {"synthetic": True}, ["0", "1", "2"], resume=False
                )
            self.assertTrue((root / "checkpoint_at_30.pt").exists())
            payload = torch.load(root / "checkpoint_at_30.pt", weights_only=False)
            self.assertEqual(payload["epoch"], 1)
            self.assertIn("input_gain", payload["learned_state"]["classifier"])
        self.assertIs(train_e03.warmup_cosine, original_schedule)
        self.assertIs(train_e03.atomic_torch_save, original_save)


if __name__ == "__main__":
    unittest.main()
