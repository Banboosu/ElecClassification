from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tcn_moment.analyze_e03 import DEFAULT_PROTOCOL, judge_candidates, read_protocol
from tcn_moment.io_utils import atomic_torch_save
from tcn_moment.moment_peft import FusionHead, PeftClassifier
from tcn_moment.train_e03 import gpu_dataset, train


class E03PilotTests(unittest.TestCase):
    def setUp(self):
        self.p = read_protocol(Path(DEFAULT_PROTOCOL))

    def scores(self):
        return {
            "statistics_mlp": 0.88,
            "frozen_moment_mlp": 0.89,
            "frozen_fusion_mlp": 0.90,
            "lora8_moment_mlp": 0.91,
            "lora8_fusion_mlp": 0.926,
            "lora16_moment_mlp": 0.91,
            "lora16_fusion_mlp": 0.92605,
        }

    def test_scope_seven_conditions_and_rank_matched_ablations(self):
        self.assertEqual(len(self.p["conditions"]), 7)
        for rank in (8, 16):
            branches = {c["branch"] for c in self.p["conditions"] if c["rank"] == rank}
            self.assertEqual(branches, {"moment", "fusion"})
        self.assertFalse(self.p["test_evaluated"])
        self.assertFalse(self.p["decision_rules"]["automatically_expand_replication"])

    def test_lower_rank_preferred_inside_predeclared_tolerance(self):
        result = judge_candidates(self.p, self.scores(), {"old": 0.91})
        self.assertEqual(result["selected_rank_for_proposed_replication"], 8)
        values = self.scores()
        values["lora16_fusion_mlp"] = 0.927
        self.assertEqual(
            judge_candidates(self.p, values, {"old": 0.91})[
                "selected_rank_for_proposed_replication"
            ],
            16,
        )

    def test_fusion_must_add_value_over_rank_matched_moment(self):
        values = self.scores()
        values["lora8_moment_mlp"] = 0.93
        values["lora16_moment_mlp"] = 0.93
        result = judge_candidates(self.p, values, {"old": 0.91})
        self.assertIsNone(result["selected_rank_for_proposed_replication"])

    def test_must_beat_strongest_not_just_weak_frozen_head(self):
        result = judge_candidates(self.p, self.scores(), {"strong_old": 0.925})
        self.assertTrue(all(not c["pilot_rule_passed"] for c in result["candidates"]))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU training-loop recovery integration")
    def test_epoch_recovery_restores_rng_optimizer_and_committed_best(self):
        import numpy as np

        from tcn_moment import train_e03

        p = copy.deepcopy(self.p)
        p["training"].update(maximum_epochs=2, minimum_epochs=2)
        torch.manual_seed(1234)

        def model():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(10043)
                return PeftClassifier(None, FusionHead([0] * 15, [1] * 15, branch="fusion")).cuda()

        rng = np.random.default_rng(9)
        train_data = gpu_dataset(
            rng.normal(size=(35, 1024)), rng.normal(size=(35, 15)), np.arange(35) % 3
        )
        val_data = gpu_dataset(
            rng.normal(size=(24, 1024)), rng.normal(size=(24, 15)), np.arange(24) % 3
        )
        metadata = {"synthetic_input_only": True}
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "complete", Path(directory) / "resumed"
            a.mkdir()
            b.mkdir()
            result_a, probs_a = train(
                model(), train_data, val_data, p, a, metadata, ["0", "1", "2"], resume=False
            )

            def interrupt_after_commit(torch_arg, value, path):
                atomic_torch_save(torch_arg, value, path)
                if path.name == "checkpoint_last.pt" and value["epoch"] == 1:
                    raise KeyboardInterrupt("simulated interruption after epoch commit")

            with (
                patch.object(train_e03, "atomic_torch_save", side_effect=interrupt_after_commit),
                self.assertRaises(KeyboardInterrupt),
            ):
                train(model(), train_data, val_data, p, b, metadata, ["0", "1", "2"], resume=False)
            # Simulate newer uncommitted files: the epoch checkpoint must win.
            extra = torch.load(b / "model_best.pt", weights_only=True)
            extra["epoch"], extra["validation_macro_f1"] = 99, 1.0
            atomic_torch_save(torch, extra, b / "model_best.pt")
            (b / "history.json").write_text(json.dumps([{"epoch": 99, "validation_macro_f1": 1}]))
            result_b, probs_b = train(
                model(), train_data, val_data, p, b, metadata, ["0", "1", "2"], resume=True
            )
            self.assertEqual(result_a["best_epoch"], result_b["best_epoch"])
            np.testing.assert_allclose(probs_a, probs_b, rtol=1e-5, atol=1e-7)
            state_a = torch.load(a / "checkpoint_last.pt", weights_only=False)["learned_state"][
                "classifier"
            ]
            state_b = torch.load(b / "checkpoint_last.pt", weights_only=False)["learned_state"][
                "classifier"
            ]
            for name in state_a:
                torch.testing.assert_close(state_a[name], state_b[name], rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
