from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tcn_moment import strong_head_lora as stage
from tcn_moment import train_e03
from tcn_moment.head_diagnostics import GainFusionHead, fan_in_gain
from tcn_moment.head_diagnostics_stable import FixedHeadGradScaler
from tcn_moment.io_utils import atomic_torch_save
from tcn_moment.moment_peft import PeftClassifier


class StrongHeadLoraTests(unittest.TestCase):
    def setUp(self):
        self.p = stage.read_protocol(Path(stage.DEFAULT_PROTOCOL))

    def test_scope_is_four_warmstart_conditions_no_test_or_expansion(self):
        self.assertEqual(len(self.p["conditions"]), 4)
        self.assertTrue(self.p["prior_test_results_known"])
        self.assertFalse(self.p["test_evaluated"])
        self.assertEqual(self.p["seed"], 43)
        for branch in ("fusion", "moment"):
            self.assertEqual(
                [c["rank"] for c in self.p["conditions"] if c["branch"] == branch], [0, 8]
            )
        for k in ("automatic_replication", "automatic_test", "automatic_budget_expansion"):
            self.assertFalse(self.p["success_rules"][k])

    def test_unexpected_seed_rank_epoch_or_scale_refused(self):
        for key, value in (("seed", 42), ("prior_test_results_known", False)):
            p = copy.deepcopy(self.p)
            p[key] = value
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "p.json"
                f.write_text(json.dumps(p))
                with self.assertRaises(ValueError):
                    stage.read_protocol(f)
        for key, value in (
            ("maximum_epochs", 31),
            ("loss_scale", 128),
            ("physical_batch_size", 32),
            ("early_stopping", True),
        ):
            p = copy.deepcopy(self.p)
            p["training"][key] = value
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "p.json"
                f.write_text(json.dumps(p))
                with self.assertRaises(ValueError):
                    stage.read_protocol(f)

    def test_fixed_additional_budget_and_both_groups_same_lr(self):
        p = stage.training_protocol(self.p)
        self.assertEqual(p["training"]["maximum_epochs"], 30)
        self.assertEqual(p["training"]["minimum_epochs"], 30)
        self.assertGreater(p["training"]["patience"], 30)
        self.assertEqual(p["training"]["head_learning_rate"], 1e-4)
        self.assertEqual(p["training"]["adapter_learning_rate"], 1e-4)
        self.assertEqual(p["training"]["physical_batch_size"], 64)

    def test_gate_needs_continued_frozen_and_strong_reference_and_matched_moment(self):
        scores = {
            "frozen_fusion": 0.92,
            "frozen_moment": 0.85,
            "lora8_fusion": 0.927,
            "lora8_moment": 0.91,
        }
        self.assertTrue(stage.decision(self.p, scores, {"old": 0.921})["pilot_gate_passed"])
        self.assertFalse(stage.decision(self.p, scores, {"old": 0.926})["pilot_gate_passed"])
        scores["frozen_fusion"] = 0.926
        self.assertFalse(stage.decision(self.p, scores, {"old": 0.92})["pilot_gate_passed"])
        scores["frozen_fusion"] = 0.92
        scores["lora8_moment"] = 0.926
        self.assertFalse(stage.decision(self.p, scores, {"old": 0.92})["pilot_gate_passed"])

    def test_prediction_audit_rejects_wrong_ids_or_argmax(self):
        a = {"sample_ids": np.array(["a", "b", "c"]), "labels": np.array([0, 1, 2])}
        probabilities = np.eye(3) * 0.8 + 0.2 / 3
        from tcn_moment.metrics import classification_metrics

        m = classification_metrics(a["labels"], np.arange(3), ["0", "1", "2"], include_details=True)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "pred.csv"
            stage.save_predictions(f, a, probabilities)
            stage.audit_predictions(f, a, m)
            frame = __import__("pandas").read_csv(f)
            frame.loc[0, "predicted_index"] = 1
            frame.to_csv(f, index=False)
            with self.assertRaises(AssertionError):
                stage.audit_predictions(f, a, m)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU epoch0 and recovery integration")
    def test_epoch0_eligible_and_committed_recovery_exact(self):
        rng = np.random.default_rng(12)
        data = train_e03.gpu_dataset(
            rng.normal(size=(35, 1024)), rng.normal(size=(35, 15)), np.arange(35) % 3
        )
        val = train_e03.gpu_dataset(
            rng.normal(size=(24, 1024)), rng.normal(size=(24, 15)), np.arange(24) % 3
        )
        p = stage.training_protocol(self.p)
        p["training"].update(maximum_epochs=2, minimum_epochs=2)

        def model():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(10043)
                return PeftClassifier(
                    None,
                    GainFusionHead(
                        [0] * 15, [1] * 15, branch="fusion", input_gain=fan_in_gain("fusion")
                    ),
                ).cuda()

        metadata = {"synthetic_input_only": True}
        with tempfile.TemporaryDirectory() as d:
            roots = [Path(d) / x for x in ("complete", "interrupted")]
            for root in roots:
                root.mkdir()
                m = model()
                initial, _ = train_e03.evaluate(m, val, 64, ["0", "1", "2"])
                stage.seed_initial_checkpoint(m, self.p, root, metadata, initial["macro_f1"])
                ck = torch.load(root / "checkpoint_last.pt", weights_only=False)
                self.assertEqual(ck["epoch"], 0)
                self.assertEqual(ck["best_epoch"], 0)
                self.assertFalse(ck["optimizer"]["state"])
                self.assertEqual(ck["scaler"]["scale"], 1024)
            with patch.object(torch.amp, "GradScaler", FixedHeadGradScaler):
                result_a, probs_a = train_e03.train(
                    model(), data, val, p, roots[0], metadata, ["0", "1", "2"], resume=True
                )

                def interrupt(torch_arg, value, path):
                    atomic_torch_save(torch_arg, value, path)
                    if path.name == "checkpoint_last.pt" and value["epoch"] == 1:
                        raise KeyboardInterrupt("synthetic interruption")

                with (
                    patch.object(train_e03, "atomic_torch_save", side_effect=interrupt),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    train_e03.train(
                        model(), data, val, p, roots[1], metadata, ["0", "1", "2"], resume=True
                    )
                newer = torch.load(roots[1] / "model_best.pt", weights_only=True)
                newer["epoch"] = 99
                newer["validation_macro_f1"] = 1
                atomic_torch_save(torch, newer, roots[1] / "model_best.pt")
                result_b, probs_b = train_e03.train(
                    model(), data, val, p, roots[1], metadata, ["0", "1", "2"], resume=True
                )
            self.assertEqual(result_a["best_epoch"], result_b["best_epoch"])
            np.testing.assert_array_equal(probs_a, probs_b)
            for name, t in torch.load(roots[0] / "checkpoint_last.pt", weights_only=False)[
                "learned_state"
            ]["classifier"].items():
                other = torch.load(roots[1] / "checkpoint_last.pt", weights_only=False)[
                    "learned_state"
                ]["classifier"][name]
                torch.testing.assert_close(t, other, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
