from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from tcn_moment import moment_peft, strong_head_lora, strong_head_lora_online, train_e03
from tcn_moment.head_diagnostics import GainFusionHead, fan_in_gain
from tcn_moment.head_diagnostics_stable import FixedHeadGradScaler
from tcn_moment.moment_peft import PeftClassifier, trainable_summary


class StrongHeadOnlineCompatibilityTests(unittest.TestCase):
    def test_no_scientific_change_in_revised_protocol(self):
        old = strong_head_lora.read_protocol(Path(strong_head_lora.DEFAULT_PROTOCOL))
        new = strong_head_lora_online.read_protocol(Path(strong_head_lora_online.DEFAULT_PROTOCOL))
        for p in (old, new):
            for k in ("suite", "smoke_report", "execution_revision"):
                p.pop(k, None)
        self.assertEqual(old, new)
        pp = strong_head_lora_online.training_protocol(new)
        self.assertEqual(pp["training"]["frozen_head_batch_size"], 64)
        self.assertEqual(pp["training"]["physical_batch_size"], 64)

    def test_frozen_provider_rejects_trainable_parameters(self):
        with self.assertRaises(ValueError):
            strong_head_lora_online.OnlineFrozenClassifier(nn.Linear(1024, 1024), nn.Identity())

    def test_online_provider_identical_outputs_and_correct_parameter_roles(self):
        encoder = nn.Linear(1024, 1024).requires_grad_(False)
        head = GainFusionHead([0] * 15, [1] * 15, branch="fusion", input_gain=fan_in_gain("fusion"))
        original = PeftClassifier(encoder, head).eval()
        fixed = strong_head_lora_online.OnlineFrozenClassifier(
            copy.deepcopy(encoder), copy.deepcopy(head)
        ).eval()
        x, s = torch.randn(5, 1024), torch.randn(5, 15)

        def forward(e, x, mask):
            return e(x)

        with (
            patch.object(moment_peft, "forward_features", side_effect=forward),
            patch.object(strong_head_lora_online, "forward_features", side_effect=forward),
            torch.inference_mode(),
        ):
            torch.testing.assert_close(original(x, s), fixed(x, s), rtol=0, atol=0)
        fixed.train_selected()
        self.assertFalse(fixed.online_frozen_encoder.training)
        self.assertTrue(fixed.classifier.training)
        self.assertIsNone(fixed.backbone)
        self.assertEqual(trainable_summary(fixed)["adapter_parameters"], 0)
        self.assertEqual(trainable_summary(fixed)["trainable_parameters"], 135555)

    @unittest.skipUnless(torch.cuda.is_available(), "Online frozen GPU training regression")
    def test_online_rank0_commits_without_fake_adapter_witness_and_uses_microbatch64(self):
        p = strong_head_lora_online.read_protocol(Path(strong_head_lora_online.DEFAULT_PROTOCOL))
        pp = strong_head_lora_online.training_protocol(p)
        pp["training"].update(maximum_epochs=1, minimum_epochs=1)
        model = strong_head_lora_online.OnlineFrozenClassifier(
            nn.Linear(1024, 1024).requires_grad_(False),
            GainFusionHead([0] * 15, [1] * 15, branch="fusion", input_gain=fan_in_gain("fusion")),
        ).cuda()
        rng = np.random.default_rng(1)
        train = train_e03.gpu_dataset(
            rng.normal(size=(130, 1024)), rng.normal(size=(130, 15)), np.arange(130) % 3
        )
        val = train_e03.gpu_dataset(
            rng.normal(size=(24, 1024)), rng.normal(size=(24, 15)), np.arange(24) % 3
        )
        batches = []

        def forward(e, x, mask):
            self.assertFalse(e.training)
            if model.classifier.training:
                batches.append(len(x))
            return e(x)

        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(strong_head_lora_online, "forward_features", side_effect=forward),
            patch.object(torch.amp, "GradScaler", FixedHeadGradScaler),
        ):
            result, _ = train_e03.train(
                model,
                train,
                val,
                pp,
                Path(d),
                {"synthetic_input_only": True},
                ["0", "1", "2"],
                resume=False,
            )
            self.assertTrue(result["all_frozen_parameters_unchanged"])
            import json

            history = json.loads((Path(d) / "history.json").read_text())
            self.assertIsNone(history[0]["adapter_B_gradient_norm_first_window"])
            self.assertEqual(history[0]["grad_scaler_scale"], 1024)
            self.assertEqual(history[0]["skipped_optimizer_steps"], 0)
        self.assertEqual(batches, [64, 64, 2])
        self.assertTrue(all(p.grad is None for p in model.online_frozen_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
