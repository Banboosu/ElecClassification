from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from tcn_moment.moment_peft import (
    FusionHead,
    LoRALinear,
    PeftClassifier,
    accumulation_windows,
    adapter_parameters,
    capture_learned_state,
    frozen_parameter_digest,
    inject_lora,
    optimizer_groups,
    restore_learned_state,
    trainable_summary,
    warmup_cosine,
)


class ToyMoment(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_len = 2
        self.config = SimpleNamespace(patch_stride_len=2)
        self.patch_embedding = nn.Linear(2, 4)
        self.encoder = nn.Module()
        self.encoder.gradient_checkpointing = False
        block, layer, attention = nn.Module(), nn.Module(), nn.Module()
        for name in ("q", "k", "v", "o"):
            setattr(attention, name, nn.Linear(4, 4, bias=False))
        layer.SelfAttention = attention
        block.layer = nn.ModuleList([layer])
        self.encoder.block = nn.ModuleList([block])
        self.head = nn.Module()
        self.head.linear = nn.Linear(4, 3)

    def embed(self, *, x_enc, input_mask, reduction):
        h = self.patch_embedding(x_enc[:, 0].unfold(-1, 2, 2))
        a = self.encoder.block[0].layer[0].SelfAttention
        h = a.o(torch.tanh(a.q(h)) + a.v(h) + a.k(h))
        return SimpleNamespace(embeddings=h[:, None])


class MomentPeftTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def head(self, branch="fusion"):
        return FusionHead(
            [0, 0, 0], [1, 1, 1], branch=branch, moment_dim=4, stats_dim=3, hidden_dim=8, dropout=0
        )

    def test_zero_initialized_lora_preserves_forward(self):
        linear = nn.Linear(4, 5)
        x = torch.randn(2, 3, 4)
        expected = linear(x).detach()
        layer = LoRALinear(linear, 2, 4, 0)
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        self.assertIs(layer.weight, linear.weight)

    def test_target_scope_count_and_repeat_rejected(self):
        backbone = ToyMoment()
        names = inject_lora(backbone, 2, 4, 0.1, expected_blocks=1)
        self.assertEqual(len(names), 2)
        self.assertEqual(sum(p.numel() for p in adapter_parameters(backbone).values()), 32)
        self.assertFalse(backbone.head.linear.weight.requires_grad)
        self.assertFalse(backbone.encoder.block[0].layer[0].SelfAttention.k.weight.requires_grad)
        with self.assertRaises(ValueError):
            inject_lora(backbone, 2, 4, 0.1, expected_blocks=1)
        other = ToyMoment()
        other.encoder.gradient_checkpointing = True
        with self.assertRaises(ValueError):
            inject_lora(other, 2, 4, 0.1, expected_blocks=1)

    def test_adapter_gradients_and_frozen_weights(self):
        backbone = ToyMoment()
        inject_lora(backbone, 2, 4, 0.1, expected_blocks=1)
        model = PeftClassifier(backbone, self.head())
        self.assertEqual(trainable_summary(model)["adapter_parameters"], 32)
        before = frozen_parameter_digest(model)
        optimizer = torch.optim.AdamW(optimizer_groups(model, 0.01, 0.01))
        x, stats, mask, y = (
            torch.randn(6, 1, 8),
            torch.randn(6, 3),
            torch.ones(6, 8),
            torch.arange(6) % 3,
        )
        for step in range(2):
            model.train_selected()
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x, stats, mask), y).backward()
            gradients = [p.grad for n, p in adapter_parameters(backbone).items() if "lora_B" in n]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
            if step == 1:
                self.assertTrue(
                    any(
                        p.grad.abs().sum() > 0
                        for n, p in adapter_parameters(backbone).items()
                        if "lora_A" in n
                    )
                )
            optimizer.step()
        self.assertEqual(before, frozen_parameter_digest(model))
        self.assertFalse(backbone.encoder.training)
        self.assertTrue(backbone.encoder.block[0].layer[0].SelfAttention.q.dropout.training)

    def test_checkpoint_contains_normalization_and_reloads_exactly(self):
        model = PeftClassifier(None, self.head())
        features, stats = torch.randn(3, 4), torch.randn(3, 3)
        model.eval()
        expected = model(features, stats).detach()
        stream = io.BytesIO()
        torch.save(capture_learned_state(model), stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        other = PeftClassifier(None, self.head())
        restore_learned_state(other, state)
        other.eval()
        torch.testing.assert_close(other(features, stats), expected, rtol=0, atol=0)
        self.assertIn("stats_mean", state["classifier"])
        broken = {**state, "adapters": {"unknown": torch.zeros(1)}}
        with self.assertRaises(ValueError):
            restore_learned_state(other, broken)

    def test_branch_masks_prevent_excluded_information(self):
        features, stats = torch.randn(3, 4), torch.randn(3, 3)
        for branch in ("statistics", "moment"):
            head = self.head(branch)
            head.eval()
            expected = head(features, stats)
            actual = head(
                features + 1000 if branch == "statistics" else features,
                stats + 1000 if branch == "moment" else stats,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_short_final_accumulation_window_matches_full_batch_gradient(self):
        a = nn.Linear(2, 3)
        b = nn.Linear(2, 3)
        b.load_state_dict(a.state_dict())
        x, y = torch.randn(10, 2), torch.arange(10) % 3
        windows = list(accumulation_windows(10, 3, 6))
        self.assertEqual(windows[-1], [(6, 9, 4), (9, 10, 4)])
        for chunks in windows:
            a.zero_grad()
            b.zero_grad()
            left, right = chunks[0][0], chunks[-1][1]
            nn.functional.cross_entropy(a(x[left:right]), y[left:right]).backward()
            for lo, hi, denominator in chunks:
                (
                    nn.functional.cross_entropy(b(x[lo:hi]), y[lo:hi]) * (hi - lo) / denominator
                ).backward()
            for p, q in zip(a.parameters(), b.parameters(), strict=True):
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)

    def test_scheduler_fixed_bounds(self):
        self.assertAlmostEqual(warmup_cosine(0, 10, 3), 1 / 3)
        self.assertAlmostEqual(warmup_cosine(2, 10, 3), 1)
        self.assertAlmostEqual(warmup_cosine(9, 10, 3), 0.1)
        with self.assertRaises(ValueError):
            warmup_cosine(10, 10, 3)


if __name__ == "__main__":
    unittest.main()
