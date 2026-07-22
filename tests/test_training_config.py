from __future__ import annotations

import unittest

from tcn_moment.config import load_config


class V100TrainingConfigTests(unittest.TestCase):
    def test_moment_strategies_keep_effective_batch_sizes_explicit(self) -> None:
        linear = load_config("configs/experiments/moment_linear_probe.yaml")
        partial = load_config("configs/experiments/moment_partial_finetune.yaml")
        full = load_config("configs/experiments/moment_full_finetune.yaml")

        self.assertEqual(linear.training.feature_extraction_batch_size, 64)
        self.assertEqual(linear.training.cached_feature_batch_size, 32)
        self.assertEqual(partial.training.batch_size, 32)
        self.assertEqual(partial.training.gradient_accumulation_steps, 1)
        self.assertEqual(full.training.batch_size, 32)
        self.assertEqual(full.training.gradient_accumulation_steps, 1)
        self.assertFalse(full.training.gradient_checkpointing)
        self.assertFalse(full.training.keep_completed_checkpoint)
        self.assertEqual(
            linear.training.cached_feature_batch_size
            * linear.training.gradient_accumulation_steps,
            partial.training.batch_size
            * partial.training.gradient_accumulation_steps,
        )
        self.assertEqual(
            partial.training.batch_size
            * partial.training.gradient_accumulation_steps,
            full.training.batch_size * full.training.gradient_accumulation_steps,
        )


if __name__ == "__main__":
    unittest.main()
