from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from tcn_moment.config import load_config
from tcn_moment.train_moment import (
    build_model,
    classification_logits_from_features,
    forward_features,
    masked_pool_embeddings,
    sequence_mask_to_patch_mask,
    set_moment_train_mode,
)
from tcn_moment.training_utils import (
    load_model_weights,
    resume_training_checkpoint,
    save_model_weights,
    save_training_checkpoint,
)


class _Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.0)
        self.linear = nn.Linear(2, 3)


class _SmallModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        self.head = _Head()


class _EmbeddingOutput:
    def __init__(self, embeddings: torch.Tensor) -> None:
        self.embeddings = embeddings


class _PipelineRecorder:
    model_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, *_args: object, **kwargs: object) -> object:
        cls.model_kwargs = dict(kwargs["model_kwargs"])
        return object()


class _EmbedModel(nn.Module):
    patch_len = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = type("Config", (), {"patch_stride_len": 2})()
        self.head = nn.Module()
        self.head.linear = nn.Linear(1, 1)

    def embed(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor,
        reduction: str,
    ) -> _EmbeddingOutput:
        del x_enc, input_mask
        if reduction != "none":
            raise AssertionError("forward_features must request unreduced embeddings")
        embeddings = torch.tensor(
            [[[[1.0], [3.0], [100.0], [200.0]]]],
        ) * self.scale
        return _EmbeddingOutput(embeddings)


class MomentPoolingTests(unittest.TestCase):
    def test_gradient_checkpointing_is_explicitly_forwarded_to_moment(self) -> None:
        config = load_config("configs/experiments/moment_full_finetune.yaml")
        build_model(config, _PipelineRecorder, num_classes=3)
        self.assertFalse(_PipelineRecorder.model_kwargs["enable_gradient_checkpointing"])

        enabled_config = replace(
            config,
            training=replace(config.training, gradient_checkpointing=True),
        )
        build_model(enabled_config, _PipelineRecorder, num_classes=3)
        self.assertTrue(_PipelineRecorder.model_kwargs["enable_gradient_checkpointing"])

    def test_patch_mask_requires_a_complete_valid_patch(self) -> None:
        input_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        actual = sequence_mask_to_patch_mask(input_mask, patch_len=2, patch_stride=2)
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, False, False, False],
            ]
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_masked_pooling_ignores_padding_patch_embeddings(self) -> None:
        embeddings = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [100.0, 200.0],
                    [300.0, 400.0],
                ]
            ]
        )
        input_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        pooled = masked_pool_embeddings(
            embeddings,
            input_mask,
            patch_len=2,
            patch_stride=2,
        )
        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 3.0]])))

    def test_masked_pooling_rejects_sequences_without_complete_patch(self) -> None:
        embeddings = torch.zeros((1, 2, 2))
        input_mask = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "no complete valid MOMENT patch"):
            masked_pool_embeddings(
                embeddings,
                input_mask,
                patch_len=2,
                patch_stride=2,
            )

    def test_cached_features_use_the_original_classification_linear_layer(self) -> None:
        model = _SmallModel()
        features = torch.tensor([[1.0, 2.0]])
        expected = model.head.linear(features)
        actual = classification_logits_from_features(model, features)
        self.assertTrue(torch.equal(actual, expected))

    def test_forward_features_keeps_backbone_gradients_and_ignores_padding(self) -> None:
        model = _EmbedModel()
        input_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        features = forward_features(model, torch.zeros((1, 1, 8)), input_mask)
        self.assertTrue(torch.equal(features, torch.tensor([[2.0]])))
        features.sum().backward()
        self.assertEqual(model.scale.grad.item(), 2.0)


class MomentCheckpointTests(unittest.TestCase):
    def test_trainable_only_weights_restore_head_without_overwriting_backbone(self) -> None:
        model = _SmallModel()
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        expected_head = {
            name: value.detach().clone()
            for name, value in model.head.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            save_model_weights(
                torch=torch,
                model=model,
                path=path,
                model_state_scope="trainable",
            )
            saved_size = path.stat().st_size
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            changed_backbone = {
                name: value.detach().clone()
                for name, value in model.backbone.state_dict().items()
            }
            scope = load_model_weights(
                torch=torch,
                model=model,
                path=path,
                device=torch.device("cpu"),
            )

        self.assertEqual(scope, "trainable")
        self.assertLess(saved_size, 100_000)
        for name, value in model.head.state_dict().items():
            self.assertTrue(torch.equal(value, expected_head[name]))
        for name, value in model.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, changed_backbone[name]))

    def test_frozen_backbone_stays_in_eval_mode_during_head_training(self) -> None:
        model = _SmallModel()
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        set_moment_train_mode(model)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.backbone[1].training)
        self.assertTrue(model.head.training)
        self.assertTrue(model.head.dropout.training)

    def test_resume_rejects_a_different_moment_protocol(self) -> None:
        model = _SmallModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        data_generator = torch.Generator().manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_training_checkpoint(
                torch=torch,
                path=path,
                epoch=1,
                model=model,
                optimizer=optimizer,
                history=[],
                best_macro_f1=0.0,
                epochs_without_improvement=0,
                data_generator=data_generator,
                metadata={"moment_protocol_version": 1},
            )
            with self.assertRaisesRegex(ValueError, "moment_protocol_version"):
                resume_training_checkpoint(
                    torch=torch,
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    data_generator=data_generator,
                    device=torch.device("cpu"),
                    expected_metadata={"moment_protocol_version": 2},
                )


if __name__ == "__main__":
    unittest.main()
