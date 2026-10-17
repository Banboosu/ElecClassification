"""Synthetic-input smoke test: random architecture, blocked loaders, distinct weights."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open

from tcn_moment.config import load_config, with_random_seed
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.train_e01c import forbid_pretrained_loading
from tcn_moment.train_moment import build_model, forward_features, require_torch_and_moment
from tcn_moment.training_utils import seed_everything


def main():
    config = with_random_seed(
        load_config("configs/experiments/improvements/feature_fusion.yaml"), 43
    )
    config = replace(config, model=replace(config.model, initialization="random"))
    torch, _, _, _, pipeline = require_torch_and_moment()
    seed_everything(torch, 43)
    with forbid_pretrained_loading():
        model = build_model(config, pipeline, 3)
        model.init()
        for parameter in model.parameters():
            parameter.requires_grad = False
        parameters = sum(parameter.numel() for parameter in model.parameters())
        key = "encoder.block.0.layer.0.SelfAttention.q.weight"
        random_probe = dict(model.named_parameters())[key][:2, :8].detach().cpu().clone()
        model.to("cuda").eval()
        x = torch.zeros((1, 1, 1024), device="cuda")
        x[0, 0, :32] = torch.sin(torch.arange(32, device="cuda", dtype=torch.float32) / 7)
        mask = torch.zeros((1, 1024), device="cuda")
        mask[:, :32] = 1
        with torch.inference_mode():
            features = forward_features(model, x, mask)
        if (
            parameters != 341243395
            or tuple(features.shape) != (1, 1024)
            or not torch.isfinite(features).all()
        ):
            raise AssertionError(
                "Random model architecture/forward differs from the declared control."
            )
    # Read a small public tensor slice for comparison only; do not load it into the random model.
    cached = try_to_load_from_cache(config.model.model_id, "model.safetensors")
    if not isinstance(cached, str):
        raise FileNotFoundError("Public checkpoint must already be cached for the probe check.")
    with safe_open(cached, framework="pt", device="cpu") as handle:
        reference_probe = handle.get_slice(key)[:2, :8]
    if torch.equal(random_probe, reference_probe):
        raise AssertionError("Random probe unexpectedly equals the public pretrained tensor.")
    report = {
        "status": "passed",
        "seed": 43,
        "synthetic_input_only": True,
        "test_evaluated": False,
        "standard_pretrained_loaders_blocked": True,
        "total_parameters": parameters,
        "feature_shape": list(features.shape),
        "finite_features": True,
        "probe_differs_from_public_checkpoint": True,
        "random_probe_sha256": hashlib.sha256(random_probe.numpy().tobytes()).hexdigest(),
        "public_probe_sha256": hashlib.sha256(reference_probe.numpy().tobytes()).hexdigest(),
        "note": "Public tensor slice was read only for equality checking after guarded construction/forward, never assigned to the random model.",
    }
    atomic_write_json(
        Path("artifacts/analysis/e01c_random_initialization_smoke_20261017.json"), report
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
