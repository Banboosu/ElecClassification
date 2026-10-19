"""Synthetic-only replacement-hardware checks. Never evaluates project test data."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from tcn_moment.config import load_config, with_random_seed
from tcn_moment.fast_svm import squared_distances
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.moment_peft import (
    FusionHead,
    PeftClassifier,
    adapter_parameters,
    capture_learned_state,
    frozen_parameter_digest,
    inject_lora,
    restore_learned_state,
    trainable_summary,
)
from tcn_moment.train_feature_fusion import verified_checkpoint
from tcn_moment.train_moment import build_model, forward_features, require_torch_and_moment
from tcn_moment.training_utils import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/deployments/autodl_20261019/synthetic_gpu_checks.json"),
    )
    args = parser.parse_args()
    seed_everything(torch, 43)
    config = with_random_seed(
        load_config("configs/experiments/improvements/feature_fusion.yaml"), 43
    )
    checkpoint = verified_checkpoint(config)
    _, _, _, _, pipeline = require_torch_and_moment()
    model = build_model(config, pipeline, 3)
    model.init()
    model.requires_grad_(False)
    head = FusionHead([0] * 15, [1] * 15, branch="fusion")
    classifier = PeftClassifier(model, head).cuda().eval()
    x = torch.zeros((4, 1, 1024), device="cuda")
    mask = torch.zeros((4, 1024), device="cuda")
    for i, n in enumerate((18, 128, 816, 256)):
        mask[i, :n] = 1
        if i:
            x[i, 0, :n] = torch.sin(torch.arange(n, device="cuda", dtype=torch.float32) / (7 + i))
    statistics = torch.randn((4, 15), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        fp32 = forward_features(model, x, mask)
        with torch.amp.autocast("cuda"):
            logits_before = classifier(x, statistics, mask).float()
    if (
        fp32.shape != (4, 1024)
        or not torch.isfinite(fp32).all()
        or not torch.isfinite(logits_before).all()
    ):
        raise AssertionError("Nonfinite/wrong model output on replacement GPU.")
    targets = inject_lora(model, 8, 16, 0.05)
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        zero = classifier(x, statistics, mask).float()
    zero_error = (zero - logits_before).abs().max().item()
    if zero_error > 1e-5:
        raise AssertionError("Zero adapter did not preserve logits.")
    before = frozen_parameter_digest(classifier)
    classifier.train_selected()
    optimizer = torch.optim.AdamW([p for p in classifier.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        loss = torch.nn.functional.cross_entropy(
            classifier(x, statistics, mask), torch.arange(4, device="cuda") % 3
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    adapters = adapter_parameters(model)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in adapters.values()):
        raise AssertionError("Missing/nonfinite adapter gradients.")
    if not any(p.grad.abs().sum() > 0 for n, p in adapters.items() if "lora_B" in n):
        raise AssertionError("No nonzero adapter B gradients.")
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in classifier.parameters() if p.requires_grad], 1
    )
    if not torch.isfinite(norm):
        raise AssertionError("Nonfinite synthetic gradient norm.")
    scaler.step(optimizer)
    scaler.update()
    if frozen_parameter_digest(classifier) != before:
        raise AssertionError("Frozen model weights changed.")
    classifier.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        expected = classifier(x, statistics, mask).float()
    stream = io.BytesIO()
    torch.save(capture_learned_state(classifier), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for p in classifier.parameters():
            if p.requires_grad:
                p.zero_()
    restore_learned_state(classifier, saved)
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        actual = classifier(x, statistics, mask).float()
    reload_error = (expected - actual).abs().max().item()
    if reload_error != 0:
        raise AssertionError("Synthetic learned-state reload changed logits.")
    rng = np.random.default_rng(43)
    a = rng.normal(size=(128, 31)).astype(np.float64)
    b = rng.normal(size=(91, 31)).astype(np.float64)
    started = time.perf_counter()
    gpu = squared_distances(a, b, "cuda")
    seconds = time.perf_counter() - started
    cpu = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    max_error = float(np.max(np.abs(gpu - cpu)))
    if not np.allclose(gpu, cpu, rtol=1e-12, atol=1e-12):
        raise AssertionError("CUDA float64 distances mismatch.")
    result = {
        "status": "passed",
        "synthetic_input_only": True,
        "test_evaluated": False,
        "training_on_project_data": False,
        "historical_results_retrained": False,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_build": torch.__version__,
        "checkpoint": checkpoint,
        "feature_shape": list(fp32.shape),
        "qv_target_count": len(targets),
        "zero_adapter_max_logit_error": zero_error,
        "reload_max_logit_error": reload_error,
        "frozen_parameters_unchanged": True,
        "adapter_gradients_finite": True,
        "parameters": trainable_summary(classifier),
        "float64_distance_max_abs_error": max_error,
        "small_distance_smoke_seconds": seconds,
        "peak_allocated_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "boundary": "Technical synthetic smoke only; not a new experiment, performance benchmark or claim of bitwise reproduction across GPUs.",
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
