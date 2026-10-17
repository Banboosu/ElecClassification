"""Real V100 LoRA gradient/frozen-weight/reload/throughput smoke, synthetic data only."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import torch
from momentfm import MOMENTPipeline

from tcn_moment.config import load_config, with_random_seed
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.moment_peft import (
    FusionHead,
    PeftClassifier,
    adapter_parameters,
    capture_learned_state,
    frozen_parameter_digest,
    inject_lora,
    optimizer_groups,
    restore_learned_state,
    trainable_summary,
)
from tcn_moment.train_feature_fusion import verified_checkpoint
from tcn_moment.train_moment import build_model
from tcn_moment.training_utils import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("amp_fp16", "fp32"), default="amp_fp16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    seed_everything(torch, 43)
    config = with_random_seed(
        load_config("configs/experiments/improvements/feature_fusion.yaml"), 43
    )
    checkpoint = verified_checkpoint(config)
    backbone = build_model(config, MOMENTPipeline, 3)
    backbone.init()
    backbone.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10043)
        head = FusionHead([0] * 15, [1] * 15, branch="fusion")
    model = PeftClassifier(backbone, head).cuda().eval()
    batch = args.batch_size
    positions = torch.arange(1024, device="cuda")[None, :]
    lengths = 18 + (torch.arange(batch, device="cuda") * 37) % 799
    mask = (positions < lengths[:, None]).float()
    x = (
        (torch.sin(positions.float() / 13) + 0.2 * torch.cos(positions.float() / 5))
        .expand(batch, -1)
        .clone()
    )
    x *= mask
    x[0].zero_()
    x = x[:, None]
    statistics = torch.randn(batch, 15, device="cuda")
    y = torch.arange(batch, device="cuda") % 3
    amp = args.precision == "amp_fp16"

    def forward():
        with torch.amp.autocast("cuda", enabled=amp):
            return model(x, statistics, mask)

    with torch.inference_mode():
        initial = forward().detach().float()
    torch.manual_seed(30043)
    targets = inject_lora(backbone, 8, 16, 0.05)
    with torch.inference_mode():
        zero_adapter = forward().detach().float()
    zero_error = (initial - zero_adapter).abs().max().item()
    if not torch.isfinite(zero_adapter).all() or zero_error > 1e-5:
        raise AssertionError(f"Zero-adapter logits changed or are not finite: {zero_error}")
    before = frozen_parameter_digest(model)
    optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-3, 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(args.steps):
        model.train_selected()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            loss = torch.nn.functional.cross_entropy(model(x, statistics, mask), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        adapters = adapter_parameters(backbone)
        if not torch.isfinite(loss) or any(
            p.grad is None or not torch.isfinite(p.grad).all() for p in adapters.values()
        ):
            raise AssertionError("Adapter gradient is missing/nonfinite.")
        if not any(p.grad.abs().sum() > 0 for name, p in adapters.items() if "lora_B" in name):
            raise AssertionError("No nonzero B gradients.")
        if step and not any(
            p.grad.abs().sum() > 0 for name, p in adapters.items() if "lora_A" in name
        ):
            raise AssertionError("No nonzero A gradients after the first update.")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if frozen_parameter_digest(model) != before:
        raise AssertionError("A frozen pretrained parameter changed.")
    model.eval()
    with torch.inference_mode():
        expected = forward().detach().float()
    stream = io.BytesIO()
    torch.save(capture_learned_state(model), stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()
    restore_learned_state(model, restored)
    with torch.inference_mode():
        actual = forward().detach().float()
    reload_error = (expected - actual).abs().max().item()
    if reload_error != 0:
        raise AssertionError(f"Learned state failed exact reload: {reload_error}")
    report = {
        "status": "passed",
        "synthetic_input_only": True,
        "test_evaluated": False,
        "precision": args.precision,
        "batch_size": batch,
        "steps": args.steps,
        "target_projections": len(targets),
        "rank": 8,
        "zero_adapter_max_logit_error": zero_error,
        "reload_max_logit_error": reload_error,
        "all_frozen_parameters_unchanged": True,
        "adapter_gradients_finite_and_nonzero": True,
        "seconds": elapsed,
        "training_samples_per_second": batch * args.steps / elapsed,
        "peak_allocated_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "parameters": trainable_summary(model),
        "checkpoint": checkpoint,
        "source_sha256": sha256_file(Path("src/tcn_moment/moment_peft.py")),
    }
    path = Path(f"artifacts/analysis/e03_peft_smoke_{args.precision}_bs{batch}.json")
    atomic_write_json(path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
