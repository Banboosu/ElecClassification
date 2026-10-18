"""Replay the failed training epoch twice; no validation evaluation or new candidate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from tcn_moment.head_diagnostics import GainFusionHead
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.moment_peft import (
    PeftClassifier,
    accumulation_windows,
    optimizer_groups,
    restore_learned_state,
)
from tcn_moment.train_e03 import gpu_dataset, take
from tcn_moment.training_utils import restore_random_state


def main():
    base = Path("artifacts/improvements/e03d_head_diagnostic")
    checkpoint = base / "e03d_head_diagnostic_20261018_v1_statistics_fanin_gain/checkpoint_last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["epoch"] != 131 or state["scaler"]["scale"] != 32768:
        raise ValueError("Unexpected failure checkpoint.")
    root = base / "e03d_head_diagnostic_20261018_v1_analysis/prepared"
    norm = json.loads((root / "normalizer.json").read_text())
    with np.load(root / "train.npz", allow_pickle=False) as arrays:
        data = gpu_dataset(arrays["features"], arrays["statistics"], arrays["labels"])
    records = []
    for loss_scale in (32768, 1024):
        model = PeftClassifier(
            None,
            GainFusionHead(
                norm["mean"], norm["scale"], branch="statistics", input_gain=32.2335229225724
            ),
        ).cuda()
        restore_learned_state(model, state["learned_state"])
        optimizer = torch.optim.AdamW(
            optimizer_groups(model, 0.001, 0.0001), weight_decay=0.01, fused=False
        )
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        for group in optimizer.param_groups:
            group["lr"] = 0.0001
        scaler = torch.amp.GradScaler("cuda", init_scale=loss_scale, growth_interval=1000000000)
        generator = torch.Generator()
        restore_random_state(torch, state["rng"], generator)
        order = torch.randperm(len(data["y"]), generator=generator).cuda()
        model.train_selected()
        failure = None
        completed = 0
        for index, chunks in enumerate(accumulation_windows(len(order), 128, 128)):
            optimizer.zero_grad(set_to_none=True)
            for lo, hi, denominator in chunks:
                x, s, mask, y = take(data, order[lo:hi])
                with torch.amp.autocast("cuda"):
                    loss = torch.nn.functional.cross_entropy(model(x, s, mask), y)
                scaler.scale(loss * (hi - lo) / denominator).backward()
            scaler.unscale_(optimizer)
            norm_value = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1)
            if not torch.isfinite(norm_value):
                failure = index
                break
            scaler.step(optimizer)
            scaler.update()
            completed += 1
        records.append(
            {
                "loss_scale": loss_scale,
                "completed_optimizer_windows": completed,
                "nonfinite_window_zero_based": failure,
            }
        )
    if (
        records[0]["nonfinite_window_zero_based"] is None
        or records[1]["nonfinite_window_zero_based"] is not None
    ):
        raise AssertionError(f"Numerical hypothesis not reproduced: {records}")
    result = {
        "status": "passed",
        "training_only": True,
        "validation_evaluated": False,
        "test_evaluated": False,
        "source_epoch": 131,
        "replayed_epoch": 132,
        "records": records,
        "note": "Both replays start from identical learned/optimizer/RNG state. Replay weights are discarded, never candidate initialization.",
    }
    atomic_write_json(Path("artifacts/analysis/e03d_fixed_scale_replay_20261018.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
