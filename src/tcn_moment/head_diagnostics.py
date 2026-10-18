"""Head-only E03D interventions; no encoder construction or adaptation."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from unittest.mock import patch

import torch

from tcn_moment.moment_peft import FusionHead, warmup_cosine


class GainFusionHead(FusionHead):
    """Change only a fixed global input gain; retain relative branch balance."""

    def __init__(self, *args, input_gain=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(input_gain) or input_gain <= 0:
            raise ValueError("Input gain must be positive and finite.")
        self.register_buffer("input_gain", torch.tensor(input_gain, dtype=torch.float32))

    def forward(self, moment, statistics):
        if moment.shape[-1] != self.moment_dim or statistics.shape != (len(moment), self.stats_dim):
            raise ValueError("Wrong diagnostic head inputs.")
        m = self.moment_norm(moment.float()) / math.sqrt(self.moment_dim)
        s = ((statistics.float() - self.stats_mean) / self.stats_scale) / math.sqrt(self.stats_dim)
        if self.branch == "statistics":
            m = torch.zeros_like(m)
        elif self.branch == "moment":
            s = torch.zeros_like(s)
        return self.network(torch.cat([m, s], dim=-1) * self.input_gain)


def fan_in_gain(branch: str, moment_dim=1024, stats_dim=15):
    if branch not in {"statistics", "moment", "fusion"}:
        raise ValueError("Unknown branch.")
    active = 2 if branch == "fusion" else 1
    return math.sqrt((moment_dim + stats_dim) / active)


def add_gain_to_state(state, gain):
    if state["adapters"]:
        raise ValueError("Head diagnosis must not use supervised-adapted encoders.")
    result = copy.deepcopy(state)
    result["classifier"]["input_gain"] = torch.tensor(gain, dtype=torch.float32)
    return result


def clone_parent_checkpoint(parent, metadata, gain=1.0):
    if parent["epoch"] != 30 or len(parent["history"]) != 30 or parent["history"][-1]["early_stop"]:
        raise ValueError("Continuation requires the declared completed 30-epoch parent.")
    result = copy.deepcopy(parent)
    result["metadata"] = metadata
    result["learned_state"] = add_gain_to_state(parent["learned_state"], gain)
    result["best_learned_state"] = add_gain_to_state(parent["best_learned_state"], gain)
    return result


def prefix_fixed_schedule(
    step, total_steps, warmup_steps, minimum_ratio, *, prefix_steps, prefix_warmup
):
    """Same original 30-epoch trajectory, then the original terminal LR forever."""
    if not 0 <= step < total_steps:
        raise ValueError("Step outside the extended budget.")
    if step < prefix_steps:
        return warmup_cosine(step, prefix_steps, prefix_warmup, minimum_ratio)
    return minimum_ratio


def training_hooks(module, root: Path, *, steps_per_epoch, prefix_epochs=30, warmup_fraction=0.1):
    """Scope interventions to a new run; never edit the parent trainer on disk."""
    from contextlib import ExitStack

    original_save = module.atomic_torch_save
    prefix_steps = steps_per_epoch * prefix_epochs

    def schedule(step, total, warmup, ratio):
        return prefix_fixed_schedule(
            step,
            total,
            warmup,
            ratio,
            prefix_steps=prefix_steps,
            prefix_warmup=int(prefix_steps * warmup_fraction),
        )

    def save(torch_module, payload, path):
        original_save(torch_module, payload, path)
        if path.name == "checkpoint_last.pt" and payload["epoch"] == prefix_epochs:
            original_save(torch_module, payload, root / "checkpoint_at_30.pt")

    stack = ExitStack()
    stack.enter_context(patch.object(module, "warmup_cosine", schedule))
    stack.enter_context(patch.object(module, "atomic_torch_save", save))
    return stack
