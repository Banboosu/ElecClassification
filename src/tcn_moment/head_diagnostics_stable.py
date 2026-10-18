"""Explicit E03D numerical revision: fixed loss scaling, no silent overflow retries."""

from __future__ import annotations

import copy

import torch

from tcn_moment.head_diagnostics import training_hooks as original_hooks

OriginalGradScaler = torch.amp.GradScaler


class FixedHeadGradScaler(OriginalGradScaler):
    def __init__(self, *args, **kwargs):
        kwargs["init_scale"] = 1024
        kwargs["growth_interval"] = 1_000_000_000
        super().__init__(*args, **kwargs)

    def load_state_dict(self, state_dict):
        state = copy.deepcopy(state_dict)
        if state:
            if float(state["scale"]) != 1024:
                raise ValueError("Stable revision must not resume a different loss scale.")
            state["growth_interval"] = 1_000_000_000
        super().load_state_dict(state)


def training_hooks(module, root, **kwargs):
    from unittest.mock import patch

    stack = original_hooks(module, root, **kwargs)
    stack.enter_context(patch.object(torch.amp, "GradScaler", FixedHeadGradScaler))
    return stack
