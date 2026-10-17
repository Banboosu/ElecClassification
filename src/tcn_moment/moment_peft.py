"""Native q/v LoRA and a shared statistics-aware head; historical trainers stay intact."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import torch
from torch import nn

from tcn_moment.train_moment import forward_features

TARGET_PATTERN = re.compile(r"encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.(q|v)")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 0 < rank <= min(
            base.in_features, base.out_features
        ):
            raise ValueError("LoRA needs a linear projection and a valid positive rank.")
        if alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA alpha/dropout.")
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(
            base.in_features, rank, bias=False, device=base.weight.device, dtype=base.weight.dtype
        )
        self.lora_B = nn.Linear(
            rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(
    backbone: nn.Module, rank: int, alpha: float, dropout: float, *, expected_blocks: int = 24
) -> list[str]:
    if bool(getattr(getattr(backbone, "encoder", None), "gradient_checkpointing", False)):
        raise ValueError(
            "This pilot requires checkpointing disabled; do not silently detach frozen inputs."
        )
    targets = [
        (name, module)
        for name, module in backbone.named_modules()
        if TARGET_PATTERN.fullmatch(name)
    ]
    expected = {
        f"encoder.block.{i}.layer.0.SelfAttention.{p}"
        for i in range(expected_blocks)
        for p in ("q", "v")
    }
    if {name for name, _ in targets} != expected or any(
        not isinstance(m, nn.Linear) for _, m in targets
    ):
        raise ValueError(
            "q/v target names/count/type differ from the declared encoder architecture."
        )
    backbone.requires_grad_(False)
    for name, module in targets:
        parent, leaf = name.rsplit(".", 1)
        setattr(backbone.get_submodule(parent), leaf, LoRALinear(module, rank, alpha, dropout))
    return [name for name, _ in targets]


def adapter_parameters(backbone: nn.Module | None) -> dict[str, nn.Parameter]:
    if backbone is None:
        return {}
    return {
        name: p
        for name, p in backbone.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }


class FusionHead(nn.Module):
    """Same topology/initial weights across ablations; excluded branches are zeroed."""

    def __init__(
        self,
        mean,
        scale,
        *,
        branch: str,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        moment_dim: int = 1024,
        stats_dim: int = 15,
        num_classes: int = 3,
    ):
        super().__init__()
        if branch not in {"statistics", "moment", "fusion"}:
            raise ValueError("Unknown feature branch.")
        mean = torch.as_tensor(mean, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if (
            mean.shape != (stats_dim,)
            or scale.shape != (stats_dim,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Statistics normalization must be finite, positive and training-only.")
        self.register_buffer("stats_mean", mean.clone())
        self.register_buffer("stats_scale", scale.clone())
        self.branch, self.moment_dim, self.stats_dim = branch, moment_dim, stats_dim
        self.moment_norm = nn.LayerNorm(moment_dim)
        self.network = nn.Sequential(
            nn.Linear(moment_dim + stats_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, moment, statistics):
        if moment.shape[-1] != self.moment_dim or statistics.shape != (len(moment), self.stats_dim):
            raise ValueError("Wrong head input dimensions.")
        m = self.moment_norm(moment.float()) / math.sqrt(self.moment_dim)
        s = ((statistics.float() - self.stats_mean) / self.stats_scale) / math.sqrt(self.stats_dim)
        if self.branch == "statistics":
            m = torch.zeros_like(m)
        elif self.branch == "moment":
            s = torch.zeros_like(s)
        return self.network(torch.cat([m, s], dim=-1))


class PeftClassifier(nn.Module):
    def __init__(self, backbone: nn.Module | None, classifier: FusionHead):
        super().__init__()
        self.backbone, self.classifier = backbone, classifier

    def forward(self, x, statistics, mask=None):
        features = x if self.backbone is None else forward_features(self.backbone, x, mask).float()
        return self.classifier(features, statistics)

    def train_selected(self):
        self.train()
        if self.backbone is not None:
            # Base attention/FF dropout stays off. Only adapter/head dropout is trained.
            # eval() does NOT disable autograd; q/v A/B still receive gradients.
            self.backbone.eval()
            for module in self.backbone.modules():
                if isinstance(module, LoRALinear):
                    module.train()
        self.classifier.train()


def trainable_summary(model: PeftClassifier) -> dict[str, Any]:
    adapters = adapter_parameters(model.backbone)
    head = dict(model.classifier.named_parameters())
    allowed = {id(p) for p in [*adapters.values(), *head.values()]}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != allowed:
        raise ValueError("Only adapters and the shared classifier may be trainable.")
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "adapter_parameters": sum(p.numel() for p in adapters.values()),
        "classifier_parameters": sum(p.numel() for p in head.values()),
    }


def capture_learned_state(model: PeftClassifier) -> dict[str, Any]:
    return {
        "classifier": {
            name: v.detach().cpu().clone() for name, v in model.classifier.state_dict().items()
        },
        "adapters": {
            name: v.detach().cpu().clone() for name, v in adapter_parameters(model.backbone).items()
        },
    }


def restore_learned_state(model: PeftClassifier, state: dict[str, Any]) -> None:
    params = adapter_parameters(model.backbone)
    if set(params) != set(state["adapters"]):
        raise ValueError("Adapter keys differ from the checkpoint architecture.")
    model.classifier.load_state_dict(state["classifier"], strict=True)
    with torch.no_grad():
        for name, p in params.items():
            value = state["adapters"][name]
            if p.shape != value.shape:
                raise ValueError("Adapter rank/shape mismatch.")
            p.copy_(value)


def frozen_parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode())
            digest.update(str((p.dtype, tuple(p.shape))).encode())
            digest.update(p.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def optimizer_groups(model: PeftClassifier, head_lr: float, adapter_lr: float):
    trainable_summary(model)
    groups = [{"params": list(model.classifier.parameters()), "lr": head_lr, "role": "head"}]
    adapters = list(adapter_parameters(model.backbone).values())
    if adapters:
        groups.append({"params": adapters, "lr": adapter_lr, "role": "adapters"})
    return groups


def accumulation_windows(total: int, physical_batch: int, effective_batch: int):
    """Exact windows: a short final group must not get divided by a full group."""
    if (
        total <= 0
        or physical_batch <= 0
        or effective_batch < physical_batch
        or effective_batch % physical_batch
    ):
        raise ValueError("Invalid physical/effective batch sizes.")
    for start in range(0, total, effective_batch):
        end = min(total, start + effective_batch)
        yield [
            (left, min(end, left + physical_batch), end - start)
            for left in range(start, end, physical_batch)
        ]


def warmup_cosine(
    step: int, total_steps: int, warmup_steps: int, minimum_ratio: float = 0.1
) -> float:
    if not 0 <= step < total_steps or not 0 <= warmup_steps < total_steps:
        raise ValueError("Scheduler step outside fixed training budget.")
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
