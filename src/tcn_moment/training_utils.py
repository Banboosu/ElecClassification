from __future__ import annotations

import random
from typing import Any

import numpy as np

from tcn_moment.io_utils import atomic_torch_save


def model_state_dict(model: Any, scope: str = "full") -> dict[str, Any]:
    if scope == "full":
        return model.state_dict()
    if scope != "trainable":
        raise ValueError(f"Unsupported model state scope: {scope}")
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not trainable_names:
        raise ValueError("Cannot save trainable model state because no parameters are trainable.")
    return {
        name: value
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def load_model_state_dict(
    model: Any,
    state_dict: dict[str, Any],
    scope: str = "full",
) -> None:
    if scope == "full":
        model.load_state_dict(state_dict)
        return
    if scope != "trainable":
        raise ValueError(f"Unsupported model state scope: {scope}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys in trainable model state: {incompatible.unexpected_keys}"
        )
    missing_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in incompatible.missing_keys
    ]
    if missing_trainable:
        raise RuntimeError(
            f"Missing trainable parameters in model state: {missing_trainable}"
        )


def save_model_weights(
    *,
    torch: Any,
    model: Any,
    path: Any,
    model_state_scope: str = "full",
) -> None:
    atomic_torch_save(
        torch,
        {
            "format_version": 2,
            "model_state_scope": model_state_scope,
            "model_state_dict": model_state_dict(model, model_state_scope),
        },
        path,
    )


def load_model_weights(*, torch: Any, model: Any, path: Any, device: Any) -> str:
    payload = torch.load(path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        scope = str(payload.get("model_state_scope", "full"))
        state_dict = payload["model_state_dict"]
    else:
        scope = "full"
        state_dict = payload
    load_model_state_dict(model, state_dict, scope)
    return scope


def seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_random_state(torch: Any, data_generator: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_loader_generator": data_generator.get_state(),
    }


def restore_random_state(torch: Any, state: dict[str, Any], data_generator: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    data_generator.set_state(state["data_loader_generator"])


def save_training_checkpoint(
    *,
    torch: Any,
    path: Any,
    epoch: int,
    model: Any,
    optimizer: Any,
    history: list[dict[str, float]],
    best_macro_f1: float,
    epochs_without_improvement: int,
    data_generator: Any,
    scheduler: Any = None,
    scaler: Any = None,
    model_state_scope: str = "full",
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "format_version": 2,
        "epoch": epoch,
        "model_state_scope": model_state_scope,
        "model_state_dict": model_state_dict(model, model_state_scope),
        "metadata": metadata or {},
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_macro_f1": best_macro_f1,
        "epochs_without_improvement": epochs_without_improvement,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "random_state": capture_random_state(torch, data_generator),
    }
    atomic_torch_save(torch, checkpoint, path)


def resume_training_checkpoint(
    *,
    torch: Any,
    path: Any,
    model: Any,
    optimizer: Any,
    data_generator: Any,
    device: Any,
    scheduler: Any = None,
    scaler: Any = None,
    expected_metadata: dict[str, Any] | None = None,
) -> tuple[int, list[dict[str, float]], float, int]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    actual_metadata = checkpoint.get("metadata", {})
    for key, expected_value in (expected_metadata or {}).items():
        if actual_metadata.get(key) != expected_value:
            raise ValueError(
                f"Checkpoint protocol mismatch for {key}: "
                f"expected {expected_value!r}, got {actual_metadata.get(key)!r}."
            )
    load_model_state_dict(
        model,
        checkpoint["model_state_dict"],
        str(checkpoint.get("model_state_scope", "full")),
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_random_state(torch, checkpoint["random_state"], data_generator)
    return (
        int(checkpoint["epoch"]) + 1,
        list(checkpoint["history"]),
        float(checkpoint["best_macro_f1"]),
        int(checkpoint.get("epochs_without_improvement", 0)),
    )
