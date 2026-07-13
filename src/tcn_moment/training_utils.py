from __future__ import annotations

import random
from typing import Any

import numpy as np

from tcn_moment.io_utils import atomic_torch_save


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
    data_generator: Any,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_macro_f1": best_macro_f1,
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
) -> tuple[int, list[dict[str, float]], float]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_random_state(torch, checkpoint["random_state"], data_generator)
    return (
        int(checkpoint["epoch"]) + 1,
        list(checkpoint["history"]),
        float(checkpoint["best_macro_f1"]),
    )
