from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from tcn_moment.config import ExperimentConfig, load_config, with_random_seed
from tcn_moment.data import DatasetBundle, load_dataset, save_label_encoder
from tcn_moment.experiment import RunContext, prepare_run
from tcn_moment.io_utils import atomic_torch_save, atomic_write_json
from tcn_moment.metrics import classification_metrics
from tcn_moment.training_utils import (
    resume_training_checkpoint,
    save_training_checkpoint,
    seed_everything,
)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalConv1d(nn.Conv1d):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        self.causal_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.causal_padding,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = super().forward(inputs)
        if self.causal_padding:
            return output[..., : -self.causal_padding]
        return output


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(inputs) + self.residual(inputs))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("TCN channels must not be empty.")
        blocks: list[nn.Module] = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        expanded_mask = input_mask.unsqueeze(1).to(dtype=features.dtype)
        pooled = (features * expanded_mask).sum(dim=-1) / expanded_mask.sum(dim=-1).clamp_min(1.0)
        return self.classifier(pooled)


class CNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not channels or kernel_size % 2 == 0:
            raise ValueError("CNN requires non-empty channels and an odd kernel_size.")
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        expanded_mask = input_mask.unsqueeze(1).to(dtype=features.dtype)
        pooled = (features * expanded_mask).sum(dim=-1) / expanded_mask.sum(dim=-1).clamp_min(1.0)
        return self.classifier(pooled)


def make_loader(
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    tensors = TensorDataset(
        torch.tensor(x[:, None, :], dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: list[str],
    *,
    include_details: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_x, batch_mask, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x, batch_mask)
            total_loss += float(loss_fn(logits, batch_y).cpu()) * len(batch_y)
            y_true.extend(batch_y.cpu().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

    result = classification_metrics(
        y_true,
        y_pred,
        class_names,
        include_details=include_details,
    )
    result["loss"] = total_loss / len(loader.dataset)
    if not include_details:
        result["val_loss"] = result.pop("loss")
    return result


def _data_record(bundle: DatasetBundle, config: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset_sha256": bundle.dataset_sha256,
        "split_manifest": str(bundle.split_path),
        "split": bundle.split_counts,
        "classes": bundle.label_encoder.classes_.tolist(),
        "short_sequences_excluded": bundle.short_sequence_count,
        "invalid_label_counts": bundle.invalid_label_counts,
        "truncated_sequences": bundle.truncated_sequence_count,
        "max_length": config.data.max_length,
        "normalization": config.data.normalize,
    }


def _train_run(
    config: ExperimentConfig,
    context: RunContext,
    *,
    resume: bool,
    model_name: str,
) -> None:
    training = config.tcn_training
    bundle = load_dataset(config.data)
    shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
    save_label_encoder(bundle.label_encoder, context.run_dir)

    data_generator = torch.Generator().manual_seed(config.data.random_state)
    train_loader = make_loader(
        bundle.x_train,
        bundle.mask_train,
        bundle.y_train,
        training.batch_size,
        training.num_workers,
        shuffle=True,
        generator=data_generator,
    )
    val_loader = make_loader(
        bundle.x_val,
        bundle.mask_val,
        bundle.y_val,
        training.batch_size,
        training.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        bundle.x_test,
        bundle.mask_test,
        bundle.y_test,
        training.batch_size,
        training.num_workers,
        shuffle=False,
    )

    device = select_device(training.device)
    model_class = TCNClassifier if model_name == "TCN" else CNNClassifier
    model = model_class(
        bundle.num_classes,
        config.tcn_model.channels,
        config.tcn_model.kernel_size,
        config.tcn_model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    class_names = [str(name) for name in bundle.label_encoder.classes_.tolist()]
    history: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    latest_path = context.run_dir / "checkpoint_latest.pt"
    best_path = context.run_dir / f"{model_name.lower()}_classifier_best.pt"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training.scheduler_factor,
        patience=training.scheduler_patience,
    )
    amp_requested = bool(training.amp and device.type == "cuda")
    amp_enabled = amp_requested
    amp_fallback_triggered = False
    amp_fallback_epoch: int | None = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if resume:
        start_epoch, history, best_macro_f1, epochs_without_improvement = (
            resume_training_checkpoint(
                torch=torch,
                path=latest_path,
                model=model,
                optimizer=optimizer,
                data_generator=data_generator,
                device=device,
                scheduler=scheduler,
                scaler=scaler,
            )
        )

    for epoch in range(start_epoch, training.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        train_loss = 0.0
        for batch_x, batch_mask, batch_y in tqdm(
            train_loader, desc=f"epoch {epoch}", leave=False
        ):
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch_x, batch_mask)
                loss = loss_fn(logits, batch_y)
            if amp_enabled and not torch.isfinite(loss):
                print(
                    f"Warning: non-finite {model_name} loss under CUDA AMP at epoch "
                    f"{epoch}; retrying this batch in FP32 and disabling AMP for this run."
                )
                amp_enabled = False
                amp_fallback_triggered = True
                amp_fallback_epoch = epoch
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x, batch_mask)
                loss = loss_fn(logits, batch_y)
            if not torch.isfinite(loss):
                inputs_finite = bool(torch.isfinite(batch_x).all().item())
                logits_finite = bool(torch.isfinite(logits).all().item())
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch {epoch} in FP32 "
                    f"(inputs_finite={inputs_finite}, logits_finite={logits_finite})."
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm) and not amp_enabled:
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradient norm detected at epoch {epoch} in FP32."
                )
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * len(batch_y)

        metrics = evaluate(model, val_loader, loss_fn, device, class_names)
        scheduler.step(metrics["macro_f1"])
        metrics["train_loss"] = train_loss / len(train_loader.dataset)
        metrics["epoch"] = float(epoch)
        metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        metrics["epoch_seconds"] = time.perf_counter() - epoch_started
        metrics["peak_gpu_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        )
        history.append(metrics)
        print(
            f"epoch={epoch} train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        if metrics["macro_f1"] > best_macro_f1 + training.early_stopping_min_delta:
            best_macro_f1 = metrics["macro_f1"]
            epochs_without_improvement = 0
            atomic_torch_save(torch, model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1
        save_training_checkpoint(
            torch=torch,
            path=latest_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            history=history,
            best_macro_f1=best_macro_f1,
            epochs_without_improvement=epochs_without_improvement,
            data_generator=data_generator,
            scheduler=scheduler,
            scaler=scaler,
        )
        atomic_write_json(context.run_dir / "metrics_partial.json", {"history": history})
        if epochs_without_improvement >= training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if not best_path.exists():
        raise RuntimeError("No best model exists. Increase training.epochs before resuming.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        class_names,
        include_details=True,
    )
    result = {
        "model": model_name,
        "run_name": context.run_name,
        "data": _data_record(bundle, config),
        "history": history,
        "best_validation_macro_f1": best_macro_f1,
        "training": {
            "stopped_epoch": int(history[-1]["epoch"]),
            "early_stopped": int(history[-1]["epoch"]) < training.epochs,
            "amp_requested": amp_requested,
            "amp_enabled": amp_enabled,
            "amp_fallback_triggered": amp_fallback_triggered,
            "amp_fallback_epoch": amp_fallback_epoch,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "total_training_seconds": sum(item["epoch_seconds"] for item in history),
        },
        "test_metrics": test_metrics,
    }
    atomic_write_json(context.run_dir / "metrics.json", result)
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"Saved artifacts to {context.run_dir}")


def train(
    config: ExperimentConfig,
    config_path: Path,
    *,
    run_name: str | None = None,
    resume_dir: Path | None = None,
    model_name: str = "TCN",
) -> None:
    seed_everything(torch, config.data.random_state)
    context = prepare_run(
        model_name=model_name,
        base_output_dir=config.tcn_training.output_dir,
        config=config,
        config_path=config_path,
        torch=torch,
        run_name=run_name,
        resume_dir=resume_dir,
    )
    try:
        _train_run(
            config,
            context,
            resume=resume_dir is not None,
            model_name=model_name,
        )
    except KeyboardInterrupt:
        resume_available = (context.run_dir / "checkpoint_latest.pt").exists()
        message = (
            "Resume this run with --resume."
            if resume_available
            else "No epoch completed; start a new run."
        )
        context.set_status("interrupted", message=message, resume_available=resume_available)
        print(f"\nTraining interrupted. {message}")
    except BaseException as exc:
        context.set_status("failed", error_type=type(exc).__name__, message=str(exc))
        raise
    else:
        context.set_status("completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCN on the unified data protocol.")
    parser.add_argument("--config", default="configs/moment.yaml", help="Path to YAML config.")
    parser.add_argument("--seed", type=int, help="Override random seed and use its split manifest.")
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run-name", help="Unique output directory name for a new run.")
    run_group.add_argument("--resume", type=Path, help="Existing run directory to resume.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.seed is not None:
        config = with_random_seed(config, args.seed)
    train(
        config,
        config_path,
        run_name=args.run_name,
        resume_dir=args.resume,
    )


if __name__ == "__main__":
    main()
