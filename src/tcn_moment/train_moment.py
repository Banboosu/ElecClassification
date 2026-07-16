from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

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


def require_torch_and_moment() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm.auto import tqdm
        from momentfm import MOMENTPipeline
    except ImportError as exc:
        message = "Missing training dependencies. Install them with:\n  uv sync --frozen\n"
        raise SystemExit(message) from exc
    return torch, DataLoader, TensorDataset, tqdm, MOMENTPipeline


def select_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: ExperimentConfig, moment_pipeline: Any, num_classes: int) -> Any:
    if not config.model.config_path.is_file():
        raise FileNotFoundError(f"MOMENT model config not found: {config.model.config_path}")
    with config.model.config_path.open("r", encoding="utf-8") as file:
        pretrained_config = json.load(file)
    return moment_pipeline.from_pretrained(
        config.model.model_id,
        config=pretrained_config,
        model_kwargs={
            "task_name": "classification",
            "seq_len": config.data.max_length,
            "n_channels": config.model.num_channels,
            "num_class": num_classes,
            "num_classes": num_classes,
            "freeze_embedder": config.model.freeze_backbone,
            "freeze_encoder": config.model.freeze_backbone,
            "enable_gradient_checkpointing": not config.model.freeze_backbone,
        },
    )


def set_num_classes(model: Any, num_classes: int) -> None:
    if hasattr(model, "num_class"):
        model.num_class = num_classes
    if hasattr(model, "num_classes"):
        model.num_classes = num_classes
    if hasattr(model, "config"):
        if hasattr(model.config, "num_class"):
            model.config.num_class = num_classes
        if hasattr(model.config, "num_classes"):
            model.config.num_classes = num_classes


def configure_trainable_parameters(model: Any, config: ExperimentConfig) -> None:
    if not config.model.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for name, parameter in model.named_parameters():
        parameter.requires_grad = "head" in name or "classification" in name
    layers_to_unfreeze = config.model.unfreeze_last_n_layers
    if layers_to_unfreeze <= 0:
        return
    blocks = getattr(getattr(model, "encoder", None), "block", None)
    if blocks is None:
        raise ValueError("Unable to locate model.encoder.block for partial unfreezing.")
    for block in list(blocks)[-layers_to_unfreeze:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def build_optimizer(torch: Any, model: Any, config: ExperimentConfig) -> Any:
    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "head" in name or "classification" in name:
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    groups = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": config.training.learning_rate})
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": config.training.backbone_learning_rate,
            }
        )
    if not groups:
        raise ValueError("No trainable MOMENT parameters were selected.")
    return torch.optim.AdamW(groups, weight_decay=config.training.weight_decay)


def forward_logits(model: Any, batch_x: Any, input_mask: Any) -> Any:
    output = model(x_enc=batch_x, input_mask=input_mask)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    return output


def make_loader(
    torch: Any,
    data_loader: Any,
    tensor_dataset: Any,
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    generator: Any = None,
) -> Any:
    dataset = tensor_dataset(
        torch.tensor(x[:, None, :], dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def evaluate(
    torch: Any,
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: Any,
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
            logits = forward_logits(model, batch_x, batch_mask)
            loss = loss_fn(logits, batch_y)
            total_loss += float(loss.detach().cpu()) * len(batch_y)
            y_true.extend(batch_y.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

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
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    tqdm: Any,
    MOMENTPipeline: Any,
    *,
    resume: bool,
) -> None:
    bundle = load_dataset(config.data)
    shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
    save_label_encoder(bundle.label_encoder, context.run_dir)
    device = select_device(torch, config.training.device)
    model = build_model(config, MOMENTPipeline, bundle.num_classes)
    set_num_classes(model, bundle.num_classes)
    model.init()
    model.to(device)
    configure_trainable_parameters(model, config)

    data_generator = torch.Generator().manual_seed(config.data.random_state)
    loader_args = (torch, DataLoader, TensorDataset)
    train_loader = make_loader(
        *loader_args,
        bundle.x_train,
        bundle.mask_train,
        bundle.y_train,
        config.training.batch_size,
        config.training.num_workers,
        shuffle=True,
        generator=data_generator,
    )
    val_loader = make_loader(
        *loader_args,
        bundle.x_val,
        bundle.mask_val,
        bundle.y_val,
        config.training.batch_size,
        config.training.num_workers,
        shuffle=False,
    )
    test_loader = make_loader(
        *loader_args,
        bundle.x_test,
        bundle.mask_test,
        bundle.y_test,
        config.training.batch_size,
        config.training.num_workers,
        shuffle=False,
    )
    optimizer = build_optimizer(torch, model, config)
    loss_fn = torch.nn.CrossEntropyLoss()
    class_names = [str(name) for name in bundle.label_encoder.classes_.tolist()]
    history: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    latest_path = context.run_dir / "checkpoint_latest.pt"
    best_path = context.run_dir / "moment_classifier_best.pt"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
    )
    amp_enabled = bool(config.training.amp and device.type == "cuda")
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

    for epoch in range(start_epoch, config.training.epochs + 1):
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
                logits = forward_logits(model, batch_x, batch_mask)
                loss = loss_fn(logits, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss detected at epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * len(batch_y)

        metrics = evaluate(torch, model, val_loader, loss_fn, device, class_names)
        scheduler.step(metrics["macro_f1"])
        metrics["train_loss"] = train_loss / len(train_loader.dataset)
        metrics["epoch"] = float(epoch)
        metrics["head_learning_rate"] = float(optimizer.param_groups[0]["lr"])
        metrics["backbone_learning_rate"] = (
            float(optimizer.param_groups[1]["lr"])
            if len(optimizer.param_groups) > 1
            else 0.0
        )
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
        if (
            metrics["macro_f1"]
            > best_macro_f1 + config.training.early_stopping_min_delta
        ):
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
        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if not best_path.exists():
        raise RuntimeError("No best model exists. Increase training.epochs before resuming.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(
        torch,
        model,
        test_loader,
        loss_fn,
        device,
        class_names,
        include_details=True,
    )
    result = {
        "model": "MOMENT",
        "run_name": context.run_name,
        "data": _data_record(bundle, config),
        "history": history,
        "best_validation_macro_f1": best_macro_f1,
        "training": {
            "stopped_epoch": int(history[-1]["epoch"]),
            "early_stopped": int(history[-1]["epoch"]) < config.training.epochs,
            "amp_enabled": amp_enabled,
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
) -> None:
    torch, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()
    seed_everything(torch, config.data.random_state)
    context = prepare_run(
        model_name="MOMENT",
        base_output_dir=config.training.output_dir,
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
            torch,
            DataLoader,
            TensorDataset,
            tqdm,
            MOMENTPipeline,
            resume=resume_dir is not None,
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
    parser = argparse.ArgumentParser(description="Train MOMENT classifier on charging power data.")
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
