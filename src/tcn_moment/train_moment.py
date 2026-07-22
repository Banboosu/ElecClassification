from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tcn_moment.config import ExperimentConfig, load_config, with_random_seed
from tcn_moment.data import DatasetBundle, load_dataset, save_label_encoder
from tcn_moment.experiment import RunContext, prepare_run
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.metrics import classification_metrics
from tcn_moment.training_utils import (
    load_model_weights,
    resume_training_checkpoint,
    save_model_weights,
    save_training_checkpoint,
    seed_everything,
)


MOMENT_PROTOCOL_VERSION = 2


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
            "enable_gradient_checkpointing": (
                not config.model.freeze_backbone
                and config.training.gradient_checkpointing
            ),
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


def build_optimizer(torch: Any, model: Any, config: ExperimentConfig) -> tuple[Any, bool]:
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
    fused_enabled = bool(
        config.training.fused_optimizer
        and any(parameter.is_cuda for parameter in trainable_parameters(model))
    )
    try:
        optimizer = torch.optim.AdamW(
            groups,
            weight_decay=config.training.weight_decay,
            fused=fused_enabled,
        )
    except (RuntimeError, TypeError) as exc:
        if not fused_enabled:
            raise
        print(f"Fused AdamW is unavailable; falling back to standard AdamW: {exc}")
        fused_enabled = False
        optimizer = torch.optim.AdamW(groups, weight_decay=config.training.weight_decay)
    return optimizer, fused_enabled


def trainable_parameters(model: Any) -> tuple[Any, ...]:
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def _output_embeddings(output: Any) -> Any:
    if hasattr(output, "embeddings"):
        return output.embeddings
    if isinstance(output, dict) and "embeddings" in output:
        return output["embeddings"]
    raise TypeError(
        "MOMENT classification output does not expose embeddings required for "
        "mask-aware pooling."
    )


def sequence_mask_to_patch_mask(
    input_mask: Any,
    patch_len: int,
    patch_stride: int,
) -> Any:
    if input_mask.ndim != 2:
        raise ValueError(
            f"Expected input_mask with shape [batch, sequence], got {input_mask.shape}."
        )
    if patch_len <= 0 or patch_stride <= 0:
        raise ValueError("MOMENT patch length and stride must be positive.")
    patch_view = input_mask.unfold(-1, patch_len, patch_stride)
    return (patch_view > 0).all(dim=-1)


def masked_pool_embeddings(
    embeddings: Any,
    input_mask: Any,
    patch_len: int,
    patch_stride: int,
) -> Any:
    if embeddings.ndim != 3:
        raise ValueError(
            f"Expected MOMENT embeddings with shape [batch, patch, feature], "
            f"got {embeddings.shape}."
        )
    patch_mask = sequence_mask_to_patch_mask(input_mask, patch_len, patch_stride)
    if patch_mask.shape[:2] != embeddings.shape[:2]:
        raise ValueError(
            "MOMENT embedding and patch-mask shapes do not match: "
            f"embeddings={embeddings.shape}, patch_mask={patch_mask.shape}."
        )
    valid_counts = patch_mask.sum(dim=1)
    if bool((valid_counts == 0).any().item()):
        raise ValueError("At least one sequence has no complete valid MOMENT patch.")
    weights = patch_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
    return (embeddings * weights).sum(dim=1) / weights.sum(dim=1)


def classification_logits_from_features(model: Any, features: Any) -> Any:
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "linear"):
        raise TypeError("MOMENT classification head does not expose a linear layer.")
    dropout = getattr(head, "dropout", None)
    if dropout is not None:
        features = dropout(features)
    return head.linear(features)


def forward_features(model: Any, batch_x: Any, input_mask: Any) -> Any:
    if hasattr(model, "embed"):
        output = model.embed(
            x_enc=batch_x,
            input_mask=input_mask,
            reduction="none",
        )
        embeddings = _output_embeddings(output)
        if embeddings.ndim != 4:
            raise ValueError(
                "Expected unreduced MOMENT embeddings with shape "
                f"[batch, channel, patch, feature], got {embeddings.shape}."
            )
        batch_size, channels, patches, feature_dim = embeddings.shape
        head_feature_dim = int(model.head.linear.in_features)
        if head_feature_dim == feature_dim:
            embeddings = embeddings.mean(dim=1)
        elif head_feature_dim == feature_dim * channels:
            embeddings = embeddings.permute(0, 2, 3, 1).reshape(
                batch_size,
                patches,
                feature_dim * channels,
            )
        else:
            raise ValueError(
                "MOMENT classification head feature dimension does not match "
                f"embeddings: head={head_feature_dim}, channels={channels}, "
                f"feature={feature_dim}."
            )
    else:
        output = model(x_enc=batch_x, input_mask=input_mask)
        embeddings = _output_embeddings(output)
    patch_len = int(getattr(model, "patch_len", 0))
    patch_stride = int(getattr(model.config, "patch_stride_len", patch_len))
    return masked_pool_embeddings(
        embeddings,
        input_mask,
        patch_len,
        patch_stride,
    )


def forward_logits(model: Any, batch_x: Any, input_mask: Any) -> Any:
    return classification_logits_from_features(
        model,
        forward_features(model, batch_x, input_mask),
    )


def set_moment_train_mode(model: Any) -> None:
    model.train()
    subtree_is_trainable: dict[int, bool] = {}
    subtree_has_parameters: dict[int, bool] = {}
    modules = list(model.modules())
    for module in reversed(modules):
        own_parameters = tuple(module.parameters(recurse=False))
        own_parameters_are_trainable = any(
            parameter.requires_grad for parameter in own_parameters
        )
        child_is_trainable = any(
            subtree_is_trainable.get(id(child), False) for child in module.children()
        )
        child_has_parameters = any(
            subtree_has_parameters.get(id(child), False) for child in module.children()
        )
        subtree_is_trainable[id(module)] = (
            own_parameters_are_trainable or child_is_trainable
        )
        subtree_has_parameters[id(module)] = bool(own_parameters) or child_has_parameters
    for module in modules:
        if (
            subtree_has_parameters[id(module)]
            and not subtree_is_trainable[id(module)]
        ):
            module.eval()


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
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Any:
    dataset = tensor_dataset(
        torch.as_tensor(x[:, None, :], dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    loader_options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "generator": generator,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor
    return data_loader(
        dataset,
        **loader_options,
    )


def make_feature_loader(
    data_loader: Any,
    tensor_dataset: Any,
    features: Any,
    y: Any,
    batch_size: int,
    *,
    shuffle: bool,
    generator: Any = None,
    pin_memory: bool = False,
) -> Any:
    return data_loader(
        tensor_dataset(features, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=pin_memory,
    )


def cache_features(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    tqdm: Any,
    description: str,
    *,
    amp_enabled: bool,
) -> tuple[Any, Any]:
    model.eval()
    cached_features = []
    cached_labels = []
    # Keep cached tensors as normal tensors: inference tensors cannot be saved by
    # the classification head for its later weight-gradient computation.
    with torch.no_grad():
        for batch_x, batch_mask, batch_y in tqdm(loader, desc=description, leave=False):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                features = forward_features(model, batch_x, batch_mask)
            cached_features.append(features.float().cpu())
            cached_labels.append(batch_y.clone())
    return torch.cat(cached_features), torch.cat(cached_labels)


def evaluate(
    torch: Any,
    model: Any,
    loader: Any,
    loss_fn: Any,
    device: Any,
    class_names: list[str],
    *,
    include_details: bool = False,
    cached_features: bool = False,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    y_true_chunks = []
    y_pred_chunks = []
    with torch.inference_mode():
        for batch in loader:
            if cached_features:
                batch_features, batch_y = batch
                batch_features = batch_features.to(device, non_blocking=True)
            else:
                batch_x, batch_mask, batch_y = batch
                batch_x = batch_x.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = (
                    classification_logits_from_features(model, batch_features)
                    if cached_features
                    else forward_logits(model, batch_x, batch_mask)
                )
                loss = loss_fn(logits, batch_y)
            total_loss += loss.double() * len(batch_y)
            y_true_chunks.append(batch_y)
            y_pred_chunks.append(torch.argmax(logits, dim=1))

    y_true = torch.cat(y_true_chunks).cpu().numpy().tolist()
    y_pred = torch.cat(y_pred_chunks).cpu().numpy().tolist()

    result = classification_metrics(
        y_true,
        y_pred,
        class_names,
        include_details=include_details,
    )
    result["loss"] = float((total_loss / len(loader.dataset)).cpu())
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
    selected_parameters = trainable_parameters(model)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    backbone_trainable = any(
        "head" not in name and "classification" not in name
        for name in trainable_names
    )
    feature_cache_enabled = bool(
        config.training.cache_frozen_features and not backbone_trainable
    )
    model_state_scope = "trainable" if config.model.freeze_backbone else "full"
    pin_memory = device.type == "cuda"
    patch_len = int(getattr(model, "patch_len", 0))
    patch_stride = int(getattr(model.config, "patch_stride_len", patch_len))
    pooled_feature_dim = int(model.head.linear.in_features)

    data_generator = torch.Generator().manual_seed(config.data.random_state)
    loader_args = (torch, DataLoader, TensorDataset)
    initial_train_batch_size = (
        config.training.feature_extraction_batch_size
        if feature_cache_enabled
        else config.training.batch_size
    )
    train_loader = make_loader(
        *loader_args,
        bundle.x_train,
        bundle.mask_train,
        bundle.y_train,
        initial_train_batch_size,
        config.training.num_workers,
        shuffle=not feature_cache_enabled,
        generator=data_generator if not feature_cache_enabled else None,
        pin_memory=pin_memory,
        persistent_workers=not feature_cache_enabled,
        prefetch_factor=config.training.prefetch_factor,
    )
    val_loader = make_loader(
        *loader_args,
        bundle.x_val,
        bundle.mask_val,
        bundle.y_val,
        config.training.evaluation_batch_size,
        0,
        shuffle=False,
        pin_memory=pin_memory,
        prefetch_factor=config.training.prefetch_factor,
    )
    test_loader = make_loader(
        *loader_args,
        bundle.x_test,
        bundle.mask_test,
        bundle.y_test,
        config.training.evaluation_batch_size,
        0,
        shuffle=False,
        pin_memory=pin_memory,
        prefetch_factor=config.training.prefetch_factor,
    )
    optimizer, fused_optimizer_enabled = build_optimizer(torch, model, config)
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
    physical_batch_size = (
        config.training.cached_feature_batch_size
        if feature_cache_enabled
        else config.training.batch_size
    )
    effective_batch_size = (
        physical_batch_size * config.training.gradient_accumulation_steps
    )
    gradient_checkpointing_enabled = bool(
        not config.model.freeze_backbone
        and config.training.gradient_checkpointing
    )
    execution_metadata = {
        "moment_protocol_version": MOMENT_PROTOCOL_VERSION,
        "mask_aware_pooling": True,
        "feature_cache_enabled": feature_cache_enabled,
        "physical_batch_size": physical_batch_size,
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "amp_enabled": amp_enabled,
        "fused_optimizer_enabled": fused_optimizer_enabled,
        "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
    }
    print(
        "MOMENT execution settings: "
        f"device={device}, amp={amp_enabled}, fused_adamw={fused_optimizer_enabled}, "
        f"batch={physical_batch_size}, effective_batch={effective_batch_size}, "
        f"eval_batch={config.training.evaluation_batch_size}, "
        f"workers={config.training.num_workers}, "
        f"gradient_checkpointing={gradient_checkpointing_enabled}."
    )
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
                expected_metadata=execution_metadata,
            )
        )

    feature_cache_seconds = 0.0
    feature_cache_peak_gpu_memory_mb = 0.0
    if feature_cache_enabled:
        print("Caching frozen MOMENT features with mask-aware pooling...")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        cache_started = time.perf_counter()
        train_features, train_y = cache_features(
            torch,
            model,
            train_loader,
            device,
            tqdm,
            "cache train features",
            amp_enabled=amp_enabled,
        )
        val_features, val_y = cache_features(
            torch,
            model,
            val_loader,
            device,
            tqdm,
            "cache validation features",
            amp_enabled=amp_enabled,
        )
        test_features, test_y = cache_features(
            torch,
            model,
            test_loader,
            device,
            tqdm,
            "cache test features",
            amp_enabled=amp_enabled,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        feature_cache_seconds = time.perf_counter() - cache_started
        feature_cache_peak_gpu_memory_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        )
        train_loader = make_feature_loader(
            DataLoader,
            TensorDataset,
            train_features,
            train_y,
            config.training.cached_feature_batch_size,
            shuffle=True,
            generator=data_generator,
            pin_memory=pin_memory,
        )
        val_loader = make_feature_loader(
            DataLoader,
            TensorDataset,
            val_features,
            val_y,
            config.training.cached_feature_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )
        test_loader = make_feature_loader(
            DataLoader,
            TensorDataset,
            test_features,
            test_y,
            config.training.cached_feature_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )
        print(
            f"Cached frozen features in {feature_cache_seconds:.1f}s "
            f"(train={tuple(train_features.shape)}, "
            f"validation={tuple(val_features.shape)}, test={tuple(test_features.shape)})."
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        set_moment_train_mode(model)
        train_loss = torch.zeros((), device=device, dtype=torch.float64)
        accumulation_steps = config.training.gradient_accumulation_steps
        total_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch}", leave=False),
            start=1,
        ):
            if feature_cache_enabled:
                batch_features, batch_y = batch
                batch_features = batch_features.to(device, non_blocking=True)
            else:
                batch_x, batch_mask, batch_y = batch
                batch_x = batch_x.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = (
                    classification_logits_from_features(model, batch_features)
                    if feature_cache_enabled
                    else forward_logits(model, batch_x, batch_mask)
                )
                loss = loss_fn(logits, batch_y)
            window_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
            window_size = min(accumulation_steps, total_batches - window_start + 1)
            scaler.scale(loss / window_size).backward()
            should_step = batch_index % accumulation_steps == 0 or batch_index == total_batches
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    selected_parameters,
                    config.training.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.detach().double() * len(batch_y)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if not bool(torch.isfinite(train_loss).item()):
            raise FloatingPointError(f"Non-finite loss detected at epoch {epoch}.")
        train_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        metrics = evaluate(
            torch,
            model,
            val_loader,
            loss_fn,
            device,
            class_names,
            cached_features=feature_cache_enabled,
            amp_enabled=amp_enabled,
        )
        validation_seconds = time.perf_counter() - validation_started
        scheduler.step(metrics["macro_f1"])
        metrics["train_loss"] = float(
            (train_loss / len(train_loader.dataset)).cpu()
        )
        metrics["epoch"] = float(epoch)
        metrics["head_learning_rate"] = float(optimizer.param_groups[0]["lr"])
        metrics["backbone_learning_rate"] = (
            float(optimizer.param_groups[1]["lr"])
            if len(optimizer.param_groups) > 1
            else 0.0
        )
        metrics["train_seconds"] = train_seconds
        metrics["validation_seconds"] = validation_seconds
        metrics["epoch_seconds"] = train_seconds + validation_seconds
        metrics["train_samples_per_second"] = len(train_loader.dataset) / train_seconds
        metrics["optimizer_steps"] = float(
            (total_batches + accumulation_steps - 1) // accumulation_steps
        )
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
            save_model_weights(
                torch=torch,
                model=model,
                path=best_path,
                model_state_scope=model_state_scope,
            )
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
            model_state_scope=model_state_scope,
            metadata=execution_metadata,
        )
        atomic_write_json(
            context.run_dir / "metrics_partial.json",
            {
                "history": history,
                "moment_protocol_version": MOMENT_PROTOCOL_VERSION,
                "mask_aware_pooling": True,
                "feature_cache_enabled": feature_cache_enabled,
                "feature_cache_seconds": feature_cache_seconds,
                "physical_batch_size": physical_batch_size,
                "effective_batch_size": effective_batch_size,
                "gradient_accumulation_steps": (
                    config.training.gradient_accumulation_steps
                ),
                "fused_optimizer_enabled": fused_optimizer_enabled,
                "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
                "checkpoint_model_state_scope": model_state_scope,
                "patch_len": patch_len,
                "patch_stride": patch_stride,
                "pooled_feature_dim": pooled_feature_dim,
            },
        )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if not best_path.exists():
        raise RuntimeError("No best model exists. Increase training.epochs before resuming.")
    load_model_weights(torch=torch, model=model, path=best_path, device=device)
    test_metrics = evaluate(
        torch,
        model,
        test_loader,
        loss_fn,
        device,
        class_names,
        include_details=True,
        cached_features=feature_cache_enabled,
        amp_enabled=amp_enabled,
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
            "moment_protocol_version": MOMENT_PROTOCOL_VERSION,
            "mask_aware_pooling": True,
            "feature_cache_enabled": feature_cache_enabled,
            "feature_cache_seconds": feature_cache_seconds,
            "feature_cache_peak_gpu_memory_mb": feature_cache_peak_gpu_memory_mb,
            "physical_batch_size": physical_batch_size,
            "effective_batch_size": effective_batch_size,
            "evaluation_batch_size": config.training.evaluation_batch_size,
            "feature_extraction_batch_size": (
                config.training.feature_extraction_batch_size
            ),
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "num_workers": config.training.num_workers,
            "prefetch_factor": config.training.prefetch_factor,
            "fused_optimizer_enabled": fused_optimizer_enabled,
            "gradient_checkpointing_enabled": gradient_checkpointing_enabled,
            "latest_checkpoint_retained": latest_path.exists(),
            "checkpoint_model_state_scope": model_state_scope,
            "patch_len": patch_len,
            "patch_stride": patch_stride,
            "pooled_feature_dim": pooled_feature_dim,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "total_training_seconds": sum(item["epoch_seconds"] for item in history),
        },
        "test_metrics": test_metrics,
    }
    atomic_write_json(context.run_dir / "metrics.json", result)
    if not config.training.keep_completed_checkpoint:
        try:
            latest_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Unable to remove completed-run checkpoint {latest_path}: {exc}")
        result["training"]["latest_checkpoint_retained"] = latest_path.exists()
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
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override training epochs for a smoke test without editing the YAML file.",
    )
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run-name", help="Unique output directory name for a new run.")
    run_group.add_argument("--resume", type=Path, help="Existing run directory to resume.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs must be positive.")
        config = replace(config, training=replace(config.training, epochs=args.epochs))
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
