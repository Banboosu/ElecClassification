"""E03 exploratory pilot, with epoch-boundary recovery and no test-evaluation path."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from tcn_moment.analyze_e03 import DEFAULT_PROTOCOL, FILES, plan_directory, run_directory, verify
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import (
    load_dataset,
    save_label_encoder,
    stratified_train_subset_indices,
    train_subset_sha256,
)
from tcn_moment.experiment import prepare_run
from tcn_moment.io_utils import atomic_torch_save, atomic_write_json, sha256_file
from tcn_moment.metrics import classification_metrics
from tcn_moment.moment_peft import (
    FusionHead,
    PeftClassifier,
    accumulation_windows,
    adapter_parameters,
    capture_learned_state,
    frozen_parameter_digest,
    inject_lora,
    optimizer_groups,
    restore_learned_state,
    trainable_summary,
    warmup_cosine,
)
from tcn_moment.train_feature_fusion import (
    raw_statistics,
    save_source_snapshot,
    verified_checkpoint,
)
from tcn_moment.train_moment import build_model, forward_features, require_torch_and_moment
from tcn_moment.training_utils import capture_random_state, restore_random_state, seed_everything


def head_digest(head):
    digest = hashlib.sha256()
    for name, value in head.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_backbone(config, rank, p):
    _, _, _, _, pipeline = require_torch_and_moment()
    seed_everything(torch, p["seeds"][0])
    backbone = build_model(config, pipeline, 3)
    backbone.init()
    backbone.requires_grad_(False)
    if rank:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p["architecture"]["adapter_seed_offset"] + p["seeds"][0])
            inject_lora(
                backbone,
                rank,
                rank * p["architecture"]["lora_alpha_per_rank"],
                p["architecture"]["lora_dropout"],
            )
    return backbone


def cached_features(p, frozen, config, bundle, indices, checkpoint):
    directory = Path(p["cache_dir"]) / "seed43_fraction40"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _cached_features_locked(p, frozen, config, bundle, indices, checkpoint)


def _cached_features_locked(p, frozen, config, bundle, indices, checkpoint):
    root = Path(p["cache_dir"]) / "seed43_fraction40"
    root.mkdir(parents=True, exist_ok=True)
    signature = {
        "protocol_sha256": frozen["protocol_sha256"],
        "dataset_sha256": bundle.dataset_sha256,
        "train_ids_sha256": train_subset_sha256(bundle.ids_train[indices]),
        "validation_ids_sha256": train_subset_sha256(bundle.ids_val),
        "checkpoint": checkpoint,
        "precision": p["training"]["precision"],
        "pooling": p["architecture"]["pooling"],
    }
    manifest_path = root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"signature": signature, "splits": {}}
    )
    if manifest["signature"] != signature:
        raise ValueError("AMP feature cache belongs to a different pilot.")
    expected = {"train": bundle.ids_train[indices], "validation": bundle.ids_val}
    arrays = {}
    backbone = None
    try:
        for split, ids in expected.items():
            path = root / f"{split}.npz"
            if split not in manifest["splits"]:
                if backbone is None:
                    backbone = build_backbone(config, 0, p).cuda().eval()
                if split == "train":
                    x, mask = bundle.x_train[indices], bundle.mask_train[indices]
                else:
                    x, mask = bundle.x_val, bundle.mask_val
                split_started = time.perf_counter()
                chunks = []
                batch_size = p["training"]["evaluation_batch_size"]
                print(f"Caching unadapted AMP {split}: {len(ids)} rows", flush=True)
                with torch.inference_mode():
                    for start in range(0, len(ids), batch_size):
                        batch_x = torch.as_tensor(
                            x[start : start + batch_size, None], device="cuda"
                        )
                        batch_mask = torch.as_tensor(
                            mask[start : start + batch_size], device="cuda"
                        )
                        with torch.amp.autocast("cuda", enabled=True):
                            value = (
                                forward_features(backbone, batch_x, batch_mask)
                                .float()
                                .cpu()
                                .numpy()
                            )
                        if not np.isfinite(value).all():
                            raise FloatingPointError("Nonfinite AMP cache features.")
                        chunks.append(value)
                features = np.concatenate(chunks)
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, features=features, sample_ids=ids.astype(str))
                os.replace(temporary, path)
                manifest["splits"][split] = {
                    "sha256": sha256_file(path),
                    "shape": list(features.shape),
                    "extraction_seconds": time.perf_counter() - split_started,
                }
                atomic_write_json(manifest_path, manifest)
            if sha256_file(path) != manifest["splits"][split]["sha256"]:
                raise ValueError("Corrupt AMP feature cache.")
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["sample_ids"], ids.astype(str))
                features = archive["features"].copy()
            if (
                features.dtype != np.float32
                or features.shape != (len(ids), 1024)
                or not np.isfinite(features).all()
            ):
                raise ValueError("Wrong AMP feature dtype/shape/values.")
            arrays[split] = features
    finally:
        del backbone
        torch.cuda.empty_cache()
    return arrays


def make_head(p, branch, scaler):
    a = p["architecture"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(a["head_seed_offset"] + p["seeds"][0])
        return FusionHead(
            scaler.mean_,
            scaler.scale_,
            branch=branch,
            hidden_dim=a["hidden_dim"],
            dropout=a["head_dropout"],
        )


def gpu_dataset(x, statistics, y, mask=None):
    return {
        "x": torch.as_tensor(np.ascontiguousarray(x), device="cuda", dtype=torch.float32),
        "statistics": torch.as_tensor(statistics, device="cuda", dtype=torch.float32),
        "y": torch.as_tensor(y, device="cuda", dtype=torch.long),
        "mask": None
        if mask is None
        else torch.as_tensor(np.ascontiguousarray(mask), device="cuda", dtype=torch.float32),
    }


def take(data, index):
    return (
        data["x"][index],
        data["statistics"][index],
        None if data["mask"] is None else data["mask"][index],
        data["y"][index],
    )


def evaluate(model, data, batch_size, classes, *, details=False):
    model.eval()
    probabilities = []
    loss_total = torch.zeros((), device="cuda", dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, len(data["y"]), batch_size):
            x, stats, mask, y = take(data, slice(start, start + batch_size))
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(x, stats, mask)
                loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation outputs.")
            loss_total += loss.double() * len(y)
            probabilities.append(torch.softmax(logits.float(), dim=1))
    probs = torch.cat(probabilities).cpu().numpy()
    truth = data["y"].cpu().numpy()
    metrics = classification_metrics(truth, probs.argmax(axis=1), classes, include_details=details)
    metrics["loss"] = float((loss_total / len(truth)).cpu())
    return metrics, probs


def train(model, train_data, val_data, p, root, metadata, classes, *, resume):
    t = p["training"]
    n = len(train_data["y"])
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, t["head_learning_rate"], t["adapter_learning_rate"]),
        weight_decay=t["weight_decay"],
        fused=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=t["grad_scaler_initial_scale"])
    generator = torch.Generator().manual_seed(
        p["architecture"]["training_seed_offset"] + p["seeds"][0]
    )
    seed_everything(torch, p["architecture"]["training_seed_offset"] + p["seeds"][0])
    frozen_before = frozen_parameter_digest(model)
    physical = (
        t["physical_batch_size"] if model.backbone is not None else t["frozen_head_batch_size"]
    )
    windows = list(accumulation_windows(n, physical, t["effective_batch_size"]))
    total_steps = len(windows) * t["maximum_epochs"]
    warm_steps = int(total_steps * t["warmup_fraction"])
    history, best, patience_best, bad, start, best_epoch = [], -1.0, -1.0, 0, 1, 0
    best_state = None
    last_path, best_path = root / "checkpoint_last.pt", root / "model_best.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["metadata"] != metadata:
            raise ValueError("Resume checkpoint metadata changed.")
        restore_learned_state(model, state["learned_state"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        restore_random_state(torch, state["rng"], generator)
        history, best, patience_best, bad, start, best_epoch = (
            state["history"],
            state["best"],
            state["patience_best"],
            state["bad"],
            state["epoch"] + 1,
            state["best_epoch"],
        )
        best_state = state["best_learned_state"]
        # Last committed epoch is authoritative even if an interrupted newer epoch
        # already replaced model_best.pt/history.json before its last checkpoint.
        atomic_write_json(root / "history.json", history)
        atomic_torch_save(
            torch,
            {
                "format_version": 1,
                "metadata": metadata,
                "learned_state": best_state,
                "epoch": best_epoch,
                "validation_macro_f1": best,
            },
            best_path,
        )
        if history and history[-1]["early_stop"]:
            start = t["maximum_epochs"] + 1
        print(f"Resumed at epoch {start}, preserving all completed epochs", flush=True)
    parameters = [v for v in model.parameters() if v.requires_grad]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for epoch in range(start, t["maximum_epochs"] + 1):
            epoch_started = time.perf_counter()
            model.train_selected()
            order = torch.randperm(n, generator=generator).cuda()
            loss_total = torch.zeros((), device="cuda", dtype=torch.float64)
            skipped = 0
            gradient_witness = None
            for window_index, chunks in enumerate(windows):
                step = (epoch - 1) * len(windows) + window_index
                factor = warmup_cosine(step, total_steps, warm_steps, t["minimum_lr_ratio"])
                for group in optimizer.param_groups:
                    base_lr = (
                        t["head_learning_rate"]
                        if group["role"] == "head"
                        else t["adapter_learning_rate"]
                    )
                    group["lr"] = base_lr * factor
                optimizer.zero_grad(set_to_none=True)
                for lo, hi, denominator in chunks:
                    x, stats, mask, y = take(train_data, order[lo:hi])
                    with torch.amp.autocast("cuda", enabled=True):
                        logits = model(x, stats, mask)
                        loss = torch.nn.functional.cross_entropy(logits, y)
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training logits/loss.")
                    loss_total += loss.detach().double() * len(y)
                    scaler.scale(loss * (hi - lo) / denominator).backward()
                scaler.unscale_(optimizer)
                if window_index == 0 and model.backbone is not None:
                    adapters = adapter_parameters(model.backbone)
                    if any(v.grad is None for v in adapters.values()):
                        raise RuntimeError("Adapter gradients missing.")
                    gradient_witness = float(
                        sum(
                            v.grad.detach().float().square().sum()
                            for key, v in adapters.items()
                            if "lora_B" in key
                        )
                        .sqrt()
                        .cpu()
                    )
                    if not math.isfinite(gradient_witness) or gradient_witness <= 0:
                        raise FloatingPointError("Adapter gradients zero/nonfinite.")
                norm = torch.nn.utils.clip_grad_norm_(parameters, t["gradient_clip_norm"])
                if not torch.isfinite(norm):
                    raise FloatingPointError(
                        "Nonfinite training gradient norm; abort rather than silently changing precision."
                    )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < previous_scale)
            val, _ = evaluate(model, val_data, t["evaluation_batch_size"], classes)
            score = val["macro_f1"]
            if score > best + 1e-12:
                best, best_epoch = score, epoch
                best_state = capture_learned_state(model)
                atomic_torch_save(
                    torch,
                    {
                        "format_version": 1,
                        "metadata": metadata,
                        "learned_state": best_state,
                        "epoch": epoch,
                        "validation_macro_f1": score,
                    },
                    best_path,
                )
            if score > patience_best + t["patience_min_delta"]:
                patience_best, bad = score, 0
            else:
                bad += 1
            stop = epoch >= t["minimum_epochs"] and bad >= t["patience"]
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": float((loss_total / n).cpu()),
                    "validation_macro_f1": score,
                    "validation_accuracy": val["accuracy"],
                    "validation_loss": val["loss"],
                    "seconds": time.perf_counter() - epoch_started,
                    "head_learning_rate": optimizer.param_groups[0]["lr"],
                    "adapter_B_gradient_norm_first_window": gradient_witness,
                    "grad_scaler_scale": scaler.get_scale(),
                    "skipped_optimizer_steps": skipped,
                    "epochs_without_patience_improvement": bad,
                    "early_stop": stop,
                    "peak_allocated_memory_mib_this_invocation": torch.cuda.max_memory_allocated()
                    / 1024**2,
                }
            )
            atomic_write_json(root / "history.json", history)
            atomic_torch_save(
                torch,
                {
                    "metadata": metadata,
                    "learned_state": capture_learned_state(model),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng": capture_random_state(torch, generator),
                    "epoch": epoch,
                    "history": history,
                    "best": best,
                    "best_epoch": best_epoch,
                    "best_learned_state": best_state,
                    "patience_best": patience_best,
                    "bad": bad,
                },
                last_path,
            )
            print(
                f"epoch={epoch:02d} val_macro_f1={score:.6f} best={best:.6f} train_loss={history[-1]['training_loss']:.4f} seconds={history[-1]['seconds']:.1f}",
                flush=True,
            )
            if stop:
                break
    finally:
        training_seconds = time.perf_counter() - started
    unchanged = frozen_parameter_digest(model) == frozen_before
    if not unchanged:
        raise RuntimeError("A frozen parameter changed during LoRA training.")
    payload = torch.load(best_path, map_location="cpu", weights_only=True)
    if payload["metadata"] != metadata:
        raise ValueError("Best checkpoint metadata changed.")
    restore_learned_state(model, payload["learned_state"])
    metrics, probs = evaluate(model, val_data, t["evaluation_batch_size"], classes, details=True)
    if abs(metrics["macro_f1"] - best) > 1e-12:
        raise RuntimeError("Best checkpoint did not reproduce selected validation score.")
    return {
        "validation_metrics": metrics,
        "best_epoch": best_epoch,
        "training_seconds": sum(r["seconds"] for r in history),
        "current_invocation_training_seconds": training_seconds,
        "all_frozen_parameters_unchanged": unchanged,
        "peak_allocated_memory_mib": max(
            [torch.cuda.max_memory_allocated() / 1024**2]
            + [r["peak_allocated_memory_mib_this_invocation"] for r in history]
        ),
        "peak_allocated_memory_mib_current_invocation": torch.cuda.max_memory_allocated() / 1024**2,
    }, probs


def run(args):
    frozen = verify(Path(args.protocol))
    p = frozen["protocol"]
    matches = [c for c in p["conditions"] if c["name"] == args.condition]
    if len(matches) != 1:
        raise ValueError("Condition outside the frozen pilot.")
    condition = matches[0]
    root = run_directory(p, condition["name"])
    lock = (plan_directory(p) / f"{condition['name']}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if root.exists() and not args.resume:
        raise FileExistsError("Run exists; explicit resume required.")
    if args.resume and not root.exists():
        raise FileNotFoundError("Cannot resume a missing run.")
    if (
        args.resume
        and (root / "status.json").exists()
        and json.loads((root / "status.json").read_text())["status"] == "completed"
    ):
        print(f"Preserved completed {condition['name']}")
        lock.close()
        return
    config_path = Path(p["training_config"])
    config = with_random_seed(load_config(config_path), 43)
    context = prepare_run(
        model_name="E03_PEFT_PILOT",
        base_output_dir=Path(p["output_dir"]),
        config=config,
        config_path=config_path,
        torch=torch,
        run_name=root.name,
        resume_dir=root if args.resume else None,
    )
    try:
        checkpoint = verified_checkpoint(config)
        print(f"Preparing {condition['name']} (validation only, test locked)", flush=True)
        bundle = load_dataset(config.data)
        parent = Path(p["parent_fusion_run"])
        if sha256_file(bundle.split_path) != sha256_file(parent / "split_manifest.json"):
            raise ValueError("Parent pilot split changed.")
        indices = stratified_train_subset_indices(bundle.y_train, bundle.ids_train, 0.4, 43)
        np.testing.assert_array_equal(
            bundle.ids_train[indices], np.load(parent / "train_ids_40.npy")
        )
        stats = raw_statistics(config, bundle)
        standardizer = StandardScaler().fit(stats["train"][indices])
        metadata = {
            "protocol_sha256": frozen["protocol_sha256"],
            "condition": condition,
            "dataset_sha256": bundle.dataset_sha256,
            "train_ids_sha256": train_subset_sha256(bundle.ids_train[indices]),
            "base_checkpoint": checkpoint,
            "classes": bundle.label_encoder.classes_.tolist(),
            "seed": 43,
            "train_fraction": 0.4,
        }
        if args.resume:
            if json.loads((root / "experiment_plan.json").read_text()) != metadata:
                raise ValueError("Resume run changed scope.")
        else:
            save_source_snapshot(root)
            for name in FILES:
                shutil.copy2(Path(__file__).with_name(name), root / "source" / name)
            shutil.copy2(plan_directory(p) / "frozen_protocol.json", root / "frozen_protocol.json")
            atomic_write_json(root / "experiment_plan.json", metadata)
            np.save(root / "train_ids.npy", bundle.ids_train[indices])
            shutil.copy2(bundle.split_path, root / "split_manifest.json")
            save_label_encoder(bundle.label_encoder, root)
        atomic_write_json(
            root / "statistics_normalizer.json",
            {
                "fitted_on": "selected training rows only",
                "n_samples": len(indices),
                "mean": standardizer.mean_.tolist(),
                "scale": standardizer.scale_.tolist(),
            },
        )
        head = make_head(p, condition["branch"], standardizer)
        initial_sha = head_digest(head)
        if condition["rank"]:
            backbone = build_backbone(config, condition["rank"], p)
            model = PeftClassifier(backbone, head).cuda()
            train_data = gpu_dataset(
                bundle.x_train[indices, None],
                stats["train"][indices],
                bundle.y_train[indices],
                bundle.mask_train[indices],
            )
            val_data = gpu_dataset(
                bundle.x_val[:, None], stats["validation"], bundle.y_val, bundle.mask_val
            )
        else:
            if condition["branch"] == "statistics":
                cached = {
                    "train": np.zeros((len(indices), 1024), np.float32),
                    "validation": np.zeros((len(bundle.y_val), 1024), np.float32),
                }
            else:
                cached = cached_features(p, frozen, config, bundle, indices, checkpoint)
            model = PeftClassifier(None, head).cuda()
            train_data = gpu_dataset(
                cached["train"], stats["train"][indices], bundle.y_train[indices]
            )
            val_data = gpu_dataset(cached["validation"], stats["validation"], bundle.y_val)
        parameter_record = trainable_summary(model)
        parameter_record["logical_pipeline_total_parameters"] = (
            341243395 if condition["branch"] != "statistics" and not condition["rank"] else 0
        ) + parameter_record["total_parameters"]
        result, probs = train(
            model,
            train_data,
            val_data,
            p,
            root,
            metadata,
            [str(c) for c in bundle.label_encoder.classes_],
            resume=args.resume,
        )
        predictions = {
            "sample_id": bundle.ids_val,
            "true_index": bundle.y_val,
            "predicted_index": probs.argmax(axis=1),
        }
        predictions.update({f"probability_{i}": probs[:, i] for i in range(3)})
        pd.DataFrame(predictions).to_csv(root / "validation_predictions.csv", index=False)
        atomic_write_json(
            root / "metrics.json",
            {
                **metadata,
                **result,
                "parameters": parameter_record,
                "initial_head_sha256": initial_sha,
                "test_evaluated": False,
                "best_checkpoint_sha256": sha256_file(root / "model_best.pt"),
            },
        )
        context.set_status("completed", test_evaluated=False)
    except BaseException as exc:
        context.set_status(
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        raise
    finally:
        lock.close()


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; restart from the last committed epoch.")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
