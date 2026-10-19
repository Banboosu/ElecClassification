"""E05B: bounded legacy-structure/early-stopping pilot, train/validation only."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.data import (parse_power_sequence, prepare_sequence, sequence_mask,
                            stratified_train_subset_indices, train_subset_sha256)
from tcn_moment.io_utils import atomic_torch_save, atomic_write_json, sha256_file
from tcn_moment.train_tcn import CausalConv1d, TCNClassifier, make_loader
from tcn_moment.training_utils import (seed_everything, save_training_checkpoint,
                                      resume_training_checkpoint)

DEFAULT = Path("configs/experiments/improvements/e05b_tcn_legacy_pilot_protocol.json")
CONDITIONS = ["current_early7", "current_fixed50", "legacy_masked50", "legacy_unmasked50"]
PACKAGES = ("torch", "numpy", "pandas", "scikit-learn")


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def protocol(path):
    p = read(path)
    if (p["experiment_id"] != "E05B" or p["seed"] != 43
        or p["train_fractions"] != [0.1, 1.0] or p["conditions"] != CONDITIONS
        or p["classes"] != ["0", "1", "2"] or p["epochs"] != 50
        or p["batch_size"] != 32 or p["evaluation_batch_size"] != 64
        or p["normalization"] != "none" or p["max_length"] != 1024
        or p["optimizer"] != "AdamW" or p["precision"] != "fp32_tf32_disabled"
        or p["test_evaluated"] or p["automatic_expansion"] or p["automatic_shutdown"]):
        raise ValueError("Unexpected bounded validation-only scope")
    return p


def root(p):
    return Path(p["output_dir"]) / p["suite"]


def case_name(fraction, condition):
    return f"fraction{round(fraction * 100):03d}_{condition}"


class LegacyTCN(nn.Module):
    """Six Conv-ReLU-Dropout layers from legacy; common Torch initialization."""
    def __init__(self, masked=True):
        super().__init__()
        layers = []
        channels = 1
        for dilation in (1, 2, 4, 8, 16, 32):
            layers.extend([CausalConv1d(channels, 128, 3, dilation),
                           nn.ReLU(), nn.Dropout(0.3)])
            channels = 128
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(128, 3)
        self.masked = masked

    def forward(self, inputs, mask):
        features = self.encoder(inputs)
        if self.masked:
            weights = mask[:, None, :].to(features.dtype)
            pooled = (features * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        else:
            pooled = features.mean(-1)
        return self.classifier(pooled)


def build_model(condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition.startswith("current_"):
        return TCNClassifier(3, (64, 64, 128, 128), 3, 0.3)
    return LegacyTCN(masked=condition == "legacy_masked50")


def state_hash(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def synthetic_check():
    torch.set_num_threads(4)
    seed_everything(torch, 43)
    conv = CausalConv1d(1, 2, 3, 4)
    x = torch.randn(2, 1, 32)
    other = x.clone()
    other[..., 17:] += 100
    with torch.no_grad():
        a, b = conv(x), conv(other)
    if a.shape != (2, 2, 32) or not torch.equal(a[..., :17], b[..., :17]):
        raise AssertionError("Causal convolution uses future samples")
    hashes = []
    for condition in CONDITIONS:
        seed_everything(torch, 43)
        model = build_model(condition).eval()
        hashes.append(state_hash(model))
        x = torch.randn(2, 1, 96)
        mask = torch.zeros(2, 96)
        mask[:, :48] = 1
        with torch.no_grad():
            a = model(x, mask)
            b = model(torch.nn.functional.pad(x, (0, 32)),
                      torch.nn.functional.pad(mask, (0, 32)))
        if not torch.isfinite(a).all() or a.shape != (2, 3):
            raise AssertionError("Nonfinite/wrong output")
        if condition != "legacy_unmasked50":
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    if hashes[0] != hashes[1] or hashes[2] != hashes[3]:
        raise AssertionError("Paired arms do not share initialization")
    return {"causality": "passed", "masked_padding": "passed",
            "paired_initialization": "passed", "project_data_used": False}


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(torch, 43)


def freeze(path):
    p = protocol(path)
    r = root(p)
    r.mkdir(parents=True, exist_ok=True)
    if (r / "frozen_protocol.json").exists():
        return verify(path)
    if list(r.glob("fraction*")):
        raise ValueError("Freeze must precede training")
    if sha256_file(Path(p["dataset"])) != p["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    manifest = read(p["split_manifest"])
    if manifest["protocol"]["random_state"] != 43 or manifest["protocol"]["dataset_sha256"] != p["dataset_sha256"]:
        raise ValueError("Split mismatch")
    sources = [Path(__file__).resolve()]
    for name in ("audit_protocol", "data", "io_utils", "train_tcn", "training_utils", "config", "experiment", "metrics"):
        sources.append(Path(__file__).with_name(name + ".py").resolve())
    files = {Path(s).relative_to(Path.cwd()) for s in sources}
    files.update([path, Path(p["dataset"]), Path(p["split_manifest"]), Path("legacy/改进TCN.py"),
                  Path("docs/experiment_records/e05b_tcn_legacy_pilot_protocol_20261019.md"),
                  Path("tests/test_tcn_legacy_pilot.py"), Path("scripts/run_e05b_remote.sh")])
    checks = synthetic_check()
    configure()
    f = {"frozen_at": now(), "protocol": p, "protocol_sha256": sha256_file(path),
         "file_sha256": {str(f): sha256_file(f) for f in sorted(files)},
         "packages": {k: importlib.metadata.version(k) for k in PACKAGES},
         "runtime": {"host": platform.node(), "python": platform.python_version(),
                     "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                     "cuda": torch.version.cuda},
         "synthetic_checks": checks, "new_training_outcomes_observed": False}
    for file in sorted(files):
        if file.parts[0] in {"src", "configs", "docs", "tests", "scripts", "legacy"}:
            dest = r / "source_snapshot" / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
    shutil.copy2(p["split_manifest"], r / "split_manifest.json")
    atomic_write_json(r / "frozen_protocol.json", f)
    print("Frozen eight train/validation conditions; no test tensors", flush=True)
    return f


def verify(path):
    p = protocol(path)
    f = read(root(p) / "frozen_protocol.json")
    if f["protocol"] != p or f["protocol_sha256"] != sha256_file(path):
        raise ValueError("Frozen protocol changed")
    for file, expected in f["file_sha256"].items():
        if sha256_file(Path(file)) != expected:
            raise ValueError(f"Frozen input changed: {file}")
    if f["packages"] != {k: importlib.metadata.version(k) for k in PACKAGES}:
        raise ValueError("Dependency versions changed")
    return f


def load_train_validation(p):
    m = read(p["split_manifest"])
    ids = {k: np.asarray(m["splits"][k], dtype=str) for k in ("train", "validation")}
    sets = [set(m["splits"][k]) for k in ("train", "validation", "test")]
    if any(len(sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Overlapping split IDs")
    if len(ids["train"]) != p["train_count"] or len(ids["validation"]) != p["validation_count"]:
        raise ValueError("Unexpected sample counts")
    # Exclude held-out rows before sequence parsing, label encoding or tensor creation.
    frame = pd.read_csv(p["dataset"], sep=" ", dtype=str).set_index("charging_station_id")
    if not frame.index.is_unique:
        raise ValueError("Nonunique sample IDs")
    frame = frame.loc[np.concatenate([ids["train"], ids["validation"]])]
    result = {}
    for split in ("train", "validation"):
        rows = frame.loc[ids[split]]
        seqs = [parse_power_sequence(v) for v in rows["charging_powers_str"]]
        if any(len(s) < 18 or len(s) > p["max_length"] for s in seqs):
            raise ValueError("Unexpected sequence length")
        labels = rows["InsertedColumn"].to_numpy(str)
        if not set(labels).issubset(p["classes"]):
            raise ValueError("Unexpected label")
        result[split] = {
            "x": np.stack([prepare_sequence(s, p["max_length"], "none") for s in seqs]),
            "mask": np.stack([sequence_mask(len(s), p["max_length"]) for s in seqs]),
            "y": np.asarray([p["classes"].index(v) for v in labels], dtype=np.int64),
            "ids": ids[split],
        }
    return result


def validation(model, loader, device):
    model.eval()
    probs, truths = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation output")
            total_loss += float(loss) * len(y)
            probs.append(logits.softmax(-1).cpu().numpy())
            truths.append(y.cpu().numpy())
    probs, y = np.concatenate(probs), np.concatenate(truths)
    pred = probs.argmax(1)
    cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3).tolist()
    return {**metrics_from_confusion(cm), "confusion_matrix": cm,
            "val_loss": total_loss / len(y)}, probs


def save_predictions(path, ids, y, probabilities):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "y_true", "y_pred", "p0", "p1", "p2"])
        for sid, label, prob in zip(ids, y, probabilities, strict=True):
            writer.writerow([sid, int(label), int(prob.argmax()), *map(float, prob)])


def run_case(p, f, fraction, condition, data, deadline):
    directory = root(p) / case_name(fraction, condition)
    directory.mkdir(exist_ok=True)
    if (directory / "result.json").exists():
        return read(directory / "result.json")
    seed_everything(torch, p["seed"])
    d, v = data["train"], data["validation"]
    indices = stratified_train_subset_indices(d["y"], d["ids"], fraction, p["seed"])
    ids = d["ids"][indices]
    np.save(directory / "train_ids.npy", ids)
    generator = torch.Generator().manual_seed(p["seed"])
    train = make_loader(d["x"][indices], d["mask"][indices], d["y"][indices], p["batch_size"], 0,
                        shuffle=True, generator=generator)
    val = make_loader(v["x"], v["mask"], v["y"], p["evaluation_batch_size"], 0, shuffle=False)
    device = torch.device("cuda")
    model = build_model(condition).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    history, selected_f1, stale, start = [], -1.0, 0, 1
    checkpoint = directory / "checkpoint_latest.pt"
    metadata = {"protocol_sha256": f["protocol_sha256"], "condition": condition,
                "fraction": fraction, "train_ids_sha256": train_subset_sha256(ids)}
    if checkpoint.exists():
        start, history, selected_f1, stale = resume_training_checkpoint(
            torch=torch, path=checkpoint, model=model, optimizer=optimizer,
            data_generator=generator, device=torch.device("cpu"), scheduler=scheduler, expected_metadata=metadata)
    peak_f1 = max((row["macro_f1"] for row in history), default=-1.0)
    for epoch in range(start, p["epochs"] + 1):
        if condition == "current_early7" and stale >= p["early_stopping_patience"]:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Frozen suite wall-time budget reached; partial results retained")
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        total_loss = 0.0
        lr = float(optimizer.param_groups[0]["lr"])
        for x, mask, y in train:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite training output; no fallback")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), p["gradient_clip_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients; no skipped step")
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
        metrics, probabilities = validation(model, val, device)
        scheduler.step(metrics["macro_f1"])
        metrics.update(epoch=epoch, train_loss=total_loss / len(ids), learning_rate_used=lr,
                       learning_rate_next=float(optimizer.param_groups[0]["lr"]),
                       optimizer_steps=len(train), epoch_seconds=time.monotonic() - started,
                       peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2)
        history.append(metrics)
        if metrics["macro_f1"] > peak_f1:
            peak_f1 = metrics["macro_f1"]
            atomic_torch_save(torch, model.state_dict(), directory / "model_peak.pt")
            save_predictions(directory / "validation_peak.csv", v["ids"], v["y"], probabilities)
        if metrics["macro_f1"] > selected_f1 + p["early_stopping_min_delta"]:
            selected_f1, stale = metrics["macro_f1"], 0
            atomic_torch_save(torch, model.state_dict(), directory / "model_selected.pt")
            save_predictions(directory / "validation_selected.csv", v["ids"], v["y"], probabilities)
        else:
            stale += 1
        save_training_checkpoint(torch=torch, path=checkpoint, epoch=epoch, model=model,
                                 optimizer=optimizer, history=history, best_macro_f1=selected_f1,
                                 epochs_without_improvement=stale, data_generator=generator,
                                 scheduler=scheduler, metadata=metadata)
        atomic_write_json(directory / "history.json", history)
        atomic_write_json(directory / "status.json", {"status": "running", "epoch": epoch, "updated_at": now(), "test_evaluated": False})
        print(f"{case_name(fraction, condition)} epoch={epoch} f1={100*metrics['macro_f1']:.4f} seconds={metrics['epoch_seconds']:.1f}", flush=True)
    metrics, probabilities = validation(model, val, device)
    save_predictions(directory / "validation_endpoint.csv", v["ids"], v["y"], probabilities)
    peak_epoch = max(history, key=lambda row: row["macro_f1"])["epoch"]
    selected_epoch = next(row["epoch"] for row in history if row["macro_f1"] == selected_f1)
    result = {**metadata, "test_evaluated": False, "train_count": len(ids),
              "validation_count": len(v["ids"]), "initial_model_sha256": initial_hash,
              "total_parameters": sum(t.numel() for t in model.parameters()),
              "epochs": len(history), "optimizer_steps": sum(row["optimizer_steps"] for row in history),
              "selected_epoch": selected_epoch, "selected_macro_f1": selected_f1,
              "peak_epoch": peak_epoch, "peak_macro_f1": peak_f1,
              "endpoint_metrics": metrics, "training_seconds": sum(row["epoch_seconds"] for row in history),
              "output_sha256": {q.name: sha256_file(q) for q in sorted(directory.iterdir()) if q.suffix in {".pt", ".csv", ".npy"} or q.name == "history.json"}}
    atomic_write_json(directory / "result.json", result)
    atomic_write_json(directory / "status.json", {"status": "completed", "updated_at": now(), "test_evaluated": False})
    del model, optimizer, scheduler, train, val
    torch.cuda.empty_cache()
    return result


def audit(path):
    f = verify(path)
    p = f["protocol"]
    r = root(p)
    data = load_train_validation(p)
    rows, manifest, prediction_rows = [], {}, 0
    for fraction in p["train_fractions"]:
        paired = []
        expected = stratified_train_subset_indices(data["train"]["y"], data["train"]["ids"], fraction, p["seed"])
        for condition in CONDITIONS:
            directory = r / case_name(fraction, condition)
            result = read(directory / "result.json")
            history = read(directory / "history.json")
            if result["test_evaluated"] or result["protocol_sha256"] != f["protocol_sha256"]:
                raise ValueError("Unexpected result protocol")
            if not np.array_equal(np.load(directory / "train_ids.npy"), data["train"]["ids"][expected]):
                raise ValueError("Training subset mismatch")
            for filename, digest in result["output_sha256"].items():
                file = directory / filename
                if sha256_file(file) != digest:
                    raise ValueError(f"Changed output {file}")
                manifest[str(file)] = digest
            if result["epochs"] != len(history) or [h["epoch"] for h in history] != list(range(1, len(history) + 1)):
                raise ValueError("Incomplete epoch history")
            if result["optimizer_steps"] != len(history) * math.ceil(len(expected) / p["batch_size"]):
                raise ValueError("Optimizer step count mismatch")
            if condition != "current_early7" and len(history) != p["epochs"]:
                raise ValueError("Fixed-budget condition incomplete")
            selected, stale, selected_epoch = -1.0, 0, 0
            for i, h in enumerate(history):
                if not all(math.isfinite(h[k]) for k in ("train_loss", "val_loss", "macro_f1")):
                    raise ValueError("Nonfinite trajectory")
                if stale >= 7 and condition == "current_early7":
                    raise ValueError("Early stopping rule violated")
                if h["macro_f1"] > selected + p["early_stopping_min_delta"]:
                    selected, stale, selected_epoch = h["macro_f1"], 0, h["epoch"]
                else:
                    stale += 1
            if condition == "current_early7" and len(history) < 50 and stale < 7:
                raise ValueError("Premature early stopping")
            if selected != result["selected_macro_f1"] or selected_epoch != result["selected_epoch"]:
                raise ValueError("Selection rule mismatch")
            if max(h["macro_f1"] for h in history) != result["peak_macro_f1"]:
                raise ValueError("Peak mismatch")
            for suffix, score in (("selected", selected), ("peak", result["peak_macro_f1"]), ("endpoint", history[-1]["macro_f1"])):
                pred = pd.read_csv(directory / f"validation_{suffix}.csv", dtype={"sample_id": str})
                v = data["validation"]
                probs = pred[["p0", "p1", "p2"]].to_numpy()
                if (not np.array_equal(pred.sample_id.to_numpy(), v["ids"])
                    or not np.array_equal(pred.y_true.to_numpy(), v["y"])
                    or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
                    or not np.allclose(probs.sum(1), 1, atol=2e-7, rtol=0)
                    or not np.array_equal(probs.argmax(1), pred.y_pred.to_numpy())):
                    raise ValueError("Validation row/probability mismatch")
                cm = np.bincount(3 * v["y"] + pred.y_pred.to_numpy(), minlength=9).reshape(3, 3).tolist()
                if abs(metrics_from_confusion(cm)["macro_f1"] - score) > 1e-12:
                    raise ValueError("Independent F1 mismatch")
                prediction_rows += len(pred)
            rows.append(result)
            paired.append((result, history))
        for i, j in ((0, 1), (2, 3)):
            if paired[i][0]["initial_model_sha256"] != paired[j][0]["initial_model_sha256"]:
                raise ValueError("Paired initialization mismatch")
        # Current arms must share the exact scientific trajectory until early stopping.
        keys = ("epoch", "train_loss", "val_loss", "macro_f1", "confusion_matrix", "learning_rate_used", "learning_rate_next", "optimizer_steps")
        for a, b in zip(paired[0][1], paired[1][1]):
            if any(a[k] != b[k] for k in keys):
                raise ValueError("Early-stop control trajectory diverged")
    summary = {"audit_status": "passed", "recorded_at": now(), "test_evaluated": False,
               "prior_test_results_known": True, "conditions": len(rows), "per_condition": rows,
               "epochs_audited": sum(row["epochs"] for row in rows),
               "prediction_rows_audited": prediction_rows,
               "limitations": "Exploratory reused seed43/validation; controlled Torch structural port, not original Keras replay or global optimization; no independent test/generalization claim.",
               "automatic_expansion": False}
    atomic_write_json(r / "output_manifest.json", manifest)
    atomic_write_json(r / "summary.json", summary)
    atomic_write_json(r / "suite_status.json", {"status": "completed", "updated_at": now(), "test_evaluated": False})
    report(p, summary)
    return summary


def report(p, s):
    lines = ["# E05B：legacy TCN结构与早停诊断结果（validation-only）", "",
             "单seed43、原始序列、统一split/AdamW/FP32；已知历史test之后的validation探索。没有新的test评价。", "",
             "| 标签预算 | 条件 | 轮数 | 选择epoch | 选择F1 % | 真峰值F1 % | 末轮F1 % | 参数 | 训练分钟 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in s["per_condition"]:
        lines.append(f"| {row['fraction']*100:g}% | {row['condition']} | {row['epochs']} | {row['selected_epoch']} | {100*row['selected_macro_f1']:.4f} | {100*row['peak_macro_f1']:.4f} | {100*row['endpoint_metrics']['macro_f1']:.4f} | {row['total_parameters']} | {row['training_seconds']/60:.2f} |")
    lines += ["", "选择F1按当前min_delta=0.0001规则；真峰值独立保存，端点不冒充最佳模型。", ""]
    for fraction in p["train_fractions"]:
        subset = {row["condition"]: row for row in s["per_condition"] if row["fraction"] == fraction}
        a, b, c, d = [subset[k] for k in CONDITIONS]
        lines.append(f"- {fraction*100:g}%标签：固定50轮−early7 **{100*(b['selected_macro_f1']-a['selected_macro_f1']):+.4f}pp**；legacy masked−当前固定50轮 **{100*(c['selected_macro_f1']-b['selected_macro_f1']):+.4f}pp**；legacy unmasked−masked **{100*(d['selected_macro_f1']-c['selected_macro_f1']):+.4f}pp**。")
    lines += ["", f"独立审计{s['epochs_audited']}epoch、{s['prediction_rows_audited']}行保存validation预测（每条件选择/峰值/端点各3510行）、参数起点、训练ID、科学轨迹前缀、更新步数与SHA均通过。", "",
              "本轮只回答这些固定训练配置下的诊断问题。legacy移植统一了初始化规则、优化器和padding，与原Keras的Glorot/Adam/val_loss调度不同；没有学习率/正则化搜索，也没有多seed验证，不能声称已穷尽原模型性能。", "",
              "seed43与validation多次复用；不据此改写E03T门槛、不自动扩seed或消费已有test。原完整标签TCN历史95.99%±0.73%是另一实验的test，不能与本表validation直接排名。", "",
              f"证据：`{root(p)}/summary.json`、`output_manifest.json`、各条件保存checkpoint/历史/预测及冻结源码。"]
    Path("docs/experiment_records/e05b_tcn_legacy_pilot_results_20261019.md").write_text("\n".join(lines) + "\n")


def run(path):
    p = protocol(path)
    r = root(p)
    r.mkdir(parents=True, exist_ok=True)
    with (r / "suite.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            f = verify(path)
            if (r / "summary.json").exists():
                return audit(path)
            configure()
            if not torch.cuda.is_available():
                raise RuntimeError("Frozen GPU pilot requires CUDA")
            previous = read(r / "suite_status.json") if (r / "suite_status.json").exists() else {}
            started_at = previous.get("started_at", now())
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
            deadline = time.monotonic() + p["wall_time_limit_hours"] * 3600 - elapsed
            atomic_write_json(r / "suite_status.json", {"status": "running", "started_at": started_at, "updated_at": now(), "test_evaluated": False})
            data = load_train_validation(p)
            for fraction in p["train_fractions"]:
                for condition in CONDITIONS:
                    run_case(p, f, fraction, condition, data, deadline)
            audit(path)
        except BaseException as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            previous = read(r / "suite_status.json") if (r / "suite_status.json").exists() else {}
            atomic_write_json(r / "suite_status.json", {**previous, "status": status, "updated_at": now(), "test_evaluated": False,
                                                       "error_type": type(error).__name__, "error": str(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "run", "audit"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.action](args.protocol)


if __name__ == "__main__":
    main()
