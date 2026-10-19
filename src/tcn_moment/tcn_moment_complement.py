"""E06: bounded, post-test train/validation TCN/MOMENT complement experiment."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from tcn_moment import replicate_tcn as replication
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import normalize_sequence, stratified_train_subset_indices, train_subset_sha256
from tcn_moment.io_utils import atomic_torch_save, atomic_write_json, sha256_file
from tcn_moment.training_utils import seed_everything

DEFAULT = Path("configs/experiments/improvements/e06_tcn_moment_complement_protocol.json")
CONDITIONS = ["tcn_only", "tcn_moment", "tcn_random"]
PACKAGES = (*replication.PACKAGES, "momentfm", "transformers")
NEW_FILES = [
    "src/tcn_moment/tcn_moment_complement.py",
    "tests/test_tcn_moment_complement.py",
    "scripts/run_e06_remote.sh",
    "scripts/audit_e06_saved_outputs.py",
    "scripts/audit_e05c_saved_outputs.py",
    "docs/experiment_records/e06_tcn_moment_complement_protocol_20261019.md",
]


def protocol(path):
    p = pilot.read(path)
    if (p["experiment_id"] != "E06_TCN_MOMENT_COMPLEMENT"
        or p["seeds"] != [42, 44, 45, 46] or p["train_fraction"] != 0.1
        or p["selected_train_count"] != 2456 or p["validation_count"] != 3510
        or p["conditions"] != CONDITIONS or p["maximum_head_fits"] != 12
        or p["parent_condition"] != "fraction010_legacy_masked50"
        or p["wall_time_limit_hours"] != 2 or p["classes"] != ["0", "1", "2"]
        or not p["prior_test_results_known"]
        or not p["validation_reused_for_parent_and_head_selection"]
        or any(p[k] for k in ("test_evaluated", "claim_total_label_efficiency",
                              "automatic_next_stage", "automatic_shutdown", "pi_wake_enabled"))):
        raise ValueError("Unexpected experiment scope")
    h = p["head"]
    if (h["type"] != "frozen_parent_logits_plus_zero_output_mlp"
        or [h[k] for k in ("tcn_dimension", "side_dimension", "hidden_dimension")] != [128, 1024, 128]
        or h["dropout"] != 0.1 or h["epochs"] != 100 or h["batch_size"] != 128
        or h["optimizer"] != "AdamW" or h["learning_rate"] != 0.001
        or h["weight_decay"] != 0.0001 or h["gradient_clip_norm"] != 1.0
        or h["scheduler"] != "none" or h["precision"] != "fp32_tf32_disabled"
        or h["early_stopping"]
        or h["selection"] != "validation_macro_f1_strict_improvement_epoch0_eligible_first_tie"):
        raise ValueError("Head budget changed")
    if p["success_rules"] != {
        "fusion_minus_matched_tcn_head_mean_pp_at_least": 0.5,
        "fusion_minus_original_tcn_mean_pp_at_least": 0.5,
        "fusion_minus_random_control_mean_pp_at_least": 0.25,
        "minimum_positive_seeds_each": 3, "all_three_required": True,
    }:
        raise ValueError("Success rules changed")
    return p


def root(p):
    return Path(p["output_dir"]) / p["suite"]


def parent_case(p, seed):
    return Path(p["parent_root"]) / f"seed{seed}" / p["parent_condition"]


def runtime():
    return {"host": platform.node(), "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "cuda": torch.version.cuda}


def finite(array):
    if not np.isfinite(array).all():
        raise FloatingPointError("Nonfinite feature/probability")
    return array


def checked_archive(path, expected_ids, expected_shape=None):
    with np.load(path, allow_pickle=False) as a:
        ids, features = a["sample_ids"], a["features"]
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"Feature ID/order mismatch: {path}")
    if features.ndim != 2 or (expected_shape and features.shape != tuple(expected_shape)):
        raise ValueError("Invalid feature shape")
    if len(features) != len(ids) or features.shape[1] != 1024:
        raise ValueError("Unexpected MOMENT dimensions")
    return finite(features)


def cache_files(p, seed, random=False):
    """Validate immutable historical caches, not a recomputation on new hardware."""
    directory = Path(p["random_cache"] if random else p["moment_cache"]) / f"seed{seed}"
    if random and seed == 42:
        if directory.exists():
            raise ValueError("Seed42 random cache unexpectedly exists; protocol must be revised")
        return []
    manifest_file = directory / "manifest.json"
    m = pilot.read(manifest_file)
    s = m["signature"]
    parent = pilot.read(p["parent_protocol"])
    split_file = Path(parent["split_manifests"][str(seed)])
    split = pilot.read(split_file)
    if (s["dataset_sha256"] != p["dataset_sha256"] or s["split_sha256"] != sha256_file(split_file)
        or s["train_ids_sha256"] != train_subset_sha256(np.asarray(split["splits"]["train"]))
        or s["validation_ids_sha256"] != train_subset_sha256(np.asarray(split["splits"]["validation"]))
        or s["max_length"] != 1024 or s["normalization"] != "zscore" or s["precision"] != "fp32"
        or s["protocol_version"] != 1 or s["moment_protocol_version"] != 2
        or s["initialization"] != ("random" if random else "pretrained") or s["seed"] != seed
        or s["model_id"] != "AutonLab/MOMENT-1-large"
        or s["model_config_sha256"] != sha256_file(Path("configs/models/moment-1-large.json"))):
        raise ValueError("Historical cache signature mismatch")
    for name, want in s["source_hashes"].items():
        if sha256_file(Path("src/tcn_moment") / name) != want:
            raise ValueError(f"Cache producer source changed: {name}")
    for name, want in s["packages"].items():
        if importlib.metadata.version(name) != want:
            raise ValueError("Cache dependency version changed")
    files = [manifest_file]
    if random:
        guard_file = directory / "pretrained_loading_guard.json"
        guard = pilot.read(guard_file)
        if (not guard["guard_enabled"] or guard["pretrained_checkpoint_loaded"]
            or guard["random_initialization_seed"] != seed
            or guard["manifest_sha256"] != sha256_file(manifest_file)
            or guard["source_sha256"] != sha256_file(Path("src/tcn_moment/train_e01c.py"))
            or s["checkpoint"] != {"initialization": "random", "seed": seed}):
            raise ValueError("Random cache loading provenance mismatch")
        files.append(guard_file)
    else:
        fingerprint = Path("artifacts/e01_model_weight_fingerprint.json")
        if s["checkpoint"] != pilot.read(fingerprint):
            raise ValueError("Pretrained fingerprint mismatch")
        files.append(fingerprint)
    for split_name in ("train", "validation"):
        file = directory / f"{split_name}.npz"
        record = m["splits"][split_name]
        if sha256_file(file) != record["sha256"]:
            raise ValueError("Corrupt feature cache")
        checked_archive(file, np.asarray(split["splits"][split_name]), record["shape"])
        files.append(file)
    return files


class ResidualHead(nn.Module):
    def __init__(self, input_dim=1152, hidden=128, dropout=0.1):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, parent_logits):
        return parent_logits + self.output(self.dropout(torch.relu(self.hidden(features))))


def branch_features(tcn_train, tcn_val, side_train, side_val):
    """Fit only selected train. None represents the zero-information side control."""
    if tcn_train.shape[1] != 128 or tcn_val.shape[1] != 128:
        raise ValueError("Invalid TCN feature dimension")
    scaler = StandardScaler().fit(finite(tcn_train))
    train = scaler.transform(tcn_train).astype(np.float32) / math.sqrt(128)
    val = scaler.transform(finite(tcn_val)).astype(np.float32) / math.sqrt(128)
    side_scaler = None
    if side_train is None and side_val is None:
        a, b = np.zeros((len(train), 1024), np.float32), np.zeros((len(val), 1024), np.float32)
    else:
        if (side_train is None or side_val is None
            or side_train.shape != (len(train), 1024) or side_val.shape != (len(val), 1024)):
            raise ValueError("Invalid side features")
        side_scaler = StandardScaler().fit(finite(side_train))
        a = side_scaler.transform(side_train).astype(np.float32) / math.sqrt(1024)
        b = side_scaler.transform(finite(side_val)).astype(np.float32) / math.sqrt(1024)
    return finite(np.concatenate([train, a], 1)), finite(np.concatenate([val, b], 1)), {
        "tcn": scaler, "side": side_scaler,
    }


def pooled_tcn(model, x, mask):
    features = model.encoder(x)
    weights = mask[:, None, :].to(features.dtype)
    return (features * weights).sum(-1) / weights.sum(-1).clamp_min(1)


def synthetic_check():
    pilot.configure()
    seed_everything(torch, 123)
    model = pilot.LegacyTCN(masked=True).eval()
    x, mask = torch.randn(3, 1, 64), torch.ones(3, 64)
    mask[:, 43:] = 0
    with torch.no_grad():
        pooled = pooled_tcn(model, x, mask)
        torch.testing.assert_close(model.classifier(pooled), model(x, mask), rtol=0, atol=0)
        head = ResidualHead().eval()
        parent = model(x, mask)
        torch.testing.assert_close(head(torch.randn(3, 1152), parent), parent, rtol=0, atol=0)
    return {"pooled_logits_equal": True, "zero_residual_restores_parent": True,
            "project_data_used": False}


def freeze(path):
    p = protocol(path)
    r = root(p)
    r.mkdir(parents=True, exist_ok=True)
    if (r / "frozen_protocol.json").exists():
        return verify(path)
    if any(r.glob("seed*")) or (r / "started.json").exists():
        raise ValueError("Freeze must precede real experiment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA server required")
    if sha256_file(Path(p["dataset"])) != p["dataset_sha256"]:
        raise ValueError("Dataset changed")
    parent_protocol = Path(p["parent_protocol"])
    if sha256_file(parent_protocol) != p["parent_protocol_sha256"]:
        raise ValueError("Parent protocol hash changed")
    parent = pilot.read(parent_protocol)
    parent_freeze = Path(p["parent_root"]) / "frozen_protocol.json"
    if pilot.read(parent_freeze)["protocol_sha256"] != p["parent_protocol_sha256"]:
        raise ValueError("Wrong frozen parent")
    if pilot.read(Path(p["parent_root"]) / "suite_status.json")["status"] != "completed":
        raise ValueError("TCN parent incomplete")
    files = {path, Path(p["dataset"]), parent_protocol, parent_freeze,
             Path(parent["parent_protocol"]), Path(p["moment_config"]),
             Path("configs/models/moment-1-large.json"), *map(Path, NEW_FILES)}
    files.update(Path("src/tcn_moment").glob("*.py"))
    files.update(Path("configs").rglob("*.yaml"))
    for seed in p["seeds"]:
        split_file = Path(parent["split_manifests"][str(seed)])
        replication.validate_split(pilot.read(split_file), pilot.read(parent["parent_protocol"]), seed)
        files.add(split_file)
        directory = parent_case(p, seed)
        result = pilot.read(directory / "result.json")
        if result["fraction"] != p["train_fraction"] or result["train_count"] != p["selected_train_count"]:
            raise ValueError("Parent train budget mismatch")
        for name in ("model_selected.pt", "validation_selected.csv", "train_ids.npy"):
            file = directory / name
            if sha256_file(file) != result["output_sha256"][name]:
                raise ValueError("Parent output changed")
            files.add(file)
        files.add(directory / "result.json")
        files.update(cache_files(p, seed))
        files.update(cache_files(p, seed, random=True))
    checks = synthetic_check()
    f = {"protocol": p, "protocol_sha256": sha256_file(path), "frozen_at": pilot.now(),
         "file_sha256": {str(file): sha256_file(file) for file in sorted(files)},
         "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
         "runtime": runtime(), "synthetic_checks": checks,
         "new_real_head_outcomes_observed": False, "prior_test_results_known": True}
    for file in files:
        if file.parts[0] in {"src", "configs", "tests", "scripts", "docs"}:
            destination = r / "source_snapshot" / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
        elif file.suffix in {".json", ".pt", ".npy"}:
            destination = r / "input_snapshot" / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    atomic_write_json(r / "frozen_protocol.json", f)
    print(f"E06 frozen {len(files)} inputs; 12 heads, no test; {f['frozen_at']}", flush=True)
    return f


def verify(path):
    p = protocol(path)
    f = pilot.read(root(p) / "frozen_protocol.json")
    if f["protocol"] != p or f["protocol_sha256"] != sha256_file(path):
        raise ValueError("Protocol changed after freeze")
    for file, expected in f["file_sha256"].items():
        if sha256_file(Path(file)) != expected:
            raise ValueError(f"Frozen input changed: {file}")
    if f["packages"] != {name: importlib.metadata.version(name) for name in PACKAGES}:
        raise ValueError("Dependencies changed")
    if f["runtime"] != runtime():
        raise ValueError("Host/GPU changed; stop before migration")
    return f


def load_selected_data(p, seed):
    parent = pilot.read(p["parent_protocol"])
    base = pilot.read(parent["parent_protocol"])
    data = pilot.load_train_validation({**base, "seed": seed,
                                       "split_manifest": parent["split_manifests"][str(seed)]})
    full_ids = data["train"]["ids"].copy()
    ix = stratified_train_subset_indices(data["train"]["y"], full_ids, p["train_fraction"], seed)
    data["train"] = {key: value[ix] for key, value in data["train"].items()}
    if len(ix) != p["selected_train_count"] or len(data["validation"]["y"]) != p["validation_count"]:
        raise ValueError("Unexpected sample count")
    if not np.array_equal(data["train"]["ids"], np.load(parent_case(p, seed) / "train_ids.npy")):
        raise ValueError("Selected train IDs differ from supervised TCN training")
    return data, full_ids, ix


def extract_tcn(model, d, batch_size, device="cuda"):
    features, logits = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(d["y"]), batch_size):
            x = torch.as_tensor(d["x"][start:start + batch_size, None, :], device=device)
            mask = torch.as_tensor(d["mask"][start:start + batch_size], device=device)
            pooled = pooled_tcn(model, x, mask)
            features.append(pooled.cpu().numpy())
            logits.append(model.classifier(pooled).cpu().numpy())
    return finite(np.concatenate(features)), finite(np.concatenate(logits))


def fresh_random(p, seed, data, directory, deadline):
    from dataclasses import replace

    from momentfm import MOMENTPipeline

    from tcn_moment.train_e01c import GUARDED_APIS, forbid_pretrained_loading
    from tcn_moment.train_moment import build_model, forward_features, set_num_classes

    if seed != 42:
        raise ValueError("Only seed42 permits a new random extraction")
    seed_everything(torch, seed)
    config = with_random_seed(load_config(Path(p["moment_config"])), seed)
    config = replace(config, model=replace(config.model, initialization="random"))
    result = {}
    started = time.monotonic()
    with forbid_pretrained_loading():
        model = build_model(config, MOMENTPipeline, 3)
        set_num_classes(model, 3)
        model.init()
        model.eval().to("cuda")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for split, d in data.items():
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(d["y"]), p["moment_extraction_batch_size"]):
                    check_deadline(deadline)
                    stop = start + p["moment_extraction_batch_size"]
                    raw, mask = d["x"][start:stop], d["mask"][start:stop]
                    x = np.zeros_like(raw)
                    for i in range(len(raw)):
                        length = int(mask[i].sum())
                        x[i, :length] = normalize_sequence(raw[i, :length], "zscore")
                    features = forward_features(model, torch.as_tensor(x[:, None, :], device="cuda"),
                                                torch.as_tensor(mask, device="cuda"))
                    chunks.append(finite(features.float().cpu().numpy()))
            result[split] = np.concatenate(chunks)
        tensors = list(model.encoder.parameters())
        marker = {"initialization": "random", "seed": seed,
                  "guarded_apis": GUARDED_APIS, "pretrained_checkpoint_loaded": False,
                  "total_parameters": sum(x.numel() for x in model.parameters()),
                  "encoder_first_values": tensors[0].detach().flatten()[:16].cpu().tolist(),
                  "config_sha256": sha256_file(config.model.config_path),
                  "seconds": time.monotonic() - started,
                  "selected_train_only": True, "test_evaluated": False}
    del model
    torch.cuda.empty_cache()
    atomic_write_json(directory / "fresh_random_provenance.json", marker)
    return result


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Frozen two-hour wall budget exhausted")


def npz_save(path, **values):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def prepare_features(p, seed, directory, deadline):
    started = time.monotonic()
    data, full_ids, ix = load_selected_data(p, seed)
    model = pilot.LegacyTCN(masked=True).to("cuda")
    model.load_state_dict(torch.load(parent_case(p, seed) / "model_selected.pt", map_location="cuda", weights_only=True))
    before = pilot.state_hash(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    extracted = {split: extract_tcn(model, d, p["tcn_extraction_batch_size"])
                 for split, d in data.items()}
    if before != pilot.state_hash(model):
        raise ValueError("Frozen TCN changed during extraction")
    del model
    torch.cuda.empty_cache()
    parent_prob = torch.from_numpy(extracted["validation"][1]).softmax(1).numpy()
    old, old_metrics = replication.read_predictions(parent_case(p, seed) / "validation_selected.csv",
                                                    data["validation"])
    drift = float(np.abs(old - parent_prob).max())
    disagreements = int(np.count_nonzero(old.argmax(1) != parent_prob.argmax(1)))
    if drift > p["restore_probability_atol"] or disagreements != p["restore_argmax_disagreements"]:
        raise ValueError("TCN checkpoint restore failed; do not relax tolerance")
    moment, random = {}, {}
    for split, d in data.items():
        expected = full_ids if split == "train" else d["ids"]
        m = checked_archive(Path(p["moment_cache"]) / f"seed{seed}" / f"{split}.npz", expected)
        moment[split] = m[ix] if split == "train" else m
        if seed != 42:
            a = checked_archive(Path(p["random_cache"]) / f"seed{seed}" / f"{split}.npz", expected)
            random[split] = a[ix] if split == "train" else a
    if seed == 42:
        random = fresh_random(p, seed, data, directory, deadline)
    for split, d in data.items():
        npz_save(directory / f"{split}_features.npz", sample_ids=d["ids"], labels=d["y"],
                 tcn=extracted[split][0], parent_logits=extracted[split][1],
                 moment=moment[split], random=random[split])
    np.save(directory / "train_ids.npy", data["train"]["ids"])
    pilot.save_predictions(directory / "original_tcn_validation.csv", data["validation"]["ids"],
                           data["validation"]["y"], parent_prob)
    atomic_write_json(directory / "feature_provenance.json", {
        "seed": seed, "train_count": len(ix), "validation_count": len(parent_prob),
        "train_ids_sha256": train_subset_sha256(data["train"]["ids"]),
        "tcn_state_sha256": before, "tcn_model_file_sha256": sha256_file(parent_case(p, seed) / "model_selected.pt"),
        "tcn_restore_max_probability_drift": drift, "tcn_restore_argmax_disagreements": disagreements,
        "original_tcn_metrics": old_metrics, "test_evaluated": False,
        "feature_preparation_seconds": time.monotonic() - started,
        "moment_cache_reused": True, "random_cache_reused": seed != 42,
    })
    print(f"seed{seed} features verified; original TCN F1={old_metrics['macro_f1']:.6f}", flush=True)


def load_features(directory):
    result = {}
    for split in ("train", "validation"):
        with np.load(directory / f"{split}_features.npz", allow_pickle=False) as a:
            result[split] = {key: a[key] for key in a.files}
    return result


def probability(head, x, parent):
    head.eval()
    with torch.no_grad():
        logits = head(x, parent)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite validation logits")
        return logits.softmax(1).cpu().numpy()


def train_head(p, frozen, seed, condition, directory, arrays, deadline):
    out = directory / condition
    out.mkdir()
    seed_everything(torch, seed)
    h = p["head"]
    t, v = arrays["train"], arrays["validation"]
    side = {"tcn_only": None, "tcn_moment": "moment", "tcn_random": "random"}[condition]
    x, z, scalers = branch_features(t["tcn"], v["tcn"], t[side] if side else None,
                                   v[side] if side else None)
    joblib.dump(scalers, out / "scalers.joblib")
    x, z = torch.from_numpy(x).cuda(), torch.from_numpy(z).cuda()
    base_train = torch.from_numpy(t["parent_logits"]).cuda()
    base_val = torch.from_numpy(v["parent_logits"]).cuda()
    labels = torch.from_numpy(t["labels"]).cuda()
    model = ResidualHead(hidden=h["hidden_dimension"], dropout=h["dropout"]).cuda()
    initial_hash = pilot.state_hash(model)
    atomic_torch_save(torch, model.state_dict(), out / "head_initial.pt")
    initial = probability(model, z, base_val)
    parent = base_val.softmax(1).cpu().numpy()
    if not np.array_equal(initial, parent):
        raise ValueError("Epoch0 is not exactly parent TCN")
    epoch0 = replication.prediction_metrics(v["labels"], initial)
    best_f1, best_epoch, best_prob = epoch0["macro_f1"], 0, initial
    best_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=h["learning_rate"],
                                  weight_decay=h["weight_decay"], fused=True)
    history = [{"epoch": 0, "validation_macro_f1": best_f1, "train_loss": None}]
    started = time.monotonic()
    for epoch in range(1, h["epochs"] + 1):
        check_deadline(deadline)
        model.train()
        order = torch.randperm(len(labels), device="cuda")
        losses = []
        for start in range(0, len(labels), h["batch_size"]):
            ids = order[start:start + h["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[ids], base_train[ids])
            loss = nn.functional.cross_entropy(logits, labels[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite head loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), h["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach() * len(ids))
        prob = probability(model, z, base_val)
        metrics = replication.prediction_metrics(v["labels"], prob)
        f1 = metrics["macro_f1"]
        history.append({"epoch": epoch, "validation_macro_f1": f1,
                        "train_loss": float(torch.stack(losses).sum() / len(labels))})
        if f1 > best_f1:
            best_f1, best_epoch, best_prob = f1, epoch, prob.copy()
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 25 == 0:
            atomic_write_json(out / "history.json", history)
            print(f"seed{seed} {condition} epoch{epoch}/100 selected={best_f1:.6f}", flush=True)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    atomic_torch_save(torch, model.state_dict(), out / "head_endpoint.pt")
    atomic_torch_save(torch, best_state, out / "head_selected.pt")
    atomic_write_json(out / "history.json", history)
    model.load_state_dict(torch.load(out / "head_selected.pt", map_location="cuda", weights_only=True))
    restored = probability(model, z, base_val)
    if not np.array_equal(best_prob, restored):
        raise ValueError("Saved best head does not restore exact probabilities")
    pilot.save_predictions(out / "validation_selected.csv", v["sample_ids"], v["labels"], restored)
    pilot.save_predictions(out / "validation_endpoint.csv", v["sample_ids"], v["labels"], prob)
    result = {"seed": seed, "condition": condition, "protocol_sha256": frozen["protocol_sha256"],
              "epochs": h["epochs"], "selected_epoch": best_epoch, "initial_head_state_sha256": initial_hash,
              "epoch0_metrics": epoch0, "selected_metrics": replication.prediction_metrics(v["labels"], restored),
              "endpoint_metrics": replication.prediction_metrics(v["labels"], prob),
              "head_fit_seconds": seconds, "head_parameters": sum(q.numel() for q in model.parameters()),
              "optimizer_steps": h["epochs"] * math.ceil(len(labels) / h["batch_size"]),
              "test_evaluated": False,
              "output_sha256": {file.name: sha256_file(file) for file in sorted(out.iterdir())}}
    atomic_write_json(out / "result.json", result)
    print(f"Completed seed{seed} {condition}: {best_f1:.6f} epoch{best_epoch}", flush=True)
    del model, optimizer
    torch.cuda.empty_cache()


def paired(values, threshold, minimum):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 3.182446305284263 * sd / math.sqrt(len(values))
    return {"per_seed_pp": values, "mean_pp": mean, "positive_seeds": sum(x > 0 for x in values),
            "descriptive_95pct_interval_pp": [mean - half, mean + half],
            "mean_threshold_pp": threshold,
            "passed": mean >= threshold and sum(x > 0 for x in values) >= minimum}


def summarize_rows(rows, p):
    by_seed = {seed: {r["condition"]: r for r in rows if r["seed"] == seed} for seed in p["seeds"]}
    aggregate = {}
    for condition in ["original_tcn", *CONDITIONS]:
        values = [by_seed[seed][condition]["metrics"]["macro_f1"] * 100 for seed in p["seeds"]]
        aggregate[condition] = {"macro_f1_mean_percent": statistics.mean(values),
                                "macro_f1_sd_percent": statistics.stdev(values), "per_seed_percent": values}
    comparisons = {}
    for reference, threshold in (("tcn_only", 0.5), ("original_tcn", 0.5), ("tcn_random", 0.25)):
        values = [(by_seed[s]["tcn_moment"]["metrics"]["macro_f1"]
                   - by_seed[s][reference]["metrics"]["macro_f1"]) * 100 for s in p["seeds"]]
        comparisons[reference] = paired(values, threshold, 3)
    return {"aggregate": aggregate, "comparisons": comparisons,
            "all_success_rules_passed": all(x["passed"] for x in comparisons.values())}


def audit(p, frozen):
    r = root(p)
    rows = []
    for seed in p["seeds"]:
        directory = r / f"seed{seed}"
        arrays = load_features(directory)
        v = {"ids": arrays["validation"]["sample_ids"], "y": arrays["validation"]["labels"]}
        parent = pilot.read(p["parent_protocol"])
        manifest = pilot.read(parent["split_manifests"][str(seed)])
        if not np.array_equal(v["ids"], np.asarray(manifest["splits"]["validation"])):
            raise ValueError("Validation feature IDs changed")
        if not np.array_equal(arrays["train"]["sample_ids"], np.load(parent_case(p, seed) / "train_ids.npy")):
            raise ValueError("Train feature IDs changed")
        _, old = replication.read_predictions(directory / "original_tcn_validation.csv", v)
        rows.append({"seed": seed, "condition": "original_tcn", "metrics": old})
        initial_hashes = []
        for condition in CONDITIONS:
            out = directory / condition
            result = pilot.read(out / "result.json")
            if (result["seed"] != seed or result["condition"] != condition
                or result["protocol_sha256"] != frozen["protocol_sha256"] or result["test_evaluated"]
                or result["epochs"] != p["head"]["epochs"]):
                raise ValueError("Head provenance mismatch")
            for name, expected in result["output_sha256"].items():
                if sha256_file(out / name) != expected:
                    raise ValueError("Head output changed")
            initial_hashes.append(result["initial_head_state_sha256"])
            history = pilot.read(out / "history.json")
            if len(history) != 101 or [x["epoch"] for x in history] != list(range(101)):
                raise ValueError("Incomplete history")
            selected = max(history, key=lambda row: row["validation_macro_f1"])
            if selected["epoch"] != result["selected_epoch"]:
                raise ValueError("Best epoch selection mismatch")
            for suffix in ("selected", "endpoint"):
                _, metrics = replication.read_predictions(out / f"validation_{suffix}.csv", v,
                                                          result[f"{suffix}_metrics"]["macro_f1"])
                for key in ("accuracy", "balanced_accuracy", "macro_f1", "battery_AP", "battery_recall", "battery_FPR"):
                    if abs(metrics[key] - result[f"{suffix}_metrics"][key]) > 1e-12:
                        raise ValueError("Independent saved CSV metric mismatch")
            rows.append({"seed": seed, "condition": condition, "metrics": result["selected_metrics"],
                         "selected_epoch": result["selected_epoch"], "head_fit_seconds": result["head_fit_seconds"]})
        if len(set(initial_hashes)) != 1:
            raise ValueError("Paired head initialization mismatch")
    completed_at = (pilot.read(r / "summary.json")["completed_at"]
                    if (r / "summary.json").exists() else pilot.now())
    summary = {"protocol_sha256": frozen["protocol_sha256"], "completed_at": completed_at,
               "audit_status": "passed", "test_evaluated": False, "post_test_validation_exploration": True,
               "seeds": p["seeds"], "train_count": 2456, "validation_count": 3510,
               "rows": rows, **summarize_rows(rows, p)}
    if (r / "summary.json").exists() and pilot.read(r / "summary.json") != summary:
        raise ValueError("Completed summary changed; refuse to overwrite")
    atomic_write_json(r / "summary.json", summary)
    lines = ["# E06 TCN + MOMENT complement (validation only)", "",
             "Post-test exploration; repeated validation selection; not independent confirmation.", "",
             "| Method | Macro-F1 mean ± sample SD (%) |", "|---|---:|"]
    for name, a in summary["aggregate"].items():
        lines.append(f"| {name} | {a['macro_f1_mean_percent']:.4f} ± {a['macro_f1_sd_percent']:.4f} |")
    lines += ["", "| MOMENT fusion minus | Mean pp | Positive seeds | Passed |", "|---|---:|---:|---|"]
    for name, comparison in summary["comparisons"].items():
        lines.append(f"| {name} | {comparison['mean_pp']:+.4f} | {comparison['positive_seeds']}/4 | {comparison['passed']} |")
    lines += ["", f"All original rules passed: {summary['all_success_rules_passed']}", "",
              "2456 training labels plus 3510 repeatedly used validation labels; no total-label-efficiency claim.",
              "No new test evaluation, threshold search, automatic expansion or shutdown."]
    (r / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def run(path):
    frozen = verify(path)
    p = frozen["protocol"]
    r = root(p)
    lock = (r / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (r / "started.json").exists():
        if (r / "suite_status.json").exists() and pilot.read(r / "suite_status.json")["status"] == "completed":
            return audit(p, frozen)
        raise ValueError("Incomplete previous run retained; no silent restart or overwrite")
    if any(r.glob("seed*")):
        raise ValueError("Unexpected real-data outputs before run")
    started = time.monotonic()
    deadline = started + p["wall_time_limit_hours"] * 3600
    pilot.configure()
    atomic_write_json(r / "started.json", {"started_at": pilot.now(), "runtime": runtime(),
                                           "protocol_sha256": frozen["protocol_sha256"]})
    atomic_write_json(r / "suite_status.json", {"status": "running", "test_evaluated": False})
    try:
        for seed in p["seeds"]:
            check_deadline(deadline)
            directory = r / f"seed{seed}"
            directory.mkdir()
            prepare_features(p, seed, directory, deadline)
            arrays = load_features(directory)
            for condition in CONDITIONS:
                check_deadline(deadline)
                train_head(p, frozen, seed, condition, directory, arrays, deadline)
        verify(path)
        summary = audit(p, frozen)
        atomic_write_json(r / "suite_status.json", {"status": "completed", "completed_at": pilot.now(),
                                                    "wall_seconds": time.monotonic() - started,
                                                    "test_evaluated": False, "automatic_expansion": False,
                                                    "all_success_rules_passed": summary["all_success_rules_passed"]})
        atomic_write_json(r / "output_manifest.json", {
            str(file.relative_to(r)): sha256_file(file) for file in sorted(r.rglob("*"))
            if file.is_file() and file.name not in {"output_manifest.json", "run.lock"}
        })
        print(json.dumps(summary["comparisons"], indent=2), flush=True)
        return summary
    except BaseException as error:
        atomic_write_json(r / "suite_status.json", {"status": "failed", "failed_at": pilot.now(),
                                                    "error": repr(error), "test_evaluated": False})
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "audit", "synthetic"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.command == "synthetic":
        print(synthetic_check())
    elif args.command == "freeze":
        freeze(args.protocol)
    elif args.command == "run":
        run(args.protocol)
    else:
        frozen = verify(args.protocol)
        audit(frozen["protocol"], frozen)


if __name__ == "__main__":
    main()
