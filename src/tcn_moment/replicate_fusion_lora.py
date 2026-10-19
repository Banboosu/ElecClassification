"""E07: compact four-seed strong-fusion-head LoRA replication, no test path."""
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
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from tcn_moment import replicate_gain_heads as parent
from tcn_moment import strong_head_lora as pilot
from tcn_moment import tcn_legacy_pilot as data_loader
from tcn_moment import train_e03
from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import stratified_train_subset_indices, train_subset_sha256
from tcn_moment.head_diagnostics import GainFusionHead, fan_in_gain
from tcn_moment.head_diagnostics_stable import FixedHeadGradScaler
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.moment_peft import (
    PeftClassifier,
    adapter_parameters,
    capture_learned_state,
    frozen_parameter_digest,
    optimizer_groups,
    restore_learned_state,
    trainable_summary,
)
from tcn_moment.strong_head_lora_online import OnlineFrozenClassifier
from tcn_moment.train_baselines import extract_statistical_features
from tcn_moment.train_feature_fusion import verified_checkpoint

DEFAULT = Path("configs/experiments/improvements/e07_fusion_lora_replication_protocol.json")
CONDITIONS = [{"name": "frozen_fusion", "branch": "fusion", "rank": 0},
              {"name": "lora8_fusion", "branch": "fusion", "rank": 8}]
PACKAGES = (*pilot.PACKAGES, "pandas", "scipy", "joblib")
NEW_FILES = ["src/tcn_moment/replicate_fusion_lora.py", "tests/test_fusion_lora_replication.py",
             "scripts/run_e07_remote.sh", "scripts/audit_e07_saved_outputs.py",
             "scripts/audit_e05c_saved_outputs.py",
             "docs/experiment_records/e07_fusion_lora_replication_protocol_20261019.md"]


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def read_protocol(path):
    p = read(path)
    if (p["experiment_id"] != "E07_FUSION_LORA_REPLICATION" or p["seeds"] != [42, 44, 45, 46]
        or p["excluded_pilot_seed"] != 43 or p["conditions"] != CONDITIONS
        or p["train_fraction"] != 0.4 or p["train_count"] != 9827 or p["validation_count"] != 3510
        or p["maximum_training_conditions"] != 8 or p["maximum_additional_epochs_total"] != 240
        or p["wall_time_limit_hours"] != 8 or p["classes"] != ["0", "1", "2"]
        or not p["prior_test_results_known"] or not p["validation_reused"]
        or any(p[k] for k in ("test_evaluated", "claim_total_label_efficiency", "automatic_next_stage",
                              "automatic_budget_expansion", "automatic_test", "automatic_shutdown", "pi_wake_enabled"))):
        raise ValueError("Unexpected compact replication scope")
    expected = read(p["pilot_protocol"])
    for section, keys in {
        "training": ("maximum_epochs", "early_stopping", "head_learning_rate", "adapter_learning_rate",
                     "physical_batch_size", "effective_batch_size", "evaluation_batch_size", "weight_decay",
                     "gradient_clip_norm", "precision", "loss_scale", "loss_scale_growth_interval",
                     "warmup_fraction", "minimum_lr_ratio"),
        "architecture": ("lora_targets", "rank", "alpha", "lora_dropout", "all_conditions_online", "gradient_checkpointing"),
    }.items():
        if any(p[section][key] != expected[section][key] for key in keys):
            raise ValueError("Cannot retune the E03F-v2 treatment in replication")
    if p["architecture"]["adapter_seed_offset"] != 30000 or p["architecture"]["training_seed_offset"] != 20000:
        raise ValueError("Seed offsets changed")
    if p["success_rules"] != {
        "minimum_mean_over_continued_frozen_fusion_pp": 0.5,
        "minimum_mean_over_strongest_nonadapted_pp": 0.5,
        "minimum_positive_seeds_each": 3, "require_both": True,
        "strongest_nonadapted_set": ["continued_frozen_fusion", "parent_fusion_nn", "parent_statistics_nn",
                                    "parent_moment_nn", "parent_fusion_svm", "parent_statistics_rf",
                                    "parent_statistics_svm", "matched_statistics_svm"],
        "strong_tcn_role": "predeclared overall context only, E05C legacy masked 40percent; not an adaptation-increment gate",
        "rank_matched_moment_ablation": "omitted to bound cost; no new claim about adaptation by statistics interaction",
    }:
        raise ValueError("Fixed comparison set or gates changed")
    if p["technical_gate"] != {
        "all_four_seeds_before_any_real_optimizer_step": True,
        "maximum_initial_prediction_disagreements": 1,
        "maximum_initial_macro_f1_difference_pp": 0.05,
        "zero_adapter_logit_tolerance": 0.00001,
        "within_seed_treatment_initial_probability_max_error": 0.0,
        "synthetic_batch_size": 64, "synthetic_optimizer_steps_per_seed": 2,
    }:
        raise ValueError("Migration/synthetic tolerances changed")
    return p


def root(p):
    return Path(p["output_dir"]) / p["suite"]


def parent_run(p, seed, branch="fusion"):
    return parent.run_directory(read(p["parent_protocol"]), seed, branch)


def old_prepared(p, seed):
    return parent.prepared_directory(read(p["parent_protocol"]), seed)


def case_dir(p, seed, condition):
    return root(p) / f"seed{seed}" / condition


def prepared_dir(p, seed):
    return root(p) / "prepared" / f"seed{seed}"


def runtime():
    return {"host": platform.node(), "python": platform.python_version(), **pilot.runtime_identity(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def training_protocol(p, seed):
    if seed not in p["seeds"]:
        raise ValueError("Seed outside replication")
    pp = copy.deepcopy(read("configs/experiments/improvements/e03_peft_pilot_protocol.json"))
    pp["seeds"] = [seed]
    for key in ("maximum_epochs", "head_learning_rate", "adapter_learning_rate", "physical_batch_size",
                "effective_batch_size", "evaluation_batch_size", "weight_decay", "gradient_clip_norm",
                "warmup_fraction", "minimum_lr_ratio"):
        pp["training"][key] = p["training"][key]
    pp["training"].update(minimum_epochs=30, patience=31, grad_scaler_initial_scale=1024,
                           frozen_head_batch_size=p["training"]["physical_batch_size"])
    pp["architecture"].update(adapter_seed_offset=30000, training_seed_offset=20000)
    return pp


def make_model(p, seed, rank):
    if seed not in p["seeds"] or rank not in (0, 8):
        raise ValueError("Invalid seed/rank")
    config = with_random_seed(load_config(p["training_config"]), seed)
    backbone = train_e03.build_backbone(config, rank, training_protocol(p, seed))
    payload = torch.load(parent_run(p, seed) / "model_best.pt", map_location="cpu", weights_only=True)
    state = payload["learned_state"]
    if state["adapters"] or payload["metadata"]["seed"] != seed or payload["metadata"]["branch"] != "fusion":
        raise ValueError("Wrong seed/branch/adapted parent")
    head = GainFusionHead(state["classifier"]["stats_mean"], state["classifier"]["stats_scale"],
                          branch="fusion", input_gain=fan_in_gain("fusion"))
    head.load_state_dict(state["classifier"], strict=True)
    model = PeftClassifier(backbone, head) if rank else OnlineFrozenClassifier(backbone, head)
    return model.cuda(), payload


def smoke(path):
    p = read_protocol(path)
    verified_checkpoint(load_config(p["training_config"]))
    reports = []
    batch = p["technical_gate"]["synthetic_batch_size"]
    positions = torch.arange(1024, device="cuda")[None, :]
    lengths = 18 + (torch.arange(batch, device="cuda") * 37) % 799
    mask = (positions < lengths[:, None]).float()
    x = (torch.sin(positions.float() / 13).expand(batch, -1) * mask)[:, None]
    x[0].zero_()
    for seed in p["seeds"]:
        model, _ = make_model(p, seed, 0)
        model.eval()
        generator = torch.Generator(device="cuda").manual_seed(seed + 40000)
        stats = model.classifier.stats_mean[None] + torch.randn(batch, 15, device="cuda", generator=generator) * model.classifier.stats_scale[None]
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            baseline = model(x, stats, mask).float().cpu()
        params = trainable_summary(model)
        if params["adapter_parameters"] or params["trainable_parameters"] != 135555:
            raise AssertionError("Frozen provider trainable role mismatch")
        del model
        torch.cuda.empty_cache()
        model, _ = make_model(p, seed, 8)
        model.eval()
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            zero = model(x, stats, mask).float().cpu()
        zero_error = float((zero - baseline).abs().max())
        if not torch.isfinite(zero).all() or zero_error > p["technical_gate"]["zero_adapter_logit_tolerance"]:
            raise AssertionError("Zero-adapter synthetic restore failed")
        before = frozen_parameter_digest(model)
        optimizer = torch.optim.AdamW(optimizer_groups(model, 1e-4, 1e-4), weight_decay=.01, fused=False)
        scaler = FixedHeadGradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            model.train_selected()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.cross_entropy(model(x, stats, mask), torch.arange(batch, device="cuda") % 3)
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite synthetic loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            adapters = adapter_parameters(model.backbone)
            if any(v.grad is None or not torch.isfinite(v.grad).all() for v in adapters.values()):
                raise AssertionError("Nonfinite/missing adapter gradient")
            for suffix in (["lora_B"] if step == 0 else ["lora_A", "lora_B"]):
                if not any(v.grad.abs().sum() > 0 for name, v in adapters.items() if suffix in name):
                    raise AssertionError("Zero adapter gradient")
            norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1)
            if not torch.isfinite(norm):
                raise AssertionError("Nonfinite synthetic norm")
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() != 1024:
                raise AssertionError("Loss scale changed")
        if before != frozen_parameter_digest(model):
            raise AssertionError("Frozen parameters changed")
        state = capture_learned_state(model)
        model.eval()
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            expected = model(x, stats, mask).float().cpu()
        with torch.no_grad():
            for value in model.parameters():
                if value.requires_grad:
                    value.zero_()
        restore_learned_state(model, state)
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            actual = model(x, stats, mask).float().cpu()
        if not torch.equal(expected, actual):
            raise AssertionError("Synthetic learned-state reload mismatch")
        reports.append({"seed": seed, "zero_adapter_max_logit_error": zero_error,
                        "reload_max_logit_error": 0., "gradients_finite_nonzero": True,
                        "frozen_parameters_unchanged": True, "scale": scaler.get_scale(),
                        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2})
        del model, optimizer, scaler, state
        torch.cuda.empty_cache()
        print(f"Synthetic strong-head batch64 checks passed seed{seed}", flush=True)
    atomic_write_json(root(p) / "synthetic_smoke.json", {
        "status": "passed", "runtime": runtime(), "synthetic_input_only": True,
        "test_evaluated": False, "batch_size": batch, "source_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(path), "per_seed": reports,
    })


def reference_paths(p, seed):
    pp = read(p["parent_protocol"])
    paths = {f"parent_{branch}_nn": parent_run(p, seed, branch) / "validation_predictions.csv"
             for branch in ("fusion", "statistics", "moment")}
    paths.update({
        "parent_fusion_svm": parent.fusion_run(seed) / "fusion_stats_svm_fraction_40/validation_predictions.csv",
        "parent_statistics_rf": parent.legacy_run(seed) / "stats_rf_fraction_40/validation_predictions.csv",
        "parent_statistics_svm": parent.legacy_run(seed) / "stats_svm_fraction_40/validation_predictions.csv",
        "matched_statistics_svm": parent.matched_run(pp, seed) / "validation_predictions.csv",
    })
    tcn = read(p["tcn_protocol"])
    paths["strong_tcn"] = Path(tcn["output_dir"]) / tcn["suite"] / f"seed{seed}/fraction040_legacy_masked50/validation_selected.csv"
    return paths


def read_predictions(file, ids, labels):
    frame = pd.read_csv(file, dtype={"sample_id": str})
    truth = "true_index" if "true_index" in frame else "y_true"
    predicted = "predicted_index" if "predicted_index" in frame else "y_pred"
    if not np.array_equal(frame.sample_id, ids) or not np.array_equal(frame[truth], labels):
        raise ValueError(f"Saved prediction ID/truth mismatch: {file}")
    values = frame[predicted].to_numpy()
    if not np.isin(values, [0, 1, 2]).all():
        raise ValueError("Invalid predicted classes")
    cm = np.bincount(labels * 3 + values, minlength=9).reshape(3, 3)
    metrics = {**metrics_from_confusion(cm.tolist()), "confusion_matrix": cm.tolist(),
               "class_f1": [float(2 * cm[i, i] / (cm[i].sum() + cm[:, i].sum())) for i in range(3)],
               "battery_recall": float(np.mean(values[labels == 2] == 2)),
               "battery_FPR": float(np.mean(values[labels != 2] == 2))}
    columns = ([f"probability_{i}" for i in range(3)] if "probability_0" in frame
               else [f"p{i}" for i in range(3)] if "p0" in frame else None)
    probabilities = None
    if columns:
        probabilities = frame[columns].to_numpy()
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-5)
            or not np.array_equal(values, probabilities.argmax(1))):
            raise ValueError("Invalid probability output")
        metrics["battery_AP"] = float(average_precision_score(labels == 2, probabilities[:, 2]))
    return metrics, probabilities


def freeze(path):
    p = read_protocol(path)
    r = root(p)
    r.mkdir(parents=True, exist_ok=True)
    if (r / "frozen_protocol.json").exists():
        return verify(path)
    if any(r.glob("seed*")) or (r / "started.json").exists():
        raise ValueError("Freeze before real training outcomes")
    pp = Path(p["parent_protocol"])
    if sha256_file(pp) != p["parent_protocol_sha256"]:
        raise ValueError("Parent protocol changed")
    pf = parent.verify(pp)
    checkpoint = verified_checkpoint(load_config(p["training_config"]))
    smoke_record = read(r / "synthetic_smoke.json")
    if (smoke_record["status"] != "passed" or not smoke_record["synthetic_input_only"]
        or smoke_record["runtime"] != runtime() or smoke_record["source_sha256"] != sha256_file(Path(__file__))
        or smoke_record["protocol_sha256"] != sha256_file(path)
        or [row["seed"] for row in smoke_record["per_seed"]] != p["seeds"]):
        raise ValueError("Exact-source four-seed strong-head smoke required")
    if sha256_file(Path(p["dataset"])) != p["dataset_sha256"]:
        raise ValueError("Dataset changed")
    files = {Path(k) for k in pf["file_sha256"]}
    files.update(Path("src/tcn_moment").glob("*.py"))
    files.update(Path("configs").rglob("*.yaml"))
    files.update([path, pp, Path(p["pilot_protocol"]), Path(p["tcn_protocol"]),
                  parent.plan_directory(pf["protocol"]) / "frozen_protocol.json",
                  r / "synthetic_smoke.json", Path("scripts/autodl_env.sh"), *map(Path, NEW_FILES)])
    references = {}
    for seed in p["seeds"]:
        split = Path(read(p["tcn_protocol"])["split_manifests"][str(seed)])
        if sha256_file(split) != sha256_file(parent.fusion_run(seed) / "split_manifest.json"):
            raise ValueError("TCN and MOMENT split bytes differ")
        files.add(split)
        prep = old_prepared(p, seed)
        dm = read(prep / "manifest.json")
        if dm["seed"] != seed or dm["protocol_sha256"] != pf["protocol_sha256"]:
            raise ValueError("Parent prepared scope changed")
        for split in ("train", "validation"):
            if sha256_file(prep / f"{split}.npz") != dm["splits"][split]["sha256"]:
                raise ValueError("Parent prepared features corrupted")
        files.update(prep / n for n in ("manifest.json", "normalizer.json", "train.npz", "validation.npz", "amp_manifest.json"))
        with np.load(prep / "train.npz", allow_pickle=False) as a:
            train_ids = a["sample_ids"]
        with np.load(prep / "validation.npz", allow_pickle=False) as a:
            ids, labels = a["sample_ids"], a["labels"]
        if len(train_ids) != 9827 or len(ids) != 3510 or train_subset_sha256(train_ids) != dm["train_ids_sha256"]:
            raise ValueError("Parent IDs/counts differ")
        references[str(seed)] = {}
        for name, file in reference_paths(p, seed).items():
            m, _ = read_predictions(file, ids, labels)
            references[str(seed)][name] = m
            files.add(file)
            if name.endswith("_nn"):
                directory = file.parent
                if read(directory / "status.json")["status"] != "completed":
                    raise ValueError("Parent NN incomplete")
                metrics = read(directory / "metrics.json")
                if abs(metrics["validation_metrics"]["macro_f1"] - m["macro_f1"]) > 1e-12:
                    raise ValueError("Parent NN metrics differ from saved predictions")
                if sha256_file(directory / "model_best.pt") != metrics["best_checkpoint_sha256"]:
                    raise ValueError("Parent NN checkpoint changed")
                files.update(directory / n for n in ("model_best.pt", "metrics.json", "status.json", "train_ids.npy"))
            elif name == "strong_tcn":
                result = read(file.parent / "result.json")
                if (sha256_file(file) != result["output_sha256"][file.name]
                    or not np.array_equal(np.load(file.parent / "train_ids.npy"), train_ids)):
                    raise ValueError("Strong TCN context budget/output differs")
                files.update([file.parent / "result.json", file.parent / "train_ids.npy"])
        payload = torch.load(parent_run(p, seed) / "model_best.pt", map_location="cpu", weights_only=True)
        if (payload["metadata"]["seed"] != seed or payload["learned_state"]["adapters"]
            or abs(payload["validation_macro_f1"] - references[str(seed)]["parent_fusion_nn"]["macro_f1"]) > 1e-12):
            raise ValueError("Parent fusion checkpoint scope/score mismatch")
    f = {"protocol": p, "protocol_sha256": sha256_file(path), "frozen_at": data_loader.now(),
         "file_sha256": {str(file): sha256_file(file) for file in sorted(files)},
         "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
         "runtime": runtime(), "public_checkpoint": checkpoint, "references": references,
         "prior_test_results_known": True, "new_real_training_outcomes_observed": False}
    for file in files:
        if file.parts[0] in {"src", "configs", "docs", "scripts", "tests", "legacy"}:
            destination = r / "source_snapshot" / file
        elif file.suffix in {".json", ".npy", ".pt"} and not file.is_relative_to(r):
            destination = r / "input_snapshot" / file
        else:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
    atomic_write_json(r / "frozen_protocol.json", f)
    print(f"Frozen E07 {len(files)} inputs: four seeds, eight cases, 240 epochs; no test", flush=True)
    return f


def verify(path):
    p = read_protocol(path)
    f = read(root(p) / "frozen_protocol.json")
    if p != f["protocol"] or sha256_file(path) != f["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    for name, expected in f["file_sha256"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("Frozen input changed: " + name)
    if f["packages"] != {name: importlib.metadata.version(name) for name in PACKAGES} or f["runtime"] != runtime():
        raise ValueError("Runtime/dependencies changed; migration requires a separate revision")
    if verified_checkpoint(load_config(p["training_config"])) != f["public_checkpoint"]:
        raise ValueError("Public backbone changed")
    return f


def select_arrays(p, seed, data):
    ix = stratified_train_subset_indices(data["train"]["y"], data["train"]["ids"], p["train_fraction"], seed)
    result = {}
    for split in ("train", "validation"):
        d = data[split]
        if split == "train":
            d = {key: value[ix] for key, value in d.items()}
        stats = extract_statistical_features(d["x"], d["mask"])
        result[split] = {"x": parent.normalize_masked_rows(d["x"], d["mask"]), "mask": d["mask"],
                         "statistics": stats, "labels": d["y"], "sample_ids": d["ids"]}
    return result


def prepare(path):
    f = verify(path)
    p = f["protocol"]
    tcn = read(p["tcn_protocol"])
    base = read(tcn["parent_protocol"])
    for seed in p["seeds"]:
        directory = prepared_dir(p, seed)
        if (directory / "manifest.json").exists():
            load_prepared(p, f, seed)
            continue
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Incomplete preparation retained; explicit recovery required")
        directory.mkdir(parents=True, exist_ok=True)
        data = data_loader.load_train_validation({**base, "seed": seed,
                                                  "split_manifest": str(parent.fusion_run(seed) / "split_manifest.json")})
        arrays = select_arrays(p, seed, data)
        del data
        if not np.array_equal(arrays["train"]["sample_ids"], np.load(parent_run(p, seed) / "train_ids.npy")):
            raise ValueError("Supervised train IDs differ from parent")
        norm = read(old_prepared(p, seed) / "normalizer.json")
        scaler = StandardScaler().fit(arrays["train"]["statistics"])
        np.testing.assert_allclose(scaler.mean_, norm["mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(scaler.scale_, norm["scale"], rtol=0, atol=1e-12)
        hashes = {}
        for split, a in arrays.items():
            with np.load(old_prepared(p, seed) / f"{split}.npz", allow_pickle=False) as old:
                for key in ("statistics", "labels", "sample_ids"):
                    np.testing.assert_array_equal(a[key], old[key])
            file = directory / f"{split}.npz"
            with file.with_suffix(".npz.tmp").open("wb") as stream:
                np.savez_compressed(stream, **a)
            file.with_suffix(".npz.tmp").replace(file)
            hashes[split] = sha256_file(file)
        shutil.copy2(parent.fusion_run(seed) / "split_manifest.json", directory / "split_manifest.json")
        atomic_write_json(directory / "manifest.json", {"seed": seed, "protocol_sha256": f["protocol_sha256"],
                                                       "splits": hashes, "classes": p["classes"],
                                                       "train_ids_sha256": train_subset_sha256(arrays["train"]["sample_ids"]),
                                                       "test_encoded": False})
        load_prepared(p, f, seed)
        print(f"Prepared E07 seed{seed}: 9827 train, 3510 validation; no test tensors", flush=True)


def load_prepared(p, f, seed):
    directory = prepared_dir(p, seed)
    manifest = read(directory / "manifest.json")
    if manifest["seed"] != seed or manifest["protocol_sha256"] != f["protocol_sha256"] or manifest["test_encoded"]:
        raise ValueError("Prepared metadata mismatch")
    arrays = {}
    for split, count in (("train", p["train_count"]), ("validation", p["validation_count"])):
        file = directory / f"{split}.npz"
        if sha256_file(file) != manifest["splits"][split]:
            raise ValueError("Prepared raw inputs changed")
        with np.load(file, allow_pickle=False) as archive:
            a = {key: archive[key].copy() for key in archive.files}
        if (a["x"].shape != (count, 1024) or a["mask"].shape != (count, 1024)
            or a["statistics"].shape != (count, 15) or a["labels"].shape != (count,)
            or a["sample_ids"].shape != (count,)
            or not all(np.isfinite(a[k]).all() for k in ("x", "mask", "statistics"))):
            raise ValueError("Prepared shapes/values invalid")
        arrays[split] = a
    if train_subset_sha256(arrays["train"]["sample_ids"]) != manifest["train_ids_sha256"]:
        raise ValueError("Training ID fingerprint changed")
    return arrays, manifest


def migration_check(p, parent_predictions, current, parent_f1):
    disagreements = int(np.count_nonzero(parent_predictions.argmax(1) != current.argmax(1)))
    if disagreements > p["technical_gate"]["maximum_initial_prediction_disagreements"]:
        raise ValueError("Initial prediction migration gate failed; do not relax tolerance")
    if abs(parent_f1[0] - parent_f1[1]) * 100 > p["technical_gate"]["maximum_initial_macro_f1_difference_pp"] + 1e-12:
        raise ValueError("Initial F1 migration gate failed")
    return {"prediction_disagreements": disagreements,
            "macro_f1_absolute_difference_pp": abs(parent_f1[0] - parent_f1[1]) * 100,
            "maximum_probability_drift": float(np.max(np.abs(parent_predictions - current)))}


def gate(path, seed):
    f = verify(path)
    p = f["protocol"]
    if seed not in p["seeds"]:
        raise ValueError("Invalid gate seed")
    directory = root(p) / "technical_gates" / f"seed{seed}"
    if (directory / "result.json").exists():
        result = read(directory / "result.json")
        for name, expected in result["output_sha256"].items():
            if sha256_file(directory / name) != expected:
                raise ValueError("Gate output changed")
        return result
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Incomplete gate retained; no silent recomputation")
    directory.mkdir(parents=True, exist_ok=True)
    arrays, _ = load_prepared(p, f, seed)
    a = arrays["validation"]
    data = train_e03.gpu_dataset(a["x"][:, None], a["statistics"], a["labels"], a["mask"])
    old_metrics, old_prob = read_predictions(parent_run(p, seed) / "validation_predictions.csv", a["sample_ids"], a["labels"])
    initials = []
    hashes = []
    migrations = []
    for c in p["conditions"]:
        model, _ = make_model(p, seed, c["rank"])
        metrics, probs = train_e03.evaluate(model, data, 64, p["classes"], details=True)
        migration = migration_check(p, old_prob, probs, (old_metrics["macro_f1"], metrics["macro_f1"]))
        migrations.append(migration)
        hashes.append(train_e03.head_digest(model.classifier))
        initials.append(probs)
        pilot.save_predictions(directory / f"{c['name']}_initial_predictions.csv", a, probs)
        atomic_write_json(directory / f"{c['name']}_initial_metrics.json", metrics)
        del model
        torch.cuda.empty_cache()
    if hashes[0] != hashes[1] or not np.array_equal(initials[0], initials[1]):
        raise ValueError("Treatments do not share exact initial head/probabilities")
    result = {"seed": seed, "status": "passed", "protocol_sha256": f["protocol_sha256"],
              "initial_head_sha256": hashes[0], "within_pair_probability_max_error": 0.,
              "migration": migrations, "test_evaluated": False,
              "output_sha256": {file.name: sha256_file(file) for file in sorted(directory.iterdir())}}
    atomic_write_json(directory / "result.json", result)
    print(f"All-treatment migration gate passed seed{seed}: {migrations}", flush=True)
    return result


def all_gates(p, f):
    results = []
    for seed in p["seeds"]:
        d = root(p) / "technical_gates" / f"seed{seed}"
        result = read(d / "result.json")
        if result["seed"] != seed or result["status"] != "passed" or result["protocol_sha256"] != f["protocol_sha256"]:
            raise ValueError("All four seed gates must pass before ANY real training")
        for name, expected in result["output_sha256"].items():
            if sha256_file(d / name) != expected:
                raise ValueError("Initial gate output changed")
        results.append(result)
    return results


def run_case(path, seed, name, resume=False):
    p = read_protocol(path)
    if seed not in p["seeds"] or name not in [c["name"] for c in CONDITIONS]:
        raise ValueError("Case outside scope")
    with (root(p) / f"seed{seed}_{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_case(path, seed, name, resume)


def _run_case(path, seed, name, resume=False):
    f = verify(path)
    p = f["protocol"]
    all_gates(p, f)
    matches = [c for c in p["conditions"] if c["name"] == name]
    if seed not in p["seeds"] or len(matches) != 1:
        raise ValueError("Case outside frozen scope")
    c = matches[0]
    directory = case_dir(p, seed, name)
    if directory.exists() != resume:
        raise ValueError("Existing case requires explicit resume; missing case cannot resume")
    if resume and read(directory / "status.json")["status"] == "completed":
        print(f"Preserved seed{seed}/{name}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    arrays, manifest = load_prepared(p, f, seed)
    model, checkpoint = make_model(p, seed, c["rank"])
    metadata = {"protocol_sha256": f["protocol_sha256"], "seed": seed, "condition": c,
                "train_fraction": 0.4, "parent_checkpoint_sha256": sha256_file(parent_run(p, seed) / "model_best.pt"),
                "parent_best_epoch": checkpoint["epoch"], "initial_head_sha256": train_e03.head_digest(model.classifier),
                "prepared_manifest_sha256": sha256_file(prepared_dir(p, seed) / "manifest.json"),
                "train_ids_sha256": manifest["train_ids_sha256"], "test_evaluated": False,
                "prior_test_results_known": True}
    gate_dir = root(p) / "technical_gates" / f"seed{seed}"
    if metadata["initial_head_sha256"] != read(gate_dir / "result.json")["initial_head_sha256"]:
        raise ValueError("Training initialization differs from gated parent")
    if resume:
        if read(directory / "experiment_plan.json") != metadata:
            raise ValueError("Resume scope changed")
    else:
        atomic_write_json(directory / "experiment_plan.json", metadata)
        shutil.copy2(root(p) / "frozen_protocol.json", directory / "frozen_protocol.json")
        np.save(directory / "train_ids.npy", arrays["train"]["sample_ids"])
        shutil.copy2(gate_dir / f"{name}_initial_predictions.csv", directory / "initial_predictions.csv")
        shutil.copy2(gate_dir / f"{name}_initial_metrics.json", directory / "initial_metrics.json")
        initial = read(directory / "initial_metrics.json")
        initial_p = {"training": p["training"], "architecture": {"training_seed": 20000 + seed}}
        pilot.seed_initial_checkpoint(model, initial_p, directory, metadata, initial["macro_f1"])
    atomic_write_json(directory / "status.json", {"status": "running", "updated_at": data_loader.now(), "test_evaluated": False})
    try:
        data = {split: train_e03.gpu_dataset(a["x"][:, None], a["statistics"], a["labels"], a["mask"])
                for split, a in arrays.items()}
        parameters = trainable_summary(model)
        with patch.object(torch.amp, "GradScaler", FixedHeadGradScaler):
            result, probabilities = train_e03.train(model, data["train"], data["validation"],
                                                     training_protocol(p, seed), directory, metadata,
                                                     p["classes"], resume=True)
        pilot.save_predictions(directory / "validation_predictions.csv", arrays["validation"], probabilities)
        last = torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        restore_learned_state(model, last["learned_state"])
        endpoint, endpoint_probabilities = train_e03.evaluate(model, data["validation"], 64, p["classes"], details=True)
        pilot.save_predictions(directory / "endpoint_predictions.csv", arrays["validation"], endpoint_probabilities)
        atomic_write_json(directory / "metrics.json", {
            **metadata, **result, "parameters": parameters, "endpoint_validation_metrics": endpoint,
            "model_best_sha256": sha256_file(directory / "model_best.pt"),
            "checkpoint_last_sha256": sha256_file(directory / "checkpoint_last.pt"),
        })
        atomic_write_json(directory / "status.json", {"status": "completed", "updated_at": data_loader.now(), "test_evaluated": False})
    except BaseException as error:
        atomic_write_json(directory / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                       "updated_at": data_loader.now(), "error": repr(error), "test_evaluated": False})
        raise


def paired(values, threshold):
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 3.182446305284263 * sd / math.sqrt(len(values))
    return {"per_seed_pp": values, "mean_pp": mean, "positive_seeds": sum(v > 0 for v in values),
            "descriptive_ci95_pp": [mean - half, mean + half], "threshold_pp": threshold,
            "rule_passed": mean >= threshold and sum(v > 0 for v in values) >= 3}


def decisions(p, scores, references):
    continued, strongest, tcn = [], [], []
    strongest_names = {}
    for seed in p["seeds"]:
        ref = references[str(seed)]
        fixed = {name: ref[name]["macro_f1"] for name in p["success_rules"]["strongest_nonadapted_set"]
                 if name != "continued_frozen_fusion"}
        fixed["continued_frozen_fusion"] = scores[seed, "frozen_fusion"]
        strongest_names[str(seed)] = max(fixed, key=fixed.get)
        continued.append(100 * (scores[seed, "lora8_fusion"] - scores[seed, "frozen_fusion"]))
        strongest.append(100 * (scores[seed, "lora8_fusion"] - max(fixed.values())))
        tcn.append(100 * (scores[seed, "lora8_fusion"] - ref["strong_tcn"]["macro_f1"]))
    primary = paired(continued, .5)
    stronger = paired(strongest, .5)
    return {"continued_frozen_rule": primary, "strongest_nonadapted_rule": stronger,
            "strongest_nonadapted_name_per_seed": strongest_names,
            "lora_minus_strong_tcn_context": {k: v for k, v in paired(tcn, 0).items() if k not in {"threshold_pp", "rule_passed"}},
            "replication_rule_passed": primary["rule_passed"] and stronger["rule_passed"]}


def summarize(path):
    f = verify(path)
    p = f["protocol"]
    all_gates(p, f)
    rows, scores = [], {}
    for seed in p["seeds"]:
        arrays, _ = load_prepared(p, f, seed)
        a = arrays["validation"]
        heads, initial_probabilities = [], []
        for c in p["conditions"]:
            directory = case_dir(p, seed, c["name"])
            m = read(directory / "metrics.json")
            history = read(directory / "history.json")
            if (read(directory / "status.json")["status"] != "completed" or m["seed"] != seed
                or m["condition"] != c or m["protocol_sha256"] != f["protocol_sha256"]
                or m["test_evaluated"] or not m["all_frozen_parameters_unchanged"]):
                raise ValueError("Incomplete/mismatched case")
            for file, key in (("model_best.pt", "model_best_sha256"), ("checkpoint_last.pt", "checkpoint_last_sha256")):
                if sha256_file(directory / file) != m[key]:
                    raise ValueError("Saved learned-state hash changed")
            if ([row["epoch"] for row in history] != list(range(1, 31))
                or any(row["grad_scaler_scale"] != 1024 or row["skipped_optimizer_steps"] or row["early_stop"] for row in history)):
                raise ValueError("Training budget/precision changed")
            if c["rank"] and (m["parameters"]["adapter_parameters"] != 786432
                              or any(not row["adapter_B_gradient_norm_first_window"] > 0 for row in history)):
                raise ValueError("LoRA gradients or parameter count invalid")
            if not c["rank"] and (m["parameters"]["adapter_parameters"] or any(row["adapter_B_gradient_norm_first_window"] is not None for row in history)):
                raise ValueError("Frozen provider entered adapter role")
            initial = read(directory / "initial_metrics.json")
            for name, expected in (("initial_predictions.csv", initial), ("validation_predictions.csv", m["validation_metrics"]),
                                   ("endpoint_predictions.csv", m["endpoint_validation_metrics"])):
                calculated, prob = read_predictions(directory / name, a["sample_ids"], a["labels"])
                for key, value in metrics_from_confusion(calculated["confusion_matrix"]).items():
                    if abs(value - expected[key]) > 1e-12:
                        raise ValueError("Saved prediction metrics differ")
                if name == "initial_predictions.csv":
                    initial_probabilities.append(prob)
                if name == "validation_predictions.csv":
                    selected_metrics = calculated
            values = [initial["macro_f1"], *[row["validation_macro_f1"] for row in history]]
            best, epoch = values[0], 0
            for i, value in enumerate(values[1:], 1):
                if value > best + 1e-12:
                    best, epoch = value, i
            if abs(best - m["validation_metrics"]["macro_f1"]) > 1e-12 or epoch != m["best_epoch"]:
                raise ValueError("Epoch0-inclusive selection mismatch")
            if abs(history[-1]["validation_macro_f1"] - m["endpoint_validation_metrics"]["macro_f1"]) > 1e-12:
                raise ValueError("Endpoint mismatch")
            heads.append(m["initial_head_sha256"])
            scores[seed, c["name"]] = selected_metrics["macro_f1"]
            rows.append({"seed": seed, "condition": c["name"], "selected": selected_metrics,
                         "initial_macro_f1": values[0], "endpoint_macro_f1": values[-1],
                         "best_epoch": epoch, "training_seconds": m["training_seconds"], "parameters": m["parameters"]})
        if heads[0] != heads[1] or not np.array_equal(*initial_probabilities):
            raise ValueError("Paired initial head/probabilities differ")
    aggregate = {}
    for c in p["conditions"]:
        values = [100 * scores[seed, c["name"]] for seed in p["seeds"]]
        aggregate[c["name"]] = {"macro_f1_mean_percent": statistics.mean(values),
                                 "macro_f1_sample_sd_percent": statistics.stdev(values), "per_seed_percent": values}
    summary = {"audit_status": "passed", "protocol_sha256": f["protocol_sha256"], "test_evaluated": False,
               "prior_test_results_known": True, "epochs_audited": 240, "prediction_rows_audited": 84240,
               "per_condition": rows, "aggregate": aggregate, "references": f["references"],
               **decisions(p, scores, f["references"]), "limitations": p["limitations"]}
    directory = root(p)
    if (directory / "summary.json").exists() and read(directory / "summary.json") != summary:
        raise ValueError("Completed scientific summary changed")
    atomic_write_json(directory / "summary.json", summary)
    lines = ["# E07 compact LoRA fusion replication (validation only)", "",
             "Post-test exploration; seed43 excluded; repeated validation selection.", "",
             "| Seed | Continued frozen % | LoRA8 % | Gain pp | Strong TCN context % |", "|---:|---:|---:|---:|---:|"]
    for seed in p["seeds"]:
        a, b = scores[seed, "frozen_fusion"], scores[seed, "lora8_fusion"]
        lines.append(f"| {seed} | {100*a:.4f} | {100*b:.4f} | {100*(b-a):+.4f} | {100*f['references'][str(seed)]['strong_tcn']['macro_f1']:.4f} |")
    lines += ["", f"Original replication gates passed: {summary['replication_rule_passed']}", "",
              "Success concerns adaptation increment, not necessarily beating TCN or external generalization."]
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"aggregate": aggregate, **decisions(p, scores, f["references"])}), flush=True)
    return summary


def launch_child(args, deadline):
    command = [sys.executable, "-m", "tcn_moment.replicate_fusion_lora", *args]
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Original eight-hour wall budget exhausted")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=remaining)
        if code:
            raise RuntimeError(f"Child failed ({code}): {args}")
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def coordinate(path, resume=False):
    f = verify(path)
    p = f["protocol"]
    directory = root(p)
    lock = (directory / "suite.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started_file = directory / "started.json"
    if started_file.exists():
        if not resume:
            raise ValueError("Previous suite exists; explicit resume required")
        started = read(started_file)
        status_file = directory / "suite_status.json"
        if status_file.exists() and read(status_file)["status"] == "completed":
            print("Completed E07 preserved; no repeated training", flush=True)
            lock.close()
            return
    else:
        if resume:
            raise ValueError("Cannot resume a missing suite")
        started = {"started_at": data_loader.now(), "wall_started_unix": time.time(),
                   "protocol_sha256": f["protocol_sha256"], "runtime": runtime(), "coordinator_pid": os.getpid()}
        atomic_write_json(started_file, started)
    if started["protocol_sha256"] != f["protocol_sha256"]:
        raise ValueError("Suite start protocol changed")
    deadline = started["wall_started_unix"] + p["wall_time_limit_hours"] * 3600

    def status(value, stage, **extra):
        atomic_write_json(directory / "suite_status.json", {"status": value, "stage": stage,
                          "updated_at": data_loader.now(), "test_evaluated": False,
                          "automatic_next_stage": False, "pi_wake_enabled": False, **extra})

    try:
        status("running", "prepare_train_validation_only")
        launch_child(["prepare", "--protocol", str(path)], deadline)
        for seed in p["seeds"]:
            status("running", f"all_seed_migration_before_training_seed{seed}")
            launch_child(["gate", "--protocol", str(path), "--seed", str(seed)], deadline)
        all_gates(p, f)
        atomic_write_json(directory / "all_technical_gates_passed.json", {
            "status": "passed", "recorded_at": data_loader.now(), "protocol_sha256": f["protocol_sha256"],
            "all_four_seeds_before_any_real_optimizer_step": True, "test_evaluated": False})
        for seed in p["seeds"]:
            for c in p["conditions"]:
                status("running", f"seed{seed}_{c['name']}")
                args = ["run-case", "--protocol", str(path), "--seed", str(seed), "--condition", c["name"]]
                if case_dir(p, seed, c["name"]).exists():
                    if not resume:
                        raise ValueError("Unexpected case before first suite run")
                    args.append("--resume")
                launch_child(args, deadline)
        status("running", "audit_complete_eight_cases")
        launch_child(["summarize", "--protocol", str(path)], deadline)
        summary = read(directory / "summary.json")
        status("completed", "completed_no_test_or_automatic_expansion", replication_rule_passed=summary["replication_rule_passed"],
               wall_seconds=time.time() - started["wall_started_unix"])
        atomic_write_json(directory / "output_manifest.json", {
            str(file.relative_to(directory)): sha256_file(file) for file in sorted(directory.rglob("*"))
            if file.is_file() and file.name not in {"suite.lock", "output_manifest.json"}
        })
        print("E07 finished. No new test, expansion or shutdown.", flush=True)
    except BaseException as error:
        status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "stopped_no_automatic_retry", error=repr(error))
        raise
    finally:
        lock.close()


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Interrupted; latest committed epoch retained, explicit resume only")

    signal.signal(signal.SIGTERM, interrupted)
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "freeze", "verify", "prepare", "gate", "run-case", "coordinate", "summarize"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition", choices=[c["name"] for c in CONDITIONS])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "run-case":
        run_case(args.protocol, args.seed, args.condition, args.resume)
    elif args.action == "coordinate":
        coordinate(args.protocol, args.resume)
    elif args.action == "gate":
        gate(args.protocol, args.seed)
    else:
        globals()[args.action](args.protocol)


if __name__ == "__main__":
    main()
