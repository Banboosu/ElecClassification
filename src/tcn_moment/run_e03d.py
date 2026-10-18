"""Frozen-cache head diagnostics; preserves E03P files and never calls an encoder."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import shutil
import signal
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from tcn_moment import train_e03
from tcn_moment.analyze_e03 import read_protocol as read_parent
from tcn_moment.analyze_e03 import run_directory as parent_run
from tcn_moment.analyze_e03 import verify as verify_parent
from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import load_dataset, save_label_encoder, train_subset_sha256
from tcn_moment.experiment import prepare_run, utc_now
from tcn_moment.head_diagnostics import (
    GainFusionHead,
    clone_parent_checkpoint,
    fan_in_gain,
    training_hooks,
)
from tcn_moment.io_utils import atomic_torch_save, atomic_write_json, sha256_file
from tcn_moment.moment_peft import PeftClassifier, restore_learned_state, trainable_summary
from tcn_moment.train_baselines import extract_statistical_features

DEFAULT_PROTOCOL = "configs/experiments/improvements/e03d_head_diagnostic_protocol.json"
PARENTS = {
    "statistics": "statistics_mlp",
    "moment": "frozen_moment_mlp",
    "fusion": "frozen_fusion_mlp",
}
PACKAGES = ("torch", "momentfm", "transformers", "numpy", "scikit-learn")


def read_protocol(path):
    p = json.loads(path.read_text())
    if (
        p["experiment_id"] != "E03D"
        or p["test_evaluated"]
        or p["seed"] != 43
        or p["train_fraction"] != 0.4
        or p["branches"] != list(PARENTS)
        or p["modes"] != ["legacy_tail", "fanin_gain"]
        or p["budgets"] != [30, 200]
        or p["training"]["maximum_epochs"] != 200
        or p["training"]["early_stopping"]
        or any(
            p["candidate_rules"][k]
            for k in ("automatic_replication", "automatic_lora_training", "automatic_test")
        )
    ):
        raise ValueError("Unexpected diagnostic scope or automatic expansion requested.")
    required = {
        "batch_size": 128,
        "evaluation_batch_size": 64,
        "head_learning_rate": 0.001,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "precision": "amp_fp16",
        "grad_scaler_initial_scale": 1024,
        "all_conditions_must_complete": True,
    }
    if any(p["training"][key] != value for key, value in required.items()):
        raise ValueError("Diagnostic settings must match the inherited parent training settings.")
    if (
        p["diagnostic_rules"]["budget_gain_threshold_pp"] != 1.0
        or p["diagnostic_rules"]["budget_endpoint_gain_threshold_pp"] != 0.5
        or p["diagnostic_rules"]["early_scale_gain_threshold_pp"] != 1.0
        or p["candidate_rules"]["minimum_gain_over_strongest_reference_pp"] != 0.5
        or p["candidate_rules"]["minimum_gain_over_same_mode_moment_pp"] != 0.25
    ):
        raise ValueError("Diagnostic thresholds differ from the declared plan.")
    return p


def plan_directory(p):
    return Path(p["output_dir"]) / f"{p['suite']}_analysis"


def cases(p):
    return [
        {"name": f"{branch}_{mode}", "branch": branch, "mode": mode}
        for mode in p["modes"]
        for branch in p["branches"]
    ]


def run_directory(p, case):
    return Path(p["output_dir"]) / f"{p['suite']}_{case['name']}"


def verify(path):
    p = read_protocol(path)
    f = json.loads((plan_directory(p) / "frozen_protocol.json").read_text())
    if f["protocol"] != p or f["protocol_sha256"] != sha256_file(path):
        raise ValueError("D protocol changed after freeze.")
    for filename, digest in f["file_sha256"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError(f"Frozen diagnostic input changed: {filename}")
    if f["packages"] != {k: importlib.metadata.version(k) for k in PACKAGES}:
        raise ValueError("Package versions changed.")
    return f


def freeze(path):
    p = read_protocol(path)
    root = plan_directory(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        verify(path)
        print("Existing E03D freeze verified.")
        return
    if any(run_directory(p, c).exists() for c in cases(p)):
        raise ValueError("Freeze before new outcomes.")
    parent = verify_parent(Path(p["parent_protocol"]))
    pp = parent["protocol"]
    files = [
        path,
        Path(p["parent_protocol"]),
        Path(__file__),
        Path(__file__).with_name("head_diagnostics.py"),
        Path("scripts/run_e03d_remote.sh"),
        Path("tests/test_head_diagnostics.py"),
        Path("docs/experiment_records/e03d_head_diagnostic_protocol_20261018.md"),
    ]
    files.extend(Path(k) for k in parent["file_sha256"])
    files.append(Path(pp["output_dir"]) / f"{pp['suite']}_analysis/frozen_protocol.json")
    for name in PARENTS.values():
        run = parent_run(pp, name)
        if json.loads((run / "status.json").read_text())["status"] != "completed":
            raise ValueError("Parent incomplete.")
        files.extend(
            run / name
            for name in (
                "checkpoint_last.pt",
                "model_best.pt",
                "metrics.json",
                "history.json",
                "statistics_normalizer.json",
                "train_ids.npy",
                "split_manifest.json",
            )
        )
    cache = Path(pp["cache_dir"]) / "seed43_fraction40"
    files.extend(cache / name for name in ("manifest.json", "train.npz", "validation.npz"))
    atomic_write_json(
        root / "frozen_protocol.json",
        {
            "frozen_at": utc_now(),
            "protocol": p,
            "protocol_sha256": sha256_file(path),
            "file_sha256": {str(k): sha256_file(k) for k in files},
            "packages": {k: importlib.metadata.version(k) for k in PACKAGES},
            "reference_macro_f1": parent["reference_macro_f1"],
            "parent_protocol_sha256": parent["protocol_sha256"],
        },
    )
    print("Frozen E03D: 6 head-only fits, 12 budget cells, no encoder/test/automatic expansion.")


def load_prepared(p, f):
    root = plan_directory(p) / "prepared"
    manifest_path = root / "manifest.json"
    m = json.loads(manifest_path.read_text())
    if m["protocol_sha256"] != f["protocol_sha256"]:
        raise ValueError("Prepared data protocol mismatch.")
    arrays = {}
    for split in ("train", "validation"):
        path = root / f"{split}.npz"
        if sha256_file(path) != m["splits"][split]["sha256"]:
            raise ValueError("Prepared input corrupted.")
        with np.load(path, allow_pickle=False) as data:
            arrays[split] = {k: data[k].copy() for k in data.files}
        a = arrays[split]
        n = len(a["sample_ids"])
        if (
            a["features"].shape != (n, 1024)
            or a["statistics"].shape != (n, 15)
            or a["labels"].shape != (n,)
            or not np.isfinite(a["features"]).all()
            or not np.isfinite(a["statistics"]).all()
            or set(np.unique(a["labels"])) != {0, 1, 2}
        ):
            raise ValueError("Invalid prepared arrays.")
    return arrays, m


def new_head(pp, branch, normalizer, gain):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(pp["architecture"]["head_seed_offset"] + 43)
        head = GainFusionHead(
            normalizer["mean"],
            normalizer["scale"],
            branch=branch,
            hidden_dim=128,
            dropout=0.1,
            input_gain=gain,
        )
    return head


def original_head_digest(head):
    digest = hashlib.sha256()
    for name, value in head.state_dict().items():
        if name != "input_gain":
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prepare(path):
    f = verify(path)
    p = f["protocol"]
    root = plan_directory(p) / "prepared"
    root.mkdir(exist_ok=True)
    if (root / "manifest.json").exists():
        load_prepared(p, f)
        print("Prepared data verified; no extraction.")
        return
    pp = read_parent(Path(p["parent_protocol"]))
    config = with_random_seed(load_config(pp["training_config"]), 43)
    print("Preparing raw statistics once; using existing unadapted AMP cache only", flush=True)
    bundle = load_dataset(replace(config.data, normalize="none"))
    parent = parent_run(pp, "frozen_fusion_mlp")
    ids = np.load(parent / "train_ids.npy")
    lookup = {str(k): i for i, k in enumerate(bundle.ids_train)}
    indices = np.asarray([lookup[str(k)] for k in ids])
    np.testing.assert_array_equal(bundle.ids_train[indices], ids)
    if sha256_file(bundle.split_path) != sha256_file(parent / "split_manifest.json"):
        raise ValueError("Split changed.")
    cache = Path(pp["cache_dir"]) / "seed43_fraction40"
    cm = json.loads((cache / "manifest.json").read_text())
    if (
        cm["signature"]["protocol_sha256"] != f["parent_protocol_sha256"]
        or cm["signature"]["dataset_sha256"] != bundle.dataset_sha256
    ):
        raise ValueError("Wrong frozen cache provenance.")
    inputs = {
        "train": (
            bundle.x_train[indices],
            bundle.mask_train[indices],
            bundle.y_train[indices],
            ids,
        ),
        "validation": (bundle.x_val, bundle.mask_val, bundle.y_val, bundle.ids_val),
    }
    splits = {}
    train_stats = None
    train_features = None
    for split, (x, mask, y, split_ids) in inputs.items():
        cp = cache / f"{split}.npz"
        if sha256_file(cp) != cm["splits"][split]["sha256"]:
            raise ValueError("Cache checksum mismatch.")
        with np.load(cp, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["sample_ids"], split_ids.astype(str))
            features = data["features"].copy()
        stats = extract_statistical_features(x, mask)
        if split == "train":
            train_stats, train_features = stats, features
        target = root / f"{split}.npz"
        temp = target.with_suffix(".npz.tmp")
        with temp.open("wb") as handle:
            np.savez(
                handle,
                features=features,
                statistics=stats,
                labels=y,
                sample_ids=split_ids.astype(str),
            )
        temp.replace(target)
        splits[split] = {"sha256": sha256_file(target), "n": len(y)}
    scaler = StandardScaler().fit(train_stats)
    norm = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "fitted_on": "selected training only",
        "n": len(ids),
    }
    for name in PARENTS.values():
        original = json.loads((parent_run(pp, name) / "statistics_normalizer.json").read_text())
        np.testing.assert_allclose(norm["mean"], original["mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(norm["scale"], original["scale"], rtol=0, atol=1e-12)
    atomic_write_json(root / "normalizer.json", norm)
    save_label_encoder(bundle.label_encoder, root)
    probes = []
    with torch.no_grad():
        x = torch.as_tensor(train_features[:1024])
        s = torch.as_tensor(train_stats[:1024])
        for branch in p["branches"]:
            for mode, gain in (("legacy", 1.0), ("fanin_gain", fan_in_gain(branch))):
                head = new_head(pp, branch, norm, gain).eval()
                captured = []
                hook = head.network[0].register_forward_pre_hook(
                    lambda module, args, captured=captured: captured.append(args[0])
                )
                head(x, s)
                hook.remove()
                z = head.network[0](captured[0])
                probes.append(
                    {
                        "branch": branch,
                        "mode": mode,
                        "gain": gain,
                        "input_rms": float(captured[0].square().mean().sqrt()),
                        "initial_fc_preactivation_rms": float(z.square().mean().sqrt()),
                    }
                )
    atomic_write_json(
        root / "training_only_scale_probe.json",
        {
            "uses_validation": False,
            "uses_labels": False,
            "rows": min(1024, len(ids)),
            "probes": probes,
        },
    )
    atomic_write_json(
        root / "manifest.json",
        {
            "protocol_sha256": f["protocol_sha256"],
            "dataset_sha256": bundle.dataset_sha256,
            "train_ids_sha256": train_subset_sha256(ids),
            "classes": bundle.label_encoder.classes_.tolist(),
            "normalizer_sha256": sha256_file(root / "normalizer.json"),
            "splits": splits,
        },
    )
    load_prepared(p, f)
    print(
        "Prepared/verified training-only scaler and unchanged cache; no encoder executed.",
        flush=True,
    )


def budget_snapshot(model, state, budget, root, metadata, val, classes, ids, best_path):
    restore_learned_state(model, state["learned_state"])
    endpoint, _ = train_e03.evaluate(model, val, 64, classes, details=True)
    restore_learned_state(model, state["best_learned_state"])
    best, probs = train_e03.evaluate(model, val, 64, classes, details=True)
    if abs(best["macro_f1"] - state["best"]) > 1e-12:
        raise ValueError("Budget best failed reproduction.")
    atomic_torch_save(
        torch,
        {
            "metadata": metadata,
            "learned_state": state["best_learned_state"],
            "epoch": state["best_epoch"],
            "budget": budget,
        },
        best_path,
    )
    pd.DataFrame(
        {
            "sample_id": ids,
            "true_index": val["y"].cpu().numpy(),
            "predicted_index": probs.argmax(axis=1),
            **{f"probability_{i}": probs[:, i] for i in range(3)},
        }
    ).to_csv(root / f"validation_predictions_{budget}.csv", index=False)
    return {
        "validation_metrics": best,
        "endpoint_metrics": endpoint,
        "best_epoch": state["best_epoch"],
        "checkpoint_sha256": sha256_file(best_path),
    }


def run(path, name, resume):
    f = verify(path)
    p = f["protocol"]
    case = next((c for c in cases(p) if c["name"] == name), None)
    if case is None:
        raise ValueError("Condition outside the frozen scope.")
    root = run_directory(p, case)
    lock = (plan_directory(p) / f"{name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if root.exists() and not resume:
        raise FileExistsError("Run exists; use explicit resume.")
    if resume and not root.exists():
        raise FileNotFoundError("Missing run.")
    if resume and json.loads((root / "status.json").read_text())["status"] == "completed":
        print(f"Preserved {name}")
        lock.close()
        return
    pp = read_parent(Path(p["parent_protocol"]))
    config_path = Path(pp["training_config"])
    config = with_random_seed(load_config(config_path), 43)
    context = prepare_run(
        model_name="E03D_HEAD_ONLY",
        base_output_dir=Path(p["output_dir"]),
        config=config,
        config_path=config_path,
        torch=torch,
        run_name=root.name,
        resume_dir=root if resume else None,
    )
    try:
        arrays, dm = load_prepared(p, f)
        prepared = plan_directory(p) / "prepared"
        if sha256_file(prepared / "normalizer.json") != dm["normalizer_sha256"]:
            raise ValueError("Scaler changed.")
        norm = json.loads((prepared / "normalizer.json").read_text())
        source = parent_run(pp, PARENTS[case["branch"]])
        source_metrics = json.loads((source / "metrics.json").read_text())
        gain = 1.0 if case["mode"] == "legacy_tail" else fan_in_gain(case["branch"])
        head = new_head(pp, case["branch"], norm, gain)
        initial = original_head_digest(head)
        if initial != source_metrics["initial_head_sha256"]:
            raise ValueError("Shared head initialization changed.")
        metadata = {
            "protocol_sha256": f["protocol_sha256"],
            "case": case,
            "input_gain": gain,
            "seed": 43,
            "train_fraction": 0.4,
            "parent_checkpoint_sha256": sha256_file(source / "checkpoint_last.pt"),
            "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
            "classes": dm["classes"],
        }
        if not resume:
            atomic_write_json(root / "experiment_plan.json", metadata)
            shutil.copy2(plan_directory(p) / "frozen_protocol.json", root / "frozen_protocol.json")
            shutil.copy2(source / "train_ids.npy", root / "train_ids.npy")
            shutil.copy2(prepared / "label_encoder.pkl", root / "label_encoder.pkl")
            (root / "source").mkdir()
            for file in (Path(__file__), Path(__file__).with_name("head_diagnostics.py")):
                shutil.copy2(file, root / "source" / file.name)
        elif json.loads((root / "experiment_plan.json").read_text()) != metadata:
            raise ValueError("Resume metadata mismatch.")
        model = PeftClassifier(None, head).cuda()
        data = {
            k: train_e03.gpu_dataset(v["features"], v["statistics"], v["labels"])
            for k, v in arrays.items()
        }
        classes = [str(v) for v in dm["classes"]]
        last = root / "checkpoint_last.pt"
        if case["mode"] == "legacy_tail" and not last.exists():
            parent = torch.load(
                source / "checkpoint_last.pt", map_location="cpu", weights_only=False
            )
            converted = clone_parent_checkpoint(parent, metadata)
            atomic_torch_save(torch, converted, last)
            atomic_torch_save(torch, converted, root / "checkpoint_at_30.pt")
            restore_learned_state(model, converted["best_learned_state"])
            check, _ = train_e03.evaluate(model, data["validation"], 64, classes)
            if abs(check["macro_f1"] - source_metrics["validation_metrics"]["macro_f1"]) > 1e-12:
                raise ValueError("Legacy30 failed exact score reproduction.")
        extended = copy.deepcopy(pp)
        extended["training"].update(maximum_epochs=200, minimum_epochs=200)
        steps = math.ceil(len(arrays["train"]["labels"]) / 128)
        print(
            f"Starting {name}: cached head only, gain={gain:.6f}, fixed30-cosine then terminal LR",
            flush=True,
        )
        with training_hooks(train_e03, root, steps_per_epoch=steps):
            result, _ = train_e03.train(
                model,
                data["train"],
                data["validation"],
                extended,
                root,
                metadata,
                classes,
                resume=last.exists(),
            )
        budgets = {}
        for budget, file in ((30, "checkpoint_at_30.pt"), (200, "checkpoint_last.pt")):
            state = torch.load(root / file, map_location="cpu", weights_only=False)
            if state["epoch"] != budget or state["metadata"] != metadata:
                raise ValueError("Budget checkpoint scope mismatch.")
            budgets[str(budget)] = budget_snapshot(
                model,
                state,
                budget,
                root,
                metadata,
                data["validation"],
                classes,
                arrays["validation"]["sample_ids"],
                root / f"model_best_{budget}.pt",
            )
        atomic_write_json(
            root / "metrics.json",
            {
                **metadata,
                "budgets": budgets,
                "parameters": trainable_summary(model),
                "initial_head_sha256_without_gain": initial,
                "test_evaluated": False,
                "encoder_executed": False,
                "recorded_epoch_seconds": result["training_seconds"],
                "additional_recorded_epoch_seconds": sum(
                    h["seconds"]
                    for h in json.loads((root / "history.json").read_text())
                    if h["epoch"] > 30 or case["mode"] == "fanin_gain"
                ),
            },
        )
        context.set_status("completed", test_evaluated=False, encoder_executed=False)
    except BaseException as exc:
        context.set_status(
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        raise
    finally:
        lock.close()


def judge(p, cells, references):
    d = p["diagnostic_rules"]
    diagnostics = []
    for branch in p["branches"]:
        old = cells[(branch, "legacy_tail", 30)]
        long = cells[(branch, "legacy_tail", 200)]
        gain = cells[(branch, "fanin_gain", 30)]
        budget_pp = (long["score"] - old["score"]) * 100
        endpoint_pp = (long["endpoint"] - old["endpoint"]) * 100
        scale_pp = (gain["score"] - old["score"]) * 100
        diagnostics.append(
            {
                "branch": branch,
                "legacy_budget_gain_pp": budget_pp,
                "legacy_endpoint_gain_pp": endpoint_pp,
                "useful_extra_budget": budget_pp >= d["budget_gain_threshold_pp"] - 1e-12
                and endpoint_pp >= d["budget_endpoint_gain_threshold_pp"] - 1e-12,
                "early_scale_gain_pp": scale_pp,
                "early_scale_sensitive": scale_pp >= d["early_scale_gain_threshold_pp"] - 1e-12,
            }
        )
    strongest = max(
        *references.values(),
        *(v["score"] for (b, m, n), v in cells.items() if b != "fusion" and n == 200),
    )
    candidates = []
    for mode in p["modes"]:
        score = cells[("fusion", mode, 200)]["score"]
        over_ref = (score - strongest) * 100
        over_mom = (score - cells[("moment", mode, 200)]["score"]) * 100
        candidates.append(
            {
                "mode": mode,
                "over_strongest_reference_pp": over_ref,
                "over_same_mode_moment_pp": over_mom,
                "eligible_for_proposed_replication": over_ref
                >= p["candidate_rules"]["minimum_gain_over_strongest_reference_pp"] - 1e-12
                and over_mom
                >= p["candidate_rules"]["minimum_gain_over_same_mode_moment_pp"] - 1e-12,
            }
        )
    return {
        "diagnostics": diagnostics,
        "strongest_reference_macro_f1": strongest,
        "candidates": candidates,
        "automatic_replication_started": False,
        "automatic_lora_started": False,
    }


def summarize(path):
    f = verify(path)
    p = f["protocol"]
    arrays, _dm = load_prepared(p, f)
    cells = {}
    rows = []
    ids = arrays["validation"]["sample_ids"].astype(str).tolist()
    truth = arrays["validation"]["labels"].tolist()
    pp = read_parent(Path(p["parent_protocol"]))
    for case in cases(p):
        root = run_directory(p, case)
        m = json.loads((root / "metrics.json").read_text())
        if (
            json.loads((root / "status.json").read_text())["status"] != "completed"
            or m["test_evaluated"]
            or m["encoder_executed"]
            or m["protocol_sha256"] != f["protocol_sha256"]
            or m["case"] != case
        ):
            raise ValueError("Incomplete diagnostic run.")
        np.testing.assert_array_equal(
            np.load(root / "train_ids.npy"), arrays["train"]["sample_ids"]
        )
        original = json.loads(
            (parent_run(pp, PARENTS[case["branch"]]) / "metrics.json").read_text()
        )
        if m["initial_head_sha256_without_gain"] != original["initial_head_sha256"]:
            raise ValueError("Head initialization differs.")
        h = json.loads((root / "history.json").read_text())
        if [r["epoch"] for r in h] != list(range(1, 201)):
            raise ValueError("Missing/duplicate epochs.")
        for budget in (30, 200):
            item = m["budgets"][str(budget)]
            if sha256_file(root / f"model_best_{budget}.pt") != item["checkpoint_sha256"]:
                raise ValueError("Budget model changed.")
            with (root / f"validation_predictions_{budget}.csv").open(newline="") as handle:
                pred = list(csv.DictReader(handle))
            if [r["sample_id"] for r in pred] != ids or [
                int(r["true_index"]) for r in pred
            ] != truth:
                raise ValueError("Diagnostic validation targets changed.")
            matrix = [[0] * 3 for _ in range(3)]
            for r in pred:
                probabilities = np.asarray([float(r[f"probability_{i}"]) for i in range(3)])
                predicted = int(r["predicted_index"])
                if (
                    not np.isfinite(probabilities).all()
                    or abs(probabilities.sum() - 1) > 1e-5
                    or (probabilities < 0).any()
                    or probabilities.argmax() != predicted
                ):
                    raise ValueError("Invalid probabilities.")
                matrix[int(r["true_index"])][predicted] += 1
            if matrix != item["validation_metrics"]["confusion_matrix"]:
                raise ValueError("Confusion matrix mismatch.")
            for key, value in metrics_from_confusion(matrix).items():
                if abs(value - item["validation_metrics"][key]) > 1e-12:
                    raise ValueError("Metric mismatch.")
            prefix = h[:budget]
            best = max(r["validation_macro_f1"] for r in prefix)
            earliest = next(r["epoch"] for r in prefix if r["validation_macro_f1"] >= best - 1e-12)
            if (
                abs(best - item["validation_metrics"]["macro_f1"]) > 1e-12
                or earliest != item["best_epoch"]
                or abs(h[budget - 1]["validation_macro_f1"] - item["endpoint_metrics"]["macro_f1"])
                > 1e-12
            ):
                raise ValueError("Budget/history selection mismatch.")
            key = (case["branch"], case["mode"], budget)
            cells[key] = {"score": best, "endpoint": item["endpoint_metrics"]["macro_f1"]}
            rows.append(
                {
                    **case,
                    "budget": budget,
                    "best_macro_f1_percent": best * 100,
                    "endpoint_macro_f1_percent": cells[key]["endpoint"] * 100,
                    "best_epoch": earliest,
                    "best_near_budget_boundary": earliest
                    > budget - p["diagnostic_rules"]["boundary_last_epochs"],
                }
            )
    result = {
        "audit_status": "passed",
        "recorded_at": utc_now(),
        "test_evaluated": False,
        "encoder_executed": False,
        "cells": rows,
        **judge(p, cells, f["reference_macro_f1"]),
        "limitations": p["limitations"],
    }
    root = plan_directory(p)
    atomic_write_json(root / "summary.json", result)
    lines = [
        "# E03D cached-head diagnostic (validation only)",
        "",
        "| Branch | Mode | Budget | Best F1 % | Endpoint F1 % | Best epoch |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['branch']} | {r['mode']} | {r['budget']} | {r['best_macro_f1_percent']:.2f} | {r['endpoint_macro_f1_percent']:.2f} | {r['best_epoch']} |"
        for r in rows
    )
    lines.extend(["", "No automatic replication, encoder training, or test."])
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "prepare", "run", "summarize"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--condition")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = Path(args.protocol)
    if args.action == "freeze":
        freeze(path)
    elif args.action == "verify":
        verify(path)
        print("E03D frozen inputs verified.")
    elif args.action == "prepare":
        prepare(path)
    elif args.action == "run":
        run(path, args.condition, args.resume)
    else:
        summarize(path)


if __name__ == "__main__":
    main()
