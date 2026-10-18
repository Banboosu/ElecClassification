"""E03E: prospectively frozen gain-head replication, excluding the seed43 pilot."""

from __future__ import annotations

import argparse
import csv
import fcntl
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
from tcn_moment.analyze_e01b import budget_rule
from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.controlled_svm import fit_controlled_rbf
from tcn_moment.data import (
    load_dataset,
    normalize_sequence,
    save_label_encoder,
    stratified_train_subset_indices,
    train_subset_sha256,
)
from tcn_moment.experiment import prepare_run, utc_now
from tcn_moment.head_diagnostics import GainFusionHead, fan_in_gain
from tcn_moment.head_diagnostics_stable import training_hooks
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.moment_peft import PeftClassifier, trainable_summary
from tcn_moment.run_e03d_stable import original_head_digest
from tcn_moment.run_e03d_stable import verify as verify_pilot
from tcn_moment.train_baselines import extract_statistical_features
from tcn_moment.train_feature_fusion import cache_signature, validate_cache, verified_checkpoint
from tcn_moment.train_moment import forward_features
from tcn_moment.train_moment_svm import _atomic_joblib_dump

DEFAULT_PROTOCOL = "configs/experiments/improvements/e03e_gain_replication_protocol.json"
PACKAGES = ("torch", "momentfm", "transformers", "numpy", "scikit-learn")


def read_protocol(path):
    p = json.loads(path.read_text())
    if (
        p["experiment_id"] != "E03E"
        or p["test_evaluated"]
        or p["replication_seeds"] != [42, 44, 45, 46]
        or p["excluded_pilot_seed"] != 43
        or p["train_fraction"] != 0.4
        or p["branches"] != ["statistics", "moment", "fusion"]
    ):
        raise ValueError("Unexpected replication scope.")
    t = p["training"]
    expected = {
        "epochs": 200,
        "minimum_epochs": 200,
        "early_stopping": False,
        "batch_size": 128,
        "evaluation_batch_size": 64,
        "feature_extraction_batch_size": 64,
        "head_learning_rate": 0.001,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "precision": "amp_fp16",
        "loss_scale": 1024,
        "loss_scale_growth_interval": 1000000000,
        "head_seed_offset": 10000,
        "training_seed_offset": 20000,
    }
    if any(t[k] != v for k, v in expected.items()):
        raise ValueError("Replication must not retune the pilot.")
    control = p["matched_statistics_control"]
    if (
        control["new_seed"] != 42
        or control["reuse_seeds"] != [44, 45, 46]
        or control["C"] != [0.1, 1.0, 10.0, 100.0, 1000.0]
        or control["gamma_tokens"] != ["reference_scale", 0.05, 5.0]
        or control["cv_folds"] != 5
        or control["cv_shuffle"]
        or control["scoring"] != "macro_f1"
    ):
        raise ValueError("Matched statistics control differs from E01C.")
    r = p["success_rules"]
    if (
        r["number_of_replication_seeds"] != 4
        or r["minimum_positive_seeds"] != 3
        or r["minimum_mean_over_strongest_reference_pp"] != 0.5
        or r["minimum_mean_over_moment_pp"] != 0.25
        or not r["require_both_comparisons"]
        or r["pilot_seed_included"]
        or any(
            r[k]
            for k in (
                "automatically_evaluate_test",
                "automatically_train_lora",
                "automatically_expand_budgets",
            )
        )
    ):
        raise ValueError("Unexpected replication gates/automatic expansion.")
    return p


def plan_directory(p):
    return Path(p["output_dir"]) / f"{p['suite']}_analysis"


def legacy_run(seed):
    suffix = "pilot" if seed == 42 else "expansion"
    return Path("artifacts/improvements/e01") / f"e01_20261017_v2_{suffix}_seed{seed}"


def fusion_run(seed):
    if seed == 42:
        return legacy_run(seed)
    return (
        Path("artifacts/improvements/e01b_high_label") / f"e01b_high_label_20261017_v1_seed{seed}"
    )


def matched_run(p, seed):
    if seed == 42:
        return plan_directory(p) / "matched_statistics_seed42"
    return (
        Path("artifacts/improvements/e01c_attribution")
        / f"e01c_attribution_20261017_v1_seed{seed}/statistics_matched_kernel_fraction_40"
    )


def run_directory(p, seed, branch):
    return Path(p["output_dir"]) / f"{p['suite']}_seed{seed}_{branch}"


def verify(path):
    p = read_protocol(path)
    f = json.loads((plan_directory(p) / "frozen_protocol.json").read_text())
    if p != f["protocol"] or sha256_file(path) != f["protocol_sha256"]:
        raise ValueError("Replication protocol changed.")
    for filename, digest in f["file_sha256"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError(f"Replication input changed: {filename}")
    if f["packages"] != {k: importlib.metadata.version(k) for k in PACKAGES}:
        raise ValueError("Replication packages changed.")
    return f


def freeze(path):
    p = read_protocol(path)
    root = plan_directory(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        verify(path)
        print("Existing replication freeze verified.")
        return
    if any(run_directory(p, s, b).exists() for s in p["replication_seeds"] for b in p["branches"]):
        raise ValueError("Freeze before new head outcomes.")
    pilot = verify_pilot(Path(p["pilot_protocol"]))
    pilot_root = Path(pilot["protocol"]["output_dir"]) / f"{pilot['protocol']['suite']}_analysis"
    summary = json.loads((pilot_root / "summary.json").read_text())
    if summary["audit_status"] != "passed" or not any(
        c["mode"] == "fanin_gain" and c["eligible_for_proposed_replication"]
        for c in summary["candidates"]
    ):
        raise ValueError("Pilot did not qualify.")
    files = [
        path,
        Path(__file__),
        Path("scripts/run_e03e_remote.sh"),
        Path("tests/test_gain_replication.py"),
        Path("docs/experiment_records/e03e_gain_replication_protocol_20261018.md"),
        pilot_root / "frozen_protocol.json",
    ]
    files.extend(Path(k) for k in pilot["file_sha256"])
    for seed in p["replication_seeds"]:
        run = fusion_run(seed)
        files.extend(run / name for name in ("train_ids_40.npy", "split_manifest.json"))
        for base in (
            run / "fusion_stats_svm_fraction_40",
            legacy_run(seed) / "stats_rf_fraction_40",
            legacy_run(seed) / "stats_svm_fraction_40",
        ):
            files.extend(base / name for name in ("metrics.json", "validation_predictions.csv"))
        cache = Path(p["fp32_reference_cache"]) / f"seed{seed}"
        files.extend(cache / name for name in ("manifest.json", "train.npz", "validation.npz"))
        if seed != 42:
            files.extend(
                matched_run(p, seed) / name
                for name in ("metrics.json", "validation_predictions.csv", "kernel_geometry.json")
            )
    atomic_write_json(
        root / "frozen_protocol.json",
        {
            "frozen_at": utc_now(),
            "protocol": p,
            "protocol_sha256": sha256_file(path),
            "file_sha256": {str(k): sha256_file(k) for k in files},
            "packages": {k: importlib.metadata.version(k) for k in PACKAGES},
            "pilot_protocol_sha256": pilot["protocol_sha256"],
            "boundary": "seed43 excluded; older reference outcomes known; 12 new heads plus one missing matched statistics control.",
        },
    )
    print("Frozen E03E: seeds42/44/45/46, 40%, 12 gain heads; test locked.")


def normalize_masked_rows(raw, mask):
    if raw.ndim != 2 or raw.shape != mask.shape:
        raise ValueError("Wrong raw/mask shape.")
    result = np.zeros_like(raw, dtype=np.float32)
    for i, row in enumerate(raw):
        valid = int(mask[i].sum())
        if valid <= 0 or not np.all(mask[i, :valid] == 1) or not np.all(mask[i, valid:] == 0):
            raise ValueError("Expected nonempty prefix mask.")
        result[i, :valid] = normalize_sequence(row[:valid], "zscore")
    return result


def prepared_directory(p, seed):
    return plan_directory(p) / f"prepared_seed{seed}"


def load_prepared(p, f, seed):
    root = prepared_directory(p, seed)
    m = json.loads((root / "manifest.json").read_text())
    if m["protocol_sha256"] != f["protocol_sha256"] or m["seed"] != seed:
        raise ValueError("Wrong prepared seed/protocol.")
    if sha256_file(root / "normalizer.json") != m["normalizer_sha256"]:
        raise ValueError("Training scaler changed.")
    arrays = {}
    for split in ("train", "validation"):
        file = root / f"{split}.npz"
        if sha256_file(file) != m["splits"][split]["sha256"]:
            raise ValueError("Prepared feature corruption.")
        with np.load(file, allow_pickle=False) as data:
            arrays[split] = {k: data[k].copy() for k in data.files}
        a = arrays[split]
        n = len(a["labels"])
        if (
            a["features"].shape != (n, 1024)
            or a["statistics"].shape != (n, 15)
            or not np.isfinite(a["features"]).all()
            or not np.isfinite(a["statistics"]).all()
        ):
            raise ValueError("Prepared shape/values invalid.")
    return arrays, m


def save_predictions(path, ids, truth, predicted, probs=None):
    frame = {"sample_id": ids, "true_index": truth, "predicted_index": predicted}
    if probs is not None:
        frame.update({f"probability_{i}": probs[:, i] for i in range(3)})
    pd.DataFrame(frame).to_csv(path, index=False)


def missing_matched_statistics(p, f, config, bundle, indices, stats):
    root = matched_run(p, 42)
    if (root / "metrics.json").exists():
        return
    root.mkdir(exist_ok=True)
    cached = validate_cache(
        Path(p["fp32_reference_cache"]) / "seed42",
        cache_signature(config, bundle),
        {"train": bundle.ids_train, "validation": bundle.ids_val},
    )
    if cached is None:
        raise ValueError("Parent FP32 reference cache required.")
    reference = np.concatenate([cached["train"][indices], stats["train"]], axis=1)
    rule = p["matched_statistics_control"]
    result = fit_controlled_rbf(
        stats["train"],
        bundle.y_train[indices],
        reference,
        kind="statistics",
        reference_dim=1024,
        c_values=rule["C"],
        gamma_tokens=rule["gamma_tokens"],
        n_jobs=rule["n_jobs"],
        device=rule["kernel_device"],
    )
    predicted = result.predict(stats["validation"])
    from tcn_moment.metrics import classification_metrics

    metrics = classification_metrics(
        bundle.y_val,
        predicted,
        [str(v) for v in bundle.label_encoder.classes_],
        include_details=True,
    )
    _atomic_joblib_dump(result.best_estimator_, root / "classifier.joblib")
    pd.DataFrame(result.cv_results_).to_csv(root / "cv_results.csv", index=False)
    atomic_write_json(root / "kernel_geometry.json", result.geometry_)
    save_predictions(root / "validation_predictions.csv", bundle.ids_val, bundle.y_val, predicted)
    atomic_write_json(
        root / "metrics.json",
        {
            "seed": 42,
            "train_fraction": 0.4,
            "protocol_sha256": f["protocol_sha256"],
            "train_ids_sha256": train_subset_sha256(bundle.ids_train[indices]),
            "test_evaluated": False,
            "validation_metrics": metrics,
            "best_params": result.best_params_,
            "cv_score": result.best_score_,
        },
    )


def prepare(path, seed):
    f = verify(path)
    p = f["protocol"]
    if seed not in p["replication_seeds"]:
        raise ValueError("Seed outside replication.")
    root = prepared_directory(p, seed)
    root.mkdir(exist_ok=True)
    if (root / "manifest.json").exists():
        load_prepared(p, f, seed)
        print(f"Prepared seed{seed} already verified.")
        return
    config = with_random_seed(load_config(p["training_config"]), seed)
    print(
        f"Preparing seed{seed}: raw rows once, unadapted AMP encoding train/validation only",
        flush=True,
    )
    bundle = load_dataset(replace(config.data, normalize="none"))
    parent = fusion_run(seed)
    if sha256_file(bundle.split_path) != sha256_file(parent / "split_manifest.json"):
        raise ValueError("Replication split mismatch.")
    indices = stratified_train_subset_indices(bundle.y_train, bundle.ids_train, 0.4, seed)
    np.testing.assert_array_equal(bundle.ids_train[indices], np.load(parent / "train_ids_40.npy"))
    stats = {
        "train": extract_statistical_features(bundle.x_train[indices], bundle.mask_train[indices]),
        "validation": extract_statistical_features(bundle.x_val, bundle.mask_val),
    }
    if seed == 42:
        missing_matched_statistics(p, f, config, bundle, indices, stats)
    pp = json.loads(
        Path("configs/experiments/improvements/e03_peft_pilot_protocol.json").read_text()
    )
    pp["seeds"] = [seed]
    checkpoint = verified_checkpoint(config)
    cache_manifest_path = root / "amp_manifest.json"
    signature = {
        "seed": seed,
        "protocol_sha256": f["protocol_sha256"],
        "checkpoint": checkpoint,
        "dataset_sha256": bundle.dataset_sha256,
        "train_ids_sha256": train_subset_sha256(bundle.ids_train[indices]),
        "precision": "amp_fp16",
        "supervised_encoder_training": False,
    }
    cm = (
        json.loads(cache_manifest_path.read_text())
        if cache_manifest_path.exists()
        else {"signature": signature, "splits": {}}
    )
    if cm["signature"] != signature:
        raise ValueError("Replication AMP cache signature mismatch.")
    inputs = {
        "train": (
            bundle.x_train[indices],
            bundle.mask_train[indices],
            bundle.y_train[indices],
            bundle.ids_train[indices],
        ),
        "validation": (bundle.x_val, bundle.mask_val, bundle.y_val, bundle.ids_val),
    }
    model = None
    try:
        for split, (raw, mask, y, ids) in inputs.items():
            file = root / f"{split}.npz"
            if split not in cm["splits"]:
                if model is None:
                    model = train_e03.build_backbone(config, 0, pp).cuda().eval()
                x = normalize_masked_rows(raw, mask)
                chunks = []
                with torch.inference_mode():
                    for start in range(0, len(y), 64):
                        xx = torch.as_tensor(
                            x[start : start + 64, None], device="cuda", dtype=torch.float32
                        )
                        mm = torch.as_tensor(
                            mask[start : start + 64], device="cuda", dtype=torch.float32
                        )
                        with torch.amp.autocast("cuda"):
                            value = forward_features(model, xx, mm).float().cpu().numpy()
                        if not np.isfinite(value).all():
                            raise FloatingPointError("Nonfinite AMP features.")
                        chunks.append(value)
                features = np.concatenate(chunks)
                temp = file.with_suffix(".npz.tmp")
                with temp.open("wb") as handle:
                    np.savez(
                        handle,
                        features=features,
                        statistics=stats[split],
                        labels=y,
                        sample_ids=ids.astype(str),
                    )
                temp.replace(file)
                cm["splits"][split] = {"sha256": sha256_file(file), "shape": list(features.shape)}
                atomic_write_json(cache_manifest_path, cm)
            if sha256_file(file) != cm["splits"][split]["sha256"]:
                raise ValueError("Corrupt AMP features.")
            with np.load(file, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["sample_ids"], ids.astype(str))
    finally:
        del model
        torch.cuda.empty_cache()
    scaler = StandardScaler().fit(stats["train"])
    atomic_write_json(
        root / "normalizer.json",
        {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "n": len(indices),
            "fitted_on": "selected train only",
        },
    )
    save_label_encoder(bundle.label_encoder, root)
    atomic_write_json(
        root / "manifest.json",
        {
            "seed": seed,
            "protocol_sha256": f["protocol_sha256"],
            "dataset_sha256": bundle.dataset_sha256,
            "classes": bundle.label_encoder.classes_.tolist(),
            "normalizer_sha256": sha256_file(root / "normalizer.json"),
            "train_ids_sha256": signature["train_ids_sha256"],
            "splits": cm["splits"],
        },
    )
    load_prepared(p, f, seed)
    print(f"Prepared seed{seed}; encoder frozen, test not encoded.", flush=True)


def new_head(p, seed, branch, norm):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(p["training"]["head_seed_offset"] + seed)
        return GainFusionHead(
            norm["mean"],
            norm["scale"],
            branch=branch,
            hidden_dim=128,
            dropout=0.1,
            input_gain=fan_in_gain(branch),
        )


def run(path, seed, branch, resume):
    f = verify(path)
    p = f["protocol"]
    if seed not in p["replication_seeds"] or branch not in p["branches"]:
        raise ValueError("Invalid replication case.")
    root = run_directory(p, seed, branch)
    lock = (plan_directory(p) / f"seed{seed}_{branch}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if root.exists() and not resume:
        raise FileExistsError("Explicit resume required.")
    if resume and not root.exists():
        raise FileNotFoundError("Missing run.")
    if resume and json.loads((root / "status.json").read_text())["status"] == "completed":
        print(f"Preserved seed{seed}/{branch}")
        lock.close()
        return
    config_path = Path(p["training_config"])
    config = with_random_seed(load_config(config_path), seed)
    context = prepare_run(
        model_name="E03E_GAIN_REPLICATION",
        base_output_dir=Path(p["output_dir"]),
        config=config,
        config_path=config_path,
        torch=torch,
        run_name=root.name,
        resume_dir=root if resume else None,
    )
    try:
        arrays, dm = load_prepared(p, f, seed)
        prep = prepared_directory(p, seed)
        norm = json.loads((prep / "normalizer.json").read_text())
        head = new_head(p, seed, branch, norm)
        metadata = {
            "protocol_sha256": f["protocol_sha256"],
            "seed": seed,
            "branch": branch,
            "train_fraction": 0.4,
            "input_gain": fan_in_gain(branch),
            "initial_head_sha256_without_gain": original_head_digest(head),
            "prepared_manifest_sha256": sha256_file(prep / "manifest.json"),
            "train_ids_sha256": dm["train_ids_sha256"],
            "classes": dm["classes"],
            "encoder_trainable": False,
        }
        if not resume:
            atomic_write_json(root / "experiment_plan.json", metadata)
            shutil.copy2(plan_directory(p) / "frozen_protocol.json", root / "frozen_protocol.json")
            shutil.copy2(prep / "label_encoder.pkl", root / "label_encoder.pkl")
            np.save(root / "train_ids.npy", arrays["train"]["sample_ids"])
            (root / "source").mkdir()
            shutil.copy2(Path(__file__), root / "source" / Path(__file__).name)
        elif json.loads((root / "experiment_plan.json").read_text()) != metadata:
            raise ValueError("Resume scope changed.")
        pp = json.loads(
            Path("configs/experiments/improvements/e03_peft_pilot_protocol.json").read_text()
        )
        pp["seeds"] = [seed]
        pp["training"].update(maximum_epochs=200, minimum_epochs=200)
        model = PeftClassifier(None, head).cuda()
        data = {
            k: train_e03.gpu_dataset(v["features"], v["statistics"], v["labels"])
            for k, v in arrays.items()
        }
        print(
            f"Training seed{seed}/{branch}, 200 epochs, fixed scale1024, no encoder gradients",
            flush=True,
        )
        with training_hooks(
            train_e03, root, steps_per_epoch=math.ceil(len(data["train"]["y"]) / 128)
        ):
            result, probs = train_e03.train(
                model,
                data["train"],
                data["validation"],
                pp,
                root,
                metadata,
                [str(v) for v in dm["classes"]],
                resume=resume,
            )
        last = torch.load(root / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        from tcn_moment.moment_peft import restore_learned_state

        restore_learned_state(model, last["learned_state"])
        endpoint, _ = train_e03.evaluate(
            model, data["validation"], 64, [str(v) for v in dm["classes"]], details=True
        )
        save_predictions(
            root / "validation_predictions.csv",
            arrays["validation"]["sample_ids"],
            arrays["validation"]["labels"],
            probs.argmax(axis=1),
            probs,
        )
        atomic_write_json(
            root / "metrics.json",
            {
                **metadata,
                **result,
                "endpoint_metrics": endpoint,
                "parameters": trainable_summary(model),
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


def audit_predictions(path, ids, truth, metrics):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [r["sample_id"] for r in rows] != ids.astype(str).tolist() or [
        int(r["true_index"]) for r in rows
    ] != truth.tolist():
        raise ValueError("Validation targets changed.")
    matrix = [[0] * 3 for _ in range(3)]
    for row in rows:
        predicted = int(row["predicted_index"])
        if predicted not in range(3):
            raise ValueError("Invalid class.")
        if "probability_0" in row:
            probs = np.asarray([float(row[f"probability_{i}"]) for i in range(3)])
            if (
                not np.isfinite(probs).all()
                or (probs < 0).any()
                or abs(probs.sum() - 1) > 1e-5
                or probs.argmax() != predicted
            ):
                raise ValueError("Invalid probabilities.")
        matrix[int(row["true_index"])][predicted] += 1
    if matrix != metrics["confusion_matrix"]:
        raise ValueError("Confusion matrix mismatch.")
    for key, value in metrics_from_confusion(matrix).items():
        if abs(value - metrics[key]) > 1e-12:
            raise ValueError("Metric mismatch.")


def decision(differences, p, threshold):
    return budget_rule(
        differences,
        {
            "number_of_replication_seeds": p["success_rules"]["number_of_replication_seeds"],
            "minimum_mean_difference_pp_per_fraction": threshold,
            "minimum_strictly_positive_seeds_per_fraction": p["success_rules"][
                "minimum_positive_seeds"
            ],
        },
    )


def summarize(path):
    f = verify(path)
    p = f["protocol"]
    rows = []
    for seed in p["replication_seeds"]:
        arrays, dm = load_prepared(p, f, seed)
        scores = {}
        endpoints = {}
        initial_heads = set()
        for branch in p["branches"]:
            root = run_directory(p, seed, branch)
            m = json.loads((root / "metrics.json").read_text())
            if (
                json.loads((root / "status.json").read_text())["status"] != "completed"
                or m["test_evaluated"]
                or m["encoder_trainable"]
                or m["protocol_sha256"] != f["protocol_sha256"]
                or m["seed"] != seed
                or m["branch"] != branch
                or m["parameters"]["adapter_parameters"] != 0
                or m["train_ids_sha256"] != dm["train_ids_sha256"]
                or sha256_file(root / "model_best.pt") != m["best_checkpoint_sha256"]
            ):
                raise ValueError("Replication case incomplete/mismatched.")
            np.testing.assert_array_equal(
                np.load(root / "train_ids.npy"), arrays["train"]["sample_ids"]
            )
            audit_predictions(
                root / "validation_predictions.csv",
                arrays["validation"]["sample_ids"],
                arrays["validation"]["labels"],
                m["validation_metrics"],
            )
            history = json.loads((root / "history.json").read_text())
            if [r["epoch"] for r in history] != list(range(1, 201)) or any(
                r["grad_scaler_scale"] != 1024 or r["skipped_optimizer_steps"] != 0 for r in history
            ):
                raise ValueError("Replication training/numerics budget changed.")
            best = max(r["validation_macro_f1"] for r in history)
            earliest = next(r["epoch"] for r in history if r["validation_macro_f1"] >= best - 1e-12)
            if (
                abs(best - m["validation_metrics"]["macro_f1"]) > 1e-12
                or m["best_epoch"] != earliest
                or abs(history[-1]["validation_macro_f1"] - m["endpoint_metrics"]["macro_f1"])
                > 1e-12
            ):
                raise ValueError("Checkpoint/endpoint selection mismatch.")
            initial_heads.add(m["initial_head_sha256_without_gain"])
            scores[branch] = best
            endpoints[branch] = m["endpoint_metrics"]["macro_f1"]
        if len(initial_heads) != 1:
            raise ValueError("Replication heads did not share seed-specific initial parameters.")
        references = {}
        locations = {
            "parent_fusion_svm": fusion_run(seed) / "fusion_stats_svm_fraction_40",
            "parent_stats_rf": legacy_run(seed) / "stats_rf_fraction_40",
            "parent_stats_svm": legacy_run(seed) / "stats_svm_fraction_40",
            "matched_statistics_svm": matched_run(p, seed),
        }
        for name, base in locations.items():
            item = json.loads((base / "metrics.json").read_text())
            if item["test_evaluated"] or item["train_ids_sha256"] != dm["train_ids_sha256"]:
                raise ValueError("Reference budget differs.")
            audit_predictions(
                base / "validation_predictions.csv",
                arrays["validation"]["sample_ids"],
                arrays["validation"]["labels"],
                item["validation_metrics"],
            )
            references[name] = item["validation_metrics"]["macro_f1"]
        strongest = max(scores["statistics"], scores["moment"], *references.values())
        rows.append(
            {
                "seed": seed,
                **{f"{k}_f1_percent": v * 100 for k, v in scores.items()},
                **{f"{k}_f1_percent": v * 100 for k, v in references.items()},
                "strongest_reference_f1_percent": strongest * 100,
                "fusion_minus_strongest_pp": (scores["fusion"] - strongest) * 100,
                "fusion_minus_moment_pp": (scores["fusion"] - scores["moment"]) * 100,
                "fusion_minus_parent_fusion_svm_pp": (
                    scores["fusion"] - references["parent_fusion_svm"]
                )
                * 100,
                "fusion_endpoint_f1_percent": endpoints["fusion"] * 100,
            }
        )
    primary = decision([r["fusion_minus_strongest_pp"] for r in rows], p, 0.5)
    side = decision([r["fusion_minus_moment_pp"] for r in rows], p, 0.25)
    result = {
        "audit_status": "passed",
        "recorded_at": utc_now(),
        "test_evaluated": False,
        "pilot_seed_included": False,
        "per_seed": rows,
        "primary_rule": primary,
        "moment_ablation_rule": side,
        "replication_rule_passed": primary["rule_passed"] and side["rule_passed"],
        "limitations": p["limitations"],
    }
    root = plan_directory(p)
    atomic_write_json(root / "summary.json", result)
    pd.DataFrame(rows).to_csv(root / "per_seed_metrics.csv", index=False)
    lines = [
        "# E03E gain-head replication (validation only)",
        "",
        "| Seed | Statistics NN | MOMENT NN | Fusion NN | Strongest reference | Gain pp |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['seed']} | {r['statistics_f1_percent']:.2f} | {r['moment_f1_percent']:.2f} | {r['fusion_f1_percent']:.2f} | {r['strongest_reference_f1_percent']:.2f} | {r['fusion_minus_strongest_pp']:+.2f} |"
        for r in rows
    )
    lines.extend(
        [
            "",
            f"Replication rule passed: {result['replication_rule_passed']}",
            "Pilot43 excluded. No automatic test, LoRA or budget expansion.",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "prepare", "run", "summarize"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--branch")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = Path(args.protocol)
    if args.action == "freeze":
        freeze(path)
    elif args.action == "verify":
        verify(path)
        print("E03E inputs verified.")
    elif args.action == "prepare":
        prepare(path, args.seed)
    elif args.action == "run":
        run(path, args.seed, args.branch, args.resume)
    else:
        summarize(path)


if __name__ == "__main__":
    main()
