"""E03T: one explicitly authorized, frozen 40%-label test evaluation; no fitting."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from tcn_moment import replicate_gain_heads as parent
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices, train_subset_sha256
from tcn_moment.experiment import collect_environment, utc_now
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.metrics import classification_metrics
from tcn_moment.moment_peft import PeftClassifier, restore_learned_state
from tcn_moment.train_baselines import extract_statistical_features
from tcn_moment.train_e03 import build_backbone
from tcn_moment.train_feature_fusion import verified_checkpoint
from tcn_moment.train_moment import forward_features
from tcn_moment.training_utils import seed_everything

DEFAULT_PROTOCOL = "configs/experiments/improvements/e03t_final_test_protocol.json"
METHODS = (
    "fusion_nn",
    "statistics_nn",
    "moment_nn",
    "fusion_svm",
    "matched_statistics_svm",
    "statistics_svm",
    "statistics_rf",
)
PACKAGES = (*parent.PACKAGES, "scipy", "pandas", "joblib", "safetensors")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_protocol(path):
    p = read_json(path)
    if (
        p["experiment_id"] != "E03T"
        or p["seeds"] != [42, 44, 45, 46]
        or p["excluded_pilot_seed"] != 43
        or p["train_fraction"] != 0.4
        or p["methods"] != list(METHODS)
        or p["training_allowed"]
        or p["tuning_allowed"]
        or p["automatic_followup_allowed"]
        or p["automatic_shutdown_allowed"]
        or p["classes"] != ["0", "1", "2"]
        or p["test_count"] != 7020
        or p["selected_train_count"] != 9827
        or p["validation_count"] != 3510
    ):
        raise ValueError("Unexpected authorized test scope.")
    rules = p["success_rules"]
    if (
        rules["primary_fixed_comparator"] != "fusion_svm"
        or rules["ablation_fixed_comparator"] != "moment_nn"
        or rules["primary_minimum_mean_difference_pp"] != 0.5
        or rules["ablation_minimum_mean_difference_pp"] != 0.25
        or rules["minimum_strictly_positive_seeds"] != 3
        or not rules["require_both"]
    ):
        raise ValueError("Test gates must not be changed to rescue outcomes.")
    if not p["preflight"]["all_seeds_must_pass_before_any_test_encoding"]:
        raise ValueError("The global pre-test barrier is required.")
    if (
        p["inference"]["backbone_batch_size"] != 64
        or p["inference"]["head_batch_size"] != 64
        or p["inference"]["allow_tf32"]
        or p["inference"]["nn_features"] != "amp_fp16"
        or p["inference"]["fusion_svm_features"] != "fp32"
    ):
        raise ValueError("Unexpected inference recipe.")
    return p


def directory(p):
    return Path(p["output_dir"]) / f"{p['suite']}_analysis"


def seed_directory(p, seed):
    return directory(p) / f"seed{seed}"


def model_directories(pp, seed):
    return {
        **{
            f"{b}_nn": parent.run_directory(pp, seed, b) for b in ("fusion", "statistics", "moment")
        },
        "fusion_svm": parent.fusion_run(seed) / "fusion_stats_svm_fraction_40",
        "matched_statistics_svm": parent.matched_run(pp, seed),
        "statistics_svm": parent.legacy_run(seed) / "stats_svm_fraction_40",
        "statistics_rf": parent.legacy_run(seed) / "stats_rf_fraction_40",
    }


def verify(path):
    p = read_protocol(path)
    f = read_json(directory(p) / "frozen_protocol.json")
    if f["protocol"] != p or f["protocol_sha256"] != sha256_file(path):
        raise ValueError("Frozen test protocol changed.")
    for name, digest in f["file_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Frozen input/model changed: {name}")
    if f["packages"] != {k: importlib.metadata.version(k) for k in PACKAGES}:
        raise ValueError("Frozen test package versions changed.")
    return f


def freeze(path):
    p = read_protocol(path)
    root = directory(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        return verify(path)
    if any(root.glob("seed*")) or (root / "test_started.json").exists():
        raise ValueError("Freeze before technical preflight and all test outcomes.")
    pf = parent.verify(Path(p["parent_protocol"]))
    pp = pf["protocol"]
    parent_summary = read_json(parent.plan_directory(pp) / "summary.json")
    if parent_summary["audit_status"] != "passed" or parent_summary["test_evaluated"]:
        raise ValueError("Wrong parent stage.")
    files = {Path(k) for k in pf["file_sha256"]}
    files.update(Path("src/tcn_moment").glob("*.py"))
    files.update(Path("configs").rglob("*.json"))
    files.update(Path("configs").rglob("*.yaml"))
    files.update(
        [
            path,
            Path("scripts/run_e03t_remote.sh"),
            Path("tests/test_final_test.py"),
            Path("docs/experiment_records/e03t_final_test_protocol_20261019.md"),
            Path("pyproject.toml"),
            Path("uv.lock"),
            Path("scripts/autodl_env.sh"),
            parent.plan_directory(pp) / "frozen_protocol.json",
            parent.plan_directory(pp) / "summary.json",
        ]
    )
    records = {}
    config = load_config(p["training_config"])
    checkpoint = verified_checkpoint(config)
    from huggingface_hub import try_to_load_from_cache

    weight_path = Path(try_to_load_from_cache(checkpoint["model_id"], checkpoint["filename"]))
    files.add(weight_path)
    files.add(weight_path.parent / "config.json")
    for seed in p["seeds"]:
        prep = parent.prepared_directory(pp, seed)
        dm = read_json(prep / "manifest.json")
        if dm["classes"] != p["classes"]:
            raise ValueError("Parent class encoding changed.")
        for name in (
            "manifest.json",
            "amp_manifest.json",
            "normalizer.json",
            "validation.npz",
            "label_encoder.pkl",
        ):
            files.add(prep / name)
        files.update(
            parent.fusion_run(seed) / name for name in ("split_manifest.json", "train_ids_40.npy")
        )
        records[str(seed)] = {}
        for method, case in model_directories(pp, seed).items():
            m = read_json(case / "metrics.json")
            if m["test_evaluated"] or m["train_ids_sha256"] != dm["train_ids_sha256"]:
                raise ValueError("Model was tested already or uses different training rows.")
            if "train_fraction" in m and m["train_fraction"] != 0.4:
                raise ValueError("Wrong baseline training budget.")
            model_file = case / ("model_best.pt" if method.endswith("_nn") else "classifier.joblib")
            files.update([model_file, case / "metrics.json", case / "validation_predictions.csv"])
            if method.endswith("_nn"):
                if read_json(case / "status.json")["status"] != "completed":
                    raise ValueError("Incomplete parent NN.")
                if sha256_file(model_file) != m["best_checkpoint_sha256"]:
                    raise ValueError("Wrong best checkpoint.")
                files.update(
                    [
                        case / "experiment_plan.json",
                        case / "label_encoder.pkl",
                        case / "status.json",
                    ]
                )
            records[str(seed)][method] = {
                "directory": str(case),
                "model": str(model_file),
                "model_sha256": sha256_file(model_file),
                "train_ids_sha256": dm["train_ids_sha256"],
                "best_epoch": m.get("best_epoch"),
                "validation_macro_f1": m["validation_metrics"]["macro_f1"],
            }
        # A comparator fixed from validation, not selected from test results.
        best_reference = max(
            (k for k in METHODS if k != "fusion_nn"),
            key=lambda k: records[str(seed)][k]["validation_macro_f1"],
        )
        if best_reference != "fusion_svm":
            raise ValueError("Predeclared strongest validation comparator no longer matches.")
    for module in (
        "momentfm.models.moment",
        "momentfm.models.layers.embed",
        "momentfm.models.layers.revin",
    ):
        import importlib

        files.add(Path(importlib.import_module(module).__file__))
    f = {
        "frozen_at": utc_now(),
        "protocol": p,
        "protocol_sha256": sha256_file(path),
        "packages": {k: importlib.metadata.version(k) for k in PACKAGES},
        "file_sha256": {str(k): sha256_file(k) for k in sorted(files)},
        "selected_models": records,
        "checkpoint": checkpoint,
        "environment": collect_environment(torch=torch),
        "test_outcomes_observed_at_freeze": False,
    }
    # Snapshot source/config/protocol/launcher; old large artifacts are hash-linked.
    import shutil

    for file in sorted(files):
        if not file.is_absolute() and file.parts[0] in {
            "src",
            "configs",
            "tests",
            "scripts",
            "docs",
        }:
            dest = root / "source_snapshot" / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
    atomic_write_json(root / "frozen_protocol.json", f)
    print(f"Frozen E03T: 28 existing models, no fitting; {len(files)} input files.", flush=True)
    return f


def hardware(p):
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name() != p["inference"]["gpu"]
        or list(torch.cuda.get_device_capability()) != p["inference"]["compute_capability"]
    ):
        raise ValueError("Frozen test hardware mismatch.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def bundle_for(p, seed):
    config = with_random_seed(load_config(p["training_config"]), seed)
    bundle = load_dataset(replace(config.data, normalize="none"))
    pp = parent.read_protocol(Path(p["parent_protocol"]))
    source = parent.fusion_run(seed)
    if sha256_file(bundle.split_path) != sha256_file(source / "split_manifest.json"):
        raise ValueError("Original split changed.")
    indices = stratified_train_subset_indices(bundle.y_train, bundle.ids_train, 0.4, seed)
    np.testing.assert_array_equal(bundle.ids_train[indices], np.load(source / "train_ids_40.npy"))
    dm = read_json(parent.prepared_directory(pp, seed) / "manifest.json")
    if (
        len(indices) != p["selected_train_count"]
        or len(bundle.y_val) != p["validation_count"]
        or len(bundle.y_test) != p["test_count"]
        or dm["train_ids_sha256"] != train_subset_sha256(bundle.ids_train[indices])
        or bundle.label_encoder.classes_.tolist() != p["classes"]
    ):
        raise ValueError("Split/budget/class mismatch.")
    return config, bundle


def load_nn(p, f, seed, method):
    pp = parent.read_protocol(Path(p["parent_protocol"]))
    norm = read_json(parent.prepared_directory(pp, seed) / "normalizer.json")
    branch = method.removesuffix("_nn")
    model = PeftClassifier(None, parent.new_head(pp, seed, branch, norm))
    rec = f["selected_models"][str(seed)][method]
    state = torch.load(rec["model"], map_location="cpu", weights_only=False)
    if (
        state["epoch"] != rec["best_epoch"]
        or state["metadata"]["seed"] != seed
        or state["metadata"]["branch"] != branch
        or state["metadata"]["classes"] != p["classes"]
        or state["metadata"]["train_ids_sha256"] != rec["train_ids_sha256"]
        or state["learned_state"]["adapters"]
    ):
        raise ValueError("NN checkpoint does not match the frozen method.")
    restore_learned_state(model, state["learned_state"])
    np.testing.assert_array_equal(
        model.classifier.stats_mean.numpy(), np.asarray(norm["mean"], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        model.classifier.stats_scale.numpy(), np.asarray(norm["scale"], dtype=np.float32)
    )
    return model.cuda().eval()


def nn_predict(model, features, statistics, batch_size=64):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device="cuda"
            )
            stats = torch.as_tensor(
                statistics[start : start + batch_size], dtype=torch.float32, device="cuda"
            )
            with torch.amp.autocast("cuda"):
                logits = model(x, stats)
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(chunks)
    validate_probabilities(probs, len(features))
    return probs.argmax(axis=1), probs


def validate_probabilities(probs, count):
    if (
        probs.shape != (count, 3)
        or not np.isfinite(probs).all()
        or (probs < 0).any()
        or (probs > 1).any()
        or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid three-class probabilities.")


def pipeline_predict(model, method, features, statistics):
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("Pipeline class order mismatch.")
    x = np.concatenate([features, statistics], axis=1) if method == "fusion_svm" else statistics
    predicted = model.predict(x)
    probs = model.predict_proba(x) if method == "statistics_rf" else None
    if probs is not None:
        validate_probabilities(probs, len(x))
    return predicted, probs


def parity(old, predicted, probs=None):
    if len(old) != len(predicted):
        raise ValueError("Parity row-count mismatch.")
    result = {"prediction_agreement": float(np.mean(old.predicted_index.to_numpy() == predicted))}
    if probs is not None:
        result["max_probability_abs_error"] = float(
            np.max(np.abs(old[[f"probability_{k}" for k in range(3)]].to_numpy() - probs))
        )
    return result


def encode(config, raw, mask, seed):
    """Same raw-prefix preprocessing and backbone, both predeclared precisions."""
    seed_everything(torch, seed)
    pp = read_json("configs/experiments/improvements/e03_peft_pilot_protocol.json")
    pp["seeds"] = [seed]
    backbone = build_backbone(config, 0, pp).cuda().eval()
    backbone.requires_grad_(False)
    x = parent.normalize_masked_rows(raw, mask)
    chunks = {"amp": [], "fp32": []}
    try:
        with torch.inference_mode():
            for start in range(0, len(x), 64):
                xx = torch.as_tensor(
                    x[start : start + 64, None], device="cuda", dtype=torch.float32
                )
                mm = torch.as_tensor(mask[start : start + 64], device="cuda", dtype=torch.float32)
                for mode, output in chunks.items():
                    with torch.amp.autocast("cuda", enabled=mode == "amp"):
                        value = forward_features(backbone, xx, mm).float().cpu().numpy()
                    if not np.isfinite(value).all():
                        raise FloatingPointError("Nonfinite test/probe embeddings.")
                    output.append(value)
                if (start // 64) % 20 == 0:
                    print(
                        f"seed{seed} encoded {min(start + 64, len(x))}/{len(x)} rows, both precisions",
                        flush=True,
                    )
    finally:
        del backbone
        torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in chunks.items()}


def preflight(path):
    f = verify(path)
    p = f["protocol"]
    root = directory(p)
    hardware(p)
    output = root / "preflight.json"
    if output.exists():
        report = read_json(output)
        if report["status"] != "passed" or report["protocol_sha256"] != f["protocol_sha256"]:
            raise ValueError("Failed/foreign technical preflight must not be silently retried.")
        return report
    if (root / "test_started.json").exists():
        raise ValueError("Cannot create a new preflight after test started.")
    results = {}
    limits = p["preflight"]
    try:
        pp = parent.read_protocol(Path(p["parent_protocol"]))
        for seed in p["seeds"]:
            config, bundle = bundle_for(p, seed)
            prep = parent.prepared_directory(pp, seed)
            with np.load(prep / "validation.npz", allow_pickle=False) as archive:
                old_amp = {k: archive[k].copy() for k in archive.files}
            cache = Path(pp["fp32_reference_cache"]) / f"seed{seed}/validation.npz"
            with np.load(cache, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["sample_ids"], bundle.ids_val.astype(str))
                old_fp32 = archive["features"].copy()
            np.testing.assert_array_equal(old_amp["sample_ids"], bundle.ids_val.astype(str))
            np.testing.assert_array_equal(old_amp["labels"], bundle.y_val)
            statistics = extract_statistical_features(bundle.x_val, bundle.mask_val)
            np.testing.assert_array_equal(statistics, old_amp["statistics"])
            n = limits["fresh_validation_probe_rows"]
            fresh = encode(config, bundle.x_val[:n], bundle.mask_val[:n], seed)
            fp32_error = float(np.max(np.abs(fresh["fp32"] - old_fp32[:n])))
            if fp32_error > limits["fresh_fp32_max_feature_abs_error"]:
                raise ValueError(f"seed{seed}: new GPU FP32 probe mismatch {fp32_error}")
            checks = {"fresh_fp32_max_abs_error": fp32_error, "methods": {}}
            results[str(seed)] = checks
            for method in p["methods"]:
                rec = f["selected_models"][str(seed)][method]
                old = pd.read_csv(
                    Path(rec["directory"]) / "validation_predictions.csv", dtype={"sample_id": str}
                )
                np.testing.assert_array_equal(old.sample_id, bundle.ids_val.astype(str))
                np.testing.assert_array_equal(old.true_index, bundle.y_val)
                if method.endswith("_nn"):
                    model = load_nn(p, f, seed, method)
                    pred, probs = nn_predict(model, old_amp["features"], statistics)
                    cached = parity(old, pred, probs)
                    f1 = classification_metrics(bundle.y_val, pred, p["classes"])["macro_f1"]
                    cached["macro_f1_difference_pp"] = 100 * (f1 - rec["validation_macro_f1"])
                    pred2, probs2 = nn_predict(model, fresh["amp"], statistics[:n])
                    probe = parity(old.iloc[:n], pred2, probs2)
                    if (
                        cached["prediction_agreement"]
                        < limits["cached_nn_min_prediction_agreement"]
                        or cached["max_probability_abs_error"]
                        > limits["cached_nn_max_probability_abs_error"]
                        or abs(cached["macro_f1_difference_pp"])
                        > limits["cached_nn_max_macro_f1_difference_pp"]
                        or probe["prediction_agreement"]
                        < limits["fresh_nn_min_prediction_agreement"]
                        or probe["max_probability_abs_error"]
                        > limits["fresh_nn_max_probability_abs_error"]
                    ):
                        raise ValueError(f"seed{seed}/{method} NN parity failed: {cached}, {probe}")
                    checks["methods"][method] = {"cached": cached, "fresh_probe": probe}
                else:
                    model = joblib.load(rec["model"])
                    pred, _ = pipeline_predict(model, method, old_fp32, statistics)
                    if not np.array_equal(pred, old.predicted_index):
                        raise ValueError(f"seed{seed}/{method}: baseline restore failed.")
                    check = {"cached_prediction_agreement": 1.0}
                    if method == "fusion_svm":
                        fresh_pred, _ = pipeline_predict(
                            model, method, fresh["fp32"], statistics[:n]
                        )
                        check["fresh_prediction_agreement"] = parity(old.iloc[:n], fresh_pred)[
                            "prediction_agreement"
                        ]
                        if (
                            check["fresh_prediction_agreement"]
                            < limits["fresh_fusion_svm_min_prediction_agreement"]
                        ):
                            raise ValueError("New GPU FP32 fusion SVM probe mismatch.")
                    checks["methods"][method] = check
                del model
            print(
                f"seed{seed}: all seven validation-only technical checks passed; test still unencoded",
                flush=True,
            )
            del bundle, old_amp, old_fp32, fresh, statistics
            gc.collect()
            torch.cuda.empty_cache()
        report = {
            "status": "passed",
            "protocol_sha256": f["protocol_sha256"],
            "completed_at": utc_now(),
            "all_seeds_passed_before_test": True,
            "test_encoded": False,
            "checks": results,
        }
        atomic_write_json(output, report)
        return report
    except BaseException as exc:
        atomic_write_json(
            output,
            {
                "status": "failed",
                "protocol_sha256": f["protocol_sha256"],
                "checks": results,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "test_encoded": False,
            },
        )
        raise


def load_test_cache(p, f, seed):
    root = seed_directory(p, seed)
    m = read_json(root / "test_features.json")
    if m["protocol_sha256"] != f["protocol_sha256"] or m["sha256"] != sha256_file(
        root / "test_features.npz"
    ):
        raise ValueError("Foreign/corrupt test feature cache.")
    with np.load(root / "test_features.npz", allow_pickle=False) as a:
        values = {k: a[k].copy() for k in a.files}
    n = p["test_count"]
    if (
        values["amp"].shape != (n, 1024)
        or values["fp32"].shape != (n, 1024)
        or values["statistics"].shape != (n, 15)
        or values["labels"].shape != (n,)
        or values["sample_ids"].shape != (n,)
        or len(np.unique(values["sample_ids"])) != n
        or not all(np.isfinite(values[k]).all() for k in ("amp", "fp32", "statistics"))
    ):
        raise ValueError("Wrong test feature shape/values/IDs.")
    return values, m


def make_test_cache(p, f, seed):
    # This guard is inside the encoder path too, not only the CLI coordinator.
    report = read_json(directory(p) / "preflight.json")
    if (
        report["status"] != "passed"
        or not report["all_seeds_passed_before_test"]
        or report["protocol_sha256"] != f["protocol_sha256"]
    ):
        raise ValueError("All four preflight seeds must pass before test encoding.")
    root = seed_directory(p, seed)
    root.mkdir(exist_ok=True)
    if (root / "test_features.json").exists():
        return load_test_cache(p, f, seed)
    config, bundle = bundle_for(p, seed)
    began = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    values = encode(config, bundle.x_test, bundle.mask_test, seed)
    values.update(
        statistics=extract_statistical_features(bundle.x_test, bundle.mask_test),
        labels=bundle.y_test,
        sample_ids=bundle.ids_test.astype(str),
    )
    temp = root / "test_features.npz.tmp"
    with temp.open("wb") as handle:
        np.savez(handle, **values)
    temp.replace(root / "test_features.npz")
    atomic_write_json(
        root / "test_features.json",
        {
            "protocol_sha256": f["protocol_sha256"],
            "seed": seed,
            "sha256": sha256_file(root / "test_features.npz"),
            "test_ids_sha256": train_subset_sha256(bundle.ids_test),
            "dataset_sha256": bundle.dataset_sha256,
            "preflight_sha256": sha256_file(directory(p) / "preflight.json"),
            "extraction_seconds": time.perf_counter() - began,
            "peak_allocated_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "environment": f["environment"],
            "precisions": ["amp_fp16", "fp32"],
            "classes": p["classes"],
        },
    )
    return load_test_cache(p, f, seed)


def result_metrics(truth, predicted, probs, classes):
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    if (
        truth.shape != predicted.shape
        or truth.ndim != 1
        or not np.isin(predicted, [0, 1, 2]).all()
        or not np.isin(truth, [0, 1, 2]).all()
    ):
        raise ValueError("Invalid labels/predictions.")
    result = classification_metrics(truth, predicted, classes, include_details=True)
    cm = np.asarray(result["confusion_matrix"])
    result["battery_argmax_recall"] = float(cm[2, 2] / cm[2].sum())
    result["battery_argmax_fpr"] = float(cm[:2, 2].sum() / cm[:2].sum())
    result["battery_average_precision"] = None
    if probs is not None:
        validate_probabilities(probs, len(truth))
        if not np.array_equal(probs.argmax(axis=1), predicted):
            raise ValueError("Probability argmax differs from prediction.")
        result["battery_average_precision"] = float(
            average_precision_score(truth == 2, probs[:, 2])
        )
    return result


def assert_metrics_equal(expected, actual):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError("Metric keys changed.")
        for key in expected:
            assert_metrics_equal(expected[key], actual[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError("Metric dimensions changed.")
        for left, right in zip(expected, actual, strict=True):
            assert_metrics_equal(left, right)
    elif isinstance(expected, (int, float)):
        if not np.isfinite(actual) or not np.isclose(expected, actual, rtol=0, atol=1e-12):
            raise ValueError("Recomputed test metric mismatch.")
    elif expected != actual:
        raise ValueError("Metric value changed.")


def audit_condition(p, f, seed, method, values):
    root = seed_directory(p, seed) / method
    m = read_json(root / "metrics.json")
    csv = root / "test_predictions.csv"
    if (
        m["protocol_sha256"] != f["protocol_sha256"]
        or m["source_model_sha256"] != f["selected_models"][str(seed)][method]["model_sha256"]
        or m["selected_best_epoch"] != f["selected_models"][str(seed)][method]["best_epoch"]
        or m["predictions_sha256"] != sha256_file(csv)
        or m["features_sha256"] != sha256_file(seed_directory(p, seed) / "test_features.npz")
    ):
        raise ValueError("Saved test result provenance/hash mismatch.")
    rows = pd.read_csv(csv, dtype={"sample_id": str})
    np.testing.assert_array_equal(rows.sample_id, values["sample_ids"])
    np.testing.assert_array_equal(rows.true_index, values["labels"])
    probs = (
        rows[[f"probability_{k}" for k in range(3)]].to_numpy() if "probability_0" in rows else None
    )
    metrics = result_metrics(
        rows.true_index.to_numpy(), rows.predicted_index.to_numpy(), probs, p["classes"]
    )
    assert_metrics_equal(metrics, m["test_metrics"])
    return m


def run(path):
    f = verify(path)
    p = f["protocol"]
    root = directory(p)
    hardware(p)
    # Preflight never sees test embeddings/outputs and must finish for every seed.
    atomic_write_json(
        root / "suite_status.json",
        {
            "status": "running",
            "stage": "validation_only_preflight",
            "updated_at": utc_now(),
            "test_encoded": False,
            "training_performed": False,
        },
    )
    preflight(path)
    if not (root / "test_started.json").exists():
        atomic_write_json(
            root / "test_started.json",
            {
                "started_at": utc_now(),
                "protocol_sha256": f["protocol_sha256"],
                "preflight_sha256": sha256_file(root / "preflight.json"),
            },
        )
    for seed in p["seeds"]:
        atomic_write_json(
            root / "suite_status.json",
            {
                "status": "running",
                "stage": "test_inference",
                "seed": seed,
                "updated_at": utc_now(),
                "training_performed": False,
            },
        )
        values, fm = make_test_cache(p, f, seed)
        for method in p["methods"]:
            out = seed_directory(p, seed) / method
            out.mkdir(exist_ok=True)
            if (out / "metrics.json").exists():
                audit_condition(p, f, seed, method, values)
                print(f"Reused verified test prediction: {seed}/{method}", flush=True)
                continue
            began = time.perf_counter()
            rec = f["selected_models"][str(seed)][method]
            if method.endswith("_nn"):
                model = load_nn(p, f, seed, method)
                predicted, probs = nn_predict(model, values["amp"], values["statistics"])
            else:
                model = joblib.load(rec["model"])
                predicted, probs = pipeline_predict(
                    model, method, values["fp32"], values["statistics"]
                )
            seconds = time.perf_counter() - began
            del model
            metrics = result_metrics(values["labels"], predicted, probs, p["classes"])
            # All outputs in a new directory; parent validation files stay immutable.
            temp = out / "test_predictions.csv.tmp"
            parent.save_predictions(temp, values["sample_ids"], values["labels"], predicted, probs)
            temp.replace(out / "test_predictions.csv")
            atomic_write_json(
                out / "metrics.json",
                {
                    "seed": seed,
                    "method": method,
                    "protocol_sha256": f["protocol_sha256"],
                    "source_model_sha256": rec["model_sha256"],
                    "selected_best_epoch": rec["best_epoch"],
                    "features_sha256": fm["sha256"],
                    "predictions_sha256": sha256_file(out / "test_predictions.csv"),
                    "test_metrics": metrics,
                    "inference_seconds_including_model_load": seconds,
                    "test_evaluated": True,
                    "training_performed": False,
                },
            )
            audit_condition(p, f, seed, method, values)
            print(f"Completed fixed test condition {seed}/{method}", flush=True)
        del values
        gc.collect()
        torch.cuda.empty_cache()
    summarize(path)
    atomic_write_json(
        root / "suite_status.json",
        {
            "status": "completed",
            "updated_at": utc_now(),
            "test_evaluated": True,
            "training_performed": False,
            "conditions": 28,
        },
    )


def descriptive(values):
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError("All four fixed seeds are mandatory.")
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    half = 3.182446305284263 * sd / 2
    return {"mean": mean, "sample_sd": sd, "descriptive_ci95": [mean - half, mean + half]}


def decision(differences, threshold):
    result = descriptive(differences)
    return {
        **result,
        "strictly_positive_seeds": sum(v > 0 for v in differences),
        "minimum_mean_difference_pp": threshold,
        "passed": bool(
            result["mean"] >= threshold - 1e-12 and sum(v > 0 for v in differences) >= 3
        ),
    }


def summarize(path):
    f = verify(path)
    p = f["protocol"]
    root = directory(p)
    rows = []
    cfg = load_config(p["training_config"])
    raw_labels = pd.read_csv(
        cfg.data.csv_path,
        delimiter=cfg.data.delimiter,
        usecols=[cfg.data.id_column, cfg.data.label_column],
    ).astype(str)
    lookup = dict(
        zip(raw_labels[cfg.data.id_column], raw_labels[cfg.data.label_column], strict=True)
    )
    per_seed = {}
    costs = []
    output_hashes = {}
    for seed in p["seeds"]:
        values, fm = load_test_cache(p, f, seed)
        costs.append(fm["extraction_seconds"])
        split = read_json(parent.fusion_run(seed) / "split_manifest.json")
        np.testing.assert_array_equal(values["sample_ids"], split["splits"]["test"])
        expected_labels = [p["classes"].index(lookup[k]) for k in values["sample_ids"]]
        np.testing.assert_array_equal(values["labels"], expected_labels)
        if fm["preflight_sha256"] != sha256_file(root / "preflight.json"):
            raise ValueError("Pre-test technical barrier changed.")
        per_seed[str(seed)] = {}
        for method in p["methods"]:
            m = audit_condition(p, f, seed, method, values)
            t = m["test_metrics"]
            per_seed[str(seed)][method] = t["macro_f1"] * 100
            row = {
                "seed": seed,
                "method": method,
                "macro_f1_percent": t["macro_f1"] * 100,
                "accuracy_percent": t["accuracy"] * 100,
                "best_epoch": m["selected_best_epoch"],
                "battery_recall_percent": t["battery_argmax_recall"] * 100,
                "battery_fpr_percent": t["battery_argmax_fpr"] * 100,
                "battery_ap_percent": None
                if t["battery_average_precision"] is None
                else t["battery_average_precision"] * 100,
                "inference_seconds_including_model_load": m[
                    "inference_seconds_including_model_load"
                ],
            }
            row.update(
                {
                    f"class_{k}_f1_percent": t["classification_report"][str(k)]["f1-score"] * 100
                    for k in range(3)
                }
            )
            rows.append(row)
            out = seed_directory(p, seed) / method
            for name in ("metrics.json", "test_predictions.csv"):
                output_hashes[str(out / name)] = sha256_file(out / name)
        for name in ("test_features.npz", "test_features.json"):
            file = seed_directory(p, seed) / name
            output_hashes[str(file)] = sha256_file(file)
    comparisons = {
        method: {
            "per_seed_difference_pp": [
                per_seed[str(s)]["fusion_nn"] - per_seed[str(s)][method] for s in p["seeds"]
            ]
        }
        for method in p["methods"]
        if method != "fusion_nn"
    }
    for data in comparisons.values():
        data.update(descriptive(data["per_seed_difference_pp"]))
    primary = decision(comparisons["fusion_svm"]["per_seed_difference_pp"], 0.5)
    ablation = decision(comparisons["moment_nn"]["per_seed_difference_pp"], 0.25)
    result = {
        "audit_status": "passed",
        "test_evaluated": True,
        "protocol_sha256": f["protocol_sha256"],
        "conditions": 28,
        "audited_prediction_rows": 28 * p["test_count"],
        "pilot_seed_included": False,
        "per_seed_f1_percent": per_seed,
        "per_run": rows,
        "aggregates": {
            method: {
                key: descriptive([r[key] for r in rows if r["method"] == method])
                for key in (
                    "macro_f1_percent",
                    "accuracy_percent",
                    "battery_recall_percent",
                    "battery_fpr_percent",
                )
            }
            for method in p["methods"]
        },
        "fusion_minus_fixed_comparators": comparisons,
        "primary_rule": primary,
        "moment_ablation_rule": ablation,
        "test_confirmation_passed": primary["passed"] and ablation["passed"],
        "extraction_seconds_both_precisions_total": sum(costs),
        "output_sha256": output_hashes,
        "completed_at": utc_now(),
        "training_performed": False,
        "boundary": "One frozen evaluation of preselected existing models. Overlapping same-dataset splits; historical tests already known; no external/bitwise cross-GPU guarantee or full-label TCN ranking.",
    }
    atomic_write_json(root / "summary.json", result)
    pd.DataFrame(rows).to_csv(root / "per_run_metrics.csv", index=False)
    lines = [
        "# E03T fixed-model formal test",
        "",
        "40% labels; seeds42/44/45/46; no fitting or test-driven selection.",
        "",
        "| Method | Macro-F1 % mean±SD | Accuracy % mean±SD |",
        "|---|---:|---:|",
    ]
    for method, data in result["aggregates"].items():
        a = data["macro_f1_percent"]
        b = data["accuracy_percent"]
        lines.append(
            f"| {method} | {a['mean']:.4f} ± {a['sample_sd']:.4f} | {b['mean']:.4f} ± {b['sample_sd']:.4f} |"
        )
    lines += [
        "",
        f"Fixed primary gate: {primary}",
        f"MOMENT ablation gate: {ablation}",
        f"Both passed: {result['test_confirmation_passed']}",
        "",
        result["boundary"],
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(
        "All 28 test predictions audited; test confirmation:",
        result["test_confirmation_passed"],
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "preflight", "run", "summarize"))
    parser.add_argument("--protocol", type=Path, default=Path(DEFAULT_PROTOCOL))
    args = parser.parse_args()
    p = read_protocol(args.protocol)
    root = directory(p)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            globals()[args.action](args.protocol)
        except BaseException as exc:
            if args.action in {"run", "preflight"}:
                atomic_write_json(
                    root / "suite_status.json",
                    {
                        "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        "action": args.action,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "updated_at": utc_now(),
                    },
                )
            raise


if __name__ == "__main__":
    main()
