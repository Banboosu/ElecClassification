"""E02P exploratory pooling pilot: frozen encoder, train/CV/validation only."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import (
    load_dataset,
    save_label_encoder,
    stratified_train_subset_indices,
    train_subset_sha256,
)
from tcn_moment.experiment import collect_environment, utc_now
from tcn_moment.fast_svm import FastRBFGridSearchCV
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.metrics import classification_metrics
from tcn_moment.pooling import (
    forward_poolings,
    length_groups,
    representation,
    training_length_edges,
)
from tcn_moment.replicate_gain_heads import normalize_masked_rows, save_predictions
from tcn_moment.run_e03d_stable import plan_directory as head_directory
from tcn_moment.run_e03d_stable import verify as verify_head
from tcn_moment.train_baselines import extract_statistical_features
from tcn_moment.train_e03 import build_backbone
from tcn_moment.train_feature_fusion import build_search, verified_checkpoint
from tcn_moment.train_moment_svm import _atomic_joblib_dump

DEFAULT_PROTOCOL = Path("configs/experiments/improvements/e02_pooling_pilot_protocol.json")
PACKAGES = ("torch", "momentfm", "transformers", "numpy", "scikit-learn", "scipy")


def read(path):
    return json.loads(Path(path).read_text())


def protocol(path):
    p = read(path)
    if (
        p["experiment_id"] != "E02P"
        or p["seed"] != 43
        or p["train_fraction"] != 0.4
        or p["poolings"] != ["mean", "mean_max", "thirds"]
        or p["branches"] != ["moment", "fusion"]
        or p["dimensions"] != {"mean": 1024, "mean_max": 2048, "thirds": 3072}
        or any(
            p[k]
            for k in (
                "test_evaluated",
                "encoder_supervised_training",
                "automatic_lora",
                "automatic_test",
                "automatic_shutdown",
            )
        )
    ):
        raise ValueError("Unexpected validation-only pooling scope.")
    s = p["search"]
    if (
        s["C"] != [0.1, 1.0, 10.0, 100.0, 1000.0]
        or s["cv_folds"] != 5
        or s["cv_shuffle"]
        or s["gamma_multipliers"] != ["scale", 0.1, 10.0]
        or s["scoring"] != "macro_f1"
        or s["kernel_precision"] != "float64"
        or p["precision"] != "fp32"
        or p["extraction_batch_size"] != 64
    ):
        raise ValueError("Search/extraction budget changed.")
    r = p["pilot_rules"]
    if (
        r["automatic_replication"]
        or r["minimum_over_reproduced_mean_fusion_pp"] != 0.5
        or r["minimum_over_strongest_prior_nonadapted_pipeline_pp"] != 0.25
        or r["eligible_poolings"] != ["mean_max", "thirds"]
        or r["tie_order"] != ["mean_max", "thirds"]
    ):
        raise ValueError("Pilot gate changed.")
    return p


def root_directory(p):
    return Path(p["output_dir"]) / f"{p['suite']}_analysis"


def cases(p):
    return [(pool, branch) for pool in p["poolings"] for branch in p["branches"]]


def case_directory(p, pool, branch):
    return root_directory(p) / f"{pool}_{branch}"


def verify(path):
    p = protocol(path)
    f = read(root_directory(p) / "frozen_protocol.json")
    if f["protocol"] != p or f["protocol_sha256"] != sha256_file(path):
        raise ValueError("Pooling protocol changed after freeze.")
    for file, want in f["file_sha256"].items():
        if sha256_file(Path(file)) != want:
            raise ValueError(f"Pooling input changed: {file}")
    if f["packages"] != {k: importlib.metadata.version(k) for k in PACKAGES}:
        raise ValueError("Pooling dependencies changed.")
    return f


def freeze(path):
    p = protocol(path)
    root = root_directory(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        return verify(path)
    if any(case_directory(p, *case).exists() for case in cases(p)) or (root / "prepared").exists():
        raise ValueError("Freeze before new features or classifier outcomes.")
    parent = verify_head(Path(p["parent_head_protocol"]))
    hd = head_directory(parent["protocol"])
    hs = read(hd / "summary.json")
    if hs["audit_status"] != "passed" or hs["test_evaluated"]:
        raise ValueError("Incomplete prior head audit.")
    gain = max(
        c["best_macro_f1_percent"] / 100
        for c in hs["cells"]
        if c["branch"] == "fusion" and c["mode"] == "fanin_gain" and c["budget"] == 200
    )
    tcn = read(Path(p["historical_tcn_metadata"]) / "validation_reference.json")
    reference_metadata = [
        read(Path(p[key]) / "metrics.json")
        for key in (
            "mean_moment_reference",
            "mean_fusion_reference",
            "matched_statistics_reference",
        )
    ]
    if any(
        m["test_evaluated"] or m["train_ids_sha256"] != tcn["train_ids_sha256"]
        for m in reference_metadata
    ):
        raise ValueError("Reference training subset mismatch.")
    files = {Path(k) for k in parent["file_sha256"]}
    files.update(Path("src/tcn_moment").glob("*.py"))
    files.update(
        [
            path,
            Path("scripts/run_e02_pooling_remote.sh"),
            Path("tests/test_pooling_pilot.py"),
            Path("docs/experiment_records/e02_pooling_pilot_protocol_20261019.md"),
            hd / "summary.json",
            hd / "frozen_protocol.json",
            Path("pyproject.toml"),
            Path("uv.lock"),
        ]
    )
    for key in ("mean_moment_reference", "mean_fusion_reference"):
        files.update(
            Path(p[key]) / n
            for n in ("metrics.json", "validation_predictions.csv", "cv_results.csv")
        )
    files.update(
        Path(p["matched_statistics_reference"]) / n
        for n in ("metrics.json", "validation_predictions.csv")
    )
    files.update(
        Path(p["split_reference"]) / n for n in ("split_manifest.json", "train_ids_40.npy")
    )
    files.update(
        Path(p["fp32_parent_cache"]) / n for n in ("manifest.json", "train.npz", "validation.npz")
    )
    files.update(
        Path(p["historical_tcn_metadata"]) / n
        for n in ("validation_reference.json", "validation_predictions.csv")
    )
    config = load_config(p["training_config"])
    weight = verified_checkpoint(config)
    from huggingface_hub import try_to_load_from_cache

    wp = Path(try_to_load_from_cache(weight["model_id"], weight["filename"]))
    files.update([wp, wp.parent / "config.json"])
    reference = {
        "prior_gain_fusion_nn": gain,
        "mean_fusion_svm": read(Path(p["mean_fusion_reference"]) / "metrics.json")[
            "validation_metrics"
        ]["macro_f1"],
        "matched_statistics_svm": read(Path(p["matched_statistics_reference"]) / "metrics.json")[
            "validation_metrics"
        ]["macro_f1"],
        "historical_tcn": tcn["validation_macro_f1"],
    }
    f = {
        "frozen_at": utc_now(),
        "protocol": p,
        "protocol_sha256": sha256_file(path),
        "packages": {k: importlib.metadata.version(k) for k in PACKAGES},
        "file_sha256": {str(k): sha256_file(k) for k in sorted(files)},
        "reference_validation_f1": reference,
        "strongest_prior_nonadapted_pipeline_f1": max(reference.values()),
        "checkpoint": weight,
        "environment": collect_environment(torch),
        "new_pooling_outcomes_observed": False,
        "historical_test_outcomes_already_known": True,
    }
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
    print(f"Frozen E02P: {len(files)} inputs, six train-CV fits; test disabled.", flush=True)
    return f


def extraction_inputs(bundle, indices):
    return {
        "train": (
            bundle.x_train[indices],
            bundle.mask_train[indices],
            bundle.y_train[indices],
            bundle.ids_train[indices],
            bundle.lengths_train[indices],
        ),
        "validation": (
            bundle.x_val,
            bundle.mask_val,
            bundle.y_val,
            bundle.ids_val,
            bundle.lengths_val,
        ),
    }


def synthetic_padding_diagnostics(model, p):
    lengths = p["padding_diagnostics"]["synthetic_valid_lengths"]
    values = {}
    for width in p["padding_diagnostics"]["padding_lengths"]:
        raw = np.zeros((len(lengths), width), dtype=np.float32)
        mask = np.zeros_like(raw)
        for index, n in enumerate(lengths):
            mask[index, :n] = 1
            raw[index, :n] = 5 if index == 0 else np.sin(np.arange(n, dtype=np.float32) / 7)
        x = normalize_masked_rows(raw, mask)
        with torch.inference_mode():
            pooled = forward_poolings(
                model, torch.tensor(x[:, None], device="cuda"), torch.tensor(mask, device="cuda")
            )
        values[str(width)] = {k: v.cpu().numpy() for k, v in pooled.items()}
    width = max(p["padding_diagnostics"]["padding_lengths"])
    junk = x.copy()
    junk[mask == 0] = p["padding_diagnostics"]["invalid_padding_junk_value"]
    with torch.inference_mode():
        pooled = forward_poolings(
            model, torch.tensor(junk[:, None], device="cuda"), torch.tensor(mask, device="cuda")
        )
    values["junk_padding"] = {k: v.cpu().numpy() for k, v in pooled.items()}
    tolerance = p["padding_diagnostics"]["descriptive_invariance_tolerance"]
    reference = values[str(width)]
    comparisons = {
        name: {k: float(np.max(np.abs(v[k] - reference[k]))) for k in reference}
        for name, v in values.items()
        if name != str(width)
    }
    return {
        "valid_lengths": lengths,
        "max_abs_feature_differences": comparisons,
        "all_within_descriptive_tolerance": all(
            error <= tolerance for row in comparisons.values() for error in row.values()
        ),
        "tolerance": tolerance,
        "constant_sequence_finite": True,
        "incomplete_tail_note": "Excluded from pooling, but external zscore/internal RevIN can still depend on valid incomplete-tail raw observations; no whole-pipeline tail-invariance claim.",
    }


def load_prepared(p, f):
    root = root_directory(p) / "prepared"
    m = read(root / "manifest.json")
    if m["protocol_sha256"] != f["protocol_sha256"]:
        raise ValueError("Foreign pooling cache.")
    data = {}
    for split in ("train", "validation"):
        file = root / f"{split}.npz"
        if sha256_file(file) != m["splits"][split]["sha256"]:
            raise ValueError("Corrupt pooling cache.")
        with np.load(file, allow_pickle=False) as a:
            data[split] = {k: a[k].copy() for k in a.files}
        n = 9827 if split == "train" else 3510
        for key, dim in (("mean", 1024), ("max", 1024), ("thirds", 3072), ("statistics", 15)):
            if data[split][key].shape != (n, dim) or not np.isfinite(data[split][key]).all():
                raise ValueError("Wrong cached shape/values.")
        if len(np.unique(data[split]["sample_ids"])) != n:
            raise ValueError("Invalid cached IDs.")
    return data, m


def prepare(p, f):
    root = root_directory(p) / "prepared"
    root.mkdir(exist_ok=True)
    if (root / "manifest.json").exists():
        return load_prepared(p, f)
    config = with_random_seed(load_config(p["training_config"]), p["seed"])
    bundle = load_dataset(replace(config.data, normalize="none"))
    source = Path(p["split_reference"])
    if sha256_file(bundle.split_path) != sha256_file(source / "split_manifest.json"):
        raise ValueError("Wrong pooling data split.")
    indices = stratified_train_subset_indices(bundle.y_train, bundle.ids_train, 0.4, p["seed"])
    np.testing.assert_array_equal(bundle.ids_train[indices], np.load(source / "train_ids_40.npy"))
    if len(indices) != 9827 or len(bundle.y_val) != 3510:
        raise ValueError("Wrong sample budget.")
    inputs = extraction_inputs(bundle, indices)
    old_manifest = read(Path(p["fp32_parent_cache"]) / "manifest.json")
    cm = (
        read(root / "extraction_progress.json")
        if (root / "extraction_progress.json").exists()
        else {"protocol_sha256": f["protocol_sha256"], "splits": {}}
    )
    if cm["protocol_sha256"] != f["protocol_sha256"]:
        raise ValueError("Partial cache belongs to another protocol.")
    pp = read("configs/experiments/improvements/e03_peft_pilot_protocol.json")
    pp["seeds"] = [p["seed"]]
    model = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for split, (raw, mask, y, ids, lengths) in inputs.items():
            file = root / f"{split}.npz"
            if split not in cm["splits"]:
                if model is None:
                    model = build_backbone(config, 0, pp).cuda().eval()
                x = normalize_masked_rows(raw, mask)
                chunks = {"mean": [], "max": [], "thirds": []}
                with torch.inference_mode():
                    for start in range(0, len(y), 64):
                        xx = torch.as_tensor(
                            x[start : start + 64, None], device="cuda", dtype=torch.float32
                        )
                        mm = torch.as_tensor(
                            mask[start : start + 64], device="cuda", dtype=torch.float32
                        )
                        out = forward_poolings(model, xx, mm)
                        for key, values in chunks.items():
                            values.append(out[key].float().cpu().numpy())
                        if (start // 64) % 30 == 0:
                            print(
                                f"Pooling {split}: {min(start + 64, len(y))}/{len(y)}", flush=True
                            )
                arrays = {key: np.concatenate(v) for key, v in chunks.items()}
                arrays.update(
                    statistics=extract_statistical_features(raw, mask),
                    labels=y,
                    sample_ids=ids.astype(str),
                    lengths=np.minimum(lengths, config.data.max_length),
                )
                temp = file.with_suffix(".npz.tmp")
                with temp.open("wb") as handle:
                    np.savez(handle, **arrays)
                temp.replace(file)
                cm["splits"][split] = {"sha256": sha256_file(file)}
                atomic_write_json(root / "extraction_progress.json", cm)
            if sha256_file(file) != cm["splits"][split]["sha256"]:
                raise ValueError("Corrupt partial features.")
            old_path = Path(p["fp32_parent_cache"]) / f"{split}.npz"
            if sha256_file(old_path) != old_manifest["splits"][split]["sha256"]:
                raise ValueError("Old reference cache corrupt.")
            with (
                np.load(old_path, allow_pickle=False) as old,
                np.load(file, allow_pickle=False) as new,
            ):
                old_ids = bundle.ids_train if split == "train" else bundle.ids_val
                np.testing.assert_array_equal(old["sample_ids"], old_ids.astype(str))
                np.testing.assert_array_equal(new["sample_ids"], ids.astype(str))
                expected = old["features"][indices] if split == "train" else old["features"]
                error = float(np.max(np.abs(expected - new["mean"])))
                if error > p["technical_gates"]["reference_feature_max_abs_error"]:
                    raise ValueError(f"Mean extraction incompatible: {error}")
                cm["splits"][split]["reference_mean_max_abs_error"] = error
        if model is None:
            model = build_backbone(config, 0, pp).cuda().eval()
        diag = synthetic_padding_diagnostics(model, p)
        atomic_write_json(root / "padding_diagnostics.json", diag)
    finally:
        del model
        torch.cuda.empty_cache()
    save_label_encoder(bundle.label_encoder, root)
    atomic_write_json(
        root / "manifest.json",
        {
            "protocol_sha256": f["protocol_sha256"],
            "splits": cm["splits"],
            "train_ids_sha256": train_subset_sha256(bundle.ids_train[indices]),
            "classes": bundle.label_encoder.classes_.tolist(),
            "length_edges": training_length_edges(
                np.minimum(bundle.lengths_train[indices], config.data.max_length)
            ),
            "length_threshold_source": "selected train only",
            "test_encoded": False,
            "encoder_trained": False,
            "extraction_seconds_including_loading_statistics_and_probes": time.perf_counter()
            - started,
            "peak_allocated_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "padding_diagnostics_sha256": sha256_file(root / "padding_diagnostics.json"),
        },
    )
    return load_prepared(p, f)


def stratified_metrics(truth, predicted, lengths, edges, classes):
    groups = length_groups(lengths, edges)
    result = []
    for group in range(3):
        take = groups == group
        y = np.asarray(truth)[take]
        pred = np.asarray(predicted)[take]
        result.append(
            {
                "bin": group,
                "n": int(take.sum()),
                "class_counts": [int((y == i).sum()) for i in range(3)],
                "metrics": classification_metrics(y, pred, classes, include_details=True)
                if len(y)
                else None,
            }
        )
    return result


def fit_case(p, f, arrays, dm, pooling, branch):
    root = case_directory(p, pooling, branch)
    root.mkdir(exist_ok=True)
    if (root / "status.json").exists() and read(root / "status.json")["status"] == "completed":
        print(f"Preserved {pooling}/{branch}", flush=True)
        return
    config = with_random_seed(load_config(p["training_config"]), p["seed"])
    name = "moment_tuned_svm" if branch == "moment" else "fusion_stats_svm"
    train = representation(arrays["train"], pooling)
    val = representation(arrays["validation"], pooling)
    dimension = train.shape[1]
    if dimension != p["dimensions"][pooling]:
        raise ValueError("Pooled dimension mismatch.")
    if branch == "fusion":
        train = np.concatenate([train, arrays["train"]["statistics"]], axis=1)
        val = np.concatenate([val, arrays["validation"]["statistics"]], axis=1)
    atomic_write_json(
        root / "status.json",
        {"status": "running", "pooling": pooling, "branch": branch, "test_evaluated": False},
    )
    base = build_search(
        name, train.shape[1], dimension, config, n_jobs=p["search"]["n_jobs"], seed=p["seed"]
    )
    search = FastRBFGridSearchCV(**base.get_params(deep=False))
    search.kernel_device = p["search"]["kernel_device"]
    (root_directory(p) / "kernel_tmp").mkdir(exist_ok=True)
    search.temporary_root = str(root_directory(p) / "kernel_tmp")
    print(
        f"Fitting {pooling}/{branch}: {dimension} pooled dimensions, 75 train-CV scores", flush=True
    )
    started = time.perf_counter()
    search.fit(train, arrays["train"]["labels"])
    predicted = search.predict(val)
    metric = classification_metrics(
        arrays["validation"]["labels"], predicted, dm["classes"], include_details=True
    )
    cv = pd.DataFrame(search.cv_results_)
    compatibility = None
    if pooling == "mean":
        reference = Path(p[f"mean_{branch}_reference"])
        old = read(reference / "metrics.json")
        old_cv = pd.read_csv(reference / "cv_results.csv")
        cols = [f"split{i}_test_score" for i in range(5)]
        cv_error = float(np.max(np.abs(cv[cols].to_numpy() - old_cv[cols].to_numpy())))
        old_predictions = pd.read_csv(
            reference / "validation_predictions.csv", dtype={"sample_id": str}
        )
        np.testing.assert_array_equal(old_predictions.sample_id, arrays["validation"]["sample_ids"])
        if (
            search.best_params_ != old["best_params"]
            or not np.array_equal(predicted, old_predictions.predicted_index)
            or cv_error > p["technical_gates"]["mean_cv_fold_scores_max_abs_error"]
        ):
            raise ValueError("Mean search/predictions did not reproduce the fixed prior baseline.")
        compatibility = {
            "best_params_equal": True,
            "validation_predictions_exact": True,
            "max_cv_fold_score_error": cv_error,
        }
    _atomic_joblib_dump(search.best_estimator_, root / "classifier.joblib")
    save_predictions(
        root / "validation_predictions.csv",
        arrays["validation"]["sample_ids"],
        arrays["validation"]["labels"],
        predicted,
    )
    cv.to_csv(root / "cv_results.csv", index=False)
    atomic_write_json(
        root / "metrics.json",
        {
            "protocol_sha256": f["protocol_sha256"],
            "pooling": pooling,
            "branch": branch,
            "pooled_dimension": dimension,
            "train_ids_sha256": dm["train_ids_sha256"],
            "best_params": search.best_params_,
            "cv_score": float(search.best_score_),
            "validation_metrics": metric,
            "length_strata": stratified_metrics(
                arrays["validation"]["labels"],
                predicted,
                arrays["validation"]["lengths"],
                dm["length_edges"],
                dm["classes"],
            ),
            "training_seconds": time.perf_counter() - started,
            "backend": search.backend_timings_,
            "baseline_compatibility": compatibility,
            "classifier_sha256": sha256_file(root / "classifier.joblib"),
            "predictions_sha256": sha256_file(root / "validation_predictions.csv"),
            "cv_results_sha256": sha256_file(root / "cv_results.csv"),
            "test_evaluated": False,
        },
    )
    atomic_write_json(root / "status.json", {"status": "completed", "test_evaluated": False})


def candidate_rule(p, scores, strongest):
    eligible = p["pilot_rules"]["tie_order"]
    chosen = max(eligible, key=lambda pool: scores[(pool, "fusion")])
    score = scores[(chosen, "fusion")]
    over_mean = (score - scores[("mean", "fusion")]) * 100
    over_strong = (score - strongest) * 100
    return {
        "selected_pooling": chosen,
        "selected_validation_f1": score,
        "over_mean_fusion_pp": over_mean,
        "over_strongest_prior_pp": over_strong,
        "eligible_for_separately_frozen_replication": bool(
            over_mean >= 0.5 - 1e-12 and over_strong >= 0.25 - 1e-12
        ),
        "automatic_replication_started": False,
    }


def summarize(path):
    f = verify(path)
    p = f["protocol"]
    arrays, dm = load_prepared(p, f)
    root = root_directory(p)
    rows = []
    scores = {}
    for pool, branch in cases(p):
        case = case_directory(p, pool, branch)
        m = read(case / "metrics.json")
        if (
            read(case / "status.json")["status"] != "completed"
            or m["protocol_sha256"] != f["protocol_sha256"]
            or m["test_evaluated"]
            or m["train_ids_sha256"] != dm["train_ids_sha256"]
        ):
            raise ValueError("Incomplete/foreign pooling result.")
        for name, key in (
            ("classifier.joblib", "classifier_sha256"),
            ("validation_predictions.csv", "predictions_sha256"),
            ("cv_results.csv", "cv_results_sha256"),
        ):
            if sha256_file(case / name) != m[key]:
                raise ValueError("Pooling artifact corruption.")
        predictions = pd.read_csv(case / "validation_predictions.csv", dtype={"sample_id": str})
        np.testing.assert_array_equal(predictions.sample_id, arrays["validation"]["sample_ids"])
        np.testing.assert_array_equal(predictions.true_index, arrays["validation"]["labels"])
        cm = [[0] * 3 for _ in range(3)]
        for row in predictions.itertuples():
            if not 0 <= int(row.predicted_index) < 3:
                raise ValueError("Invalid predicted class.")
            cm[int(row.true_index)][int(row.predicted_index)] += 1
        if cm != m["validation_metrics"]["confusion_matrix"]:
            raise ValueError("Confusion mismatch.")
        for key, value in metrics_from_confusion(cm).items():
            if abs(value - m["validation_metrics"][key]) > 1e-12:
                raise ValueError("Independent metric mismatch.")
        expected = stratified_metrics(
            arrays["validation"]["labels"],
            predictions.predicted_index,
            arrays["validation"]["lengths"],
            dm["length_edges"],
            dm["classes"],
        )
        if expected != m["length_strata"]:
            raise ValueError("Length bins or metrics changed.")
        scores[(pool, branch)] = m["validation_metrics"]["macro_f1"]
        rows.append(
            {
                "pooling": pool,
                "branch": branch,
                "macro_f1_percent": scores[(pool, branch)] * 100,
                "accuracy_percent": m["validation_metrics"]["accuracy"] * 100,
                "training_seconds": m["training_seconds"],
                "best_params": m["best_params"],
                "length_strata": expected,
            }
        )
    result = {
        "audit_status": "passed",
        "recorded_at": utc_now(),
        "test_evaluated": False,
        "conditions": 6,
        "audited_prediction_rows": 6 * 3510,
        "reference_validation_f1": f["reference_validation_f1"],
        "per_condition": rows,
        "length_edges": dm["length_edges"],
        "padding_diagnostics": read(root / "prepared/padding_diagnostics.json"),
        "pilot_decision": candidate_rule(p, scores, f["strongest_prior_nonadapted_pipeline_f1"]),
        "limitations": p["limitations"],
    }
    atomic_write_json(root / "summary.json", result)
    lines = [
        "# E02P validation-only pooling pilot",
        "",
        "| Pool | Branch | Macro-F1 % | Accuracy % |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f"| {r['pooling']} | {r['branch']} | {r['macro_f1_percent']:.4f} | {r['accuracy_percent']:.4f} |"
        for r in rows
    ]
    lines += [
        "",
        str(result["pilot_decision"]),
        "",
        "Historical tests already known; no new test inference, automatic replication, LoRA, or budget expansion.",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("All six pooling conditions audited:", result["pilot_decision"], flush=True)
    return result


def run(path):
    f = verify(path)
    p = f["protocol"]
    root = root_directory(p)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    atomic_write_json(
        root / "suite_status.json",
        {
            "status": "running",
            "stage": "pooling_features_train_validation_only",
            "test_evaluated": False,
        },
    )
    arrays, dm = prepare(p, f)
    for pool, branch in cases(p):
        atomic_write_json(
            root / "suite_status.json",
            {
                "status": "running",
                "stage": "svm_cv",
                "pooling": pool,
                "branch": branch,
                "test_evaluated": False,
            },
        )
        fit_case(p, f, arrays, dm, pool, branch)
    summarize(path)
    atomic_write_json(
        root / "suite_status.json",
        {
            "status": "completed",
            "completed_at": utc_now(),
            "conditions": 6,
            "test_evaluated": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "run", "summarize"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    p = protocol(args.protocol)
    root = root_directory(p)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            globals()[args.action](args.protocol)
        except BaseException as exc:
            if args.action in {"run", "summarize"}:
                atomic_write_json(
                    root / "suite_status.json",
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "test_evaluated": False,
                        "updated_at": utc_now(),
                    },
                )
            raise


if __name__ == "__main__":
    main()
