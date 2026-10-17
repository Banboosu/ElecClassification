"""E01: leakage-safe low-label baselines and frozen MOMENT/statistics fusion.

Run with ``python -m tcn_moment.train_feature_fusion``. Test evaluation is opt-in;
validation-only runs neither extract test embeddings nor write test metrics.
Historical paper-aligned trainers are deliberately left unchanged.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tcn_moment.config import ExperimentConfig, load_config, with_random_seed
from tcn_moment.data import (
    DatasetBundle,
    load_dataset,
    save_label_encoder,
    stratified_train_subset_indices,
    train_subset_sha256,
)
from tcn_moment.experiment import prepare_run
from tcn_moment.feature_transforms import BranchScaler
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.metrics import classification_metrics
from tcn_moment.train_baselines import FEATURE_NAMES, extract_statistical_features
from tcn_moment.train_moment import (
    MOMENT_PROTOCOL_VERSION,
    build_model,
    forward_features,
    make_loader,
    require_torch_and_moment,
    select_device,
    set_num_classes,
)
from tcn_moment.train_moment_svm import _atomic_joblib_dump
from tcn_moment.training_utils import seed_everything

PROTOCOL_VERSION = 1
MODEL_NAMES = (
    "stats_rf",
    "stats_svm",
    "moment_paper_svm",
    "moment_tuned_svm",
    "fusion_meanstd_svm",
    "fusion_stats_svm",
)
BASELINE_NAMES = MODEL_NAMES[:4]
DEPENDENCY_FILES = (
    "config.py",
    "data.py",
    "train_baselines.py",
    "train_moment.py",
    "train_feature_fusion.py",
    "feature_transforms.py",
    "metrics.py",
    "training_utils.py",
    "train_moment_svm.py",
    "io_utils.py",
    "experiment.py",
)


def source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {name: sha256_file(directory / name) for name in DEPENDENCY_FILES}


def save_source_snapshot(run_dir: Path) -> None:
    target = run_dir / "source"
    target.mkdir()
    for name in DEPENDENCY_FILES:
        shutil.copy2(Path(__file__).parent / name, target / name)
    for path in Path("configs").rglob("*"):
        if path.suffix not in {".yaml", ".json"}:
            continue
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    atomic_write_json(run_dir / "source_hashes.json", source_hashes())
    provenance = Path("artifacts/deployment_provenance.json")
    if provenance.exists():
        shutil.copy2(provenance, run_dir / provenance.name)
    fingerprint = Path("artifacts/e01_model_weight_fingerprint.json")
    if fingerprint.exists():
        shutil.copy2(fingerprint, run_dir / fingerprint.name)


def verified_checkpoint(
    config: ExperimentConfig,
    manifest_path: Path = Path("artifacts/e01_model_weight_fingerprint.json"),
) -> dict[str, Any]:
    """Require a pinned, locally cached checkpoint; never fetch a moving main here."""
    if config.model.initialization == "random":
        return {"initialization": "random", "seed": config.data.random_state}
    from huggingface_hub import try_to_load_from_cache

    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record["model_id"] != config.model.model_id:
        raise ValueError("Checkpoint fingerprint belongs to a different model.")
    cached_path = try_to_load_from_cache(config.model.model_id, record["filename"])
    if not isinstance(cached_path, str) or not Path(cached_path).is_file():
        raise FileNotFoundError("Verified MOMENT weights must be cached before starting E01.")
    path = Path(cached_path)
    if path.parent.name != record["revision"]:
        raise ValueError("Cached main revision differs from the pinned checkpoint.")
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError("Cached MOMENT checkpoint checksum mismatch.")
    return record


def cache_signature(config: ExperimentConfig, bundle: DatasetBundle) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "moment_protocol_version": MOMENT_PROTOCOL_VERSION,
        "dataset_sha256": bundle.dataset_sha256,
        "split_sha256": sha256_file(bundle.split_path),
        "train_ids_sha256": train_subset_sha256(bundle.ids_train),
        "validation_ids_sha256": train_subset_sha256(bundle.ids_val),
        "max_length": config.data.max_length,
        "normalization": config.data.normalize,
        "model_id": config.model.model_id,
        "checkpoint": verified_checkpoint(config),
        "model_config_sha256": sha256_file(config.model.config_path),
        "initialization": config.model.initialization,
        "seed": config.data.random_state,
        "precision": "fp32",
        "source_hashes": source_hashes(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("momentfm", "torch", "numpy", "transformers")
        },
    }


def validate_cache(
    directory: Path,
    signature: dict[str, Any],
    split_ids: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["signature"] != signature:
        raise ValueError(
            f"Feature cache protocol mismatch: {directory}; use a new cache directory."
        )
    loaded = {}
    for split, ids in split_ids.items():
        record = manifest["splits"].get(split)
        if record is None:
            return None
        path = directory / f"{split}.npz"
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Corrupt feature cache: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["sample_ids"], ids.astype(str)):
                raise ValueError(f"Feature cache sample order mismatch: {path}")
            features = archive["features"]
        if features.ndim != 2 or features.shape != tuple(record["shape"]):
            raise ValueError(f"Invalid feature shape in cache: {path}")
        if len(features) != len(ids) or not np.isfinite(features).all():
            raise ValueError(f"Invalid feature values in cache: {path}")
        loaded[split] = features
    return loaded


def frozen_features(
    config: ExperimentConfig,
    bundle: DatasetBundle,
    cache_dir: Path,
    *,
    evaluate_test: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    split_values = {
        "train": (bundle.x_train, bundle.mask_train, bundle.y_train, bundle.ids_train),
        "validation": (bundle.x_val, bundle.mask_val, bundle.y_val, bundle.ids_val),
    }
    if evaluate_test:
        split_values["test"] = (
            bundle.x_test,
            bundle.mask_test,
            bundle.y_test,
            bundle.ids_test,
        )
    signature = cache_signature(config, bundle)
    split_ids = {key: values[3] for key, values in split_values.items()}
    cached = validate_cache(cache_dir, signature, split_ids)
    if cached is not None:
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        return cached, {"cache_hit": True, "cache_dir": str(cache_dir), **manifest}

    torch, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()
    device = select_device(torch, config.training.device)
    model = build_model(config, MOMENTPipeline, bundle.num_classes)
    set_num_classes(model, bundle.num_classes)
    model.init()
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"signature": signature, "splits": {}}
    )
    extraction_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for split, (x, mask, labels, ids) in split_values.items():
        # Reuse verified existing splits when a later formal run adds test features.
        existing = validate_cache(cache_dir, signature, {split: ids})
        if existing is not None:
            continue
        loader = make_loader(
            torch,
            DataLoader,
            TensorDataset,
            x,
            mask,
            labels,
            config.training.feature_extraction_batch_size,
            config.training.num_workers if split == "train" else 0,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        started = time.perf_counter()
        chunks = []
        with torch.inference_mode():
            for batch_x, batch_mask, _ in tqdm(loader, desc=f"FP32 {split} embeddings"):
                features = (
                    forward_features(
                        model,
                        batch_x.to(device, non_blocking=True),
                        batch_mask.to(device, non_blocking=True),
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                if not np.isfinite(features).all():
                    raise FloatingPointError(f"Non-finite FP32 MOMENT features in {split}.")
                chunks.append(features)
        array = np.concatenate(chunks)
        path = cache_dir / f"{split}.npz"
        temporary = cache_dir / f"{split}.npz.tmp"
        with temporary.open("wb") as file:
            np.savez(file, features=array, sample_ids=ids.astype(str))
        os.replace(temporary, path)
        manifest["splits"][split] = {
            "shape": list(array.shape),
            "sha256": sha256_file(path),
            "extraction_seconds": time.perf_counter() - started,
        }
        atomic_write_json(manifest_path, manifest)
    manifest["last_extraction"] = {
        "seconds": time.perf_counter() - extraction_started,
        "device": str(device),
        "amp_enabled": False,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0
        ),
    }
    atomic_write_json(manifest_path, manifest)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cached = validate_cache(cache_dir, signature, split_ids)
    assert cached is not None
    return cached, {"cache_hit": False, "cache_dir": str(cache_dir), **manifest}


def raw_statistics(config: ExperimentConfig, bundle: DatasetBundle) -> dict[str, np.ndarray]:
    raw = load_dataset(replace(config.data, normalize="none"))
    for split in ("train", "val", "test"):
        if not np.array_equal(getattr(raw, f"ids_{split}"), getattr(bundle, f"ids_{split}")):
            raise ValueError("Raw/statistical and MOMENT sample IDs differ.")
    # Test statistics, like test embeddings, are assembled only by the explicit test path.
    return {
        "train": extract_statistical_features(raw.x_train, raw.mask_train),
        "validation": extract_statistical_features(raw.x_val, raw.mask_val),
    }


def assemble_features(
    model_name: str,
    moment: np.ndarray | None,
    stats: np.ndarray,
) -> np.ndarray:
    if model_name.startswith("stats_"):
        return stats
    if moment is None:
        raise ValueError("MOMENT features required for this model.")
    if len(moment) != len(stats):
        raise ValueError("MOMENT/statistical sample counts differ.")
    if model_name.startswith("fusion_"):
        columns = [FEATURE_NAMES.index("mean"), FEATURE_NAMES.index("std")]
        side = stats[:, columns] if model_name == "fusion_meanstd_svm" else stats
        return np.concatenate([moment, side], axis=1)
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    return moment


def build_search(
    model_name: str,
    feature_dim: int,
    moment_dim: int,
    config: ExperimentConfig,
    *,
    n_jobs: int,
    seed: int,
) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=config.svm.cv_folds, shuffle=False)
    scoring: Any = make_scorer(f1_score, average="macro", zero_division=0)
    if model_name == "stats_rf":
        estimator = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
        grid = {"max_features": ["sqrt", 1.0], "min_samples_leaf": [1, 3]}
    elif model_name == "moment_paper_svm":
        estimator = SVC(
            kernel="rbf",
            gamma=config.svm.gamma,
            cache_size=config.svm.cache_size_mb,
            max_iter=config.svm.max_iter,
        )
        grid = {"C": list(config.svm.c_values)}
        scoring = "accuracy"
    else:
        transform: Any = StandardScaler()
        gamma_scale = 1.0 / feature_dim
        if model_name.startswith("fusion_"):
            transform = BranchScaler(split_at=moment_dim)
            gamma_scale = 0.5  # two unit-total-variance branches
        estimator = Pipeline(
            [
                ("scale", transform),
                (
                    "svc",
                    SVC(
                        kernel="rbf",
                        cache_size=config.svm.cache_size_mb,
                        max_iter=config.svm.max_iter,
                    ),
                ),
            ]
        )
        grid = {
            "svc__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "svc__gamma": ["scale", gamma_scale * 0.1, gamma_scale * 10],
        }
    return GridSearchCV(
        estimator,
        grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )


def validation_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["model"], row["train_fraction"]): row for row in results}
    candidates = {}
    for model_name in ("fusion_meanstd_svm", "fusion_stats_svm"):
        comparisons = []
        for fraction in (0.01, 0.05, 0.1):
            keys = [
                (name, fraction)
                for name in ("stats_rf", "stats_svm", "moment_tuned_svm", model_name)
            ]
            if not all(key in lookup for key in keys):
                continue
            best_single = max(lookup[key]["validation_metrics"]["macro_f1"] for key in keys[:-1])
            score = lookup[(model_name, fraction)]["validation_metrics"]["macro_f1"]
            comparisons.append(
                {
                    "train_fraction": fraction,
                    "fusion_macro_f1": score,
                    "best_single_branch_macro_f1": best_single,
                    "difference_pp": (score - best_single) * 100,
                    "passes": score - best_single >= 0.01 - 1e-12,
                }
            )
        passed = len(comparisons) == 3 and sum(row["passes"] for row in comparisons) >= 2
        candidates[model_name] = {"comparisons": comparisons, "expand": passed}
    return {
        "selection_data": "validation only; no test metrics read",
        "rule": ">=1 pp over strongest single-branch baseline at >=2 of 1%/5%/10%",
        "candidates": candidates,
        "models_to_expand": list(BASELINE_NAMES)
        + [name for name, record in candidates.items() if record["expand"]],
    }


def run_seed(args: argparse.Namespace, seed: int) -> None:
    config_path = Path(args.config)
    config = with_random_seed(load_config(config_path), seed)
    if config.data.train_fraction != 1 or not config.model.freeze_backbone:
        raise ValueError("E01 requires the full training pool and a frozen MOMENT backbone.")
    if config.model.unfreeze_last_n_layers != 0:
        raise ValueError("E01 does not support partially unfrozen encoders.")
    import torch

    seed_everything(torch, seed)
    run_name = f"{args.run_name}_seed{seed}"
    context = prepare_run(
        model_name="FEATURE_FUSION_E01",
        base_output_dir=Path(args.output_dir),
        config=config,
        config_path=config_path,
        torch=torch,
        run_name=run_name,
        resume_dir=None,
    )
    try:
        atomic_write_json(
            context.run_dir / "experiment_plan.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "models": args.models,
                "fractions": args.fractions,
                "evaluate_test": args.evaluate_test,
                "n_jobs": args.n_jobs,
                "selection": "training CV only; pilot gate uses validation only",
                "feature_precision": "fp32",
                "source_hashes": source_hashes(),
            },
        )
        save_source_snapshot(context.run_dir)
        bundle = load_dataset(config.data)
        shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
        save_label_encoder(bundle.label_encoder, context.run_dir)
        stats = raw_statistics(config, bundle)
        if args.evaluate_test:
            raw = load_dataset(replace(config.data, normalize="none"))
            stats["test"] = extract_statistical_features(raw.x_test, raw.mask_test)
            del raw
        moment: dict[str, np.ndarray] = {}
        extraction: dict[str, Any] = {"used": False}
        if any(not name.startswith("stats_") for name in args.models):
            moment, extraction = frozen_features(
                config,
                bundle,
                Path(args.cache_dir) / f"seed{seed}",
                evaluate_test=args.evaluate_test,
            )
        atomic_write_json(context.run_dir / "feature_extraction.json", extraction)
        class_names = [str(value) for value in bundle.label_encoder.classes_]
        results = []
        for fraction in args.fractions:
            indices = stratified_train_subset_indices(
                bundle.y_train,
                bundle.ids_train,
                fraction,
                seed,
            )
            if len(indices) > config.svm.max_samples:
                raise ValueError("Requested label budget exceeds declared SVM limit.")
            if np.bincount(bundle.y_train[indices]).min() < config.svm.cv_folds:
                raise ValueError("Not enough examples per class for the declared CV folds.")
            tag = f"{fraction * 100:g}".replace(".", "p")
            np.save(context.run_dir / f"train_ids_{tag}.npy", bundle.ids_train[indices])
            subset_labels = bundle.y_train[indices]
            fold_assignment = np.full(len(indices), -1, dtype=np.int8)
            cv = StratifiedKFold(n_splits=config.svm.cv_folds, shuffle=False)
            for fold, (_, held_out) in enumerate(cv.split(indices, subset_labels)):
                fold_assignment[held_out] = fold
            np.save(context.run_dir / f"cv_folds_{tag}.npy", fold_assignment)
            atomic_write_json(
                context.run_dir / f"train_subset_{tag}.json",
                {
                    "requested_fraction": fraction,
                    "selected_count": len(indices),
                    "full_training_count": len(bundle.y_train),
                    "class_counts": np.bincount(subset_labels).tolist(),
                    "sample_ids_sha256": train_subset_sha256(bundle.ids_train[indices]),
                    "cv_folds": config.svm.cv_folds,
                    "cv_shuffle": False,
                },
            )
            for model_name in args.models:
                print(
                    f"Starting {model_name} seed={seed} fraction={fraction:g} n={len(indices)}",
                    flush=True,
                )
                destination = context.run_dir / f"{model_name}_fraction_{tag}"
                destination.mkdir()
                features = {
                    split: assemble_features(model_name, moment.get(split), values)
                    for split, values in stats.items()
                }
                moment_dim = moment["train"].shape[1] if moment else 0
                search = build_search(
                    model_name,
                    features["train"].shape[1],
                    moment_dim,
                    config,
                    n_jobs=args.n_jobs,
                    seed=seed,
                )
                started = time.perf_counter()
                search.fit(features["train"][indices], bundle.y_train[indices])
                if np.any(getattr(search.best_estimator_, "fit_status_", 0)):
                    raise RuntimeError("SVM did not converge.")
                if (
                    isinstance(search.best_estimator_, Pipeline)
                    and search.best_estimator_.named_steps["svc"].fit_status_ != 0
                ):
                    raise RuntimeError("Pipeline SVM did not converge.")
                fit_seconds = time.perf_counter() - started
                search_results = pd.DataFrame(search.cv_results_)
                search_results.to_csv(destination / "cv_results.csv", index=False)
                _atomic_joblib_dump(search.best_estimator_, destination / "classifier.joblib")
                row = {
                    "model": model_name,
                    "seed": seed,
                    "train_fraction": fraction,
                    "train_samples": len(indices),
                    "train_ids_sha256": train_subset_sha256(bundle.ids_train[indices]),
                    "dataset_sha256": bundle.dataset_sha256,
                    "split_sha256": sha256_file(bundle.split_path),
                    "best_params": search.best_params_,
                    "cv_score": float(search.best_score_),
                    "selection_metric": "accuracy"
                    if model_name == "moment_paper_svm"
                    else "macro_f1",
                    "fit_seconds": fit_seconds,
                    "feature_dimension": features["train"].shape[1],
                    "test_evaluated": args.evaluate_test,
                }
                eval_splits = ["validation"] + (["test"] if args.evaluate_test else [])
                for split in eval_splits:
                    labels = bundle.y_val if split == "validation" else bundle.y_test
                    ids = bundle.ids_val if split == "validation" else bundle.ids_test
                    predicted = search.predict(features[split])
                    row[f"{split}_metrics"] = classification_metrics(
                        labels,
                        predicted,
                        class_names,
                        include_details=True,
                    )
                    pd.DataFrame(
                        {
                            "sample_id": ids,
                            "true_index": labels,
                            "predicted_index": predicted,
                            "true_label": [class_names[int(value)] for value in labels],
                            "predicted_label": [class_names[int(value)] for value in predicted],
                        }
                    ).to_csv(destination / f"{split}_predictions.csv", index=False)
                atomic_write_json(destination / "metrics.json", row)
                results.append(row)
                atomic_write_json(context.run_dir / "metrics_partial.json", {"results": results})
                print(
                    f"Completed {model_name} fraction={fraction:g} "
                    f"val_macro_f1={row['validation_metrics']['macro_f1']:.6f} "
                    f"fit_seconds={fit_seconds:.1f}",
                    flush=True,
                )
        atomic_write_json(
            context.run_dir / "metrics.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "results": results,
                "test_evaluated": args.evaluate_test,
            },
        )
        if not args.evaluate_test:
            atomic_write_json(context.run_dir / "validation_gate.json", validation_gate(results))
        context.set_status("completed", test_evaluated=args.evaluate_test)
    except KeyboardInterrupt:
        context.set_status("interrupted", message="Preserve outputs; use a new run name to retry.")
        raise
    except BaseException as exc:
        context.set_status("failed", error_type=type(exc).__name__, message=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/improvements/feature_fusion.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="artifacts/improvements/e01")
    parser.add_argument("--cache-dir", default="artifacts/improvements/e01_features_v1")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicit formal evaluation; omit this throughout exploration.",
    )
    args = parser.parse_args()
    if args.n_jobs <= 0 or len(set(args.models)) != len(args.models):
        parser.error("Use positive n-jobs and distinct model names.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct.")
    if (
        not args.fractions
        or any(not 0 < f <= 1 for f in args.fractions)
        or args.fractions != sorted(set(args.fractions))
    ):
        parser.error("Fractions must be unique, ascending, and in (0, 1].")
    for seed in args.seeds:
        run_seed(args, seed)


if __name__ == "__main__":
    main()
