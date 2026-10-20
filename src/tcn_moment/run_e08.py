"""E08: strong dynamics-feature baselines and MOMENT stacking (validation only).

The experiment answers two questions with a frozen protocol:

1. How much does the 15-feature statistical branch gain from a compact, frozen set of
   raw-scale dynamics descriptors?
2. Does a frozen MOMENT representation add anything on top of that stronger feature
   model, when both are evaluated with the same training labels, CV folds, and
   validation set?

No test split is encoded, predicted, or evaluated anywhere in this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import (
    DatasetBundle,
    load_dataset,
    save_label_encoder,
    stratified_train_subset_indices,
    train_subset_sha256,
)
from tcn_moment.dynamics_features import (
    BASE_FEATURE_COUNT,
    DYNAMICS_FEATURE_NAMES,
    FEATURE_GROUPS,
    extract_dynamics_features,
    group_indices,
    select_feature_set,
)
from tcn_moment.experiment import prepare_run
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.metrics import classification_metrics

DEFAULT_PROTOCOL = "configs/experiments/improvements/e08_strong_features_protocol.json"
DEFAULT_CONFIG = "configs/moment.yaml"
DEFAULT_SUITE = "e08_strong_features_20261020_v1"
DEFAULT_OUTPUT_ROOT = "artifacts/improvements/e08_strong_features"

SOURCE_FILES = (
    "src/tcn_moment/dynamics_features.py",
    "src/tcn_moment/statistical_features.py",
    "src/tcn_moment/train_baselines.py",
    "src/tcn_moment/run_e08.py",
    "src/tcn_moment/e08_analysis.py",
    "src/tcn_moment/data.py",
    "src/tcn_moment/config.py",
    "src/tcn_moment/metrics.py",
    "src/tcn_moment/io_utils.py",
    "src/tcn_moment/experiment.py",
    "tests/test_dynamics_features.py",
    "tests/test_e08_analysis.py",
    "scripts/analyze_e08_results.py",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(f"Frozen E08 source file is missing: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def _feature_contract() -> dict[str, Any]:
    return {
        "base_feature_count": BASE_FEATURE_COUNT,
        "feature_names": list(DYNAMICS_FEATURE_NAMES),
        "groups": {name: list(names) for name, names in FEATURE_GROUPS.items()},
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("no_test_access") is not True:
        raise ValueError("E08 protocol must explicitly forbid test access.")
    if protocol.get("seeds") != [42, 44, 45, 46]:
        raise ValueError("E08 fixes seeds 42/44/45/46; seed43 is the pilot and must stay out.")
    fractions = protocol.get("fractions")
    if not fractions or fractions != sorted(set(fractions)) or any(not 0 < f <= 1 for f in fractions):
        raise ValueError("E08 fractions must be unique, ascending, and in (0, 1].")
    models = protocol.get("models")
    allowed = {"f15_hgb", "dyn_hgb", "f15_svm", "dyn_svm", "moment_logit", "stack"}
    if not models or not set(models) <= allowed:
        raise ValueError("E08 model list is invalid.")
    contract = _feature_contract()
    if protocol.get("feature_names") != contract["feature_names"]:
        raise ValueError("Protocol feature names differ from the frozen feature module.")
    if protocol.get("feature_groups") != contract["groups"]:
        raise ValueError("Protocol feature groups differ from the frozen feature module.")
    for name, group in protocol["feature_groups"].items():
        unknown = set(group) - set(protocol["feature_names"])
        if unknown:
            raise ValueError(f"Unknown features in group {name}: {sorted(unknown)}")


def _load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        protocol = json.load(file)
    _validate_protocol(protocol)
    return protocol


def _suite_root(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / args.suite_name


def _require_frozen(root: Path, protocol: dict[str, Any], *, allow_drift: bool) -> dict[str, Any]:
    frozen_path = root / "frozen_protocol.json"
    if not frozen_path.exists():
        raise FileNotFoundError(
            f"Frozen protocol missing: {frozen_path}. Run `run_e08 freeze --suite-name ...` first."
        )
    with frozen_path.open("r", encoding="utf-8") as file:
        frozen = json.load(file)
    protocol_sha = _canonical_sha256(protocol)
    if frozen.get("protocol_sha256") != protocol_sha:
        raise ValueError("Protocol file changed after freeze; use a new suite name.")
    if frozen.get("feature_contract_sha256") != _canonical_sha256(_feature_contract()):
        raise ValueError("Feature contract changed after freeze; use a new suite name.")
    current = _source_hashes()
    if frozen.get("source_hashes") != current:
        changed = sorted(
            name
            for name in set(current) | set(frozen.get("source_hashes", {}))
            if current.get(name) != frozen.get("source_hashes", {}).get(name)
        )
        if not allow_drift:
            raise ValueError(f"Source files changed after freeze: {changed}")
        frozen["source_drift"] = changed
    return frozen


def freeze(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol)
    protocol = _load_protocol(protocol_path)
    root = _suite_root(args)
    root.mkdir(parents=True, exist_ok=True)
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "feature_contract": _feature_contract(),
        "feature_contract_sha256": _canonical_sha256(_feature_contract()),
        "source_hashes": _source_hashes(),
        "test_evaluated": False,
    }
    destination = root / "frozen_protocol.json"
    if destination.exists():
        with destination.open("r", encoding="utf-8") as file:
            previous = json.load(file)
        if previous.get("protocol_sha256") != frozen["protocol_sha256"]:
            raise ValueError("A different protocol is already frozen in this suite; pick a new suite name.")
        if previous.get("source_hashes") != frozen["source_hashes"]:
            raise ValueError("Sources changed; use a new suite name instead of re-freezing.")
        print(f"E08 protocol already frozen: {destination}")
        return
    atomic_write_json(destination, frozen)
    print(f"Froze E08 protocol: {destination}")
    print(f"Feature contract SHA-256: {frozen['feature_contract_sha256']}")


def _atomic_write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_moment_split(
    cache_dir: Path,
    *,
    seed: int,
    split: str,
    ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    directory = cache_dir / f"seed{seed}"
    path = directory / f"{split}.npz"
    manifest_path = directory / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"MOMENT feature cache missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        cache_ids = archive["sample_ids"].astype(str)
    if features.ndim != 2 or features.shape[0] != len(cache_ids):
        raise ValueError(f"Invalid MOMENT feature shape: {features.shape}")
    if len(set(cache_ids)) != len(cache_ids):
        raise ValueError(f"Duplicate sample IDs in MOMENT cache: {path}")
    if not np.array_equal(cache_ids, ids.astype(str)):
        raise ValueError(
            f"MOMENT cache order does not match the E08 split for {split}: {path}. "
            "Do not silently reindex; rebuild or use the matching cache."
        )
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite MOMENT features: {path}")
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(features.shape),
    }
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        signature = manifest.get("signature", {})
        record["signature"] = {
            key: signature.get(key)
            for key in (
                "dataset_sha256",
                "split_sha256",
                "train_ids_sha256",
                "validation_ids_sha256",
                "normalization",
                "model_id",
                "seed",
                "precision",
            )
        }
    return features, record


def _cv_select(
    factory: Callable[[dict[str, Any]], Any],
    grid: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None
    for params in grid:
        scores = []
        for train_index, valid_index in splitter.split(x, y):
            model = factory(params)
            model.fit(x[train_index], y[train_index])
            scores.append(
                float(f1_score(y[valid_index], model.predict(x[valid_index]), average="macro"))
            )
        record = {
            "params": params,
            "fold_scores": scores,
            "mean_macro_f1": float(np.mean(scores)),
        }
        results.append(record)
        if best is None or record["mean_macro_f1"] > best[0]:
            best = (record["mean_macro_f1"], params)
    if best is None:
        raise RuntimeError("Empty hyperparameter grid.")
    return best[1], results


def _hgb_factory(params: dict[str, Any], seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        **params,
        early_stopping=False,
        random_state=seed,
    )


def _svm_factory(params: dict[str, Any]) -> Any:
    return make_pipeline(
        StandardScaler(),
        SVC(
            C=params["C"],
            gamma=params["gamma"],
            cache_size=1000,
            tol=1e-3,
        ),
    )


def _logit_factory(params: dict[str, Any], seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=params["C"],
            max_iter=5000,
            random_state=seed,
        ),
    )


def _battery_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    battery_index: int,
) -> dict[str, float]:
    positive = y_true == battery_index
    true_positive = int(np.sum(positive & (y_pred == battery_index)))
    false_negative = int(np.sum(positive & (y_pred != battery_index)))
    false_positive = int(np.sum(~positive & (y_pred == battery_index)))
    true_negative = int(np.sum(~positive & (y_pred != battery_index)))
    recall = true_positive / max(true_positive + false_negative, 1)
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "support": int(positive.sum()),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "precision": float(precision),
        "f1": float(f1),
        "average_precision": float(average_precision_score(positive, scores[:, battery_index])),
    }


def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    battery_index: int,
) -> dict[str, Any]:
    metrics = classification_metrics(y_true, y_pred, class_names, include_details=True)
    metrics["battery"] = _battery_metrics(y_true, y_pred, scores, battery_index)
    return metrics


def _write_condition(
    run_dir: Path,
    *,
    tag: str,
    method: str,
    ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    battery_index: int,
) -> str:
    directory = run_dir / "conditions" / tag / method
    directory.mkdir(parents=True, exist_ok=False)
    header = [
        "sample_id",
        "split",
        "y_true",
        "y_pred",
        "score_0",
        "score_1",
        "score_2",
        "score_battery",
    ]
    rows = [
        [
            str(sample_id),
            "validation",
            int(true),
            int(predicted),
            float(score[0]),
            float(score[1]),
            float(score[2]),
            float(score[battery_index]),
        ]
        for sample_id, true, predicted, score in zip(ids, y_true, y_pred, scores, strict=True)
    ]
    path = directory / "predictions.csv"
    _atomic_write_csv(path, header, rows)
    return str(path)


def _run_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    grid: list[dict[str, Any]],
    folds: int,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    selected, cv_results = _cv_select(
        lambda params: _hgb_factory(params, seed),
        grid,
        x_train,
        y_train,
        folds=folds,
        seed=seed,
    )
    model = _hgb_factory(selected, seed)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_val)
    return model, model.predict(x_val), scores, selected, cv_results


def _run_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    c_values: list[float],
    gamma: str,
    folds: int,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    grid = [{"C": float(c), "gamma": gamma} for c in c_values]
    selected, cv_results = _cv_select(
        _svm_factory,
        grid,
        x_train,
        y_train,
        folds=folds,
        seed=seed,
    )
    model = _svm_factory(selected)
    model.fit(x_train, y_train)
    if model.named_steps["svc"].fit_status_ != 0:
        raise RuntimeError("SVM did not converge.")
    scores = np.asarray(model.decision_function(x_val), dtype=np.float64)
    if scores.ndim != 2:
        raise RuntimeError("Expected one-vs-rest decision scores for three classes.")
    return model, model.predict(x_val), scores, selected, cv_results


def _run_logit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    c_values: list[float],
    folds: int,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    grid = [{"C": float(c)} for c in c_values]
    selected, cv_results = _cv_select(
        lambda params: _logit_factory(params, seed),
        grid,
        x_train,
        y_train,
        folds=folds,
        seed=seed,
    )
    model = _logit_factory(selected, seed)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_val)
    return model, model.predict(x_val), scores, selected, cv_results


def _run_stack(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    moment_train: np.ndarray,
    moment_val: np.ndarray,
    *,
    hgb_params: dict[str, Any],
    folds: int,
    moment_c: float,
    meta_c_values: list[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]], float]:
    started = time.perf_counter()
    oof = np.zeros((len(y_train), 6), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, valid_index in splitter.split(x_train, y_train):
        hgb = _hgb_factory(hgb_params, seed)
        hgb.fit(x_train[train_index], y_train[train_index])
        oof[valid_index, :3] = hgb.predict_proba(x_train[valid_index])
        logistic = _logit_factory({"C": moment_c}, seed)
        logistic.fit(moment_train[train_index], y_train[train_index])
        oof[valid_index, 3:] = logistic.predict_proba(moment_train[valid_index])

    meta_grid = [{"C": float(c)} for c in meta_c_values]
    meta_selected, meta_cv = _cv_select(
        lambda params: LogisticRegression(
            C=params["C"],
            max_iter=5000,
            random_state=seed,
        ),
        meta_grid,
        oof,
        y_train,
        folds=3,
        seed=seed,
    )
    meta = LogisticRegression(C=meta_selected["C"], max_iter=5000, random_state=seed)
    meta.fit(oof, y_train)

    hgb = _hgb_factory(hgb_params, seed).fit(x_train, y_train)
    logistic = _logit_factory({"C": moment_c}, seed).fit(moment_train, y_train)
    stacked_val = np.hstack([hgb.predict_proba(x_val), logistic.predict_proba(moment_val)])
    scores = meta.predict_proba(stacked_val)
    details = {
        "hgb_params": hgb_params,
        "moment_c": moment_c,
        "meta_params": meta_selected,
        "meta_cv": meta_cv,
        "oof_macro_f1": float(f1_score(y_train, meta.predict(oof), average="macro")),
    }
    return meta.predict(stacked_val), scores, details, meta_cv, time.perf_counter() - started


def _group_ablation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    params: dict[str, Any],
    seed: int,
    full_macro_f1: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    all_indices = list(range(len(DYNAMICS_FEATURE_NAMES)))
    for group in FEATURE_GROUPS:
        kept = [index for index in all_indices if index not in set(group_indices(group))]
        dropped_model = _hgb_factory(params, seed).fit(x_train[:, kept], y_train)
        dropped = float(f1_score(y_val, dropped_model.predict(x_val[:, kept]), average="macro"))
        only_indices = group_indices(group)
        only_model = _hgb_factory(params, seed).fit(x_train[:, only_indices], y_train)
        only = float(f1_score(y_val, only_model.predict(x_val[:, only_indices]), average="macro"))
        records.append(
            {
                "group": group,
                "n_features": len(FEATURE_GROUPS[group]),
                "drop_group_macro_f1": dropped,
                "drop_delta_pp": (dropped - full_macro_f1) * 100.0,
                "only_group_macro_f1": only,
            }
        )
    return records


def _permutation_importance(
    model: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    result = permutation_importance(
        model,
        x_val,
        y_val,
        scoring="f1_macro",
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    )
    order = np.argsort(-result.importances_mean)
    top = [
        {
            "feature": DYNAMICS_FEATURE_NAMES[int(index)],
            "mean_drop": float(result.importances_mean[index]),
            "std_drop": float(result.importances_std[index]),
        }
        for index in order[:20]
    ]
    return {
        "usage": "descriptive validation importance, not used for feature selection",
        "n_repeats": 3,
        "top20": top,
    }


def _tag(fraction: float) -> str:
    return f"{fraction * 100:g}".replace(".", "p") + "p"


def _model_budget_allowed(model: str, fraction: float, protocol: dict[str, Any]) -> bool:
    if model in {"f15_hgb", "dyn_hgb"}:
        return True
    if model in {"f15_svm", "dyn_svm"}:
        return fraction in protocol["svm"]["budgets"]
    if model == "moment_logit":
        return fraction in protocol["logit"]["budgets"]
    if model == "stack":
        return fraction in protocol["stack"]["budgets"]
    raise ValueError(f"Unknown model: {model}")


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol)
    protocol = _load_protocol(protocol_path)
    root = _suite_root(args)
    frozen = _require_frozen(root, protocol, allow_drift=args.allow_source_drift)
    if args.seed not in protocol["seeds"]:
        raise ValueError(f"Seed {args.seed} is not in the frozen protocol.")
    fractions = [float(value) for value in args.fractions] if args.fractions else protocol["fractions"]
    if any(value not in protocol["fractions"] for value in fractions):
        raise ValueError("Requested fraction is not in the frozen protocol.")
    models = list(args.models) if args.models else protocol["models"]
    if not set(models) <= set(protocol["models"]):
        raise ValueError("Requested model is not in the frozen protocol.")

    import torch

    from tcn_moment.training_utils import seed_everything

    seed_everything(torch, args.seed)
    config_path = Path(args.config)
    config = with_random_seed(load_config(config_path), args.seed)
    run_name = f"seed{args.seed}"
    context = prepare_run(
        model_name="E08_STRONG_FEATURES",
        base_output_dir=root,
        config=config,
        config_path=config_path,
        torch=torch,
        run_name=run_name,
        resume_dir=None,
    )
    try:
        raw = load_dataset(replace(config.data, normalize="none"))
        if raw.label_encoder.classes_.tolist() != ["0", "1", "2"]:
            raise ValueError("E08 expects exactly the 0/1/2 classes.")
        if raw.x_train.shape[1] != config.data.max_length:
            raise ValueError("Unexpected max_length.")
        if set(raw.ids_test) & (set(raw.ids_train) | set(raw.ids_val)):
            raise RuntimeError("Split overlap detected; refusing to continue.")

        class_names = [str(value) for value in raw.label_encoder.classes_]
        battery_index = class_names.index("2")
        save_label_encoder(raw.label_encoder, context.run_dir)
        split_manifest_path = context.run_dir / "split_manifest.json"
        split_manifest_path.write_bytes(raw.split_path.read_bytes())

        started = time.perf_counter()
        train_features = extract_dynamics_features(
            raw.x_train, raw.mask_train, max_length=config.data.max_length
        )
        validation_features = extract_dynamics_features(
            raw.x_val, raw.mask_val, max_length=config.data.max_length
        )
        feature_seconds = time.perf_counter() - started
        if train_features.shape[1] != len(DYNAMICS_FEATURE_NAMES):
            raise RuntimeError("Feature extraction returned an unexpected width.")

        _atomic_savez(
            context.run_dir / "features.npz",
            train_features=train_features,
            validation_features=validation_features,
            train_ids=raw.ids_train.astype(str),
            validation_ids=raw.ids_val.astype(str),
            train_labels=raw.y_train,
            validation_labels=raw.y_val,
            feature_names=np.asarray(DYNAMICS_FEATURE_NAMES, dtype=str),
        )
        atomic_write_json(
            context.run_dir / "feature_contract.json",
            {
                **_feature_contract(),
                "base_feature_count": BASE_FEATURE_COUNT,
                "train_shape": list(train_features.shape),
                "validation_shape": list(validation_features.shape),
                "max_length": config.data.max_length,
                "extraction_seconds": feature_seconds,
            },
        )
        atomic_write_json(
            context.run_dir / "protocol_snapshot.json",
            {
                "protocol": protocol,
                "protocol_sha256": frozen["protocol_sha256"],
                "feature_contract_sha256": frozen["feature_contract_sha256"],
                "frozen_at": frozen["frozen_at"],
                "source_hashes": frozen["source_hashes"],
                "source_drift": frozen.get("source_drift", []),
                "test_evaluated": False,
            },
        )
        atomic_write_json(
            context.run_dir / "experiment_plan.json",
            {
                "seed": args.seed,
                "fractions": fractions,
                "models": models,
                "config": str(config_path),
                "moment_cache_dir": None if args.skip_moment else str(Path(args.moment_cache_dir)),
                "skip_moment": bool(args.skip_moment),
                "selection": "training-subset CV only; validation evaluated once per condition",
                "test_evaluated": False,
            },
        )

        moment: dict[str, np.ndarray] = {}
        moment_record: dict[str, Any] = {"used": False}
        if not args.skip_moment and any(model in models for model in ("moment_logit", "stack")):
            cache_dir = Path(args.moment_cache_dir)
            moment["train"], train_record = _load_moment_split(
                cache_dir, seed=args.seed, split="train", ids=raw.ids_train
            )
            moment["validation"], validation_record = _load_moment_split(
                cache_dir, seed=args.seed, split="validation", ids=raw.ids_val
            )
            signature_verified = False
            manifest_path = cache_dir / f"seed{args.seed}" / "manifest.json"
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as file:
                    manifest = json.load(file)
                signature = manifest.get("signature", {})
                expected = {
                    "dataset_sha256": raw.dataset_sha256,
                    "split_sha256": sha256_file(raw.split_path),
                    "train_ids_sha256": train_subset_sha256(raw.ids_train),
                    "validation_ids_sha256": train_subset_sha256(raw.ids_val),
                    "normalization": "zscore",
                    "model_id": protocol["moment_cache"]["model_id"],
                    "seed": args.seed,
                }
                mismatch = {
                    key: {"expected": value, "actual": signature.get(key)}
                    for key, value in expected.items()
                    if signature.get(key) != value
                }
                if mismatch:
                    raise ValueError(f"MOMENT cache signature mismatch: {mismatch}")
                signature_verified = True
            moment_record = {
                "used": True,
                "cache_dir": str(cache_dir),
                "train": train_record,
                "validation": validation_record,
                "signature_verified": signature_verified,
            }
        elif not args.skip_moment:
            raise ValueError("MOMENT cache is required for the frozen model list.")
        atomic_write_json(context.run_dir / "moment_cache.json", moment_record)

        records: list[dict[str, Any]] = []
        fitted: dict[tuple[str, str], Any] = {}
        subsets: dict[str, dict[str, Any]] = {}
        for fraction in fractions:
            tag = _tag(fraction)
            indices = stratified_train_subset_indices(
                raw.y_train, raw.ids_train, fraction, args.seed
            )
            if np.bincount(raw.y_train[indices]).min() < max(protocol["hgb"]["folds"], protocol["svm"]["folds"]):
                raise ValueError(f"Not enough examples per class at fraction {fraction}.")
            subsets[tag] = {
                "fraction": fraction,
                "selected_count": int(len(indices)),
                "full_training_count": int(len(raw.y_train)),
                "class_counts": np.bincount(raw.y_train[indices]).tolist(),
                "sample_ids_sha256": train_subset_sha256(raw.ids_train[indices]),
            }
            atomic_write_json(context.run_dir / f"train_subset_{tag}.json", subsets[tag])
            np.save(context.run_dir / f"train_ids_{tag}.npy", raw.ids_train[indices])

            x_train_dyn = train_features[indices]
            x_train_f15 = select_feature_set(x_train_dyn, "f15")
            y_train = raw.y_train[indices]
            x_val_dyn = validation_features
            x_val_f15 = select_feature_set(x_val_dyn, "f15")
            y_val = raw.y_val

            for model_name in models:
                if not _model_budget_allowed(model_name, fraction, protocol):
                    continue
                if args.skip_moment and model_name in {"moment_logit", "stack"}:
                    continue
                condition_started = time.perf_counter()
                selected: dict[str, Any] = {}
                cv_results: list[dict[str, Any]] = []
                extra: dict[str, Any] = {}
                if model_name in {"f15_hgb", "dyn_hgb"}:
                    x_train = x_train_f15 if model_name == "f15_hgb" else x_train_dyn
                    x_val = x_val_f15 if model_name == "f15_hgb" else x_val_dyn
                    fitted_model, prediction, scores, selected, cv_results = _run_hgb(
                        x_train,
                        y_train,
                        x_val,
                        grid=protocol["hgb"]["grid"],
                        folds=protocol["hgb"]["folds"],
                        seed=args.seed,
                    )
                    score_type = "probability"
                    fitted[(model_name, tag)] = fitted_model
                elif model_name in {"f15_svm", "dyn_svm"}:
                    x_train = x_train_f15 if model_name == "f15_svm" else x_train_dyn
                    x_val = x_val_f15 if model_name == "f15_svm" else x_val_dyn
                    fitted_model, prediction, scores, selected, cv_results = _run_svm(
                        x_train,
                        y_train,
                        x_val,
                        c_values=protocol["svm"]["c_values"],
                        gamma=protocol["svm"]["gamma"],
                        folds=protocol["svm"]["folds"],
                        seed=args.seed,
                    )
                    score_type = "decision_function"
                elif model_name == "moment_logit":
                    fitted_model, prediction, scores, selected, cv_results = _run_logit(
                        moment["train"][indices],
                        y_train,
                        moment["validation"],
                        c_values=protocol["logit"]["c_values"],
                        folds=protocol["logit"]["folds"],
                        seed=args.seed,
                    )
                    score_type = "probability"
                elif model_name == "stack":
                    if ("dyn_hgb", tag) not in fitted:
                        raise ValueError("Stack requires dyn_hgb at the same label budget.")
                    hgb_params = next(
                        record["selected_params"]
                        for record in records
                        if record["method"] == "dyn_hgb" and record["tag"] == tag
                    )
                    prediction, scores, extra, meta_cv, stack_seconds = _run_stack(
                        x_train_dyn,
                        y_train,
                        x_val_dyn,
                        moment["train"][indices],
                        moment["validation"],
                        hgb_params=hgb_params,
                        folds=protocol["stack"]["folds"],
                        moment_c=protocol["stack"]["moment_c"],
                        meta_c_values=protocol["stack"]["meta_c_values"],
                        seed=args.seed,
                    )
                    selected = extra
                    cv_results = meta_cv
                    score_type = "probability"
                    extra["stack_seconds"] = stack_seconds
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                metrics = _evaluate(y_val, prediction, scores, class_names, battery_index)
                prediction_path = _write_condition(
                    context.run_dir,
                    tag=tag,
                    method=model_name,
                    ids=raw.ids_val,
                    y_true=y_val,
                    y_pred=prediction,
                    scores=scores,
                    battery_index=battery_index,
                )
                record = {
                    "tag": tag,
                    "fraction": fraction,
                    "method": model_name,
                    "feature_set": "moment" if model_name == "moment_logit" else (
                        "stack" if model_name == "stack" else model_name.split("_")[0]
                    ),
                    "n_train": int(len(indices)),
                    "class_counts": np.bincount(y_train).tolist(),
                    "selected_params": selected,
                    "cv_results": cv_results,
                    "score_type": score_type,
                    "validation": metrics,
                    "predictions": prediction_path,
                    "fit_seconds": time.perf_counter() - condition_started,
                    **extra,
                }
                records.append(record)
                atomic_write_json(
                    context.run_dir / "conditions" / tag / model_name / "metrics.json",
                    record,
                )
                print(
                    f"E08 seed={args.seed} {model_name} {tag}: "
                    f"macro-F1={metrics['macro_f1'] * 100:.4f} "
                    f"battery_recall={metrics['battery']['recall'] * 100:.2f} "
                    f"battery_fpr={metrics['battery']['false_positive_rate'] * 100:.2f}",
                    flush=True,
                )

        ablation: list[dict[str, Any]] = []
        importance: dict[str, Any] = {}
        if 0.4 in fractions and ("dyn_hgb", "40p") in fitted:
            indices = stratified_train_subset_indices(raw.y_train, raw.ids_train, 0.4, args.seed)
            dyn_record = next(
                record for record in records if record["method"] == "dyn_hgb" and record["tag"] == "40p"
            )
            ablation = _group_ablation(
                train_features[indices],
                raw.y_train[indices],
                validation_features,
                raw.y_val,
                params=dyn_record["selected_params"],
                seed=args.seed,
                full_macro_f1=dyn_record["validation"]["macro_f1"],
            )
            importance = _permutation_importance(
                fitted[("dyn_hgb", "40p")],
                validation_features,
                raw.y_val,
                seed=args.seed,
            )
            atomic_write_json(context.run_dir / "group_ablation_40p.json", ablation)
            atomic_write_json(context.run_dir / "permutation_importance_40p.json", importance)

        summary = {
            "seed": args.seed,
            "dataset_sha256": raw.dataset_sha256,
            "split_sha256": sha256_file(raw.split_path),
            "feature_contract_sha256": frozen["feature_contract_sha256"],
            "protocol_sha256": frozen["protocol_sha256"],
            "source_hashes": frozen["source_hashes"],
            "source_drift": frozen.get("source_drift", []),
            "test_evaluated": False,
            "fractions": fractions,
            "models": models,
            "subsets": subsets,
            "moment_cache": moment_record,
            "conditions": records,
            "group_ablation_40p": ablation,
            "permutation_importance_40p": importance,
        }
        atomic_write_json(context.run_dir / "seed_results.json", summary)
        context.set_status(
            "completed",
            conditions=len(records),
            test_evaluated=False,
            feature_seconds=feature_seconds,
        )
        print(f"E08 seed={args.seed} completed: {len(records)} conditions", flush=True)
    except BaseException as error:
        context.set_status(
            "failed",
            error=f"{type(error).__name__}: {error}",
            test_evaluated=False,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze", help="Write and hash the frozen protocol.")
    freeze_parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    freeze_parser.add_argument("--suite-name", default=DEFAULT_SUITE)
    freeze_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    freeze_parser.set_defaults(func=freeze)

    run_parser = subparsers.add_parser("run", help="Run one validation-only seed.")
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    run_parser.add_argument("--config", default=DEFAULT_CONFIG)
    run_parser.add_argument("--suite-name", default=DEFAULT_SUITE)
    run_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument(
        "--moment-cache-dir",
        default="artifacts/improvements/e01_20261017_v2_features",
    )
    run_parser.add_argument("--skip-moment", action="store_true")
    run_parser.add_argument("--fractions", nargs="+", type=float)
    run_parser.add_argument("--models", nargs="+")
    run_parser.add_argument("--allow-source-drift", action="store_true")
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
