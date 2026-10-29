"""E13: noise and quantization robustness of clean-trained final systems.

Train clean, evaluate perturbed validation only.  DYN features, MOMENT/diff
embeddings and TCN inputs are recomputed from the perturbed raw sequence; labels
are assumed unchanged.  No test split is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tcn_moment import run_e08
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.config import load_config, replace, with_random_seed
from tcn_moment.data import load_dataset, normalize_sequence
from tcn_moment.dynamics_features import extract_dynamics_features
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.label_semantics import BATTERY_LABEL, class_index
from tcn_moment.train_moment import (
    build_model,
    forward_features,
    require_torch_and_moment,
    set_num_classes,
)

PACKAGES = ("torch", "numpy", "pandas", "scikit-learn")
METHODS = ("dyn_hgb", "md_logit", "tcn", "tcn_dyn", "dyn_md", "full")
TAGS = ("10p", "40p")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root(p: dict[str, Any]) -> Path:
    return Path(p["output_dir"]) / p["suite"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(p: dict[str, Any]) -> None:
    if p.get("experiment_id") != "E13":
        raise ValueError("Unexpected experiment id")
    if p.get("seeds") != [42, 44, 45, 46]:
        raise ValueError("E13 fixes seeds 42/44/45/46")
    if p.get("fractions") not in ([0.1, 0.4], [0.4, 0.1], [0.1], [0.4]):
        raise ValueError("E13 supports 10% and/or 40%")
    if p.get("methods") != list(METHODS):
        raise ValueError("E13 fixes six methods")
    perturbations = p.get("perturbations", {})
    if "clean" not in perturbations or perturbations["clean"].get("kind") != "none":
        raise ValueError("E13 requires a clean reference")
    for name, spec in perturbations.items():
        if spec.get("kind") not in {"none", "noise", "quantization"}:
            raise ValueError(f"Unknown perturbation: {name}")
        if spec["kind"] == "noise" and not 0 < float(spec["sigma_fraction"]) < 1:
            raise ValueError(f"Bad noise level: {name}")
        if spec["kind"] == "quantization" and float(spec["step_w"]) <= 0:
            raise ValueError(f"Bad quantization step: {name}")
    if p.get("test_evaluated") or p.get("automatic_expansion"):
        raise ValueError("E13 is validation-only and does not auto-expand")


def _load(path: Path) -> dict[str, Any]:
    p = _read(path)
    _validate(p)
    return p


def _sources(protocol_path: Path) -> list[Path]:
    directory = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        directory / "run_e08.py",
        directory / "label_semantics.py",
        directory / "run_e09_diff_stream.py",
        directory / "run_e12_tcn_stack.py",
        directory / "tcn_legacy_pilot.py",
        directory / "train_moment.py",
        directory / "data.py",
        directory / "dynamics_features.py",
        directory / "config.py",
        directory / "io_utils.py",
        Path("tests/test_e13_robustness.py"),
        protocol_path,
    ]


def synthetic_check() -> dict[str, Any]:
    x = np.zeros((2, 8), dtype=np.float32)
    mask = np.zeros((2, 8), dtype=bool)
    x[0, :5] = [0.0, 1.0, 2.0, 3.0, 4.0]
    x[1, :4] = [10.0, 10.0, 10.0, 10.0]
    mask[0, :5] = True
    mask[1, :4] = True
    rng = np.random.default_rng(7)
    noisy = _perturb(x, mask, {"kind": "noise", "sigma_fraction": 0.05}, rng, 10.0, 0.0)
    quantized = _perturb(x, mask, {"kind": "quantization", "step_w": 2.0}, rng, 10.0, 0.0)
    if not np.all(noisy >= 0) or not np.allclose(x[1, :4], 10.0):
        raise AssertionError("Noise perturbation check failed")
    if not np.allclose(quantized[0, :5], [0.0, 0.0, 2.0, 4.0, 4.0]):
        raise AssertionError("Quantization perturbation check failed")
    normalized = _normalize_pad(np.asarray([1.0, 2.0, 3.0], dtype=np.float64), 8)
    differences, _ = _diff_pad(np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64), 8)
    if not np.isfinite(normalized).all() or not np.isfinite(differences).all():
        raise AssertionError("Normalization/difference check failed")
    return {
        "noise_deterministic": True,
        "quantization": quantized[0, :5].tolist(),
        "normalized_nonzero": int((normalized != 0).sum()),
        "diff_nonzero": int((differences != 0).sum()),
    }


def freeze(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    root = _root(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        return verify(protocol_path)
    if list(root.glob("seed*")):
        raise ValueError("Freeze must precede running")
    if _sha256(Path(p["dataset"])) != p["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    for seed in p["seeds"]:
        manifest = _read(Path(p["split_manifests"][str(seed)]))
        if (
            manifest["protocol"]["random_state"] != seed
            or manifest["protocol"]["dataset_sha256"] != p["dataset_sha256"]
            or manifest["counts"]["train"] != p["train_count"]
            or manifest["counts"]["validation"] != p["validation_count"]
        ):
            raise ValueError(f"Split mismatch for seed {seed}")
    files = set(_sources(protocol_path))
    files.update(Path(p["split_manifests"][str(seed)]) for seed in p["seeds"])
    files.add(Path(p["dataset"]))
    for seed in p["seeds"]:
        files.update(
            [
                Path(p["e08_suite"]) / f"seed{seed}" / "features.npz",
                Path(p["e08_suite"]) / f"seed{seed}" / "seed_results.json",
                Path(p["e01_cache"]) / f"seed{seed}" / "train.npz",
                Path(p["e01_cache"]) / f"seed{seed}" / "validation.npz",
                Path(p["e01_cache"]) / f"seed{seed}" / "manifest.json",
                Path(p["e09_cache"]) / f"seed{seed}" / "train.npz",
                Path(p["e09_cache"]) / f"seed{seed}" / "validation.npz",
                Path(p["e09_cache"]) / f"seed{seed}" / "manifest.json",
            ]
        )
        for tag in TAGS:
            suite = Path(p["e12_suites"][tag])
            files.update(
                [
                    suite / f"seed{seed}" / f"oof_{tag}.npz",
                    suite / f"seed{seed}" / "models" / tag / "full.pt",
                ]
            )
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Missing frozen input: {file}")
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol": p,
        "protocol_sha256": _sha256(protocol_path),
        "file_sha256": {str(file): _sha256(file) for file in sorted(files)},
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "cuda": torch.version.cuda,
        },
        "synthetic_checks": synthetic_check(),
        "test_evaluated": False,
    }
    for file in sorted(files):
        if file.parts[0] in {"src", "configs", "tests", "scripts", "docs"}:
            destination = root / "source_snapshot" / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    atomic_write_json(root / "frozen_protocol.json", frozen)
    print(f"Froze E13: {root / 'frozen_protocol.json'}", flush=True)
    return frozen


def verify(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    frozen = _read(_root(p) / "frozen_protocol.json")
    if frozen["protocol"] != p or frozen["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("Frozen protocol changed")
    for file, expected in frozen["file_sha256"].items():
        if _sha256(Path(file)) != expected:
            raise ValueError(f"Frozen input changed: {file}")
    if frozen["packages"] != {name: importlib.metadata.version(name) for name in PACKAGES}:
        raise ValueError("Dependency versions changed")
    return frozen


def _perturb(
    x: np.ndarray,
    mask: np.ndarray,
    spec: dict[str, Any],
    rng: np.random.Generator,
    sigma_base: float,
    clip_min: float | None,
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    mask_bool = np.asarray(mask, dtype=bool)
    kind = spec["kind"]
    if kind == "none":
        return out
    for index in range(len(out)):
        valid = mask_bool[index]
        values = out[index, valid].astype(np.float64)
        if kind == "noise":
            values = values + rng.normal(0.0, float(spec["sigma_fraction"]) * sigma_base, size=len(values))
            if clip_min is not None:
                values = np.maximum(values, float(clip_min))
        elif kind == "quantization":
            step = float(spec["step_w"])
            values = np.round(values / step) * step
        else:
            raise ValueError(f"Unknown perturbation kind: {kind}")
        out[index, valid] = values.astype(np.float32)
    return out


def _normalize_pad(values: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    values = np.asarray(values, dtype=np.float64)
    if len(values) > 0:
        mean = float(values.mean())
        std = float(values.std())
        if std >= 1e-8:
            out[: len(values)] = ((values - mean) / std).astype(np.float32)
    return out


def _diff_pad(values: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    differences = np.diff(np.asarray(values, dtype=np.float64))
    out = np.zeros(length, dtype=np.float32)
    mask = np.zeros(length, dtype=bool)
    if len(differences) > 0:
        mean = float(differences.mean())
        std = float(differences.std())
        if std >= 1e-8:
            differences = (differences - mean) / std
        count = min(len(differences), length)
        out[:count] = differences[:count].astype(np.float32)
        mask[:count] = True
    return out, mask


def _moment_embed(
    model: Any, x: np.ndarray, mask: np.ndarray, device: torch.device, batch_size: int = 128
) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start : start + batch_size, None, :], dtype=torch.float32, device=device)
            batch_mask = torch.as_tensor(mask[start : start + batch_size], dtype=torch.float32, device=device)
            features = forward_features(model, batch_x, batch_mask).float().cpu().numpy()
            if not np.isfinite(features).all():
                raise FloatingPointError("Nonfinite MOMENT features")
            chunks.append(features)
    return np.concatenate(chunks, axis=0)


def _tcn_prob(model: Any, x: np.ndarray, mask: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start : start + batch_size, None, :], dtype=torch.float32, device=device)
            batch_mask = torch.as_tensor(mask[start : start + batch_size], dtype=torch.float32, device=device)
            probabilities = model(batch_x, batch_mask).softmax(-1).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("Nonfinite TCN probabilities")
            chunks.append(probabilities)
    return np.concatenate(chunks, axis=0)


def _load_tcn(path: Path, device: torch.device) -> Any:
    model = pilot.build_model("legacy_masked50").to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _meta_model(oof: np.ndarray, y: np.ndarray, seed: int) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    grid = [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    selected, cv_results = run_e08._cv_select(
        lambda params: LogisticRegression(C=params["C"], max_iter=5000, random_state=seed),
        grid,
        oof,
        y,
        folds=3,
        seed=seed,
    )
    model = LogisticRegression(C=selected["C"], max_iter=5000, random_state=seed).fit(oof, y)
    return model, selected, cv_results


def _read_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = [row["sample_id"] for row in rows]
    y = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in rows],
        dtype=np.float64,
    )
    return ids, y, probabilities, np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)


def _align(reference_ids: list[str], ids: list[str], probabilities: np.ndarray) -> np.ndarray:
    mapping = {sample_id: index for index, sample_id in enumerate(ids)}
    if set(reference_ids) != set(ids):
        raise ValueError("Validation ID sets differ")
    return probabilities[[mapping[sample_id] for sample_id in reference_ids]]


def _write_predictions(
    root: Path,
    tag: str,
    perturbation: str,
    method: str,
    ids: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
) -> str:
    directory = root / "conditions" / tag / perturbation / method
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "split", "y_true", "y_pred", "score_0", "score_1", "score_2"])
        for sample_id, truth, prediction, probability in zip(ids, y, pred, scores, strict=True):
            writer.writerow(
                [sample_id, "validation", int(truth), int(prediction), *[f"{value:.12f}" for value in probability]]
            )
    return str(directory / "predictions.csv")


def run_seed(protocol_path: Path, seed: int, moments: Any) -> dict[str, Any]:
    frozen = verify(protocol_path)
    p = frozen["protocol"]
    if seed not in p["seeds"]:
        raise ValueError("Seed not in protocol")
    root = _root(p) / f"seed{seed}"
    if (root / "seed_summary.json").exists():
        return _read(root / "seed_summary.json")
    device = torch.device("cuda")
    config = with_random_seed(load_config(Path(p["config"])), seed)
    raw = load_dataset(replace(config.data, normalize="none"))
    if not np.array_equal(raw.mask_val, raw.mask_val.astype(bool)):
        raise ValueError("Unexpected mask values")
    with np.load(Path(p["e08_suite"]) / f"seed{seed}" / "features.npz", allow_pickle=False) as archive:
        dyn_train = archive["train_features"]
        dyn_val_clean = archive["validation_features"]
        if not np.array_equal(archive["train_ids"].astype(str), raw.ids_train.astype(str)):
            raise ValueError("E08 train IDs mismatch")
        if not np.array_equal(archive["validation_ids"].astype(str), raw.ids_val.astype(str)):
            raise ValueError("E08 validation IDs mismatch")
    moment_train, _ = run_e08._load_moment_split(
        Path(p["e01_cache"]), seed=seed, split="train", ids=raw.ids_train
    )
    moment_val_clean, _ = run_e08._load_moment_split(
        Path(p["e01_cache"]), seed=seed, split="validation", ids=raw.ids_val
    )
    diff_train, _ = run_e08._load_moment_split(
        Path(p["e09_cache"]), seed=seed, split="train", ids=raw.ids_train
    )
    diff_val_clean, _ = run_e08._load_moment_split(
        Path(p["e09_cache"]), seed=seed, split="validation", ids=raw.ids_val
    )
    train_values = raw.x_train[raw.mask_train.astype(bool)]
    sigma_base = float(np.sqrt(np.mean(np.square(train_values, dtype=np.float64))))
    seed_results = _read(Path(p["e08_suite"]) / f"seed{seed}" / "seed_results.json")
    hgb_params_by_tag = {
        record["tag"]: record["selected_params"]
        for record in seed_results["conditions"]
        if record["method"] == "dyn_hgb"
    }
    indices_by_tag = {
        tag: np.asarray(
            __import__("tcn_moment.data", fromlist=["stratified_train_subset_indices"]).stratified_train_subset_indices(
                raw.y_train, raw.ids_train, fraction, seed
            ),
            dtype=np.int64,
        )
        for tag, fraction in (("10p", 0.1), ("40p", 0.4))
        if fraction in p["fractions"]
    }
    hgb_by_tag = {}
    md_model_by_tag = {}
    metas: dict[str, dict[str, Any]] = {}
    for tag, indices in indices_by_tag.items():
        hgb_by_tag[tag] = run_e08._hgb_factory(hgb_params_by_tag[tag], seed).fit(
            dyn_train[indices], raw.y_train[indices]
        )
        md_model_by_tag[tag] = run_e08._logit_factory({"C": 1.0}, seed).fit(
            np.hstack([moment_train[indices], diff_train[indices]]), raw.y_train[indices]
        )
        with np.load(Path(p["e12_suites"][tag]) / f"seed{seed}" / f"oof_{tag}.npz", allow_pickle=False) as archive:
            oof_dyn = archive["oof_dyn"]
            oof_md = archive["oof_md"]
            oof_tcn = archive["oof_tcn"]
            y_oof = archive["y_train"]
        metas[tag] = {}
        for name, matrix in (
            ("tcn_dyn", np.hstack([oof_tcn, oof_dyn])),
            ("dyn_md", np.hstack([oof_dyn, oof_md])),
            ("full", np.hstack([oof_dyn, oof_md, oof_tcn])),
        ):
            model, selected, cv_results = _meta_model(matrix, y_oof, seed)
            metas[tag][name] = {"model": model, "selected": selected, "cv_results": cv_results}

    class_names = [str(value) for value in raw.label_encoder.classes_]
    battery_index = class_index(class_names, BATTERY_LABEL)
    tcn_models = {
        tag: _load_tcn(Path(p["e12_suites"][tag]) / f"seed{seed}" / "models" / tag / "full.pt", device)
        for tag in indices_by_tag
    }
    clean_tcn_by_tag = {}
    for tag in indices_by_tag:
        ids, y, probs, _ = _read_predictions(
            Path(p["e12_suites"][tag]) / f"seed{seed}" / "conditions" / tag / "tcn_full" / "predictions.csv"
        )
        if not np.array_equal(y, raw.y_val):
            raise ValueError("Clean TCN labels differ")
        clean_tcn_by_tag[tag] = _align(raw.ids_val.astype(str).tolist(), ids, probs)

    conditions: list[dict[str, Any]] = []
    for perturbation, spec in p["perturbations"].items():
        started = time.perf_counter()
        if spec["kind"] == "none":
            x_pert = raw.x_val
            moment_val = moment_val_clean
            diff_val = diff_val_clean
            dyn_val = dyn_val_clean
        else:
            rng = np.random.default_rng(seed * 1000 + len(conditions))
            x_pert = _perturb(
                raw.x_val,
                raw.mask_val,
                spec,
                rng,
                sigma_base,
                p.get("noise_clip_min"),
            )
            x_norm = np.zeros_like(x_pert)
            for index in range(len(x_pert)):
                valid = x_pert[index, raw.mask_val[index].astype(bool)]
                x_norm[index] = _normalize_pad(valid, x_pert.shape[1])
            moment_val = _moment_embed(moments, x_norm, raw.mask_val.astype(bool), device)
            diff_rows = []
            diff_masks = []
            for index in range(len(x_pert)):
                values = x_pert[index, raw.mask_val[index].astype(bool)]
                row, row_mask = _diff_pad(values, x_pert.shape[1])
                diff_rows.append(row)
                diff_masks.append(row_mask)
            diff_matrix = np.stack(diff_rows)
            diff_mask = np.stack(diff_masks)
            diff_val = _moment_embed(moments, diff_matrix, diff_mask, device)
            dyn_val = extract_dynamics_features(x_pert.astype(np.float32), raw.mask_val.astype(bool))
        for tag in indices_by_tag:
            dyn_prob = hgb_by_tag[tag].predict_proba(dyn_val)
            md_prob = md_model_by_tag[tag].predict_proba(np.hstack([moment_val, diff_val]))
            if spec["kind"] == "none":
                tcn_prob = clean_tcn_by_tag[tag]
            else:
                tcn_prob = _tcn_prob(tcn_models[tag], x_pert.astype(np.float32), raw.mask_val, device)
            base = {
                "dyn_hgb": dyn_prob,
                "md_logit": md_prob,
                "tcn": tcn_prob,
            }
            base["tcn_dyn"] = metas[tag]["tcn_dyn"]["model"].predict_proba(np.hstack([tcn_prob, dyn_prob]))
            base["dyn_md"] = metas[tag]["dyn_md"]["model"].predict_proba(np.hstack([dyn_prob, md_prob]))
            base["full"] = metas[tag]["full"]["model"].predict_proba(np.hstack([dyn_prob, md_prob, tcn_prob]))
            for method in METHODS:
                probabilities = base[method]
                prediction = probabilities.argmax(1)
                metrics = run_e08._evaluate(raw.y_val, prediction, probabilities, class_names, battery_index)
                prediction_path = _write_predictions(
                    root, tag, perturbation, method, raw.ids_val, raw.y_val, prediction, probabilities
                )
                record = {
                    "seed": seed,
                    "tag": tag,
                    "perturbation": perturbation,
                    "perturbation_spec": spec,
                    "sigma_base": sigma_base,
                    "method": method,
                    "n_validation": len(raw.y_val),
                    "clean_training": True,
                    "test_evaluated": False,
                    "validation": metrics,
                    "predictions": prediction_path,
                }
                conditions.append(record)
                atomic_write_json(
                    root / "conditions" / tag / perturbation / method / "metrics.json", record
                )
                print(
                    f"E13 seed={seed} {tag} {perturbation} {method}: "
                    f"{100*metrics['macro_f1']:.2f} battery "
                    f"{100*metrics['battery']['recall']:.2f}/{100*metrics['battery']['false_positive_rate']:.2f}",
                    flush=True,
                )
        print(f"E13 seed={seed} {perturbation} done in {time.perf_counter()-started:.1f}s", flush=True)
    summary = {
        "seed": seed,
        "fractions": list(indices_by_tag),
        "sigma_base": sigma_base,
        "conditions": conditions,
        "test_evaluated": False,
        "clean_training": True,
    }
    atomic_write_json(root / "seed_summary.json", summary)
    return summary


def _aggregate(summaries: list[dict[str, Any]], p: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tag in [t for t in TAGS if t in {row["tag"] for s in summaries for row in s["conditions"]}]:
        for method in METHODS:
            for perturbation in p["perturbations"]:
                values = [
                    row["validation"]["macro_f1"]
                    for s in summaries
                    for row in s["conditions"]
                    if row["tag"] == tag and row["method"] == method and row["perturbation"] == perturbation
                ]
                recalls = [
                    row["validation"]["battery"]["recall"]
                    for s in summaries
                    for row in s["conditions"]
                    if row["tag"] == tag and row["method"] == method and row["perturbation"] == perturbation
                ]
                fprs = [
                    row["validation"]["battery"]["false_positive_rate"]
                    for s in summaries
                    for row in s["conditions"]
                    if row["tag"] == tag and row["method"] == method and row["perturbation"] == perturbation
                ]
                clean_values = [
                    row["validation"]["macro_f1"]
                    for s in summaries
                    for row in s["conditions"]
                    if row["tag"] == tag and row["method"] == method and row["perturbation"] == "clean"
                ]
                deltas = [100 * (value - clean) for value, clean in zip(values, clean_values, strict=True)]
                rows.append(
                    {
                        "tag": tag,
                        "method": method,
                        "perturbation": perturbation,
                        "n": len(values),
                        "mean": 100 * float(np.mean(values)),
                        "sample_sd": 100 * float(np.std(values, ddof=1)),
                        "per_seed": [100 * value for value in values],
                        "delta_mean": float(np.mean(deltas)),
                        "delta_sd": float(np.std(deltas, ddof=1)),
                        "worse_seeds": int(sum(delta < 0 for delta in deltas)),
                        "battery_recall_mean": 100 * float(np.mean(recalls)),
                        "battery_fpr_mean": 100 * float(np.mean(fprs)),
                    }
                )
    return rows


def _markdown(rows: list[dict[str, Any]], audit: dict[str, Any], p: dict[str, Any], sigma_base: float) -> str:
    lookup = {(row["tag"], row["method"], row["perturbation"]): row for row in rows}
    def label(name: str) -> str:
        spec = p["perturbations"][name]
        if spec["kind"] == "none":
            return "Clean"
        if spec["kind"] == "noise":
            return f"Noise {100*float(spec['sigma_fraction']):.2f}%"
        return f"Quant {float(spec['step_w']):.0f} W"

    labels = {name: label(name) for name in p["perturbations"]}
    noise_names = sorted(
        [name for name, spec in p["perturbations"].items() if spec["kind"] == "noise"],
        key=lambda name: float(p["perturbations"][name]["sigma_fraction"]),
    )
    quant_names = sorted(
        [name for name, spec in p["perturbations"].items() if spec["kind"] == "quantization"],
        key=lambda name: float(p["perturbations"][name]["step_w"]),
    )
    worst_noise = noise_names[-1] if noise_names else "clean"
    worst_quant = quant_names[-1] if quant_names else "clean"
    noise_levels = sorted(
        float(spec["sigma_fraction"])
        for spec in p["perturbations"].values()
        if spec["kind"] == "noise"
    )
    quant_steps = sorted(
        float(spec["step_w"])
        for spec in p["perturbations"].values()
        if spec["kind"] == "quantization"
    )
    noise_text = "/".join(f"{100*level:g}%" for level in noise_levels)
    quant_text = "/".join(f"{step:g}" for step in quant_steps)
    lines = [
        "# E13：噪声与量化鲁棒性（clean 训练，perturbed validation）",
        "",
        f"**状态：{audit['conditions']} 条件 / {audit['rows']:,} 行；仅 validation，无 test。**",
        "",
        f"训练集有效观测 RMS = {sigma_base:.2f} W；噪声 sigma = {noise_text} × 该 RMS，并在 0 W 截断；"
        f"量化步长 {quant_text} W（原始数据最小非零步长 1 W）。",
        "",
        "模型只在 clean 训练集上训练；扰动只作用于 validation。DYN、MOMENT、ΔP 和 TCN 输入都从扰动后的原始序列重新计算。",
        "",
        "## Macro-F1 %（40% 标签）",
        "",
        "| 方法 | " + " | ".join(labels[name] for name in p["perturbations"]) + " |",
        "|---|" + "---:|" * len(p["perturbations"]),
    ]
    for method in METHODS:
        cells = []
        for name in p["perturbations"]:
            row = lookup[("40p", method, name)]
            cells.append(f"{row['mean']:.2f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "## Macro-F1 %（10% 标签）", "", "| 方法 | " + " | ".join(labels[name] for name in p["perturbations"]) + " |", "|---|" + "---:|" * len(p["perturbations"])]
    for method in METHODS:
        cells = []
        for name in p["perturbations"]:
            row = lookup[("10p", method, name)]
            cells.append(f"{row['mean']:.2f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "## 相对 clean 的下降（pp；负值为下降，四 seed 均值 ± SD）", "",
              f"| 预算 | 方法 | {labels[worst_noise]} | {labels[worst_quant]} |", "|---|---|---:|---:|"]
    for tag in ("40p", "10p"):
        for method in METHODS:
            noise = lookup[(tag, method, worst_noise)]
            quant = lookup[(tag, method, worst_quant)]
            lines.append(
                f"| {tag} | {method} | {noise['delta_mean']:+.2f} ± {noise['delta_sd']:.2f} | "
                f"{quant['delta_mean']:+.2f} ± {quant['delta_sd']:.2f} |"
            )
    lines += ["", "## 电池异常（40%，默认 argmax）", "",
              "| 扰动 | full Recall/FPR | TCN+DYN Recall/FPR | TCN Recall/FPR |",
              "|---|---:|---:|---:|"]
    for name in p["perturbations"]:
        full = lookup[("40p", "full", name)]
        td = lookup[("40p", "tcn_dyn", name)]
        tcn = lookup[("40p", "tcn", name)]
        lines.append(
            f"| {labels[name]} | {full['battery_recall_mean']:.2f}/{full['battery_fpr_mean']:.2f} | "
            f"{td['battery_recall_mean']:.2f}/{td['battery_fpr_mean']:.2f} | "
            f"{tcn['battery_recall_mean']:.2f}/{tcn['battery_fpr_mean']:.2f} |"
        )
    lines += [
        "",
        "## 审计",
        "",
        f"- {audit['conditions']} 条件 / {audit['rows']} 行；Macro-F1 复算最大误差 {audit['max_macro_f1_error']:.3g}；0 test。",
        "",
        "## 边界",
        "",
        "- 只在 clean 数据上训练，没有噪声/量化增广；测的是推理鲁棒性，不是鲁棒训练后的上限。",
        "- 噪声/量化模拟采集与记录条件变化，标签保持不变；不是物理状态变化或故障演化。",
        "- 噪声是全局 RMS 比例的加性高斯，0 W 截断；量化步长是绝对瓦数。其他噪声类型/缺失块/前缀未覆盖。",
        "- 四 seed split 重叠、validation 反复使用；无 test、无外部数据；不是部署保证。",
    ]
    return "\n".join(lines)


def _audit(summaries: list[dict[str, Any]], p: dict[str, Any]) -> dict[str, Any]:
    count = 0
    total = 0
    maximum = 0.0
    for summary in summaries:
        seed = summary["seed"]
        for condition in summary["conditions"]:
            path = Path(condition["predictions"])
            with path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            if len(rows) != p["validation_count"] or any(row["split"] != "validation" for row in rows):
                raise ValueError(f"Bad prediction rows: {path}")
            y = np.asarray([int(row["y_true"]) for row in rows])
            pred = np.asarray([int(row["y_pred"]) for row in rows])
            maximum = max(maximum, abs(float(f1_score(y, pred, average="macro")) - condition["validation"]["macro_f1"]))
            total += len(rows)
            count += 1
    return {"conditions": count, "rows": total, "max_macro_f1_error": maximum, "test_rows": 0}


def run(protocol_path: Path) -> None:
    frozen = freeze(protocol_path)
    p = frozen["protocol"]
    if not torch.cuda.is_available():
        raise RuntimeError("E13 requires CUDA")
    root = _root(p)
    if (root / "summary.json").exists():
        print(f"Already complete: {root}", flush=True)
        return
    device = torch.device("cuda")
    config = load_config(Path(p["config"]))
    torch_mod, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()
    moments = build_model(config, MOMENTPipeline, 3)
    set_num_classes(moments, 3)
    moments.init()
    moments.to(device)
    moments.eval()
    for parameter in moments.parameters():
        parameter.requires_grad = False
    summaries = [run_seed(protocol_path, seed, moments) for seed in p["seeds"]]
    del moments
    torch.cuda.empty_cache()
    rows = _aggregate(summaries, p)
    audit = _audit(summaries, p)
    sigma_base = float(np.mean([summary["sigma_base"] for summary in summaries]))
    summary = {
        "test_evaluated": False,
        "clean_training": True,
        "sigma_base_mean": sigma_base,
        "conditions": len(summaries) * len(p["fractions"]) * len(p["perturbations"]) * len(METHODS),
        "aggregate": rows,
        "audit": audit,
        "protocol": p,
    }
    atomic_write_json(root / "summary.json", summary)
    markdown = _markdown(rows, audit, p, sigma_base)
    (root / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/experiments/improvements/e13_robustness_protocol.json")
    parser.add_argument("--action", choices=["freeze", "verify", "run", "report"], default="run")
    args = parser.parse_args()
    path = Path(args.protocol)
    if args.action == "freeze":
        freeze(path)
    elif args.action == "verify":
        verify(path)
    elif args.action == "report":
        p = _load(path)
        summary = _read(_root(p) / "summary.json")
        markdown = _markdown(summary["aggregate"], summary["audit"], p, summary["sigma_base_mean"])
        (_root(p) / "summary.md").write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
    else:
        run(path)


if __name__ == "__main__":
    main()
