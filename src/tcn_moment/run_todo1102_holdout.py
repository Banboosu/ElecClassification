"""TODO1102-1A: sealed, cluster-aware hold-out evaluation.

Motivation (from ``todo1102.md`` §1a): the historical test split was already
consumed during development, so headline numbers cannot answer "was the method
selected on the same rows it is scored on?".  This module carves a *sealed* set
of the 35,100 valid rows that is never used for fitting, tuning, early stopping,
threshold selection or temperature scaling, then re-runs the main systems on the
remaining rows and scores the sealed set once per seed and budget.

Design decisions that keep the audit honest:

* the sealed set is built at six-character station-cluster granularity, so rows
  from one deployment cluster never straddle the boundary;
* the sealed set is identical across seeds; only the train/validation partition
  of the remaining rows changes per seed;
* all three branches are refitted from scratch; no cached probability from the
  historical split is reused;
* HGB and MOMENT-logistic hyper-parameters are the ones frozen by E08 on the
  historical split (no tuning on the new rows), and the TCN uses a fixed
  50-epoch cosine schedule without validation-based epoch selection;
* the historical test split is never read.

Sub-commands: ``build-split``, ``extract``, ``run``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from tcn_moment import run_e08
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.config import load_config
from tcn_moment.data import (
    normalize_sequence,
    parse_power_sequence,
    prepare_sequence,
    sequence_mask,
    stratified_train_subset_indices,
)
from tcn_moment.dynamics_features import extract_dynamics_features
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.label_semantics import BATTERY_LABEL, class_index
from tcn_moment.run_e09_diff_stream import _diff_arrays
from tcn_moment.run_e12_tcn_stack import _tcn_predict, _train_tcn
from tcn_moment.train_moment import (
    build_model,
    forward_features,
    require_torch_and_moment,
    set_num_classes,
)

CLASSES = ["0", "1", "2"]
CLUSTER_PREFIX = 6
BATTERY_INDEX = class_index(CLASSES, BATTERY_LABEL)
DEFAULT_ROOT = Path("artifacts/analysis/todo1102_holdout")
DEFAULT_CONFIG = "configs/moment.yaml"
E08_SUITE = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
SEEDS = (42, 44, 45, 46)
FOLDS = 5
EPOCHS = 50


# --------------------------------------------------------------------------- split


def _cluster_of(sample_id: str) -> str:
    return sample_id[:CLUSTER_PREFIX]


def build_split(
    root: Path,
    *,
    sealed_fraction: float,
    sealed_seed: int,
    dataset: Path,
    max_length: int,
    min_length: int,
) -> dict[str, Any]:
    frame = pd.read_csv(dataset, sep=" ", dtype=str)
    frame = frame[frame["InsertedColumn"].isin(CLASSES)]
    lengths = frame["charging_powers_str"].map(lambda value: len(parse_power_sequence(value)))
    frame = frame[(lengths >= min_length) & (lengths <= max_length)]
    ids = frame["charging_station_id"].to_numpy(str)
    labels = frame["InsertedColumn"].to_numpy(str)
    clusters = np.asarray([_cluster_of(value) for value in ids])
    class_totals = {name: int((labels == name).sum()) for name in CLASSES}

    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    per_cluster = np.zeros((len(unique_clusters), len(CLASSES)), dtype=np.int64)
    for index, label in enumerate(CLASSES):
        per_cluster[:, index] = np.bincount(
            inverse[labels == label], minlength=len(unique_clusters)
        )
    generator = np.random.default_rng(sealed_seed)
    order = generator.permutation(len(unique_clusters))
    target = np.array([sealed_fraction * class_totals[name] for name in CLASSES])
    cap = target * 1.25
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    sealed_cluster = np.zeros(len(unique_clusters), dtype=bool)
    for position in order:
        candidate = per_cluster[position]
        if np.any(counts + candidate > cap):
            continue
        sealed_cluster[position] = True
        counts += candidate
        if np.all(counts >= target * 0.9):
            break
    if not np.all(counts >= target * 0.9):
        raise RuntimeError(f"Sealed set missed the class targets: {counts} vs {target}")

    sealed_rows = sealed_cluster[inverse]
    sealed_labels = labels[sealed_rows]
    remaining_ids = ids[~sealed_rows]
    remaining_labels = labels[~sealed_rows]
    remaining_clusters = clusters[~sealed_rows]

    record: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "cluster_prefix": CLUSTER_PREFIX,
        "sealed_fraction_target": sealed_fraction,
        "sealed_seed": sealed_seed,
        "max_length": max_length,
        "min_length": min_length,
        "counts": {
            "universe": int(len(ids)),
            "sealed": int(sealed_rows.sum()),
            "remaining": int(len(remaining_ids)),
        },
        "class_totals": class_totals,
        "sealed_class_counts": {
            name: int((sealed_labels == name).sum()) for name in CLASSES
        },
        "sealed_clusters": int(len(np.unique(clusters[sealed_rows]))),
        "remaining_clusters": int(len(np.unique(remaining_clusters))),
        "sealed_ids": [str(value) for value in ids[sealed_rows]],
        "remaining_ids": [str(value) for value in remaining_ids],
        "test_evaluated": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "sealed_split.json", record)

    partitions: dict[str, Any] = {}
    for partition_seed in SEEDS:
        rng = np.random.default_rng(partition_seed)
        unique, inverse_remaining = np.unique(remaining_clusters, return_inverse=True)
        per_cluster_class = np.zeros((len(unique), len(CLASSES)), dtype=np.int64)
        for index, label in enumerate(CLASSES):
            per_cluster_class[:, index] = np.bincount(
                inverse_remaining[remaining_labels == label], minlength=len(unique)
            )
        target_val = np.array(
            [0.2 * int((remaining_labels == name).sum()) for name in CLASSES]
        )
        counts_val = np.zeros(len(CLASSES), dtype=np.int64)
        val_cluster = np.zeros(len(unique), dtype=bool)
        for position in rng.permutation(len(unique)):
            candidate = per_cluster_class[position]
            if np.any(counts_val + candidate > target_val * 1.2):
                continue
            val_cluster[position] = True
            counts_val += candidate
            if np.all(counts_val >= target_val * 0.85):
                break
        rows_val = val_cluster[inverse_remaining]
        partition = {
            "seed": partition_seed,
            "sealed": "sealed_split.json",
            "train_count": int((~rows_val).sum()),
            "validation_count": int(rows_val.sum()),
            "train_class_counts": {
                name: int((remaining_labels[~rows_val] == name).sum()) for name in CLASSES
            },
            "validation_class_counts": {
                name: int((remaining_labels[rows_val] == name).sum()) for name in CLASSES
            },
            "train_ids": [str(value) for value in remaining_ids[~rows_val]],
            "validation_ids": [str(value) for value in remaining_ids[rows_val]],
        }
        partitions[str(partition_seed)] = partition
        atomic_write_json(root / f"partition_seed{partition_seed}.json", partition)
    record["partitions"] = {
        key: {
            name: value[name]
            for name in (
                "train_count",
                "validation_count",
                "train_class_counts",
                "validation_class_counts",
            )
        }
        for key, value in partitions.items()
    }
    atomic_write_json(root / "sealed_split.json", record)
    print(
        json.dumps(
            {
                "counts": record["counts"],
                "class_totals": record["class_totals"],
                "sealed_class_counts": record["sealed_class_counts"],
                "sealed_clusters": record["sealed_clusters"],
                "partitions": record["partitions"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return record


# ------------------------------------------------------------------------- loading


def _load_arrays(dataset: Path, ids: list[str], max_length: int) -> dict[str, Any]:
    frame = pd.read_csv(dataset, sep=" ", dtype=str).set_index("charging_station_id")
    rows = frame.loc[ids]
    sequences = [parse_power_sequence(value) for value in rows["charging_powers_str"]]
    lengths = np.asarray([len(value) for value in sequences], dtype=np.int64)
    if (lengths < 18).any() or (lengths > max_length).any():
        raise ValueError("Sequence length outside the allowed range")
    x = np.stack([prepare_sequence(value, max_length, "none") for value in sequences])
    mask = np.stack([sequence_mask(length, max_length) for length in lengths])
    z = np.zeros_like(x)
    for index in range(len(x)):
        valid = mask[index].astype(bool)
        z[index, valid] = normalize_sequence(x[index][valid], "zscore")
    labels = rows["InsertedColumn"].to_numpy(str)
    y = np.asarray([CLASSES.index(value) for value in labels], dtype=np.int64)
    return {
        "x": x,
        "z": z,
        "mask": mask,
        "y": y,
        "ids": np.asarray(ids, dtype=str),
        "lengths": lengths,
    }


def _chunk_dyn(payload: tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    x, mask, max_length = payload
    return extract_dynamics_features(x, mask, max_length=max_length)


def _dyn_features(x: np.ndarray, mask: np.ndarray, max_length: int, jobs: int) -> np.ndarray:
    if jobs <= 1 or len(x) <= 256:
        return extract_dynamics_features(x, mask, max_length=max_length)
    chunk = max(128, len(x) // (jobs * 6))
    payloads = [
        (x[start : start + chunk], mask[start : start + chunk], max_length)
        for start in range(0, len(x), chunk)
    ]
    with Pool(processes=jobs) as pool:
        return np.vstack(pool.map(_chunk_dyn, payloads))


def _moment_features(
    x: np.ndarray, mask: np.ndarray, model: Any, device: Any, batch_size: int = 64
) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start : start + batch_size, None], device=device)
            batch_mask = torch.as_tensor(mask[start : start + batch_size], device=device)
            value = forward_features(model, batch_x, batch_mask).float().cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError("Nonfinite MOMENT features")
            chunks.append(value)
    return np.concatenate(chunks)


def extract(root: Path, seed: int, *, config_path: str, jobs: int) -> None:
    split = json.loads((root / "sealed_split.json").read_text(encoding="utf-8"))
    partition = json.loads((root / f"partition_seed{seed}.json").read_text(encoding="utf-8"))
    dataset = Path(split["dataset"])
    config = load_config(Path(config_path))
    max_length = int(split["max_length"])
    target = root / f"features_seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    splits = {
        "train": partition["train_ids"],
        "validation": partition["validation_ids"],
        "sealed": split["sealed_ids"],
    }
    arrays = {name: _load_arrays(dataset, ids, max_length) for name, ids in splits.items()}
    dyn = {
        name: _dyn_features(value["x"], value["mask"], max_length, jobs)
        for name, value in arrays.items()
    }
    print(f"seed {seed}: DYN done in {time.perf_counter() - started:.0f}s", flush=True)

    _, _, _, _, pipeline = require_torch_and_moment()
    model = build_model(config, pipeline, len(CLASSES))
    set_num_classes(model, len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    moment = {
        name: _moment_features(value["z"], value["mask"], model, device)
        for name, value in arrays.items()
    }
    print(f"seed {seed}: MOMENT P done in {time.perf_counter() - started:.0f}s", flush=True)
    diff = {}
    for name, value in arrays.items():
        diff_x, diff_mask = _diff_arrays(value["x"], value["mask"])
        diff[name] = _moment_features(diff_x, diff_mask, model, device)
    print(f"seed {seed}: MOMENT ΔP done in {time.perf_counter() - started:.0f}s", flush=True)

    for name, value in arrays.items():
        np.savez_compressed(
            target / f"{name}.npz",
            ids=value["ids"].astype(str),
            y=value["y"],
            dyn=dyn[name],
            moment=moment[name],
            diff=diff[name],
        )
    atomic_write_json(
        target / "manifest.json",
        {
            "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": split["dataset_sha256"],
            "config": config_path,
            "max_length": max_length,
            "counts": {name: int(len(value["ids"])) for name, value in arrays.items()},
            "seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "test_evaluated": False,
        },
    )
    print(f"seed {seed}: cached to {target}", flush=True)


# --------------------------------------------------------------------------- run


def _tag(fraction: float) -> str:
    return f"{int(round(fraction * 100))}p"


def _hgb_params(seed: int, tag: str) -> dict[str, Any]:
    record = json.loads((E08_SUITE / f"seed{seed}" / "seed_results.json").read_text(encoding="utf-8"))
    matches = [
        row for row in record["conditions"] if row["method"] == "dyn_hgb" and row["tag"] == tag
    ]
    if len(matches) != 1:
        raise ValueError(f"Missing frozen HGB params for seed={seed} tag={tag}")
    return dict(matches[0]["selected_params"])


def _meta(
    oof: np.ndarray, y: np.ndarray, evaluation: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
    return model.predict(evaluation), model.predict_proba(evaluation), selected


def _write_predictions(
    path: Path, ids: np.ndarray, y: np.ndarray, scores: np.ndarray, prediction: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "y_true", "y_pred", "score_0", "score_1", "score_2"])
        for index in range(len(ids)):
            writer.writerow(
                [
                    str(ids[index]),
                    int(y[index]),
                    int(prediction[index]),
                    f"{scores[index, 0]:.6f}",
                    f"{scores[index, 1]:.6f}",
                    f"{scores[index, 2]:.6f}",
                ]
            )


def run_seed(root: Path, seed: int, *, fractions: list[float]) -> None:
    split = json.loads((root / "sealed_split.json").read_text(encoding="utf-8"))
    partition = json.loads((root / f"partition_seed{seed}.json").read_text(encoding="utf-8"))
    dataset = Path(split["dataset"])
    max_length = int(split["max_length"])
    feature_root = root / f"features_seed{seed}"
    archives = {
        name: np.load(feature_root / f"{name}.npz", allow_pickle=False)
        for name in ("train", "validation", "sealed")
    }
    if not np.array_equal(archives["train"]["ids"].astype(str), np.asarray(partition["train_ids"])):
        raise ValueError("Cached train ids mismatch")
    if not np.array_equal(
        archives["validation"]["ids"].astype(str), np.asarray(partition["validation_ids"])
    ):
        raise ValueError("Cached validation ids mismatch")
    if not np.array_equal(archives["sealed"]["ids"].astype(str), np.asarray(split["sealed_ids"])):
        raise ValueError("Cached sealed ids mismatch")

    raw = {
        "train": _load_arrays(dataset, partition["train_ids"], max_length),
        "validation": _load_arrays(dataset, partition["validation_ids"], max_length),
        "sealed": _load_arrays(dataset, split["sealed_ids"], max_length),
    }
    train, validation, sealed = archives["train"], archives["validation"], archives["sealed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pilot.configure()
    output = root / f"run_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)

    for fraction in fractions:
        tag = _tag(fraction)
        indices = stratified_train_subset_indices(
            train["y"], train["ids"].astype(str), fraction, seed
        )
        y_train = train["y"][indices]
        dyn_train = train["dyn"][indices]
        md_train = np.hstack([train["moment"][indices], train["diff"][indices]])
        x_subset = raw["train"]["x"][indices]
        mask_subset = raw["train"]["mask"][indices]

        params = _hgb_params(seed, tag)
        oof_dyn = np.zeros((len(indices), 3))
        oof_md = np.zeros((len(indices), 3))
        oof_tcn = np.zeros((len(indices), 3))
        splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        models_dir = output / "models" / tag
        models_dir.mkdir(parents=True, exist_ok=True)
        for fold, (fit_index, hold_index) in enumerate(splitter.split(x_subset, y_train)):
            tcn = _train_tcn(
                x_subset[fit_index],
                mask_subset[fit_index],
                y_train[fit_index],
                epochs=EPOCHS,
                seed=seed * 100 + fold,
                device=device,
            )
            torch.save(
                {key: value.cpu() for key, value in tcn.state_dict().items()},
                models_dir / f"fold{fold}.pt",
            )
            oof_tcn[hold_index] = _tcn_predict(
                tcn, x_subset[hold_index], mask_subset[hold_index], device
            )
            hgb = run_e08._hgb_factory(params, seed).fit(dyn_train[fit_index], y_train[fit_index])
            oof_dyn[hold_index] = hgb.predict_proba(dyn_train[hold_index])
            logistic = run_e08._logit_factory({"C": 1.0}, seed).fit(
                md_train[fit_index], y_train[fit_index]
            )
            oof_md[hold_index] = logistic.predict_proba(md_train[hold_index])
            print(f"seed {seed} {tag} fold {fold} done", flush=True)

        tcn_full = _train_tcn(
            x_subset, mask_subset, y_train, epochs=EPOCHS, seed=seed * 100 + 99, device=device
        )
        torch.save(
            {key: value.cpu() for key, value in tcn_full.state_dict().items()},
            models_dir / "full.pt",
        )
        hgb_full = run_e08._hgb_factory(params, seed).fit(dyn_train, y_train)
        logistic_full = run_e08._logit_factory({"C": 1.0}, seed).fit(md_train, y_train)

        oof = {
            "tcn_dyn": np.hstack([oof_dyn, oof_tcn]),
            "full": np.hstack([oof_dyn, oof_md, oof_tcn]),
        }
        evaluation: dict[str, dict[str, Any]] = {}
        for split_name, archive in (("validation", validation), ("sealed", sealed)):
            branch = {
                "dyn": hgb_full.predict_proba(archive["dyn"]),
                "md": logistic_full.predict_proba(np.hstack([archive["moment"], archive["diff"]])),
                "tcn": _tcn_predict(
                    tcn_full, raw[split_name]["x"], raw[split_name]["mask"], device
                ),
            }
            matrices = {
                "dyn": branch["dyn"],
                "md": branch["md"],
                "tcn": branch["tcn"],
                "tcn_dyn": np.hstack([branch["dyn"], branch["tcn"]]),
                "full": np.hstack([branch["dyn"], branch["md"], branch["tcn"]]),
            }
            block: dict[str, Any] = {}
            for method, matrix in matrices.items():
                if method in ("dyn", "md", "tcn"):
                    prediction, scores = matrix.argmax(1), matrix
                else:
                    prediction, scores, _ = _meta(oof[method], y_train, matrix, seed)
                metrics = run_e08._evaluate(archive["y"], prediction, scores, CLASSES, BATTERY_INDEX)
                _write_predictions(
                    output / "predictions" / split_name / tag / f"{method}.csv",
                    archive["ids"],
                    archive["y"],
                    scores,
                    prediction,
                )
                block[method] = metrics
                print(
                    f"seed {seed} {tag} {split_name} {method}: "
                    f"macro-F1={100 * metrics['macro_f1']:.3f} "
                    f"battery recall={100 * metrics['battery']['recall']:.2f}% "
                    f"FPR={100 * metrics['battery']['false_positive_rate']:.2f}%",
                    flush=True,
                )
            evaluation[split_name] = block

        atomic_write_json(
            output / "conditions" / f"{tag}.json",
            {
                "seed": seed,
                "tag": tag,
                "fraction": fraction,
                "n_train": int(len(indices)),
                "n_validation": int(len(validation["y"])),
                "n_sealed": int(len(sealed["y"])),
                "hgb_params": params,
                "epochs": EPOCHS,
                "folds": FOLDS,
                "evaluation": evaluation,
                "test_evaluated": False,
            },
        )
        np.savez_compressed(
            output / f"oof_{tag}.npz",
            indices=indices.astype(np.int64),
            y_train=y_train,
            oof_dyn=oof_dyn,
            oof_md=oof_md,
            oof_tcn=oof_tcn,
        )


def summarise(root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"test_evaluated": False, "by_tag": {}}
    for tag in ("10p", "40p"):
        block: dict[str, Any] = {}
        for split_name in ("validation", "sealed"):
            per_method: dict[str, list[dict[str, float]]] = {}
            for seed in SEEDS:
                path = root / f"run_seed{seed}" / "conditions" / f"{tag}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                for method, metrics in record["evaluation"][split_name].items():
                    per_method.setdefault(method, []).append(
                        {
                            "macro_f1": metrics["macro_f1"],
                            "battery_recall": metrics["battery"]["recall"],
                            "battery_fpr": metrics["battery"]["false_positive_rate"],
                        }
                    )
            block[split_name] = {
                method: {
                    "n": len(values),
                    "macro_f1_mean": float(np.mean([value["macro_f1"] for value in values]) * 100),
                    "macro_f1_std": float(np.std([value["macro_f1"] for value in values]) * 100),
                    "battery_recall_mean": float(
                        np.mean([value["battery_recall"] for value in values]) * 100
                    ),
                    "battery_fpr_mean": float(
                        np.mean([value["battery_fpr"] for value in values]) * 100
                    ),
                }
                for method, values in per_method.items()
            }
        summary["by_tag"][tag] = block
    atomic_write_json(root / "summary.json", summary)
    lines = ["# TODO1102-1A sealed hold-out", ""]
    for tag, block in summary["by_tag"].items():
        lines += [f"## {tag}", "", "| 方法 | 验证集 Macro-F1 | 封存集 Macro-F1 | 封存集电池召回/FPR |", "|---|---:|---:|---:|"]
        for method in ("tcn", "dyn", "md", "tcn_dyn", "full"):
            val = block["validation"].get(method)
            sealed = block["sealed"].get(method)
            if not val or not sealed:
                continue
            lines.append(
                f"| {method} | {val['macro_f1_mean']:.2f} ± {val['macro_f1_std']:.2f} | "
                f"{sealed['macro_f1_mean']:.2f} ± {sealed['macro_f1_std']:.2f} | "
                f"{sealed['battery_recall_mean']:.2f} / {sealed['battery_fpr_mean']:.2f} |"
            )
        lines.append("")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("build-split")
    split_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    split_parser.add_argument("--dataset", type=Path, default=Path("data/raw/最新多.csv"))
    split_parser.add_argument("--sealed-fraction", type=float, default=0.10)
    split_parser.add_argument("--sealed-seed", type=int, default=42)
    split_parser.add_argument("--max-length", type=int, default=1024)
    split_parser.add_argument("--min-length", type=int, default=18)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    extract_parser.add_argument("--seed", type=int, required=True)
    extract_parser.add_argument("--config", default=DEFAULT_CONFIG)
    extract_parser.add_argument("--jobs", type=int, default=32)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.4])

    summary_parser = subparsers.add_parser("summarise")
    summary_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)

    args = parser.parse_args()
    if args.command == "build-split":
        build_split(
            args.root,
            sealed_fraction=args.sealed_fraction,
            sealed_seed=args.sealed_seed,
            dataset=args.dataset,
            max_length=args.max_length,
            min_length=args.min_length,
        )
    elif args.command == "extract":
        extract(args.root, args.seed, config_path=args.config, jobs=args.jobs)
    elif args.command == "run":
        run_seed(args.root, args.seed, fractions=args.fractions)
    else:
        summarise(args.root)


if __name__ == "__main__":
    main()
