"""TODO1102-2C: noise robustness of the random-convolution-kernel baselines.

`todo1102.md` §2 asks for a "when to use which" decision table spanning label
budget x noise tolerance x cost.  Our own branches have E13 / 4a / 4b numbers for
the noise column, but MultiRocket and MiniRocket had none, so that column read
"untested".  This script fills the gap with the same perturbation protocol used by
E13 so the numbers are comparable:

* noise sigma is a fraction of the training-set RMS over valid observations, added
  to valid observations only and clipped at zero;
* features are recomputed from the perturbed sequence through the frozen
  preprocessing (per-series z-score + linear interpolation to 256 points);
* the classifier is the frozen E14/E15 configuration (RidgeClassifierCV over the
  same alpha grid, training-only CV);
* an optional noisy copy of the labelled rows tests whether augmentation helps
  random kernels the way it helped the DYN/fusion systems;
* validation only, no test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path("vendor")))
import minirocket.minirocket as minirocket  # noqa: E402
import multirocket.multirocket as multirocket  # noqa: E402

from tcn_moment.config import load_config, replace  # noqa: E402
from tcn_moment.data import load_dataset, stratified_train_subset_indices  # noqa: E402

SEEDS = (42, 44, 45, 46)
BUDGETS = (0.1, 0.4)
EVAL_LEVELS = (0.0, 0.001, 0.0025, 0.005, 0.01)
AUG_LEVEL = 0.005
LENGTH = 256
NUM_FEATURES = 2000
N_FEATURES_PER_KERNEL = 4
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
RIDGE_CV = 3
DEFAULT_OUTPUT = Path("artifacts/analysis/todo1102_20261102/rocket_noise")
FIELDNAMES = [
    "seed",
    "tag",
    "method",
    "arm",
    "noise_fraction",
    "n_fit",
    "seconds",
    "sigma_base",
    "macro_f1",
    "class0_recall",
    "class0_fpr",
    "class1_recall",
    "class1_fpr",
    "class2_recall",
    "class2_fpr",
]


def _interp_z(x: np.ndarray, mask: np.ndarray, length: int) -> np.ndarray:
    """Per-series z-score then linear interpolation to ``length`` points."""
    output = np.zeros((len(x), length), dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    for index in range(len(x)):
        values = np.asarray(x[index][mask_bool[index]], dtype=np.float64)
        if len(values) < 2:
            continue
        std = float(values.std())
        normalized = (values - values.mean()) / std if std >= 1e-8 else np.zeros_like(values)
        output[index] = np.interp(
            np.linspace(0.0, 1.0, length),
            np.linspace(0.0, 1.0, len(normalized)),
            normalized,
        )
    return output


def _noise(
    values: np.ndarray,
    mask: np.ndarray,
    sigma_fraction: float,
    sigma_base: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if sigma_fraction <= 0.0:
        return out
    mask_bool = np.asarray(mask, dtype=bool)
    for index in range(len(out)):
        valid = mask_bool[index]
        draw = out[index, valid].astype(np.float64)
        draw = draw + rng.normal(0.0, sigma_fraction * sigma_base, size=len(draw))
        out[index, valid] = np.maximum(draw, 0.0).astype(np.float32)
    return out


def _rng(seed: int, stream: int, level_bp: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream, level_bp, 0])


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for truth, predicted in zip(y, prediction, strict=True):
        matrix[int(truth), int(predicted)] += 1
    result = {"macro_f1": float(f1_score(y, prediction, average="macro"))}
    for label in range(3):
        tp = float(matrix[label, label])
        fn = float(matrix[label].sum() - tp)
        fp = float(matrix[:, label].sum() - tp)
        tn = float(matrix.sum() - tp - fn - fp)
        result[f"class{label}_recall"] = tp / (tp + fn) if tp + fn else 0.0
        result[f"class{label}_fpr"] = fp / (fp + tn) if fp + tn else 0.0
    return result


def _fit_multirocket(train: np.ndarray, seed: int) -> tuple[Any, Any]:
    np.random.seed(seed)
    base = multirocket.fit(train, num_features=NUM_FEATURES)
    diff = multirocket.fit(np.diff(train, axis=1), num_features=NUM_FEATURES)
    return base, diff


def _transform_multirocket(model: tuple[Any, Any], values: np.ndarray) -> np.ndarray:
    base, diff = model
    return np.nan_to_num(
        multirocket.transform(
            values, np.diff(values, axis=1), base, diff, N_FEATURES_PER_KERNEL
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
        help="Seeds handled by this process; run one process per seed for wall-clock speed.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip fitting; aggregate existing per-seed CSVs into summary.json/md.",
    )
    parser.add_argument("--methods", nargs="+", default=["multirocket", "minirocket"])
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument(
        "--aug-budgets",
        nargs="+",
        type=float,
        default=[0.1],
        help="Budgets that also get an augmented arm (40%% doubles ~10k rows and is slow).",
    )
    parser.add_argument(
        "--mini-augment",
        dest="skip_minirocket_aug",
        action="store_false",
        default=True,
        help="Also fit MiniRocket on the augmented rows (default: clean only, it is the weak baseline).",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "settings.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": "data/raw/最新多.csv",
                "length": LENGTH,
                "num_features": NUM_FEATURES,
                "n_features_per_kernel": N_FEATURES_PER_KERNEL,
                "ridge_alphas": RIDGE_ALPHAS.tolist(),
                "ridge_cv": RIDGE_CV,
                "eval_levels": list(EVAL_LEVELS),
                "aug_level": AUG_LEVEL,
                "budgets": list(BUDGETS),
                "seeds": args.seeds,
                "methods": args.methods,
                "augment": args.augment,
                "aug_budgets": args.aug_budgets,
                "noise_scale": "train_valid_rms",
                "split": "artifacts/splits/unified_split.json",
                "test_evaluated": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if args.merge_only:
        _summarise(args.output, args.methods)
        return

    config = load_config(Path("configs/moment.yaml"))
    raw = load_dataset(replace(config.data, normalize="none"))
    sigma_base = float(
        np.sqrt(np.mean(np.square(raw.x_train[raw.mask_train.astype(bool)], dtype=np.float64)))
    )
    print(f"sigma_base={sigma_base:.3f}", flush=True)
    x_val_clean = _interp_z(raw.x_val, raw.mask_val, LENGTH)

    for seed in args.seeds:
        rows = run_one_seed(raw, sigma_base, x_val_clean, seed, args)
        path = args.output / f"conditions_seed{seed}.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in FIELDNAMES})
        print(f"seed {seed}: wrote {path}", flush=True)


def run_one_seed(
    raw: Any, sigma_base: float, x_val_clean: np.ndarray, seed: int, args: argparse.Namespace
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation_by_level: dict[float, np.ndarray] = {0.0: x_val_clean}
    for level in EVAL_LEVELS[1:]:
        level_bp = int(round(level * 1e4))
        noisy = _noise(raw.x_val, raw.mask_val, level, sigma_base, _rng(seed, 2, level_bp))
        validation_by_level[level] = _interp_z(noisy, raw.mask_val, LENGTH)

    for fraction in BUDGETS:
        tag = f"{int(round(fraction * 100))}p"
        indices = stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
        x_train = _interp_z(raw.x_train[indices], raw.mask_train[indices], LENGTH)
        arms: dict[str, np.ndarray] = {"clean": x_train}
        if args.augment and fraction in args.aug_budgets:
            level_bp = int(round(AUG_LEVEL * 1e4))
            noisy = _noise(
                raw.x_train[indices],
                raw.mask_train[indices],
                AUG_LEVEL,
                sigma_base,
                _rng(seed, 1, level_bp),
            )
            arms["aug"] = np.vstack(
                [x_train, _interp_z(noisy, raw.mask_train[indices], LENGTH)]
            )

        for method in args.methods:
            if method == "minirocket" and args.skip_minirocket_aug:
                arms_to_run = {"clean": arms["clean"]}
            else:
                arms_to_run = arms
            for arm, features in arms_to_run.items():
                started = time.perf_counter()
                labels = (
                    np.concatenate([raw.y_train[indices], raw.y_train[indices]])
                    if arm == "aug"
                    else raw.y_train[indices]
                )
                if method == "multirocket":
                    model = _fit_multirocket(features, seed * 100 + 2)
                    train_transform = _transform_multirocket(model, features)
                    transforms = {
                        level: _transform_multirocket(model, values)
                        for level, values in validation_by_level.items()
                    }
                elif method == "minirocket":
                    np.random.seed(seed * 100 + 3)
                    dense = features.astype(np.float32)
                    parameters = minirocket.fit(dense, num_features=NUM_FEATURES)
                    train_transform = np.nan_to_num(minirocket.transform(dense, parameters))
                    transforms = {
                        level: np.nan_to_num(
                            minirocket.transform(values.astype(np.float32), parameters)
                        )
                        for level, values in validation_by_level.items()
                    }
                else:
                    raise ValueError(f"Unknown method: {method}")
                classifier = make_pipeline(
                    StandardScaler(),
                    RidgeClassifierCV(alphas=RIDGE_ALPHAS, cv=RIDGE_CV),
                )
                classifier.fit(train_transform, labels)
                for level in EVAL_LEVELS:
                    scores = _softmax(classifier.decision_function(transforms[level]))
                    metrics = _metrics(raw.y_val, scores.argmax(1))
                    rows.append(
                        {
                            "seed": seed,
                            "tag": tag,
                            "method": method,
                            "arm": arm,
                            "noise_fraction": level,
                            "n_fit": int(len(labels)),
                            "seconds": time.perf_counter() - started,
                            "sigma_base": sigma_base,
                            **metrics,
                        }
                    )
                clean = [
                    row["macro_f1"]
                    for row in rows
                    if row["tag"] == tag
                    and row["method"] == method
                    and row["arm"] == arm
                    and row["noise_fraction"] == 0.0
                ][-1]
                print(
                    f"seed {seed} {tag} {method} {arm}: clean={100 * clean:.2f}% "
                    f"({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )
    return rows


def _summarise(output: Path, methods: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("conditions_seed*.csv")):
        with path.open("r", encoding="utf-8", newline="") as file:
            for raw_row in csv.DictReader(file):
                rows.append(
                    {
                        key: raw_row[key] if key in ("method", "arm", "tag") else float(raw_row[key])
                        for key in FIELDNAMES
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No per-seed CSVs under {output}")
    sigma_base = rows[0]["sigma_base"]
    aggregates: list[dict[str, Any]] = []
    for method in methods:
        for tag in ("10p", "40p"):
            for arm in ("clean", "aug"):
                for level in EVAL_LEVELS:
                    selected = [
                        row
                        for row in rows
                        if row["method"] == method
                        and row["tag"] == tag
                        and row["arm"] == arm
                        and row["noise_fraction"] == level
                    ]
                    if not selected:
                        continue
                    aggregates.append(
                        {
                            "method": method,
                            "tag": tag,
                            "arm": arm,
                            "noise_fraction": level,
                            "n_seeds": len(selected),
                            "macro_f1_mean": float(
                                np.mean([row["macro_f1"] for row in selected]) * 100
                            ),
                            "macro_f1_std": float(
                                np.std([row["macro_f1"] for row in selected]) * 100
                            ),
                            "class1_recall_mean": float(
                                np.mean([row["class1_recall"] for row in selected]) * 100
                            ),
                            "class1_fpr_mean": float(
                                np.mean([row["class1_fpr"] for row in selected]) * 100
                            ),
                        }
                    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sigma_base": sigma_base,
                "aggregates": aggregates,
                "test_evaluated": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lookup = {
        (row["method"], row["tag"], row["arm"], row["noise_fraction"]): row
        for row in aggregates
    }
    lines = [
        "# TODO1102-2C random-kernel noise robustness",
        "",
        f"训练集有效观测 RMS = {sigma_base:.2f}; 噪声 sigma 取该值的比例，并在 0 处截断。",
        "",
        "| 方法 | 预算 | 训练 | 干净 | 0.1% | 0.25% | 0.5% | 1% |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        for tag in ("10p", "40p"):
            for arm in ("clean", "aug"):
                if (method, tag, arm, 0.0) not in lookup:
                    continue
                cells = [
                    f"{lookup[(method, tag, arm, level)]['macro_f1_mean']:.2f}"
                    if (method, tag, arm, level) in lookup
                    else "—"
                    for level in EVAL_LEVELS
                ]
                lines.append(f"| {method} | {tag} | {arm} | " + " | ".join(cells) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for row in aggregates:
        print(
            f"{row['method']:>12} {row['tag']:>4} {row['arm']:>5} "
            f"{100 * row['noise_fraction']:6.2f}% macro_f1={row['macro_f1_mean']:6.2f} "
            f"battery_recall={row['class1_recall_mean']:6.2f}",
            flush=True,
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
