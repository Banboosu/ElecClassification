"""TODO1102-4B: fusion-side noise augmentation (0.5% of train RMS).

E13b showed the fusion system collapses under acquisition noise, and 4a showed the
DYN branch can be repaired on the feature side.  This audit asks the next question:
if *all three* branches are trained with one noisy copy of the labelled training
sequences (σ = 0.5% of the training RMS), does the stacked system keep its accuracy
under noise, and what does that cost on clean data?

Both arms are trained and scored inside this script, so the paired difference uses
the same noise draws for the clean-trained and augmented models.  Validation only;
the historical test split is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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
from tcn_moment.dynamics_features import FEATURE_GROUPS, DYNAMICS_FEATURE_NAMES, extract_dynamics_features
from tcn_moment.io_utils import atomic_write_json
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
BATTERY_INDEX = class_index(CLASSES, BATTERY_LABEL)
E08_SUITE = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
SEEDS = (42, 44, 45, 46)
BUDGETS = (0.1, 0.4)
EVAL_LEVELS = (0.0, 0.001, 0.0025, 0.005, 0.01)
DEFAULT_EVAL_LEVELS = EVAL_LEVELS
AUG_LEVEL = 0.005
FOLDS = 5
EPOCHS = 50
DEFAULT_OUTPUT = Path("artifacts/analysis/todo1102_20261102/fusion_augmentation")


def _tag(fraction: float) -> str:
    return f"{int(round(fraction * 100))}p"


def _load(dataset: Path, max_length: int) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(dataset, sep=" ", dtype=str)
    frame = frame[frame["InsertedColumn"].isin(CLASSES)]
    sequences = [parse_power_sequence(value) for value in frame["charging_powers_str"]]
    lengths = np.asarray([len(value) for value in sequences])
    keep = (lengths >= 18) & (lengths <= max_length)
    frame = frame[keep]
    sequences = [value for value, flag in zip(sequences, keep) if flag]
    ids = frame["charging_station_id"].to_numpy(str)
    y = np.asarray([CLASSES.index(value) for value in frame["InsertedColumn"]], dtype=np.int64)
    x = np.stack([prepare_sequence(value, max_length, "none") for value in sequences])
    mask = np.stack([sequence_mask(len(value), max_length) for value in sequences])
    z = np.zeros_like(x)
    for index in range(len(x)):
        valid = mask[index].astype(bool)
        z[index, valid] = normalize_sequence(x[index][valid], "zscore")
    return {"ids": ids, "y": y, "x": x, "mask": mask, "z": z}


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
    valid_mask = np.asarray(mask, dtype=bool)
    for index in range(len(out)):
        valid = valid_mask[index]
        draw = out[index, valid].astype(np.float64)
        draw = draw + rng.normal(0.0, sigma_fraction * sigma_base, size=len(draw))
        out[index, valid] = np.maximum(draw, 0.0).astype(np.float32)
    return out


def _rng(seed: int, stream: int, level_bp: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream, level_bp, 0])


def _moment_streams(
    model: Any, device: Any, raw_x: np.ndarray, mask: np.ndarray, max_length: int, batch: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    z = np.zeros_like(raw_x)
    for index in range(len(raw_x)):
        valid = mask[index].astype(bool)
        z[index, valid] = normalize_sequence(raw_x[index][valid], "zscore")
    diff_x, diff_mask = _diff_arrays(raw_x, mask)
    return (
        _forward(model, z, mask, device, batch),
        _forward(model, diff_x, diff_mask, device, batch),
    )


def _forward(model: Any, x: np.ndarray, mask: np.ndarray, device: Any, batch: int) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            bx = torch.as_tensor(x[start : start + batch, None], device=device)
            bm = torch.as_tensor(mask[start : start + batch], device=device)
            value = forward_features(model, bx, bm).float().cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError("Nonfinite MOMENT features")
            chunks.append(value)
    return np.concatenate(chunks)


def _hgb_params(seed: int, tag: str) -> dict[str, Any]:
    record = json.loads((E08_SUITE / f"seed{seed}" / "seed_results.json").read_text(encoding="utf-8"))
    matches = [
        row for row in record["conditions"] if row["method"] == "dyn_hgb" and row["tag"] == tag
    ]
    if len(matches) != 1:
        raise ValueError(f"Missing frozen HGB params for seed={seed} tag={tag}")
    return dict(matches[0]["selected_params"])


def _meta(oof: np.ndarray, y: np.ndarray, evaluation: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    grid = [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    selected, _ = run_e08._cv_select(
        lambda params: LogisticRegression(C=params["C"], max_iter=5000, random_state=seed),
        grid,
        oof,
        y,
        folds=3,
        seed=seed,
    )
    model = LogisticRegression(C=selected["C"], max_iter=5000, random_state=seed).fit(oof, y)
    return model.predict(evaluation), model.predict_proba(evaluation)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/最新多.csv"))
    parser.add_argument("--config", default="configs/moment.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", type=Path, default=Path("artifacts/splits/unified_split.json"))
    parser.add_argument(
        "--dyn-groups",
        type=str,
        default=None,
        help="Comma-separated DYN feature groups to keep (default: all 60).",
    )
    parser.add_argument(
        "--eval-levels",
        nargs="+",
        type=float,
        default=None,
        help="Noise levels to evaluate (default: 0, 0.1%%, 0.25%%, 0.5%%, 1%%).",
    )
    parser.add_argument(
        "--arms",
        choices=("both", "clean", "aug"),
        default="both",
        help="Train both the clean and augmented arms, or only the clean arm.",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=list(BUDGETS),
        help="Restrict to a subset of label budgets, e.g. --budgets 0.1",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SEEDS),
        help="Seeds handled by this process; one process per seed is much faster.",
    )
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()
    global EVAL_LEVELS
    if args.eval_levels:
        EVAL_LEVELS = tuple(args.eval_levels)
    budgets = tuple(args.budgets)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "settings.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(args.dataset),
                "aug_level": AUG_LEVEL,
                "eval_levels": list(EVAL_LEVELS),
                "budgets": list(BUDGETS),
                "seeds": args.seeds,
                "folds": FOLDS,
                "epochs": EPOCHS,
                "noise_scale": "train_valid_rms",
                "clip_min": 0.0,
                "dyn_groups": args.dyn_groups,
                "arms": args.arms,
                "test_evaluated": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_ids = np.asarray(split["splits"]["train"], dtype=str)
    val_ids = np.asarray(split["splits"]["validation"], dtype=str)
    data = _load(args.dataset, load_config(Path(args.config)).data.max_length)
    position = {value: index for index, value in enumerate(data["ids"])}
    train_index = np.asarray([position[value] for value in train_ids])
    val_index = np.asarray([position[value] for value in val_ids])
    max_length = data["x"].shape[1]
    dyn_groups = None
    if args.dyn_groups:
        wanted = [value.strip() for value in args.dyn_groups.split(",") if value.strip()]
        unknown = [value for value in wanted if value not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(f"Unknown DYN groups: {unknown}")
        keep = {name for group in wanted for name in FEATURE_GROUPS[group]}
        dyn_groups = [index for index, name in enumerate(DYNAMICS_FEATURE_NAMES) if name in keep]
        print(f"DYN groups {wanted}: {len(dyn_groups)} features", flush=True)

    _, _, _, _, pipeline = require_torch_and_moment()
    config = load_config(Path(args.config))
    model = build_model(config, pipeline, len(CLASSES))
    set_num_classes(model, len(CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    pilot.configure()

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        validation_perturbed: dict[float, dict[str, np.ndarray]] = {}
        for level in EVAL_LEVELS:
            level_bp = int(round(level * 1e4))
            noisy = _noise(
                data["x"][val_index],
                data["mask"][val_index],
                level,
                float(np.sqrt(np.mean(np.square(data["x"][train_index][data["mask"][train_index].astype(bool)])))),
                _rng(seed, 2, level_bp),
            )
            p_features, dp_features = _moment_streams(model, device, noisy, data["mask"][val_index], max_length)
            dyn_val = extract_dynamics_features(
                noisy, data["mask"][val_index], max_length=max_length
            )
            if dyn_groups is not None:
                dyn_val = dyn_val[:, dyn_groups]
            validation_perturbed[level] = {
                "x": noisy,
                "mask": data["mask"][val_index],
                "dyn": dyn_val,
                "md": np.hstack([p_features, dp_features]),
            }
            print(f"seed {seed}: validation level {level:.4f} features ready", flush=True)

        for fraction in budgets:
            tag = _tag(fraction)
            indices = stratified_train_subset_indices(
                data["y"][train_index], train_ids, fraction, seed
            )
            subset_index = train_index[indices]
            y_train = data["y"][subset_index]
            x_train = data["x"][subset_index]
            mask_train = data["mask"][subset_index]
            sigma_base = float(np.sqrt(np.mean(np.square(x_train[mask_train.astype(bool)]))))
            noisy_train = _noise(x_train, mask_train, AUG_LEVEL, sigma_base, _rng(seed, 1, int(AUG_LEVEL * 1e4)))
            dyn_clean = extract_dynamics_features(x_train, mask_train, max_length=max_length)
            dyn_noisy = extract_dynamics_features(noisy_train, mask_train, max_length=max_length)
            if dyn_groups is not None:
                dyn_clean = dyn_clean[:, dyn_groups]
                dyn_noisy = dyn_noisy[:, dyn_groups]
            p_clean, dp_clean = _moment_streams(model, device, x_train, mask_train, max_length)
            p_noisy, dp_noisy = _moment_streams(model, device, noisy_train, mask_train, max_length)
            md_clean = np.hstack([p_clean, dp_clean])
            md_noisy = np.hstack([p_noisy, dp_noisy])
            params = _hgb_params(seed, tag)

            splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
            folds = list(splitter.split(x_train, y_train))
            arms = {
                "clean": {
                    "dyn": (dyn_clean, y_train),
                    "md": (md_clean, y_train),
                    "x": x_train,
                    "mask": mask_train,
                    "y": y_train,
                },
                "aug": {
                    "dyn": (np.vstack([dyn_clean, dyn_noisy]), np.concatenate([y_train, y_train])),
                    "md": (np.vstack([md_clean, md_noisy]), np.concatenate([y_train, y_train])),
                    "x": np.concatenate([x_train, noisy_train]),
                    "mask": np.concatenate([mask_train, mask_train]),
                    "y": np.concatenate([y_train, y_train]),
                },
            }
            selected_arms = (
                ("clean",) if args.arms == "clean" else ("aug",) if args.arms == "aug" else ("clean", "aug")
            )
            for arm in selected_arms:
                payload = arms[arm]
                started = time.perf_counter()
                dyn_x, dyn_y = payload["dyn"]
                md_x, md_y = payload["md"]
                arm_x = payload["x"]
                arm_mask = payload["mask"]
                arm_y = payload["y"]
                oof_dyn = np.zeros((len(y_train), 3))
                oof_md = np.zeros((len(y_train), 3))
                oof_tcn = np.zeros((len(y_train), 3))
                n_rows = len(y_train)
                fold_rows: list[dict[str, int]] = []
                for fold, (fit_index, hold_index) in enumerate(folds):
                    # The augmented arm must *train* on the noisy copies; the OOF
                    # probabilities stay clean-input estimates so that the
                    # meta-learner learns branch reliability on clean inputs, which
                    # is what the earlier E12 protocol does.
                    if arm == "aug":
                        fit_rows = np.concatenate([fit_index, fit_index + n_rows])
                    else:
                        fit_rows = fit_index
                    if arm == "aug" and len(fit_rows) != 2 * len(fit_index):
                        raise AssertionError("Augmented fold must see both copies")
                    hold_rows = hold_index
                    tcn = _train_tcn(
                        arm_x[fit_rows],
                        arm_mask[fit_rows],
                        arm_y[fit_rows],
                        epochs=EPOCHS,
                        seed=seed * 100 + fold,
                        device=device,
                    )
                    oof_tcn[hold_rows] = _tcn_predict(
                        tcn, x_train[hold_rows], mask_train[hold_rows], device
                    )
                    hgb = run_e08._hgb_factory(params, seed).fit(dyn_x[fit_rows], dyn_y[fit_rows])
                    oof_dyn[hold_rows] = hgb.predict_proba(dyn_clean[hold_rows])
                    logistic = run_e08._logit_factory({"C": 1.0}, seed).fit(
                        md_x[fit_rows], md_y[fit_rows]
                    )
                    oof_md[hold_rows] = logistic.predict_proba(md_clean[hold_rows])
                    fold_rows.append(
                        {
                            "fold": fold,
                            "n_fit_rows": int(len(fit_rows)),
                            "n_hold_rows": int(len(hold_rows)),
                        }
                    )
                tcn_full = _train_tcn(
                    arm_x, arm_mask, arm_y, epochs=EPOCHS, seed=seed * 100 + 99, device=device
                )
                hgb_full = run_e08._hgb_factory(params, seed).fit(dyn_x, dyn_y)
                logistic_full = run_e08._logit_factory({"C": 1.0}, seed).fit(md_x, md_y)
                for level in EVAL_LEVELS:
                    block = validation_perturbed[level]
                    branch = {
                        "dyn": hgb_full.predict_proba(block["dyn"]),
                        "md": logistic_full.predict_proba(block["md"]),
                        "tcn": _tcn_predict(tcn_full, block["x"], block["mask"], device),
                    }
                    full_prediction = _meta(
                        np.hstack([oof_dyn, oof_md, oof_tcn]),
                        y_train,
                        np.hstack([branch["dyn"], branch["md"], branch["tcn"]]),
                        seed,
                    )[0]
                    predictions = {"dyn": branch["dyn"].argmax(1), "full": full_prediction}
                    for method, prediction in predictions.items():
                        metrics = _metrics(data["y"][val_index], prediction)
                        rows.append(
                            {
                                "seed": seed,
                                "tag": tag,
                                "arm": arm,
                                "method": method,
                                "noise_fraction": level,
                                "n_fit": int(len(arm_y)),
                                "fold_fit_rows": [entry["n_fit_rows"] for entry in fold_rows],
                                "seconds": time.perf_counter() - started,
                                "sigma_base": sigma_base,
                                **metrics,
                            }
                        )
                print(f"seed {seed} {tag} {arm}: done in {time.perf_counter() - started:.0f}s", flush=True)

    fieldnames = [
        "seed", "tag", "arm", "method", "noise_fraction", "n_fit", "fold_fit_rows_min",
        "fold_fit_rows_max", "seconds", "sigma_base",
        "macro_f1", "class0_recall", "class0_fpr", "class1_recall", "class1_fpr",
        "class2_recall", "class2_fpr",
    ]
    output_csv = args.output / ("conditions.csv" if len(args.seeds) > 1 else f"conditions_seed{args.seeds[0]}.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["fold_fit_rows_min"] = min(row["fold_fit_rows"])
            row["fold_fit_rows_max"] = max(row["fold_fit_rows"])
            writer.writerow({key: row[key] for key in fieldnames})

    merged: list[dict[str, Any]] = []
    for path in sorted(args.output.glob("conditions_seed*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for raw_row in csv.DictReader(handle):
                merged.append(
                    {
                        "tag": raw_row["tag"],
                        "arm": raw_row["arm"],
                        "method": raw_row["method"],
                        "noise_fraction": float(raw_row["noise_fraction"]),
                        "seed": int(raw_row["seed"]),
                        "macro_f1": float(raw_row["macro_f1"]),
                        "class1_recall": float(raw_row["class1_recall"]),
                        "class1_fpr": float(raw_row["class1_fpr"]),
                        "n_fit": int(raw_row["n_fit"]),
                    }
                )
    if merged:
        rows = merged
        print(f"aggregating {len(merged)} rows from per-seed CSVs", flush=True)

    aggregates: list[dict[str, Any]] = []
    for tag in ("10p", "40p"):
        for arm in ("clean", "aug"):
            for method in ("dyn", "full"):
                for level in EVAL_LEVELS:
                    selected = [
                        row for row in rows
                        if row["tag"] == tag and row["arm"] == arm
                        and row["method"] == method and row["noise_fraction"] == level
                    ]
                    if not selected:
                        continue
                    aggregates.append(
                        {
                            "tag": tag,
                            "arm": arm,
                            "method": method,
                            "noise_fraction": level,
                            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in selected]) * 100),
                            "macro_f1_std": float(np.std([row["macro_f1"] for row in selected]) * 100),
                            "class1_recall_mean": float(np.mean([row["class1_recall"] for row in selected]) * 100),
                            "class1_fpr_mean": float(np.mean([row["class1_fpr"] for row in selected]) * 100),
                        }
                    )
    paired: list[dict[str, Any]] = []
    for tag in ("10p", "40p"):
        for method in ("dyn", "full"):
            for level in EVAL_LEVELS:
                clean = {
                    row["seed"]: row["macro_f1"]
                    for row in rows
                    if row["tag"] == tag and row["arm"] == "clean"
                    and row["method"] == method and row["noise_fraction"] == level
                }
                aug = {
                    row["seed"]: row["macro_f1"]
                    for row in rows
                    if row["tag"] == tag and row["arm"] == "aug"
                    and row["method"] == method and row["noise_fraction"] == level
                }
                shared = sorted(set(clean) & set(aug))
                if not shared:
                    continue
                diffs = np.asarray([aug[key] - clean[key] for key in shared])
                paired.append(
                    {
                        "tag": tag,
                        "method": method,
                        "noise_fraction": level,
                        "delta_mean_pp": float(diffs.mean() * 100),
                        "delta_std_pp": float(diffs.std(ddof=0) * 100),
                        "positive_seeds": int((diffs > 0).sum()),
                        "n_seeds": int(len(diffs)),
                    }
                )

    atomic_write_json(
        args.output / "summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "aggregates": aggregates,
            "paired": paired,
            "test_evaluated": False,
        },
    )
    lines = ["# TODO1102-4B fusion-side augmentation", "", "| 预算 | 训练 | 方法 | 噪声 | Macro-F1 | 电池召回 |", "|---|---|---|---:|---:|---:|"]
    for row in aggregates:
        lines.append(
            f"| {row['tag']} | {row['arm']} | {row['method']} | {100 * row['noise_fraction']:.2f}% | "
            f"{row['macro_f1_mean']:.2f} ± {row['macro_f1_std']:.2f} | {row['class1_recall_mean']:.2f} |"
        )
    lines += ["", "| 预算 | 方法 | 噪声 | 增广 − 干净 | 种子 |", "|---|---|---:|---:|---:|"]
    for row in paired:
        lines.append(
            f"| {row['tag']} | {row['method']} | {100 * row['noise_fraction']:.2f}% | "
            f"{row['delta_mean_pp']:+.2f} ± {row['delta_std_pp']:.2f} | "
            f"{row['positive_seeds']}/{row['n_seeds']} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for row in aggregates:
        print(
            f"{row['tag']:>4} {row['arm']:>5} {row['method']:>4} {100 * row['noise_fraction']:6.2f}% "
            f"macro_f1={row['macro_f1_mean']:6.2f} battery_recall={row['class1_recall_mean']:6.2f}",
            flush=True,
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
