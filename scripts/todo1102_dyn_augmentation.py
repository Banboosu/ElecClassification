"""TODO1102-4A: feature-side noise augmentation for the DYN branch.

E13b showed that the clean-trained DYN+HGB branch is the weak link under
acquisition noise: at 10% labels the branch falls from 92.81% to 38.36% at 0.5%
train-RMS noise.  This audit asks whether augmenting the *feature side* alone —
recomputing the 60-D descriptors on noisy copies of the labelled training
sequences before fitting the same HGB — recovers that collapse.

Scope and discipline:

* validation-only, no test split is read;
* the HGB configuration is the one E08 already selected per seed and budget
  (no new hyper-parameter search);
* the clean-trained arm is re-scored under the *same* noise draws as the
  augmented arm, so the comparison is paired within this audit and does not
  depend on E13's random stream;
* E13b remains the reference for "clean model, E13 draws" and is not overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score

from tcn_moment.data import parse_power_sequence
from tcn_moment.dynamics_features import FEATURE_GROUPS, extract_dynamics_features

DEFAULT_PROTOCOL = Path("configs/experiments/improvements/todo1102_dyn_augmentation_protocol.json")
DEFAULT_OUTPUT = Path("artifacts/analysis/todo1102_20261102/dyn_augmentation")
BATTERY_INDEX = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_features(payload: tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    x, mask, max_length = payload
    return extract_dynamics_features(x, mask, max_length=max_length)


def _extract_many(
    x: np.ndarray, mask: np.ndarray, max_length: int, jobs: int
) -> np.ndarray:
    if jobs <= 1 or len(x) <= 256:
        return extract_dynamics_features(x, mask, max_length=max_length)
    chunk = max(64, len(x) // (jobs * 4))
    payloads = [
        (x[start : start + chunk], mask[start : start + chunk], max_length)
        for start in range(0, len(x), chunk)
    ]
    with Pool(processes=jobs) as pool:
        parts = pool.map(_chunk_features, payloads)
    return np.vstack(parts)


def _add_noise(
    x: np.ndarray,
    mask: np.ndarray,
    sigma_fraction: float,
    sigma_base: float,
    clip_min: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    valid_mask = np.asarray(mask, dtype=bool)
    if sigma_fraction <= 0.0:
        return out
    for index in range(len(out)):
        valid = valid_mask[index]
        values = out[index, valid].astype(np.float64)
        values = values + rng.normal(0.0, sigma_fraction * sigma_base, size=len(values))
        values = np.maximum(values, clip_min)
        out[index, valid] = values.astype(np.float32)
    return out


def _seed_rng(seed: int, stream: int, level_bp: int, copy: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream, level_bp, copy])


def _metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    result: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "confusion_matrix": matrix.tolist(),
    }
    for label in range(num_classes):
        tp = float(matrix[label, label])
        fn = float(matrix[label, :].sum() - tp)
        fp = float(matrix[:, label].sum() - tp)
        tn = float(matrix.sum() - tp - fn - fp)
        result[f"class{label}_recall"] = tp / (tp + fn) if tp + fn else 0.0
        result[f"class{label}_fpr"] = fp / (fp + tn) if fp + tn else 0.0
    return result


def _load_sequences(path: Path, delimiter: str, id_column: str, sequence_column: str,
                    label_column: str, max_length: int) -> dict[str, Any]:
    frame = pd.read_csv(path, delimiter=delimiter)
    ids = frame[id_column].astype(str).to_numpy()
    labels = frame[label_column].astype(str).to_numpy()
    sequences = frame[sequence_column].map(parse_power_sequence).to_numpy()
    values = np.zeros((len(ids), max_length), dtype=np.float32)
    mask = np.zeros((len(ids), max_length), dtype=bool)
    lengths = np.zeros(len(ids), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        array = np.asarray(sequence, dtype=np.float32)[:max_length]
        values[index, : len(array)] = array
        mask[index, : len(array)] = True
        lengths[index] = len(array)
    return {
        "ids": ids,
        "labels": labels,
        "values": values,
        "mask": mask,
        "lengths": lengths,
    }


def _select(raw: dict[str, Any], ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = {value: index for index, value in enumerate(raw["ids"])}
    missing = [value for value in ids if value not in position]
    if missing:
        raise KeyError(f"{len(missing)} ids missing from raw data, first={missing[0]}")
    index = np.array([position[value] for value in ids], dtype=np.int64)
    return raw["values"][index], raw["mask"][index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-suite", type=Path, default=None)
    parser.add_argument(
        "--feature-groups",
        type=str,
        default=None,
        help="Comma-separated DYN groups to keep (default: all 60 features).",
    )
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("no_test_access") is not True or protocol.get("test_evaluated") is not False:
        raise ValueError("Protocol must forbid test access")
    if protocol["seeds"] != [42, 44, 45, 46]:
        raise ValueError("This audit fixes seeds 42/44/45/46")
    dataset_path = Path(protocol["dataset"])
    if _sha256(dataset_path) != protocol["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    source_suite = Path(args.source_suite or protocol["source_suite"])
    max_length = int(protocol["max_length"])
    num_classes = len(protocol["classes"])
    classes = [str(value) for value in protocol["classes"]]
    if classes[BATTERY_INDEX] != "1":
        raise ValueError("Battery class must be raw label 1 under the corrected semantics")

    feature_names = None
    keep_columns: list[int] | None = None
    if args.feature_groups:
        groups = [value.strip() for value in args.feature_groups.split(",") if value.strip()]
        unknown = [value for value in groups if value not in FEATURE_GROUPS]
        if unknown:
            raise ValueError(f"Unknown feature groups: {unknown}")
        wanted = {name for group in groups for name in FEATURE_GROUPS[group]}
        feature_names = sorted(wanted)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(
        args.protocol.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (args.output / "invocation.json").write_text(
        json.dumps(
            {
                "protocol": str(args.protocol),
                "source_suite": str(source_suite),
                "dataset": str(dataset_path),
                "feature_groups": args.feature_groups,
                "jobs": args.jobs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    raw = _load_sequences(
        dataset_path,
        protocol.get("delimiter", " "),
        protocol["id_column"],
        protocol["sequence_column"],
        protocol["label_column"],
        max_length,
    )
    raw_label_index = {value: index for index, value in enumerate(classes)}
    raw_label_by_id = {
        value: raw_label_index[label]
        for value, label in zip(raw["ids"], raw["labels"], strict=True)
        if label in raw_label_index
    }
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(args.protocol),
        "protocol_sha256": _sha256(args.protocol),
        "source_suite": str(source_suite),
        "dataset_sha256": protocol["dataset_sha256"],
        "inputs": {},
        "sanity": {},
    }
    aug_levels = [float(value) for value in protocol["augmentation"]["levels"]]
    aug_copies = int(protocol["augmentation"]["copies"])
    eval_levels = [float(value) for value in protocol["evaluation"]["noise_levels"]]
    clip_min = float(protocol["augmentation"]["clip_min"])

    for seed in protocol["seeds"]:
        seed_root = source_suite / f"seed{seed}"
        archive = np.load(seed_root / "features.npz", allow_pickle=True)
        train_ids = archive["train_ids"].astype(str)
        validation_ids = archive["validation_ids"].astype(str)
        train_features = archive["train_features"]
        validation_features = archive["validation_features"]
        train_labels_raw = archive["train_labels"]
        validation_labels = archive["validation_labels"]
        expected_train_labels = np.array(
            [raw_label_by_id[value] for value in train_ids], dtype=np.int64
        )
        if feature_names is not None and keep_columns is None:
            keep_columns = [
                index for index, name in enumerate(archive["feature_names"].tolist())
                if name in set(feature_names)
            ]
            if len(keep_columns) != len(feature_names):
                raise ValueError("Requested feature groups do not match cached names")
        if keep_columns is not None:
            train_features = train_features[:, keep_columns]
            validation_features = validation_features[:, keep_columns]
        if not np.array_equal(np.sort(np.unique(train_labels_raw)), np.arange(num_classes)):
            raise ValueError(f"Unexpected label encoding for seed {seed}")
        if not np.array_equal(train_labels_raw, expected_train_labels):
            raise ValueError(f"Raw label mapping mismatch for seed {seed}")

        x_train, mask_train = _select(raw, train_ids)
        x_val, mask_val = _select(raw, validation_ids)
        recomputed = _extract_many(x_train[:64], mask_train[:64], max_length, jobs=1)
        cached_reference = archive["train_features"][:64]
        if keep_columns is not None:
            cached_reference = cached_reference[:, keep_columns]
        if not np.allclose(recomputed[:, keep_columns] if keep_columns else recomputed,
                           cached_reference, atol=1e-4):
            raise ValueError(f"Cached train features do not reproduce for seed {seed}")
        train_values = x_train[mask_train.astype(bool)]
        sigma_base = float(np.sqrt(np.mean(np.square(train_values, dtype=np.float64))))

        augmentation_started = time.perf_counter()
        train_blocks = [train_features]
        train_label_blocks = [train_labels_raw]
        for level in aug_levels:
            for copy_index in range(aug_copies):
                level_bp = int(round(level * 1e4))
                rng = _seed_rng(seed, 1, level_bp, copy_index)
                noisy = _add_noise(x_train, mask_train, level, sigma_base, clip_min, rng)
                train_blocks.append(_extract_many(noisy, mask_train, max_length, args.jobs))
                train_label_blocks.append(train_labels_raw)
        if keep_columns is not None:
            train_blocks = [
                block[:, keep_columns] if block.shape[1] != len(keep_columns) else block
                for block in train_blocks
            ]
        augmented_features = np.vstack(train_blocks)
        augmented_labels = np.concatenate(train_label_blocks)
        n_blocks = len(train_blocks)
        augmentation_seconds = time.perf_counter() - augmentation_started

        validation_features_by_level: dict[float, np.ndarray] = {0.0: validation_features}
        for level in eval_levels:
            level_bp = int(round(level * 1e4))
            rng = _seed_rng(seed, 2, level_bp, 0)
            noisy = _add_noise(x_val, mask_val, level, sigma_base, clip_min, rng)
            extracted = _extract_many(noisy, mask_val, max_length, args.jobs)
            validation_features_by_level[level] = (
                extracted[:, keep_columns] if keep_columns is not None else extracted
            )

        seed_results = json.loads((seed_root / "seed_results.json").read_text(encoding="utf-8"))
        params_by_tag = {
            record["tag"]: record["selected_params"]
            for record in seed_results["conditions"]
            if record["method"] == "dyn_hgb"
        }

        for fraction in protocol["label_fractions"]:
            tag = f"{int(round(fraction * 100))}p"
            subset_ids = np.load(seed_root / f"train_ids_{tag}.npy", allow_pickle=True).astype(str)
            subset_set = set(subset_ids.tolist())
            subset_mask = np.array([value in subset_set for value in train_ids])
            if int(subset_mask.sum()) != len(subset_ids):
                raise ValueError(f"Subset alignment failed for seed {seed} {tag}")
            params = dict(params_by_tag[tag])
            y_subset = train_labels_raw[subset_mask]

            arms = {
                "clean_trained": (train_features[subset_mask], y_subset),
                "aug_trained": (
                    augmented_features[np.tile(subset_mask, n_blocks)],
                    augmented_labels[np.tile(subset_mask, n_blocks)],
                ),
            }
            for arm, (x_fit, y_fit) in arms.items():
                started = time.perf_counter()
                model = HistGradientBoostingClassifier(
                    **params, early_stopping=False, random_state=seed
                )
                model.fit(x_fit, y_fit)
                fit_seconds = time.perf_counter() - started
                for level in [0.0] + eval_levels:
                    features = validation_features_by_level[level]
                    prediction = model.predict(features)
                    scores = _metrics(validation_labels, prediction, num_classes)
                    rows.append(
                        {
                            "seed": seed,
                            "tag": tag,
                            "arm": arm,
                            "noise_fraction": level,
                            "n_fit_rows": int(len(x_fit)),
                            "n_features": int(x_fit.shape[1]),
                            "fit_seconds": fit_seconds,
                            "sigma_base": sigma_base,
                            **{key: value for key, value in scores.items() if key != "confusion_matrix"},
                        }
                    )
                print(
                    f"seed {seed} {tag} {arm}: clean="
                    f"{rows[-len(eval_levels) - 1]['macro_f1'] * 100:.2f}% "
                    f"fit={fit_seconds:.1f}s",
                    flush=True,
                )
        manifest["inputs"][str(seed)] = {
            "features_sha256": _sha256(seed_root / "features.npz"),
            "seed_results_sha256": _sha256(seed_root / "seed_results.json"),
            "sigma_base": sigma_base,
            "augmentation_seconds": augmentation_seconds,
            "n_augmented_rows": int(len(augmented_features)),
        }
        manifest["sanity"][str(seed)] = {
            "feature_reproduction_atol": 1e-4,
            "raw_label_mapping": "ok",
        }

    fieldnames = [
        "seed",
        "tag",
        "arm",
        "noise_fraction",
        "n_fit_rows",
        "n_features",
        "fit_seconds",
        "sigma_base",
        "macro_f1",
        "accuracy",
        "class0_recall",
        "class0_fpr",
        "class1_recall",
        "class1_fpr",
        "class2_recall",
        "class2_fpr",
    ]
    with (args.output / "conditions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    aggregates: list[dict[str, Any]] = []
    for tag in sorted({row["tag"] for row in rows}, key=lambda value: int(value[:-1])):
        for arm in ("clean_trained", "aug_trained"):
            for level in sorted({row["noise_fraction"] for row in rows}):
                selected = [
                    row
                    for row in rows
                    if row["tag"] == tag and row["arm"] == arm and row["noise_fraction"] == level
                ]
                if not selected:
                    continue
                scores = np.array([row["macro_f1"] for row in selected])
                battery_recall = np.array([row["class1_recall"] for row in selected])
                battery_fpr = np.array([row["class1_fpr"] for row in selected])
                aggregates.append(
                    {
                        "tag": tag,
                        "arm": arm,
                        "noise_fraction": level,
                        "n_seeds": len(selected),
                        "macro_f1_mean": float(scores.mean()),
                        "macro_f1_std": float(scores.std(ddof=0)),
                        "macro_f1_mean_pct": float(scores.mean() * 100.0),
                        "macro_f1_std_pct": float(scores.std(ddof=0) * 100.0),
                        "battery_recall_mean": float(battery_recall.mean()),
                        "battery_fpr_mean": float(battery_fpr.mean()),
                        "battery_recall_mean_pct": float(battery_recall.mean() * 100.0),
                        "battery_fpr_mean_pct": float(battery_fpr.mean() * 100.0),
                    }
                )

    paired: list[dict[str, Any]] = []
    for tag in sorted({row["tag"] for row in rows}, key=lambda value: int(value[:-1])):
        for level in sorted({row["noise_fraction"] for row in rows}):
            clean = {
                row["seed"]: row["macro_f1"]
                for row in rows
                if row["tag"] == tag and row["arm"] == "clean_trained" and row["noise_fraction"] == level
            }
            aug = {
                row["seed"]: row["macro_f1"]
                for row in rows
                if row["tag"] == tag and row["arm"] == "aug_trained" and row["noise_fraction"] == level
            }
            diffs = np.array([aug[seed] - clean[seed] for seed in sorted(clean)])
            paired.append(
                {
                    "tag": tag,
                    "noise_fraction": level,
                    "delta_mean_pp": float(diffs.mean() * 100.0),
                    "delta_std_pp": float(diffs.std(ddof=0) * 100.0),
                    "positive_seeds": int((diffs > 0).sum()),
                    "n_seeds": int(len(diffs)),
                }
            )

    summary = {
        "generated_at": manifest["generated_at"],
        "protocol": str(args.protocol),
        "protocol_sha256": manifest["protocol_sha256"],
        "dataset_sha256": protocol["dataset_sha256"],
        "source_suite": str(source_suite),
        "seeds": protocol["seeds"],
        "label_fractions": protocol["label_fractions"],
        "feature_groups": args.feature_groups,
        "augmentation": protocol["augmentation"],
        "evaluation": protocol["evaluation"],
        "units": {
            "macro_f1_mean": "fraction (0-1)",
            "macro_f1_mean_pct": "percent",
            "battery_recall_mean": "fraction (0-1)",
            "battery_recall_mean_pct": "percent",
            "delta_mean_pp": "percentage points",
            "noise_fraction": "fraction of train RMS",
        },
        "aggregates": aggregates,
        "paired": paired,
        "test_evaluated": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# TODO1102-4A DYN feature-side augmentation",
        "",
        f"- Protocol: `{args.protocol}`",
        f"- Seeds: {protocol['seeds']}",
        f"- Augmentation: levels {aug_levels}, {aug_copies} copies each, clip >= {clip_min}",
        "",
        "| Budget | Arm | Noise | Macro-F1 | Battery recall | Battery FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['tag']} | {row['arm']} | {100 * row['noise_fraction']:.2f}% | "
            f"{row['macro_f1_mean'] * 100:.2f} ± {row['macro_f1_std'] * 100:.2f} | "
            f"{row['battery_recall_mean'] * 100:.2f} | {row['battery_fpr_mean'] * 100:.2f} |"
        )
    lines += ["", "| Budget | Noise | Aug − clean (pp) | Seeds |", "|---|---:|---:|---:|"]
    for row in paired:
        lines.append(
            f"| {row['tag']} | {100 * row['noise_fraction']:.2f}% | "
            f"{row['delta_mean_pp']:+.2f} ± {row['delta_std_pp']:.2f} | "
            f"{row['positive_seeds']}/{row['n_seeds']} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for row in aggregates:
        print(
            f"{row['tag']:>4} {row['arm']:>14} {100 * row['noise_fraction']:6.2f}% "
            f"macro_f1={row['macro_f1_mean'] * 100:6.2f} "
            f"battery_recall={row['battery_recall_mean'] * 100:6.2f} "
            f"battery_fpr={row['battery_fpr_mean'] * 100:5.2f}",
            flush=True,
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
