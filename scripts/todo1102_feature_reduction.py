"""TODO1102-6A: group-level DYN feature-budget screen on frozen E08 features.

This script only reads the E08 feature cache and split-derived 40% subsets.
It trains HGB with the parameters already selected by E08 for each seed; no
hyperparameter search, no raw-sequence rescan, and no test split access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from tcn_moment.dynamics_features import FEATURE_GROUPS

DEFAULT_PROTOCOL = Path(
    "configs/experiments/improvements/todo1102_feature_reduction_protocol.json"
)
DEFAULT_OUTPUT = Path("artifacts/analysis/todo1102_20261102/feature_reduction")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_hgb_params(seed_root: Path, seed: int, tag: str) -> dict[str, Any]:
    results = _load_json(seed_root / f"seed{seed}" / "seed_results.json")
    matches = [
        row
        for row in results["conditions"]
        if row["method"] == "dyn_hgb" and row["tag"] == tag
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one dyn_hgb/{tag} record for seed {seed}")
    return dict(matches[0]["selected_params"])


def _subset_columns(feature_names: np.ndarray, groups: tuple[str, ...]) -> list[int]:
    selected = {name for group in groups for name in FEATURE_GROUPS[group]}
    return [index for index, name in enumerate(feature_names) if name in selected]


def _score_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    columns: list[int],
    params: dict[str, Any],
    seed: int,
) -> tuple[float, float]:
    started = time.perf_counter()
    model = HistGradientBoostingClassifier(
        **params,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x_train[:, columns], y_train)
    score = float(f1_score(y_val, model.predict(x_val[:, columns]), average="macro"))
    return score, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tag", type=str, default=None, help="Budget tag override, e.g. 10p.")
    parser.add_argument(
        "--fraction", type=float, default=None, help="Budget fraction override, e.g. 0.1."
    )
    parser.add_argument(
        "--source-suite",
        type=Path,
        default=None,
        help="Override the E08 suite directory (e.g. when artifacts live under imports/).",
    )
    args = parser.parse_args()

    protocol = _load_json(args.protocol)
    if protocol.get("no_test_access") is not True or protocol.get("test_evaluated") is not False:
        raise ValueError("Protocol must forbid test access")
    if protocol["seeds"] != [42, 44, 45, 46]:
        raise ValueError("This audit fixes seeds 42/44/45/46")
    source_suite = Path(args.source_suite or protocol["source_suite"])
    tag = args.tag or protocol["tag"]
    fraction = float(args.fraction if args.fraction is not None else protocol["label_fraction"])
    if abs(fraction - int(round(fraction * 100)) / 100.0) > 1e-9:
        raise ValueError("Fraction must map to a whole-percent budget tag")
    low, high = protocol["target_feature_range"]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(
        args.protocol.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (args.output / "invocation.json").write_text(
        json.dumps(
            {
                "protocol": str(args.protocol),
                "source_suite": str(source_suite),
                "tag": tag,
                "fraction": fraction,
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    group_names = tuple(FEATURE_GROUPS)
    group_combinations = [
        combo
        for size in range(1, len(group_names) + 1)
        for combo in itertools.combinations(group_names, size)
    ]
    if len(group_combinations) != 2 ** len(group_names) - 1:
        raise AssertionError("Group combination enumeration is incomplete")

    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(args.protocol),
        "protocol_sha256": _sha256(args.protocol),
        "source_suite": str(source_suite),
        "inputs": {},
    }

    for seed in protocol["seeds"]:
        seed_root = source_suite / f"seed{seed}"
        features_path = seed_root / "features.npz"
        ids_path = seed_root / f"train_ids_{int(round(fraction * 100))}p.npy"
        params = _seed_hgb_params(source_suite, seed, tag)
        archive = np.load(features_path, allow_pickle=True)
        train_features = archive["train_features"]
        validation_features = archive["validation_features"]
        train_ids = archive["train_ids"]
        train_labels = archive["train_labels"]
        validation_labels = archive["validation_labels"]
        feature_names = archive["feature_names"]
        subset_ids = np.load(ids_path, allow_pickle=True)
        subset_set = set(subset_ids.tolist())
        subset_mask = np.array([value in subset_set for value in train_ids])
        x_train = train_features[subset_mask]
        y_train = train_labels[subset_mask]
        if len(x_train) != len(subset_ids):
            raise RuntimeError(f"Subset alignment failed for seed {seed}")
        if len(feature_names) != train_features.shape[1]:
            raise RuntimeError(f"Feature name mismatch for seed {seed}")
        if set(feature_names) != {
            name for names in FEATURE_GROUPS.values() for name in names
        }:
            raise RuntimeError(f"Feature groups do not match cache for seed {seed}")

        full_columns = list(range(len(feature_names)))
        full_score, full_seconds = _score_subset(
            x_train,
            y_train,
            validation_features,
            validation_labels,
            full_columns,
            params,
            seed,
        )
        baseline_rows.append(
            {
                "seed": seed,
                "n_features": len(full_columns),
                "macro_f1": full_score,
                "fit_seconds": full_seconds,
                "selected_params": params,
                "n_train": len(x_train),
            }
        )
        print(f"seed {seed}: full={full_score:.6f} ({len(x_train)} rows)", flush=True)

        for combo in group_combinations:
            columns = _subset_columns(feature_names, combo)
            score, seconds = _score_subset(
                x_train,
                y_train,
                validation_features,
                validation_labels,
                columns,
                params,
                seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "groups": "+".join(combo),
                    "n_groups": len(combo),
                    "n_features": len(columns),
                    "macro_f1": score,
                    "fit_seconds": seconds,
                    "selected_params": params,
                    "n_train": len(x_train),
                }
            )
        manifest["inputs"][str(seed)] = {
            "features_sha256": _sha256(features_path),
            "subset_sha256": _sha256(ids_path),
            "seed_results_sha256": _sha256(seed_root / "seed_results.json"),
        }

    baseline_mean = float(np.mean([row["macro_f1"] for row in baseline_rows]))
    baseline_std = float(np.std([row["macro_f1"] for row in baseline_rows], ddof=0))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["groups"], []).append(row)
    aggregates: list[dict[str, Any]] = []
    for groups, group_rows in grouped.items():
        scores = [row["macro_f1"] for row in group_rows]
        aggregates.append(
            {
                "groups": groups,
                "n_groups": group_rows[0]["n_groups"],
                "n_features": group_rows[0]["n_features"],
                "mean_macro_f1": float(np.mean(scores)),
                "std_macro_f1": float(np.std(scores, ddof=0)),
                "min_macro_f1": float(np.min(scores)),
                "max_macro_f1": float(np.max(scores)),
                "delta_vs_full_pp": float((np.mean(scores) - baseline_mean) * 100.0),
                "per_seed": {str(row["seed"]): row["macro_f1"] for row in group_rows},
            }
        )
    aggregates.sort(key=lambda row: (-row["mean_macro_f1"], row["n_features"], row["groups"]))

    eligible = [
        row for row in aggregates if low <= row["n_features"] <= high
    ]
    if not eligible:
        raise RuntimeError("No feature subset falls in the requested range")
    best_eligible = eligible[0]
    tie_tolerance = float(protocol["selection"]["tie_tolerance_pp"]) / 100.0
    lean = [
        row
        for row in eligible
        if row["mean_macro_f1"] >= best_eligible["mean_macro_f1"] - tie_tolerance
    ]
    selected = min(lean, key=lambda row: (row["n_features"], -row["mean_macro_f1"], row["groups"]))

    within_report_tolerance = [
        row
        for row in aggregates
        if row["mean_macro_f1"]
        >= baseline_mean - float(protocol["selection"]["report_tolerance_pp"]) / 100.0
    ]

    summary = {
        "generated_at": manifest["generated_at"],
        "protocol": str(args.protocol),
        "protocol_sha256": manifest["protocol_sha256"],
        "seeds": protocol["seeds"],
        "label_fraction": fraction,
        "tag": tag,
        "n_train": len(x_train),
        "baseline_full": {
            "mean_macro_f1": baseline_mean,
            "std_macro_f1": baseline_std,
            "per_seed": {str(row["seed"]): row["macro_f1"] for row in baseline_rows},
            "per_seed_rows": baseline_rows,
        },
        "target_feature_range": [low, high],
        "best_eligible": best_eligible,
        "selected": selected,
        "within_0p20pp_of_full": within_report_tolerance,
        "aggregates": aggregates,
        "test_evaluated": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (args.output / "subset_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "groups",
                "n_groups",
                "n_features",
                "macro_f1",
                "fit_seconds",
                "n_train",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        "# TODO1102-6A feature-budget screen",
        "",
        f"- Protocol: `{args.protocol}`",
        f"- Seeds: {protocol['seeds']}",
        f"- Training subset: {tag} ({len(x_train)} rows)",
        f"- Full 60-D mean: {baseline_mean * 100:.2f}% ± {baseline_std * 100:.2f} pp",
        f"- Target range: {low}–{high} features",
        f"- Best eligible: {best_eligible['groups']} ({best_eligible['n_features']} D, "
        f"{best_eligible['mean_macro_f1'] * 100:.2f}%)",
        f"- Selected lean subset: {selected['groups']} ({selected['n_features']} D, "
        f"{selected['mean_macro_f1'] * 100:.2f}%)",
        "",
        "| Groups | Dim | Mean Macro-F1 | Δ vs full | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['groups']} | {row['n_features']} | "
            f"{row['mean_macro_f1'] * 100:.2f} | {row['delta_vs_full_pp']:+.2f} | "
            f"{row['min_macro_f1'] * 100:.2f} | {row['max_macro_f1'] * 100:.2f} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"full mean={baseline_mean * 100:.3f}%")
    print(
        f"best eligible={best_eligible['groups']} {best_eligible['n_features']}D "
        f"{best_eligible['mean_macro_f1'] * 100:.3f}%"
    )
    print(
        f"selected={selected['groups']} {selected['n_features']}D "
        f"{selected['mean_macro_f1'] * 100:.3f}%"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
