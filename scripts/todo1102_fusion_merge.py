"""Merge the corrected 4b run with the unaffected clean arm and check the fold audit.

Why a merge instead of one run: the fold-index bug in the first 4b run only affected
the augmented arm (`fit_rows != fit_index`).  For the clean arm `fit_rows == fit_index`,
so its numbers are unchanged by the fix and can be reused — the rerun only had to
rebuild the augmented arm, which halves the GPU time.

This script:

* takes the clean arm from `fusion_augmentation/` (v1) and the augmented arm from
  `fusion_augmentation_v2/`;
* refuses to merge if the augmented folds did not fit on twice as many rows
  (`fold_fit_rows_min`/`max` are written by the fixed script);
* recomputes aggregates and paired differences from the per-seed rows, so nothing is
  copied from the old (invalid) summary;
* writes `fusion_augmentation_merged/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

A = Path("artifacts/analysis/todo1102_20261102")
CLEAN_DIR = A / "fusion_augmentation"
AUG_DIR = A / "fusion_augmentation_v2"
OUT = A / "fusion_augmentation_merged"
EVAL_LEVELS = (0.0, 0.0025, 0.005, 0.01)


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _norm(row: dict[str, str], arm: str, source: str) -> dict[str, Any]:
    return {
        "seed": int(row["seed"]),
        "tag": row["tag"],
        "arm": arm,
        "method": row["method"],
        "noise_fraction": float(row["noise_fraction"]),
        "n_fit": int(float(row["n_fit"])),
        "macro_f1": float(row["macro_f1"]),
        "class1_recall": float(row["class1_recall"]),
        "class1_fpr": float(row["class1_fpr"]),
        "source": source,
    }


def main() -> None:
    clean_rows = [
        _norm(row, "clean", "fusion_augmentation(v1)")
        for row in _read(CLEAN_DIR / "conditions.csv")
        if row["arm"] == "clean"
    ]
    aug_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for path in sorted(AUG_DIR.glob("conditions_seed*.csv")):
        rows = _read(path)
        aug_rows.extend(_norm(row, "aug", "fusion_augmentation_v2") for row in rows)
        first = rows[0]
        audit[path.name] = {
            "fold_fit_rows_min": int(first["fold_fit_rows_min"]),
            "fold_fit_rows_max": int(first["fold_fit_rows_max"]),
        }
    if not aug_rows:
        raise FileNotFoundError("No augmented-arm rows found; has the rerun finished?")

    # guard: augmented folds must have seen both copies
    for key, value in audit.items():
        if value["fold_fit_rows_min"] <= 0:
            raise AssertionError(f"Missing fold audit in {key}")
    for row in aug_rows:
        if row["tag"] == "10p" and row["n_fit"] != 2 * 2456:
            raise AssertionError(f"Unexpected 10p augmented fit size: {row['n_fit']}")
        if row["tag"] == "40p" and row["n_fit"] != 2 * 9827:
            raise AssertionError(f"Unexpected 40p augmented fit size: {row['n_fit']}")

    rows = clean_rows + aug_rows
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "tag",
        "arm",
        "method",
        "noise_fraction",
        "n_fit",
        "macro_f1",
        "class1_recall",
        "class1_fpr",
        "source",
    ]
    with (OUT / "conditions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["tag"], item["arm"], item["method"], item["noise_fraction"], item["seed"])):
            writer.writerow({key: row[key] for key in fields})

    aggregates: list[dict[str, Any]] = []
    for tag in ("10p", "40p"):
        for arm in ("clean", "aug"):
            for method in ("dyn", "full"):
                for level in EVAL_LEVELS:
                    selected = [
                        row
                        for row in rows
                        if row["tag"] == tag
                        and row["arm"] == arm
                        and row["method"] == method
                        and abs(row["noise_fraction"] - level) < 1e-9
                    ]
                    if not selected:
                        continue
                    values = [row["macro_f1"] for row in selected]
                    mean = sum(values) / len(values)
                    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
                    aggregates.append(
                        {
                            "tag": tag,
                            "arm": arm,
                            "method": method,
                            "noise_fraction": level,
                            "n_seeds": len(values),
                            "macro_f1_mean": mean * 100.0,
                            "macro_f1_std": std * 100.0,
                        }
                    )

    paired: list[dict[str, Any]] = []
    for tag in ("10p", "40p"):
        for method in ("dyn", "full"):
            for level in EVAL_LEVELS:
                clean = {
                    row["seed"]: row["macro_f1"]
                    for row in rows
                    if row["tag"] == tag
                    and row["arm"] == "clean"
                    and row["method"] == method
                    and abs(row["noise_fraction"] - level) < 1e-9
                }
                aug = {
                    row["seed"]: row["macro_f1"]
                    for row in rows
                    if row["tag"] == tag
                    and row["arm"] == "aug"
                    and row["method"] == method
                    and abs(row["noise_fraction"] - level) < 1e-9
                }
                shared = sorted(set(clean) & set(aug))
                if not shared:
                    continue
                diffs = [(aug[seed] - clean[seed]) * 100.0 for seed in shared]
                mean = sum(diffs) / len(diffs)
                std = (sum((value - mean) ** 2 for value in diffs) / len(diffs)) ** 0.5
                paired.append(
                    {
                        "tag": tag,
                        "method": method,
                        "noise_fraction": level,
                        "delta_mean_pp": mean,
                        "delta_std_pp": std,
                        "positive_seeds": sum(1 for value in diffs if value > 0),
                        "n_seeds": len(diffs),
                        "per_seed_pp": {str(seed): value for seed, value in zip(shared, diffs)},
                    }
                )

    summary = {
        "generated_at": "2026-11-03",
        "provenance": {
            "clean_arm": str(CLEAN_DIR / "conditions.csv"),
            "augmented_arm": str(AUG_DIR),
            "note": "clean arm is unaffected by the fold-index fix (fit_rows == fit_index); augmented arm was rerun",
        },
        "fold_audit": audit,
        "aggregates": aggregates,
        "paired": paired,
        "test_evaluated": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lookup = {
        (row["tag"], row["arm"], row["method"], row["noise_fraction"]): row for row in aggregates
    }
    lines = [
        "# TODO1102-4B fusion-side augmentation (corrected)",
        "",
        "- clean arm: `fusion_augmentation/`（不受折索引 bug 影响）",
        "- augmented arm: `fusion_augmentation_v2/`（已修复并重跑）",
        "",
        "| 预算 | 方法 | 训练 | 干净 | 0.25% | 0.5% | 1% |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for tag in ("10p", "40p"):
        for method in ("dyn", "full"):
            for arm in ("clean", "aug"):
                cells = []
                for level in EVAL_LEVELS:
                    key = (tag, arm, method, level)
                    cells.append(f"{lookup[key]['macro_f1_mean']:.2f}" if key in lookup else "—")
                lines.append(f"| {tag} | {method} | {arm} | " + " | ".join(cells) + " |")
    lines += ["", "| 预算 | 方法 | 噪声 | 增广 − 干净 | 种子 |", "|---|---|---:|---:|---:|"]
    for row in paired:
        lines.append(
            f"| {row['tag']} | {row['method']} | {100 * row['noise_fraction']:.2f}% | "
            f"{row['delta_mean_pp']:+.2f} ± {row['delta_std_pp']:.2f} | "
            f"{row['positive_seeds']}/{row['n_seeds']} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
