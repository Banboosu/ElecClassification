"""TODO1102-6A(fusion): does a 36-D DYN block change the fused system?

The 6a screen showed the 60-D dynamics block can be cut to 36 features
(length+trend+roughness+shape) at a cost of 0.03 pp at 40% labels and 0.05 pp at
10%.  The open question was whether the *fused* system keeps that insensitivity.

Retraining TCNs to answer that costs hours of GPU time and adds a fresh source of
variance.  The TCN and MOMENT/ΔP blocks do not depend on the DYN feature count, so
this audit holds them fixed at the frozen E12/E09 artifacts and refits only the DYN
block (same 5-fold assignment, same frozen HGB parameters) plus the meta-learner.
That isolates the feature-count effect on the fused system.

Validation only; the test split is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tcn_moment import run_e08
from tcn_moment.data import stratified_train_subset_indices
from tcn_moment.dynamics_features import DYNAMICS_FEATURE_NAMES, FEATURE_GROUPS

SEEDS = (42, 44, 45, 46)
TAGS = ("10p", "40p")
E08 = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
E09 = Path("artifacts/improvements/e09_diff_stream")
E12 = {
    "10p": Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_10p_20261020_v1"),
    "40p": Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_20261020_v1"),
}
GROUPS = ("length", "trend", "roughness", "shape")
DEFAULT_OUTPUT = Path("artifacts/analysis/todo1102_20261102/fusion_dyn36")


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = np.asarray([row["sample_id"] for row in rows], dtype=str)
    y = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    probs = np.asarray(
        [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in rows],
        dtype=np.float64,
    )
    return ids, y, probs


def _branch(path: Path, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids, y, probs = _read_predictions(path)
    if np.array_equal(ids, reference):
        return probs, y
    mapping = {value: index for index, value in enumerate(ids)}
    if set(ids.tolist()) != set(reference.tolist()):
        raise ValueError(f"Prediction ID sets differ: {path}")
    return probs[[mapping[value] for value in reference]], y


def _meta(
    oof: np.ndarray, y: np.ndarray, evaluation: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
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
    return model.predict(evaluation), selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--e08", type=Path, default=E08)
    parser.add_argument("--e09", type=Path, default=E09)
    parser.add_argument("--e12-root", type=Path, default=None)
    parser.add_argument("--groups", nargs="+", default=list(GROUPS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    e08 = args.e08
    e09 = args.e09
    e12 = (
        {
            "10p": args.e12_root / "e12_tcn_stack_10p_20261020_v1",
            "40p": args.e12_root / "e12_tcn_stack_20261020_v1",
        }
        if args.e12_root is not None
        else dict(E12)
    )
    keep = {name for group in args.groups for name in FEATURE_GROUPS[group]}
    columns = [index for index, name in enumerate(DYNAMICS_FEATURE_NAMES) if name in keep]
    print(f"keeping {len(columns)} of {len(DYNAMICS_FEATURE_NAMES)} DYN features", flush=True)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        archive = np.load(e08 / f"seed{seed}" / "features.npz", allow_pickle=True)
        dyn36_train = archive["train_features"][:, columns]
        dyn36_val = archive["validation_features"][:, columns]
        train_ids = archive["train_ids"].astype(str)
        val_ids = archive["validation_ids"].astype(str)
        seed_results = json.loads(
            (e08 / f"seed{seed}" / "seed_results.json").read_text(encoding="utf-8")
        )
        params_by_tag = {
            row["tag"]: row["selected_params"]
            for row in seed_results["conditions"]
            if row["method"] == "dyn_hgb"
        }

        for tag in TAGS:
            fraction = 0.1 if tag == "10p" else 0.4
            indices = stratified_train_subset_indices(
                archive["train_labels"], train_ids, fraction, seed
            )
            oof = np.load(e12[tag] / f"seed{seed}" / f"oof_{tag}.npz", allow_pickle=False)
            if not np.array_equal(oof["indices"], indices):
                raise ValueError(f"E12 OOF indices differ for seed={seed} tag={tag}")
            folds = oof["fold_assignment"]
            y_train = oof["y_train"]

            md_val, md_y = _branch(
                e09 / f"e09_diff_stream_20261020_seed{seed}" / "conditions" / tag
                / "moment_diff_logit" / "predictions.csv",
                val_ids,
            )
            tcn_val, tcn_y = _branch(
                e12[tag] / f"seed{seed}" / "conditions" / tag / "tcn_full" / "predictions.csv",
                val_ids,
            )
            dyn60_val, dyn_y = _branch(
                e08 / f"seed{seed}" / "conditions" / tag / "dyn_hgb" / "predictions.csv",
                val_ids,
            )
            if not (np.array_equal(md_y, tcn_y) and np.array_equal(md_y, dyn_y)):
                raise ValueError("Validation labels disagree across branches")

            subset = dyn36_train[indices]
            params = params_by_tag[tag]
            oof_dyn36 = np.zeros((len(indices), 3))
            for fold in np.unique(folds):
                hold = folds == fold
                model = run_e08._hgb_factory(params, seed).fit(subset[~hold], y_train[~hold])
                oof_dyn36[hold] = model.predict_proba(subset[hold])
            hgb_full = run_e08._hgb_factory(params, seed).fit(subset, y_train)
            dyn36_val_prob = hgb_full.predict_proba(dyn36_val)

            variants = {
                "full_60": (
                    np.hstack([oof["oof_dyn"], oof["oof_md"], oof["oof_tcn"]]),
                    np.hstack([dyn60_val, md_val, tcn_val]),
                    60,
                ),
                "full_36": (
                    np.hstack([oof_dyn36, oof["oof_md"], oof["oof_tcn"]]),
                    np.hstack([dyn36_val_prob, md_val, tcn_val]),
                    len(columns),
                ),
                "tcn_dyn_60": (
                    np.hstack([oof["oof_dyn"], oof["oof_tcn"]]),
                    np.hstack([dyn60_val, tcn_val]),
                    60,
                ),
                "tcn_dyn_36": (
                    np.hstack([oof_dyn36, oof["oof_tcn"]]),
                    np.hstack([dyn36_val_prob, tcn_val]),
                    len(columns),
                ),
            }
            for name, (oof_matrix, val_matrix, dim) in variants.items():
                prediction, selected = _meta(oof_matrix, y_train, val_matrix, seed)
                score = float(f1_score(md_y, prediction, average="macro"))
                rows.append(
                    {
                        "seed": seed,
                        "tag": tag,
                        "variant": name,
                        "n_features": dim,
                        "macro_f1": score,
                        "meta_C": float(selected["C"]),
                    }
                )
                print(f"seed {seed} {tag} {name}: {100 * score:.3f}%", flush=True)

    with (args.output / "conditions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["seed", "tag", "variant", "n_features", "macro_f1", "meta_C"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary: dict[str, Any] = {"test_evaluated": False, "groups": args.groups, "by_tag": {}}
    for tag in TAGS:
        block: dict[str, Any] = {}
        for variant in ("full_60", "full_36", "tcn_dyn_60", "tcn_dyn_36"):
            values = [
                row["macro_f1"] for row in rows if row["tag"] == tag and row["variant"] == variant
            ]
            block[variant] = {
                "n": len(values),
                "macro_f1_mean": float(np.mean(values) * 100),
                "macro_f1_std": float(np.std(values, ddof=0) * 100),
            }
        for wide, narrow in (("full_60", "full_36"), ("tcn_dyn_60", "tcn_dyn_36")):
            diffs = [
                (
                    row["macro_f1"]
                    - next(
                        other["macro_f1"]
                        for other in rows
                        if other["seed"] == row["seed"]
                        and other["tag"] == tag
                        and other["variant"] == narrow
                    )
                )
                * 100
                for row in rows
                if row["tag"] == tag and row["variant"] == wide
            ]
            block[f"{wide}_minus_36"] = {
                "mean_pp": float(np.mean(diffs)),
                "std_pp": float(np.std(diffs)),
                "positive_seeds": int(sum(1 for value in diffs if value > 0)),
                "n": len(diffs),
            }
        summary["by_tag"][tag] = block

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["# TODO1102-6A(fusion) 36-D DYN inside the fused system", ""]
    for tag, block in summary["by_tag"].items():
        lines += [f"## {tag}", "", "| 变体 | DYN 维数 | Macro-F1 |", "|---|---:|---:|"]
        for variant in ("full_60", "full_36", "tcn_dyn_60", "tcn_dyn_36"):
            entry = block[variant]
            dim = 60 if variant.endswith("60") else len(columns)
            lines.append(
                f"| {variant} | {dim} | {entry['macro_f1_mean']:.2f} ± {entry['macro_f1_std']:.2f} |"
            )
        for wide in ("full_60", "tcn_dyn_60"):
            entry = block[f"{wide}_minus_36"]
            lines.append(
                f"\n- {wide} − 36 维：{entry['mean_pp']:+.2f} ± {entry['std_pp']:.2f} pp"
                f"（{entry['positive_seeds']}/{entry['n']} 种子 60 维更高）"
            )
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
