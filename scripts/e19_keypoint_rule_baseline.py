"""E19: charging key-point rule features + simple classifiers (non-deep-learning baseline).

Protocol: configs/experiments/improvements/e19_keypoint_rule_baseline_protocol.json.
Uses the same seeds, seed-specific train/validation partitions and nested labelled
subsets as E18, so every result is paired with the saved E18 validation results.
CPU only, validation only; the test split is never scored.

Run from the project root:
    PYTHONPATH=src python scripts/e19_keypoint_rule_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.dynamics_features import extract_dynamics_features
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.label_semantics import BATTERY_LABEL, class_index

PROTOCOL = Path("configs/experiments/improvements/e19_keypoint_rule_baseline_protocol.json")
DEFAULT_OUTPUT = Path("artifacts/analysis/e19_keypoint_rule_20261105")
KEYPOINT_NAMES = (
    "length",
    "peak_power",
    "cc_fraction",
    "cc_rise",
    "cc_slope",
    "cv_slope",
    "half_decay",
    "reaches_half",
    "end_ratio",
    "max_drop",
    "profile_residual",
)
E18_METHODS = ("f15_hgb", "dyn_hgb", "tcn_dyn", "full", "minirocket", "multirocket")
OWN_METHODS = ("keypoint_lr", "keypoint_hgb", "dyn_lr", "dyn_hgb_repro")


def _tag(fraction: float) -> str:
    return f"{round(fraction * 100):g}p"


def _ols(index: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Least-squares line; constant fit when fewer than three points."""
    if len(values) < 3:
        return 0.0, float(values.mean()) if len(values) else 0.0
    t = index - index.mean()
    slope = float((t * (values - values.mean())).sum() / (t * t).sum())
    return slope, float(values.mean() - slope * index.mean())


def keypoint_features(x: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    padded = np.concatenate([np.full(2, np.nan), x, np.full(2, np.nan)])
    smooth = np.nanmedian(sliding_window_view(padded, 5), axis=1)
    t_pk = int(np.argmax(smooth))
    p_pk = float(smooth[t_pk])
    scale = p_pk if abs(p_pk) > 1e-9 else 1e-9
    index = np.arange(n, dtype=np.float64)

    cc_slope, cc_icpt = _ols(index[: t_pk + 1], x[: t_pk + 1])
    cv_slope, cv_icpt = _ols(index[t_pk:], x[t_pk:])
    fitted = np.where(index <= t_pk, cc_slope * index + cc_icpt, cv_slope * index + cv_icpt)
    residual = float(np.sqrt(np.mean((x - fitted) ** 2)))

    below = np.flatnonzero(smooth[t_pk + 1 :] <= 0.5 * p_pk)
    reaches = float(len(below) > 0)
    half = float(below[0] + 1) if len(below) else float(n - t_pk)
    drops = -np.diff(x)
    max_drop = max(float(drops.max()), 0.0) if len(drops) else 0.0
    return [
        float(n),
        p_pk,
        t_pk / n,
        (p_pk - float(np.median(x[:5]))) / scale,
        cc_slope / scale,
        cv_slope / scale,
        half,
        reaches,
        float(np.median(x[-5:])) / scale,
        max_drop / scale,
        residual / scale,
    ]


def keypoint_matrix(x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.asarray(
        [keypoint_features(row[:length]) for row, length in zip(x, lengths, strict=True)]
    )


def _logit(c: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=c, max_iter=5000, random_state=seed)
    )


def _hgb(params: dict[str, Any], seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**params, random_state=seed)


def fit_logit_cv(x: np.ndarray, y: np.ndarray, grid: list[float], seed: int) -> tuple[Any, float]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_c, best_score = grid[0], -1.0
    for c in grid:
        scores = [
            f1_score(y[va], _logit(c, seed).fit(x[tr], y[tr]).predict(x[va]), average="macro")
            for tr, va in splitter.split(x, y)
        ]
        if float(np.mean(scores)) > best_score:
            best_c, best_score = c, float(np.mean(scores))
    return _logit(best_c, seed).fit(x, y), best_c


def evaluate(y: np.ndarray, prob: np.ndarray, battery: int) -> dict[str, float]:
    pred = prob.argmax(axis=1)
    negatives = y != battery
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_f1": [float(v) for v in f1_score(y, pred, average=None)],
        "battery_recall": float(np.mean(pred[y == battery] == battery)),
        "battery_fpr": float(np.mean(pred[negatives] == battery)),
    }


def e18_dir(protocol: dict[str, Any], tag: str) -> Path:
    key = "e18_100p" if tag == "100p" else "e18"
    return Path(protocol["comparison_suite"][key])


def e18_metric(protocol: dict[str, Any], seed: int, tag: str, method: str) -> dict[str, Any]:
    path = e18_dir(protocol, tag) / f"seed{seed}" / "conditions" / tag / method / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def check_validation_rows(
    protocol: dict[str, Any], seed: int, ids: np.ndarray, y: np.ndarray
) -> None:
    for tag in ("10p", "100p"):
        path = (
            e18_dir(protocol, tag)
            / f"seed{seed}"
            / "conditions"
            / tag
            / "dyn_hgb"
            / "predictions.csv"
        )
        with path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if [r["sample_id"] for r in rows] != ids.tolist():
            raise RuntimeError(f"Validation ids differ from E18 for seed {seed} ({tag})")
        if [int(r["y_true"]) for r in rows] != y.tolist():
            raise RuntimeError(f"Validation labels differ from E18 for seed {seed} ({tag})")


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = a - b
    half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return {
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)),
        "positive_seeds": int((d > 0).sum()),
        "ci95": [float(d.mean() - half), float(d.mean() + half)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("test_evaluated") or protocol.get("automatic_expansion"):
        raise ValueError("E19 is validation-only")
    if (args.output / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    hgb_params = dict(protocol["hgb_params"])
    grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    fractions = [float(v) for v in protocol["label_fractions"]]

    records: list[dict[str, Any]] = []
    for seed in protocol["seeds"]:
        config = with_random_seed(load_config(Path("configs/moment.yaml")), seed)
        split_path = args.output / "splits" / config.data.split_path.name
        raw = load_dataset(replace(config.data, normalize="none", split_path=split_path))
        battery = class_index(list(raw.label_encoder.classes_), BATTERY_LABEL)
        check_validation_rows(protocol, seed, raw.ids_val, raw.y_val)
        kp_train = keypoint_matrix(raw.x_train, raw.lengths_train)
        kp_val = keypoint_matrix(raw.x_val, raw.lengths_val)
        dyn_train = extract_dynamics_features(raw.x_train, raw.mask_train)
        dyn_val = extract_dynamics_features(raw.x_val, raw.mask_val)
        probabilities: dict[str, np.ndarray] = {}
        for fraction in fractions:
            tag = _tag(fraction)
            idx = stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
            y = raw.y_train[idx]
            if len(idx) != e18_metric(protocol, seed, tag, "full")["n_train"]:
                raise RuntimeError(f"Training subset size differs from E18 for seed {seed} {tag}")
            kp_lr, kp_c = fit_logit_cv(kp_train[idx], y, grid, seed)
            dyn_lr, dyn_c = fit_logit_cv(dyn_train[idx], y, grid, seed)
            models = {
                "keypoint_lr": (kp_lr, kp_val, {"C": kp_c}),
                "keypoint_hgb": (_hgb(hgb_params, seed).fit(kp_train[idx], y), kp_val, {}),
                "dyn_lr": (dyn_lr, dyn_val, {"C": dyn_c}),
                "dyn_hgb_repro": (_hgb(hgb_params, seed).fit(dyn_train[idx], y), dyn_val, {}),
            }
            for method, (model, x_val, params) in models.items():
                prob = model.predict_proba(x_val)
                probabilities[f"{tag}/{method}"] = prob.astype(np.float32)
                record = {"seed": seed, "tag": tag, "method": method, "n_train": len(idx), **params}
                record.update(evaluate(raw.y_val, prob, battery))
                records.append(record)
                print(
                    f"E19 seed{seed} {tag} {method}: {100 * record['macro_f1']:.2f} {params}",
                    flush=True,
                )
        np.savez_compressed(
            args.output / f"predictions_seed{seed}.npz",
            sample_id=raw.ids_val.astype(str),
            y_true=raw.y_val,
            **{k.replace("/", "__"): v for k, v in probabilities.items()},
        )

    summarize(protocol, records, fractions, args.output)


def summarize(
    protocol: dict[str, Any], records: list[dict[str, Any]], fractions: list[float], out: Path
) -> None:
    seeds = protocol["seeds"]
    tags = [_tag(f) for f in fractions]
    own = {(r["tag"], r["method"]): {} for r in records}
    for r in records:
        own[(r["tag"], r["method"])][r["seed"]] = r
    table: dict[str, dict[str, Any]] = {}
    for tag in tags:
        table[tag] = {}
        for method in OWN_METHODS:
            rows = [own[(tag, method)][s] for s in seeds]
            table[tag][method] = {
                "per_seed": [100 * r["macro_f1"] for r in rows],
                "battery_recall": float(np.mean([100 * r["battery_recall"] for r in rows])),
                "battery_fpr": float(np.mean([100 * r["battery_fpr"] for r in rows])),
            }
        for method in E18_METHODS:
            rows = [e18_metric(protocol, s, tag, method) for s in seeds]
            table[tag][f"e18_{method}"] = {
                "per_seed": [100 * r["macro_f1"] for r in rows],
                "battery_recall": float(np.mean([100 * r["battery_recall"] for r in rows])),
                "battery_fpr": float(np.mean([100 * r["battery_fpr"] for r in rows])),
            }
        for entry in table[tag].values():
            v = np.asarray(entry["per_seed"])
            entry["mean"], entry["sd"] = float(v.mean()), float(v.std(ddof=1))

    fidelity = {
        tag: float(
            np.max(
                np.abs(
                    np.asarray(table[tag]["dyn_hgb_repro"]["per_seed"])
                    - np.asarray(table[tag]["e18_dyn_hgb"]["per_seed"])
                )
            )
        )
        for tag in tags
    }
    comparisons = {}
    for tag in tags:
        kp = np.asarray(table[tag]["keypoint_lr"]["per_seed"])
        comparisons[tag] = {
            ref: paired(np.asarray(table[tag][ref]["per_seed"]), kp)
            for ref in (
                "e18_full",
                "e18_tcn_dyn",
                "e18_dyn_hgb",
                "e18_f15_hgb",
                "e18_multirocket",
                "keypoint_hgb",
            )
        }
    c1 = comparisons["1p"]
    criteria = protocol["interpretation_criteria_fixed_before_running"]
    if c1["e18_full"]["mean"] <= 1.0:
        verdict = "challenges_necessity"
    elif all(
        min(
            np.asarray(table["1p"][ref]["per_seed"])
            - np.asarray(table["1p"]["keypoint_lr"]["per_seed"])
        )
        >= 2.0
        for ref in ("e18_tcn_dyn", "e18_dyn_hgb")
    ):
        verdict = "supports_feature_engineering"
    else:
        verdict = "otherwise"

    summary = {
        "experiment_id": "E19",
        "protocol_sha256": sha256_file(PROTOCOL),
        "test_evaluated": False,
        "keypoint_feature_names": list(KEYPOINT_NAMES),
        "table": table,
        "paired_minus_keypoint_lr": comparisons,
        "fidelity_max_abs_diff_dyn_hgb_pp": fidelity,
        "verdict_at_1p": {
            "label": verdict,
            "criterion": criteria.get(verdict, criteria["otherwise"]),
        },
        "records": records,
    }
    atomic_write_json(out / "summary.json", summary)

    names = {
        "keypoint_lr": "Key points + LR",
        "keypoint_hgb": "Key points + HGB",
        "dyn_lr": "DYN + LR",
        "dyn_hgb_repro": "DYN + HGB (E19 rerun)",
        "e18_f15_hgb": "F15 + HGB (E18)",
        "e18_dyn_hgb": "DYN + HGB (E18)",
        "e18_tcn_dyn": "TCN + DYN (E18)",
        "e18_full": "Full fusion (E18)",
        "e18_minirocket": "MiniRocket (E18)",
        "e18_multirocket": "MultiRocket (E18)",
    }
    lines = [
        "# E19 key-point rule baseline (validation Macro-F1, %, 4 seeds)",
        "",
        "| Method | " + " | ".join(tags) + " |",
        "|---|" + "---:|" * len(tags),
    ]
    for key, label in names.items():
        cells = [f"{table[t][key]['mean']:.2f} ± {table[t][key]['sd']:.2f}" for t in tags]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Paired difference (reference − key points + LR), mean [95% CI] (positive seeds):",
        "",
    ]
    lines.append("| Reference | " + " | ".join(tags) + " |")
    lines.append("|---|" + "---:|" * len(tags))
    for ref in comparisons[tags[0]]:
        cells = [
            f"{comparisons[t][ref]['mean']:+.2f} [{comparisons[t][ref]['ci95'][0]:+.2f}, "
            f"{comparisons[t][ref]['ci95'][1]:+.2f}] ({comparisons[t][ref]['positive_seeds']}/4)"
            for t in tags
        ]
        lines.append(f"| {names.get(ref, ref)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Battery (label 1) recall / FPR, key points + LR: "
        + ", ".join(
            f"{t} {table[t]['keypoint_lr']['battery_recall']:.2f}/{table[t]['keypoint_lr']['battery_fpr']:.2f}"
            for t in tags
        ),
        "",
        "Fidelity: max |E19 DYN+HGB − E18 dyn_hgb| per budget (pp): "
        + ", ".join(f"{t} {fidelity[t]:.3f}" for t in tags),
        "",
        f"Pre-registered verdict at 1%: **{verdict}** — {summary['verdict_at_1p']['criterion']}",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
