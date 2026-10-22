"""E18: unified low-label benchmark at 1/5/10/20/40%.

Trains F15/DYN HGB, frozen MOMENT orig/diff/dual logistic, legacy masked TCN,
TCN+DYN, DYN+MOMENT, DYN+MOMENT/diff and full stacking, plus MultiRocket and
MiniRocket.  Validation only; no test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
sys.path.insert(0, str(Path("vendor")))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import multirocket.multirocket as multirocket
import minirocket.minirocket as minirocket
from tcn_moment import run_e08
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.config import load_config, replace, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.run_e12_tcn_stack import _tcn_predict, _train_tcn

SEEDS = (42, 44, 45, 46)
FRACTIONS = (0.01, 0.05, 0.1, 0.2, 0.4)
E08 = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
E01 = Path("artifacts/improvements/e01_20261017_v2_features")
E09 = Path("artifacts/improvements/e09_diff_features")
METHODS = (
    "f15_hgb",
    "dyn_hgb",
    "moment_logit",
    "diff_logit",
    "moment_diff_logit",
    "tcn",
    "tcn_dyn",
    "dyn_moment",
    "dyn_moment_diff",
    "full",
    "multirocket",
    "minirocket",
)
HGB_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}


def _tag(fraction: float) -> str:
    return f"{int(round(fraction * 100))}p"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _interp_z(x: np.ndarray, mask: np.ndarray, length: int = 256) -> np.ndarray:
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


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _fit_multi(train: np.ndarray, val: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    np.random.seed(seed)
    base = multirocket.fit(train, num_features=2000)
    diff = multirocket.fit(np.diff(train, axis=1), num_features=2000)
    train_t = multirocket.transform(train, np.diff(train, axis=1), base, diff, 4)
    val_t = multirocket.transform(val, np.diff(val, axis=1), base, diff, 4)
    model = make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(alphas=np.asarray([100.0, 1000.0]), cv=3),
    ).fit(np.nan_to_num(train_t), y)
    return _softmax(model.decision_function(np.nan_to_num(val_t)))


def _fit_mini(train: np.ndarray, val: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    np.random.seed(seed)
    parameters = minirocket.fit(train.astype(np.float32), num_features=10000)
    train_t = minirocket.transform(train.astype(np.float32), parameters)
    val_t = minirocket.transform(val.astype(np.float32), parameters)
    model = make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(alphas=np.asarray([100.0, 1000.0]), cv=3),
    ).fit(np.nan_to_num(train_t), y)
    return _softmax(model.decision_function(np.nan_to_num(val_t)))


def _meta(oof: np.ndarray, y: np.ndarray, val: np.ndarray, seed: int) -> np.ndarray:
    grid = [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    selected, _ = run_e08._cv_select(
        lambda params: LogisticRegression(C=params["C"], max_iter=5000, random_state=seed),
        grid,
        oof,
        y,
        folds=3,
        seed=seed,
    )
    return LogisticRegression(
        C=selected["C"], max_iter=5000, random_state=seed
    ).fit(oof, y).predict_proba(val)


def _evaluate(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    pred = prob.argmax(1)
    metrics = run_e08._evaluate(y, pred, prob, ["0", "1", "2"], 2)
    return {"macro_f1": float(metrics["macro_f1"]), "battery_recall": float(metrics["battery"]["recall"]),
            "battery_fpr": float(metrics["battery"]["false_positive_rate"]), "pred": pred, "prob": prob}


def _write(root: Path, tag: str, seed: int, method: str, ids: np.ndarray, y: np.ndarray, prob: np.ndarray) -> None:
    path = root / f"seed{seed}" / "conditions" / tag / method / "predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "split", "y_true", "y_pred", "score_0", "score_1", "score_2"])
        for sample_id, truth, prediction, scores in zip(ids, y, prob.argmax(1), prob, strict=True):
            writer.writerow([sample_id, "validation", int(truth), int(prediction), *[f"{v:.12f}" for v in scores]])


def run(protocol_path: Path) -> None:
    p = json.loads(protocol_path.read_text())
    if p.get("test_evaluated") or p.get("automatic_expansion"):
        raise ValueError("E18 is validation-only")
    root = Path(p["output_dir"]) / p["suite"]
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("E18 requires CUDA")
    summary = {"test_evaluated": False, "protocol_sha256": _sha(protocol_path), "conditions": []}
    for seed in SEEDS:
        config = with_random_seed(load_config(Path("configs/moment.yaml")), seed)
        raw = load_dataset(replace(config.data, normalize="none"))
        with np.load(E08 / f"seed{seed}" / "features.npz", allow_pickle=False) as archive:
            dyn_train = archive["train_features"]
            dyn_val = archive["validation_features"]
        moment_train, _ = run_e08._load_moment_split(E01, seed=seed, split="train", ids=raw.ids_train)
        moment_val, _ = run_e08._load_moment_split(E01, seed=seed, split="validation", ids=raw.ids_val)
        diff_train, _ = run_e08._load_moment_split(E09, seed=seed, split="train", ids=raw.ids_train)
        diff_val, _ = run_e08._load_moment_split(E09, seed=seed, split="validation", ids=raw.ids_val)
        md_train = np.hstack([moment_train, diff_train])
        md_val = np.hstack([moment_val, diff_val])
        x_interp_train = _interp_z(raw.x_train, raw.mask_train)
        x_interp_val = _interp_z(raw.x_val, raw.mask_val)
        for fraction in FRACTIONS:
            tag = _tag(fraction)
            indices = stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
            y_train = raw.y_train[indices]
            # OOF branches
            oof_dyn = np.zeros((len(indices), 3))
            oof_moment = np.zeros((len(indices), 3))
            oof_diff = np.zeros((len(indices), 3))
            oof_tcn = np.zeros((len(indices), 3))
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (tr, va) in enumerate(splitter.split(indices, y_train)):
                dyn_h = run_e08._hgb_factory(HGB_PARAMS, seed).fit(dyn_train[indices][tr], y_train[tr])
                oof_dyn[va] = dyn_h.predict_proba(dyn_train[indices][va])
                mom = run_e08._logit_factory({"C": 1.0}, seed).fit(moment_train[indices][tr], y_train[tr])
                oof_moment[va] = mom.predict_proba(moment_train[indices][va])
                dif = run_e08._logit_factory({"C": 1.0}, seed).fit(diff_train[indices][tr], y_train[tr])
                oof_diff[va] = dif.predict_proba(diff_train[indices][va])
                tcn_fold = _train_tcn(
                    raw.x_train[indices][tr], raw.mask_train[indices][tr], y_train[tr],
                    epochs=50, seed=seed * 100 + fold, device=device,
                )
                oof_tcn[va] = _tcn_predict(tcn_fold, raw.x_train[indices][va], raw.mask_train[indices][va], device)
            # Full branches
            dyn_h = run_e08._hgb_factory(HGB_PARAMS, seed).fit(dyn_train[indices], y_train)
            f15_h = run_e08._hgb_factory(HGB_PARAMS, seed).fit(dyn_train[indices][:, :15], y_train)
            mom = run_e08._logit_factory({"C": 1.0}, seed).fit(moment_train[indices], y_train)
            dif = run_e08._logit_factory({"C": 1.0}, seed).fit(diff_train[indices], y_train)
            md_logit = run_e08._logit_factory({"C": 1.0}, seed).fit(md_train[indices], y_train)
            tcn = _train_tcn(
                raw.x_train[indices], raw.mask_train[indices], y_train,
                epochs=50, seed=seed * 100 + 99, device=device,
            )
            probs = {
                "f15_hgb": f15_h.predict_proba(dyn_val[:, :15]),
                "dyn_hgb": dyn_h.predict_proba(dyn_val),
                "moment_logit": mom.predict_proba(moment_val),
                "diff_logit": dif.predict_proba(diff_val),
                "moment_diff_logit": md_logit.predict_proba(md_val),
                "tcn": _tcn_predict(tcn, raw.x_val, raw.mask_val, device),
            }
            probs["tcn_dyn"] = _meta(np.hstack([oof_tcn, oof_dyn]), y_train, np.hstack([probs["tcn"], probs["dyn_hgb"]]), seed)
            probs["dyn_moment"] = _meta(np.hstack([oof_dyn, oof_moment]), y_train, np.hstack([probs["dyn_hgb"], probs["moment_logit"]]), seed)
            probs["dyn_moment_diff"] = _meta(np.hstack([oof_dyn, oof_moment, oof_diff]), y_train, np.hstack([probs["dyn_hgb"], probs["moment_logit"], probs["diff_logit"]]), seed)
            probs["full"] = _meta(np.hstack([oof_dyn, oof_moment, oof_diff, oof_tcn]), y_train, np.hstack([probs["dyn_hgb"], probs["moment_logit"], probs["diff_logit"], probs["tcn"]]), seed)
            # Rocket baselines
            probs["multirocket"] = _fit_multi(x_interp_train[indices], x_interp_val, y_train, seed)
            probs["minirocket"] = _fit_mini(x_interp_train[indices], x_interp_val, y_train, seed)
            for method in METHODS:
                result = _evaluate(raw.y_val, probs[method])
                _write(root, tag, seed, method, raw.ids_val, raw.y_val, result["prob"])
                record = {
                    "seed": seed, "fraction": fraction, "tag": tag, "method": method,
                    "n_train": len(indices),
                    "macro_f1": result["macro_f1"],
                    "battery_recall": result["battery_recall"],
                    "battery_fpr": result["battery_fpr"],
                    "test_evaluated": False,
                }
                summary["conditions"].append(record)
                atomic_write_json(root / f"seed{seed}" / "conditions" / tag / method / "metrics.json", record)
                print(f"E18 {seed} {tag} {method}: {100*result['macro_f1']:.2f}", flush=True)
            atomic_write_json(root / f"seed{seed}" / f"seed_{tag}.json", {"fraction": fraction, "tag": tag, "n_train": len(indices)})
    _summarize(root, summary)


def _summarize(root: Path, summary: dict[str, Any]) -> None:
    rows = []
    for fraction in FRACTIONS:
        tag = _tag(fraction)
        for method in METHODS:
            selected = [c for c in summary["conditions"] if c["tag"] == tag and c["method"] == method]
            values = [100 * c["macro_f1"] for c in selected]
            recalls = [100 * c["battery_recall"] for c in selected]
            fprs = [100 * c["battery_fpr"] for c in selected]
            rows.append({"tag": tag, "method": method, "n": len(values), "mean": float(np.mean(values)),
                         "sample_sd": float(np.std(values, ddof=1)),
                         "per_seed": values, "battery_recall": float(np.mean(recalls)),
                         "battery_fpr": float(np.mean(fprs))})
    summary["aggregate"] = rows
    summary["audit"] = {"conditions": len(summary["conditions"]), "rows": len(summary["conditions"]) * 3510,
                        "test_rows": 0}
    atomic_write_json(root / "summary.json", summary)
    lines = ["# E18：统一下采样低标签基准（validation-only）", "",
             "**1%/5%/10%/20%/40%；4 seed；固定超参；0 test。**", "",
             "| 方法 | 1% | 5% | 10% | 20% | 40% |", "|---|---:|---:|---:|---:|---:|"]
    lookup = {(r["tag"], r["method"]): r for r in rows}
    for method in METHODS:
        cells = [f"{lookup[(_tag(f), method)]['mean']:.2f} ± {lookup[(_tag(f), method)]['sample_sd']:.2f}" for f in FRACTIONS]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "## 审计", "", f"- {summary['audit']['conditions']} 条件 / {summary['audit']['rows']} 行；0 test。", "",
              "## 固定设置", "",
              "- HGB: max_iter400/lr0.08/leaves31/min_samples20/l2=1.0，所有预算固定；F15/DYN 共用。",
              "- MOMENT logistic C=1.0；TCN 50 epoch、5折OOF+全量、无validation选模；meta C在OOF内选。",
              "- MultiRocket 2000 features、MiniRocket 10000 features；interp_z 256；RidgeCV {100,1000} 训练内3折。",
              "- 所有方法共用同seed训练子集和validation行；validation未用于超参选择（meta C用OOF）。"]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/experiments/improvements/e18_low_label_protocol.json")
    args = parser.parse_args()
    run(Path(args.protocol))


if __name__ == "__main__":
    main()
