"""E18b: TCN+meta / TCN+MOM controls at all E18 budgets using 5-fold OOF.

TCN fold models are retrained; DYN/MOMENT OOF is cheap. Validation base probs are
reused from E18 saved predictions.  Validation only, no test.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("vendor")))
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from tcn_moment import run_e08
from tcn_moment.config import load_config, replace, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.label_semantics import BATTERY_LABEL, SEMANTICS_VERSION
from tcn_moment.run_e12_tcn_stack import _tcn_predict, _train_tcn
from tcn_moment.run_e18_low_label import (
    E01,
    E08,
    E09,
    FRACTIONS,
    HGB_PARAMS,
    SEEDS,
    _evaluate,
    _meta,
    _tag,
    _write,
)

OUT = Path("artifacts/improvements/e18b_tcn_controls/e18b_tcn_controls_20261020_v1")
E18 = Path("artifacts/improvements/e18_low_label/e18_low_label_20261020_v1")
METHODS = ("tcn_meta", "tcn_md", "tcn_moment")


def _read(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["sample_id"] for r in rows]
    y = np.array([int(r["y_true"]) for r in rows])
    p = np.array([[float(r["score_0"]), float(r["score_1"]), float(r["score_2"])] for r in rows])
    return ids, y, p


def _align(ids_ref, ids, p):
    m = {s: i for i, s in enumerate(ids)}
    if set(ids_ref) != set(ids):
        raise ValueError("ID mismatch")
    return p[[m[s] for s in ids_ref]]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("E18b requires CUDA")
    device = torch.device("cuda")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Preserve historical E18b; use a new suite for semantic v2")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"test_evaluated": False, "conditions": []}
    for seed in SEEDS:
        config = with_random_seed(load_config(Path("configs/moment.yaml")), seed)
        raw = load_dataset(replace(config.data, normalize="none"))
        with np.load(E08 / f"seed{seed}" / "features.npz", allow_pickle=False) as a:
            dyn_train = a["train_features"]
        moment_train, _ = run_e08._load_moment_split(E01, seed=seed, split="train", ids=raw.ids_train)
        diff_train, _ = run_e08._load_moment_split(E09, seed=seed, split="train", ids=raw.ids_train)
        md_train = np.hstack([moment_train, diff_train])
        # validation base probs from E18
        base = {}
        ids_ref = None
        for e18_method, key in (("tcn", "tcn"), ("dyn_hgb", "dyn"), ("moment_logit", "moment"), ("moment_diff_logit", "md")):
            path = E18 / f"seed{seed}" / "conditions" / "1p" / e18_method / "predictions.csv"
            parts = {tag: _read(E18 / f"seed{seed}" / "conditions" / tag / e18_method / "predictions.csv") for tag in ("1p", "5p", "10p", "20p", "40p")}
            base[key] = parts
        for fraction in FRACTIONS:
            tag = _tag(fraction)
            indices = stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
            y_train = raw.y_train[indices]
            oof = {k: np.zeros((len(indices), 3)) for k in ("tcn", "dyn", "moment", "md")}
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (tr, va) in enumerate(splitter.split(indices, y_train)):
                h = run_e08._hgb_factory(HGB_PARAMS, seed).fit(dyn_train[indices][tr], y_train[tr])
                oof["dyn"][va] = h.predict_proba(dyn_train[indices][va])
                mo = run_e08._logit_factory({"C": 1.0}, seed).fit(moment_train[indices][tr], y_train[tr])
                oof["moment"][va] = mo.predict_proba(moment_train[indices][va])
                md = run_e08._logit_factory({"C": 1.0}, seed).fit(md_train[indices][tr], y_train[tr])
                oof["md"][va] = md.predict_proba(md_train[indices][va])
                t = _train_tcn(raw.x_train[indices][tr], raw.mask_train[indices][tr], y_train[tr], epochs=50, seed=seed * 100 + fold, device=device)
                oof["tcn"][va] = _tcn_predict(t, raw.x_train[indices][va], raw.mask_train[indices][va], device)
            np.savez_compressed(OUT / f"oof_seed{seed}_{tag}.npz", **oof, y=y_train)
            ids_ref = base["tcn"][tag][0]
            y = base["tcn"][tag][1]
            p = {k: _align(ids_ref, base[k][tag][0], base[k][tag][2]) for k in base}
            probs = {
                "tcn_meta": _meta(oof["tcn"], y_train, p["tcn"], seed),
                "tcn_md": _meta(np.hstack([oof["tcn"], oof["md"]]), y_train, np.hstack([p["tcn"], p["md"]]), seed),
                "tcn_moment": _meta(np.hstack([oof["tcn"], oof["moment"]]), y_train, np.hstack([p["tcn"], p["moment"]]), seed),
            }
            for method, prob in probs.items():
                res = _evaluate(y, prob)
                _write(OUT, tag, seed, method, np.asarray(ids_ref), y, res["prob"])
                rec = {"seed": seed, "fraction": fraction, "tag": tag, "method": method, "n_train": len(indices), "macro_f1": res["macro_f1"], "battery_recall": res["battery_recall"], "battery_fpr": res["battery_fpr"], "test_evaluated": False}
                rec.update(battery_label=BATTERY_LABEL, label_semantics_version=SEMANTICS_VERSION)
                summary["conditions"].append(rec)
                atomic_write_json(OUT / f"seed{seed}" / "conditions" / tag / method / "metrics.json", rec)
                print(f"E18b {seed} {tag} {method}: {100*res['macro_f1']:.2f}", flush=True)
    atomic_write_json(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
