"""E18d: F15/DYN RBF-SVM at 5% and 20% (fills E08 gap)."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from tcn_moment import run_e08
from tcn_moment.config import load_config, replace, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.io_utils import atomic_write_json

E08 = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
OUT = Path("artifacts/improvements/e18d_svm/e18d_svm_20261020_v1")
SEEDS = (42, 44, 45, 46)
FRACTIONS = (0.05, 0.2)
CFG = {"f15_svm": 100.0, "dyn_svm": 10.0}


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"test_evaluated": False, "conditions": []}
    for seed in SEEDS:
        cfg = with_random_seed(load_config(Path("configs/moment.yaml")), seed)
        raw = load_dataset(replace(cfg.data, normalize="none"))
        with np.load(E08 / f"seed{seed}" / "features.npz", allow_pickle=False) as a:
            train = a["train_features"]
            val = a["validation_features"]
        for fraction in FRACTIONS:
            tag = f"{int(round(fraction*100))}p"
            idx = stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
            for method, c in CFG.items():
                x = train[idx][:, :15] if method == "f15_svm" else train[idx]
                v = val[:, :15] if method == "f15_svm" else val
                model = run_e08._svm_factory({"C": c, "gamma": "scale"}).fit(x, raw.y_train[idx])
                scores = softmax(model.decision_function(v))
                pred = scores.argmax(1)
                metrics = run_e08._evaluate(raw.y_val, pred, scores, ["0", "1", "2"], 2)
                path = OUT / f"seed{seed}" / "conditions" / tag / method
                path.mkdir(parents=True, exist_ok=True)
                with (path / "predictions.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["sample_id", "split", "y_true", "y_pred", "score_0", "score_1", "score_2"])
                    for sid, y, p, s in zip(raw.ids_val, raw.y_val, pred, scores, strict=True):
                        w.writerow([sid, "validation", int(y), int(p), *[f"{z:.12f}" for z in s]])
                rec = {"seed": seed, "fraction": fraction, "tag": tag, "method": method, "C": c,
                       "macro_f1": float(metrics["macro_f1"]),
                       "battery_recall": float(metrics["battery"]["recall"]),
                       "battery_fpr": float(metrics["battery"]["false_positive_rate"]),
                       "test_evaluated": False}
                summary["conditions"].append(rec)
                atomic_write_json(path / "metrics.json", rec)
                print(f"E18d {seed} {tag} {method}: {100*rec['macro_f1']:.2f}", flush=True)
    atomic_write_json(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
