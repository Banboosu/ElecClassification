"""Deep analysis of E18 saved predictions: per-class, battery AP, paired stats, length strata."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from tcn_moment.label_semantics import BATTERY_LABEL, SEMANTICS_VERSION

ROOT = Path("artifacts/improvements/e18_low_label/e18_low_label_20261020_v1")
E08 = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
OUT = Path("artifacts/improvements/e18_deep_analysis_semantics_v2")
TAGS = ("1p", "5p", "10p", "20p", "40p")
METHODS = ("f15_hgb", "dyn_hgb", "moment_logit", "diff_logit", "moment_diff_logit", "tcn", "tcn_dyn", "dyn_moment", "dyn_moment_diff", "full", "multirocket", "minirocket")
SEEDS = (42, 44, 45, 46)


def read(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if any(r["split"] != "validation" for r in rows):
        raise ValueError("Only validation predictions allowed")
    ids = [r["sample_id"] for r in rows]
    y = np.array([int(r["y_true"]) for r in rows])
    p = np.array([[float(r["score_0"]), float(r["score_1"]), float(r["score_2"])] for r in rows])
    pred = np.array([int(r["y_pred"]) for r in rows])
    return y, p, pred, ids


def macro(y, pred):
    return 100 * f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half, "positive": int((d > 0).sum()), "n": n, "diffs": d.tolist()}


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Use a new output directory; preserve historical E18 deep analysis")
    result = {"test_evaluated": False, "edges_by_seed": {}, "budgets": {}, "length_strata": {},
              "battery_label": BATTERY_LABEL, "label_semantics_version": SEMANTICS_VERSION}
    for tag in TAGS:
        result["budgets"][tag] = {}
        result["length_strata"][tag] = {}
        values = {m: [] for m in METHODS}
        preds = {m: {} for m in METHODS}
        for seed in SEEDS:
            with np.load(E08 / f"seed{seed}" / "features.npz", allow_pickle=False) as archive:
                train_lengths = archive["train_features"][:, 0]
                val_lengths = archive["validation_features"][:, 0]
                expected_ids = archive["validation_ids"].astype(str).tolist()
            edges = np.quantile(train_lengths, [1 / 3, 2 / 3])
            result["edges_by_seed"][str(seed)] = edges.tolist()
            strata = [val_lengths <= edges[0], (val_lengths > edges[0]) & (val_lengths <= edges[1]), val_lengths > edges[1]]
            for m in METHODS:
                y, p, pred, ids = read(ROOT / f"seed{seed}" / "conditions" / tag / m / "predictions.csv")
                if ids != expected_ids:
                    raise ValueError("Feature/prediction validation IDs or order differ")
                if m == "dyn_hgb" and seed == 42:
                    ref_y = y
                values[m].append(macro(y, pred))
                preds[m][seed] = pred
                if m in ("full", "tcn", "tcn_dyn", "multirocket", "dyn_hgb"):
                    result["length_strata"][tag].setdefault(m, [])
                    for index, mask in enumerate(strata):
                        result["length_strata"][tag][m].append(macro(y[mask], pred[mask]))
        for m in METHODS:
            y, p, pred, ids = read(ROOT / f"seed42" / "conditions" / tag / m / "predictions.csv")
            per_class = {str(c): 100 * f1_score(y, pred, labels=[c], average="macro") for c in (0, 1, 2)}
            result["budgets"][tag][m] = {
                "mean": float(np.mean(values[m])),
                "sd": float(np.std(values[m], ddof=1)),
                "per_class_f1": per_class,
                "battery_ap": 100 * float(average_precision_score((y == int(BATTERY_LABEL)).astype(int), p[:, int(BATTERY_LABEL)])),
                "battery_recall": 100 * float((pred[y == int(BATTERY_LABEL)] == int(BATTERY_LABEL)).mean()),
                "battery_fpr": 100 * float((pred[y != int(BATTERY_LABEL)] == int(BATTERY_LABEL)).mean()),
                "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            }
        result["budgets"][tag]["paired_vs_full"] = {
            m: paired(values[m], values["full"]) for m in ("multirocket", "minirocket", "tcn", "tcn_dyn", "dyn_hgb")
        }
        b = int(np.sum((preds["full"][42] == ref_y) & (preds["multirocket"][42] != ref_y)))
        c = int(np.sum((preds["full"][42] != ref_y) & (preds["multirocket"][42] == ref_y)))
        result["budgets"][tag]["mcnemar_full_vs_multi_seed42"] = {
            "full_fixes": b, "multi_fixes": c,
            "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
        }
    # average length strata across seeds
    for tag in TAGS:
        for m, values in result["length_strata"][tag].items():
            arr = np.asarray(values).reshape(4, 3)
            result["length_strata"][tag][m] = {
                "per_seed": arr.tolist(),
                "mean": arr.mean(0).tolist(),
                "sd": arr.std(0, ddof=1).tolist(),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# E18 深度分析（逐类、电池 AP、配对统计、长度分层）", "",
             "基于 E18 已保存的 240 份 validation 预测；0 test。", "",
             "## 逐类 F1 / 电池 AP（seed42，描述性）", "",
             "| 预算 | 方法 | F1-0正常 | F1-1电池 | F1-2充电器 | Battery AP | Recall | FPR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for tag in TAGS:
        for m in METHODS:
            r = result["budgets"][tag][m]
            lines.append(f"| {tag} | {m} | {r['per_class_f1']['0']:.1f} | {r['per_class_f1']['1']:.1f} | {r['per_class_f1']['2']:.1f} | {r['battery_ap']:.1f} | {r['battery_recall']:.1f} | {r['battery_fpr']:.2f} |")
    lines += ["", "## 相对 Full 的配对差（4 seed，95% t 区间）", "", "| 预算 | 方法 | 差值 pp (95% CI) | 正向 seed |", "|---|---|---:|---:|"]
    for tag in TAGS:
        for m, r in result["budgets"][tag]["paired_vs_full"].items():
            lines.append(f"| {tag} | {m} | {r['mean']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | {r['positive']}/4 |")
    lines += ["", "## 长度分层 Macro-F1（train 三分位；四 seed 均值±SD）", "", "| 预算 | 方法 | ≤低 | 中 | >高 |", "|---|---|---:|---:|---:|"]
    for tag in TAGS:
        for m, r in result["length_strata"][tag].items():
            mean = r["mean"]; sd = r["sd"]
            lines.append(f"| {tag} | {m} | {mean[0]:.2f}±{sd[0]:.2f} | {mean[1]:.2f}±{sd[1]:.2f} | {mean[2]:.2f}±{sd[2]:.2f} |")
    lines += ["", "## 边界", "",
              "- 逐类/混淆单 seed 描述；配对统计四 seed split 重叠，不能当独立外部确认。",
              "- McNemar 只用 seed42（同一 validation 行），不跨 seed 合并。",
              "- 长度边界和validation掩码逐seed从其自身特征构造并核对ID，已纠正旧版将seed42掩码用于其他seed的问题；不是因果解释。"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
