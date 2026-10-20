"""Aggregate E12 cross-fitted TCN + DYN + MOMENT/ΔP stacking (validation only)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from tcn_moment.e08_analysis import paired_summary

METHODS = ("tcn_full", "tcn_foldavg", "stack_no_tcn", "stack_tcn")
SEEDS = (42, 44, 45, 46)
E08_SUITE = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load(root: Path) -> list[dict[str, Any]]:
    results = [_read(root / f"seed{seed}" / "seed_results.json") for seed in SEEDS]
    return sorted(results, key=lambda result: result["seed"])


def _aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        values = []
        recall = []
        fpr = []
        for result in results:
            record = next(row for row in result["conditions"] if row["method"] == method)
            values.append(100 * record["validation"]["macro_f1"])
            recall.append(100 * record["validation"]["battery"]["recall"])
            fpr.append(100 * record["validation"]["battery"]["false_positive_rate"])
        rows.append(
            {
                "method": method,
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "per_seed": values,
                "battery_recall_mean": float(np.mean(recall)),
                "battery_fpr_mean": float(np.mean(fpr)),
            }
        )
    return rows


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    lookup = {row["method"]: row for row in rows}
    return paired_summary(
        [a - b for a, b in zip(lookup[left]["per_seed"], lookup[right]["per_seed"], strict=True)]
    )


def _e08_stack(results: list[dict[str, Any]], tag: str) -> list[float]:
    values = []
    for result in results:
        saved = _read(E08_SUITE / f"seed{result['seed']}" / "seed_results.json")
        record = next(
            row for row in saved["conditions"] if row["method"] == "stack" and row["tag"] == tag
        )
        values.append(100 * record["validation"]["macro_f1"])
    return values


def _audit(results: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    rows = 0
    maximum = 0.0
    for result in results:
        seed = result["seed"]
        with np.load(E08_SUITE / f"seed{seed}" / "features.npz", allow_pickle=False) as archive:
            expected = set(archive["validation_ids"].astype(str))
        for record in result["conditions"]:
            path = root / f"seed{seed}" / "conditions" / record["tag"] / record["method"] / "predictions.csv"
            with path.open("r", encoding="utf-8", newline="") as file:
                predictions = list(csv.DictReader(file))
            if len(predictions) != 3510:
                raise ValueError(f"Unexpected rows: {path}")
            sample_ids = [row["sample_id"] for row in predictions]
            if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != expected:
                raise ValueError(f"ID mismatch: {path}")
            if any(row["split"] != "validation" for row in predictions):
                raise ValueError(f"Non-validation row: {path}")
            y_true = np.asarray([int(row["y_true"]) for row in predictions])
            y_pred = np.asarray([int(row["y_pred"]) for row in predictions])
            scores = np.asarray(
                [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in predictions]
            )
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite scores: {path}")
            maximum = max(maximum, abs(float(f1_score(y_true, y_pred, average="macro")) - record["validation"]["macro_f1"]))
            rows += len(predictions)
    return {"conditions": sum(len(result["conditions"]) for result in results), "rows": rows, "max_macro_f1_error": maximum}


def _markdown(rows: list[dict[str, Any]], paired: dict[str, dict[str, Any]], audit: dict[str, Any], tag: str) -> str:
    lookup = {row["method"]: row for row in rows}
    lines = [
        f"# E12：cross-fitted TCN + DYN + MOMENT/ΔP 堆叠（{tag} 标签，validation-only）",
        "",
        "四 seed；40% 训练子集内 5 折交叉拟合；TCN 每折/全量固定 50 轮 cosine，无 validation 选模；",
        "DYN-HGB 与 MOMENT+ΔP logistic 使用同一折分配；meta 只在 OOF 概率上拟合。无 test。",
        "",
        "| 方法 | Macro-F1 均值 ± SD % | 逐 seed | 电池 recall/FPR % |",
        "|---|---:|---|---:|",
    ]
    for method in METHODS:
        row = lookup[method]
        per_seed = ", ".join(f"{value:.2f}" for value in row["per_seed"])
        lines.append(
            f"| {method} | {row['mean']:.2f} ± {row['sample_sd']:.2f} | {per_seed} | "
            f"{row['battery_recall_mean']:.2f}/{row['battery_fpr_mean']:.2f} |"
        )
    lines += ["", "## 配对差", "", "| 比较 | 均值 ± SD pp | 正向 seed |", "|---|---:|---:|"]
    for label, record in paired.items():
        lines.append(
            f"| {label} | {record['mean']:+.2f} ± {record['sample_sd']:.2f} | "
            f"{record['positive_seeds']}/{record['n']} |"
        )
    lines += [
        "",
        "## Gate 与审计",
        "",
        f"- Gate（stack_tcn − tcn_full ≥ +0.3pp 且 ≥3/4 正向）：**{'通过' if paired['stack_tcn − tcn_full']['mean'] >= 0.3 and paired['stack_tcn − tcn_full']['positive_seeds'] >= 3 else '未通过'}**。",
        f"- 审计：{audit['conditions']} 条件 / {audit['rows']} 行；Macro-F1 最大误差 {audit['max_macro_f1_error']:.3e}；"
        "全部 validation，0 test。",
        "",
        "## 边界",
        "",
        "- 四 seed split 重叠、validation 反复使用；这是已有 test 已知后的 validation 探索，不是独立盲测。",
        "- TCN 与 stack 的训练/选择预算不同；stack 的 meta 只在 OOF 上拟合，TCN 未使用 validation 选 checkpoint。",
        f"- 本轮只执行 {tag} 预算；其他预算未在本轮执行，不能外推。",
        "- 没有做校准/保形、鲁棒性或同机端到端延迟。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/improvements/e12_tcn_stack/e12_tcn_stack_20261020_v1")
    parser.add_argument("--tag", default="40p")
    args = parser.parse_args()
    root = Path(args.root)
    results = _load(root)
    rows = _aggregate(results)
    e08 = _e08_stack(results, args.tag)
    lookup = {row["method"]: row for row in rows}
    paired = {
        "stack_tcn − tcn_full": _paired(rows, "stack_tcn", "tcn_full"),
        "stack_tcn − tcn_foldavg": _paired(rows, "stack_tcn", "tcn_foldavg"),
        "stack_tcn − stack_no_tcn": _paired(rows, "stack_tcn", "stack_no_tcn"),
        "tcn_full − tcn_foldavg": _paired(rows, "tcn_full", "tcn_foldavg"),
        "stack_tcn − E08 stack": paired_summary(
            [a - b for a, b in zip(lookup["stack_tcn"]["per_seed"], e08, strict=True)]
        ),
    }
    audit = _audit(results, root)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(
            {"aggregate": rows, "paired": paired, "audit": audit, "test_evaluated": False},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown = _markdown(rows, paired, audit, args.tag)
    (output / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
