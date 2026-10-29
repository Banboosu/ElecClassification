"""Recalculate E10 class-1 battery descriptive validation working points.

No model fitting, temperature refitting, inference or test scoring. Thresholds are
constructed and evaluated on the SAME validation labels: never deployment claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.label_semantics import BATTERY_LABEL, SEMANTICS_VERSION

DEFAULT = Path("artifacts/analysis/label_semantics_correction_20261029_v2/summary.json")


def quantile(values: list[float], q: float) -> float:
    """Linear empirical quantile (numpy's default), including tied scores."""
    ordered = sorted(values)
    if not ordered or not 0 <= q <= 1:
        raise ValueError("Nonempty values and q in [0,1] required")
    index = (len(ordered) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def rates(y: list[int], selected: list[bool]) -> dict[str, float]:
    positive = int(BATTERY_LABEL)
    total_positive = sum(v == positive for v in y)
    total_negative = len(y) - total_positive
    if not total_positive or not total_negative:
        raise ValueError("Both battery and non-battery validation records required")
    tp = sum(chosen and label == positive for chosen, label in zip(selected, y, strict=True))
    fp = sum(chosen and label != positive for chosen, label in zip(selected, y, strict=True))
    return {"recall": tp / total_positive, "fpr": fp / total_negative,
            "precision": tp / max(tp + fp, 1), "alert_rate": (tp + fp) / len(y)}


def working_points(y: list[int], scores: list[float]) -> dict[str, Any]:
    if not y or len(y) != len(scores) or any(not math.isfinite(v) for v in scores):
        raise ValueError("Invalid labels/scores")
    positive = int(BATTERY_LABEL)
    positive_scores = [s for v, s in zip(y, scores, strict=True) if v == positive]
    negative_scores = [s for v, s in zip(y, scores, strict=True) if v != positive]
    if not positive_scores or not negative_scores:
        raise ValueError("Both battery and non-battery records required")
    prevalence = len(positive_scores) / len(y)
    result: dict[str, Any] = {
        "prevalence": prevalence,
        "perfect_ranking_fpr_at_50pct_alert": max(0., (.5 - prevalence) / (1. - prevalence)),
        "fixed_fpr": [], "fixed_recall": [], "alert_curve": [],
    }
    for target in (.005, .01, .02):
        threshold = quantile(negative_scores, 1. - target)
        result["fixed_fpr"].append({"target": target, "threshold": threshold,
                                    **rates(y, [s >= threshold for s in scores])})
    for target in (.95, .98, .99):
        threshold = quantile(positive_scores, 1. - target)
        result["fixed_recall"].append({"target": target, "threshold": threshold,
                                       **rates(y, [s >= threshold for s in scores])})
    # Stable tie break by saved row order; fixed counts, not deployable thresholds.
    order = sorted(range(len(y)), key=lambda i: scores[i], reverse=True)
    for rate in (.05, .1, .2, .3, .5):
        selected = set(order[:int(round(rate * len(y)))])
        result["alert_curve"].append({"target": rate,
                                      **rates(y, [i in selected for i in range(len(y))])})
    return result


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Preserve existing outputs: {output}")
    summary = json.loads(source.read_text(encoding="utf-8"))
    if summary["semantics_version"] != SEMANTICS_VERSION or summary["battery_label"] != BATTERY_LABEL:
        raise ValueError("Audited class-1 summary required")
    records = []
    for rec in summary["conditions"]:
        if rec["suite"] != "e12" or rec["method"] != "stack_tcn":
            continue
        path = Path(rec["source_predictions"])
        if sha256_file(path) != rec["source_sha256"]:
            raise ValueError(f"Changed audited prediction: {path}")
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 3510 or any(row["split"] != "validation" for row in rows):
            raise ValueError("Only the audited validation split is allowed")
        y = [int(row["y_true"]) for row in rows]
        scores = [float(row[f"score_{BATTERY_LABEL}"]) for row in rows]
        records.append({"seed": rec["seed"], "tag": rec["tag"], "source_predictions": str(path),
                        "source_sha256": rec["source_sha256"], **working_points(y, scores)})
    if {(r["tag"], r["seed"]) for r in records} != {(t, s) for t in ("10p", "40p") for s in (42, 44, 45, 46)} or len(records) != 8:
        raise ValueError("Exactly 8 E12 final-stack conditions required")
    aggregates = {}
    for tag in ("10p", "40p"):
        selected = [r for r in records if r["tag"] == tag]
        combined = {key: statistics.mean(r[key] for r in selected) for key in ("prevalence", "perfect_ranking_fpr_at_50pct_alert")}
        for kind in ("fixed_fpr", "fixed_recall", "alert_curve"):
            combined[kind] = []
            for index in range(len(selected[0][kind])):
                cells = [r[kind][index] for r in selected]
                combined[kind].append({"target": cells[0]["target"], **{
                    key: statistics.mean(cell[key] for cell in cells)
                    for key in ("recall", "fpr", "precision", "alert_rate")
                }})
        aggregates[tag] = combined
    result = {"label_semantics_version": SEMANTICS_VERSION, "battery_label": BATTERY_LABEL,
              "descriptive_only": True, "threshold_selection_split": "validation",
              "evaluation_split": "same_validation", "calibration_refitted": False,
              "test_evaluated": False, "conditions": 8, "rows": 8 * 3510,
              "source_summary": str(source), "source_summary_sha256": sha256_file(source),
              "per_seed": records, "aggregate": aggregates}
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "summary.json", result)
    (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# E10 电池类别纠正：描述性工作点（类别1）", "",
             "**阈值在validation上构造并原地评价；不是独立校准、锁定阈值或部署保证。**",
             "只使用8份已审计E12最终stack的validation预测；不拟合模型、不重拟合温度、不打开test。", ""]
    for tag, rec in aggregates.items():
        lines += [f"## {tag}", "", f"电池占比 {100*rec['prevalence']:.2f}%；固定50%报警时理想排序FPR下界 {100*rec['perfect_ranking_fpr_at_50pct_alert']:.2f}%。", "",
                  "| 约束 | 目标 % | 实际Recall % | 实际FPR % | 报警比例 % |", "|---|---:|---:|---:|---:|"]
        for kind in ("fixed_fpr", "fixed_recall", "alert_curve"):
            for row in rec[kind]:
                lines.append(f"| {kind} | {100*row['target']:.2f} | {100*row['recall']:.2f} | {100*row['fpr']:.2f} | {100*row['alert_rate']:.2f} |")
        lines.append("")
    lines += ["- 正类与分数列同时改为 `y==1` / `score_1`；旧类别2阈值不沿用。",
              "- 线性分位数、score≥threshold；并列分数可能使实际目标略有偏差。",
              "- 固定报警数量时并列分数按保存行顺序稳定打破；与独立阈值部署不是一回事。",
              "- 总体ECE/NLL等不因类别名称更正而改变，本次不重复拟合校准器。",
              "- 四seed split重叠、validation被反复使用，不作外部确认或电池物理机理解释。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/e10_battery_semantics_v2_20261029"))
    args = parser.parse_args()
    run(args.input, args.output_dir)
