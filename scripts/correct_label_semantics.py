"""Audit saved validation predictions and recalculate class-1 battery metrics.

Standard-library only. Numeric labels, predictions and scores are never swapped.
Inputs are enumerated from preservation manifests, not from test predictions.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any

from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.label_semantics import (
    BATTERY_LABEL, CHARGER_LABEL, CLASS_LABELS, LABEL_NAMES_ZH, SEMANTICS_VERSION,
)

DEFAULT = Path("configs/experiments/improvements/label_semantics_correction_20261029.json")


def ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def average_precision(y: list[int], scores: list[float], positive: int) -> float:
    """Non-interpolated AP with score ties grouped, matching sklearn's AP."""
    total = sum(value == positive for value in y)
    if not total:
        return 0.0
    ranked = sorted(zip(scores, y, strict=True), key=lambda pair: pair[0], reverse=True)
    tp = count = 0
    ap = 0.0
    for _, group in groupby(ranked, key=lambda pair: pair[0]):
        rows = list(group)
        new_tp = sum(value == positive for _, value in rows)
        tp += new_tp
        count += len(rows)
        ap += new_tp / total * tp / count
    return ap


def evaluate(y: list[int], pred: list[int], scores: list[list[float]]) -> dict[str, Any]:
    if not y or len(y) != len(pred) or len(y) != len(scores):
        raise ValueError("Nonempty y, predictions and scores must have matching lengths")
    cm = [[0] * 3 for _ in range(3)]
    for truth, guessed in zip(y, pred, strict=True):
        if truth not in (0, 1, 2) or guessed not in (0, 1, 2):
            raise ValueError("Only raw labels 0/1/2 are allowed")
        cm[truth][guessed] += 1
    per_class = {}
    for c, label in enumerate(CLASS_LABELS):
        tp = cm[c][c]
        support = sum(cm[c])
        predicted = sum(row[c] for row in cm)
        fp = predicted - tp
        fn = support - tp
        tn = len(y) - tp - fp - fn
        per_class[label] = {
            "name_zh": LABEL_NAMES_ZH[label], "support": support,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": ratio(tp, predicted), "recall": ratio(tp, support),
            "fpr": ratio(fp, fp + tn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "average_precision": average_precision(y, [row[c] for row in scores], c),
        }
    return {
        "rows": len(y), "macro_f1": statistics.mean(r["f1"] for r in per_class.values()),
        "accuracy": sum(cm[c][c] for c in range(3)) / len(y),
        "class_order": list(CLASS_LABELS), "confusion_matrix": cm,
        "per_class": per_class,
        "battery": per_class[BATTERY_LABEL], "charger": per_class[CHARGER_LABEL],
    }


def read_validation(
    path: Path, expected_ids: list[str], truths: dict[str, int],
    *, score_type: str = "probability",
) -> tuple[list[int], list[int], list[list[float]]]:
    if score_type not in ("probability", "decision_function"):
        raise ValueError(f"Unknown score type: {score_type}")
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    ids = [r["sample_id"] for r in rows]
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError(f"Validation ID/order/uniqueness mismatch: {path}")
    y, pred, scores = [], [], []
    for r in rows:
        if r["split"] != "validation":
            raise ValueError(f"Non-validation prediction refused: {path}")
        truth, guessed = int(r["y_true"]), int(r["y_pred"])
        if truth != truths[r["sample_id"]]:
            raise ValueError(f"Truth differs from raw numeric label: {path}")
        values = [float(r[f"score_{c}"]) for c in range(3)]
        if guessed not in (0, 1, 2) or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid prediction/nonfinite score: {path}")
        if score_type == "probability":
            if not all(0 <= v <= 1 for v in values):
                raise ValueError(f"Out-of-range score: {path}")
            if abs(sum(values) - 1.0) > 1e-5:
                raise ValueError(f"Score sum mismatch: {path}")
            if max(values) - values[guessed] > 1e-10:
                raise ValueError(f"Argmax mismatch: {path}")
        # SVC.predict uses pairwise votes; its OVR decision scores need not be
        # normalized or have the same argmax. Keep the recorded prediction and
        # rank the raw class-specific decision score for AP, without softmax.
        y.append(truth)
        pred.append(guessed)
        scores.append(values)
    return y, pred, scores


def load_truths(dataset: Path, wanted_ids: set[str]) -> dict[str, int]:
    # Scan the space-delimited CSV but do not parse any power sequence or score test.
    csv.field_size_limit(10**8)
    truths = {}
    with dataset.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=" "):
            sid = row["charging_station_id"]
            if sid not in wanted_ids:
                continue
            if sid in truths:
                raise ValueError(f"Duplicate validation ID in raw data: {sid}")
            truths[sid] = int(row["InsertedColumn"])
    if set(truths) != wanted_ids:
        raise ValueError("Raw data is missing validation IDs")
    return truths


def aggregate(conditions: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in conditions:
        groups[(rec["suite"], rec["tag"], rec["method"])].append(rec)
    result = []
    for (suite, tag, method), selected in sorted(groups.items()):
        selected.sort(key=lambda rec: rec["seed"])
        if [r["seed"] for r in selected] != sorted(seeds):
            raise ValueError(f"Missing or duplicate formal seeds: {suite}/{tag}/{method}")
        row: dict[str, Any] = {"suite": suite, "tag": tag, "method": method, "n": len(selected)}
        for metric in ("macro_f1", "accuracy"):
            values = [r[metric] for r in selected]
            row[metric] = {"mean": statistics.mean(values), "sd": statistics.stdev(values)}
        for cls in ("battery", "charger"):
            row[cls] = {}
            for metric in ("recall", "fpr", "f1", "average_precision"):
                values = [r[cls][metric] for r in selected]
                row[cls][metric] = {"mean": statistics.mean(values), "sd": statistics.stdev(values)}
        result.append(row)
    return result


def markdown(result: dict[str, Any]) -> str:
    audit = result["audit"]
    lines = [
        "# 类别语义纠正复评（不重训，validation-only）", "",
        "正确映射（用户确认）：**0=正常、1=电池异常、2=充电器故障**。", "",
        "原始数字标签、score_0/1/2 列及模型预测均不变。旧 battery_* 实际对应类别 2，不能当电池指标。",
        "本表按各 seed 分别复算，再汇总四 seed 均值；所有数值单位为 %。", "",
        f"- 审计：{audit['conditions']} 条件 / {audit['rows']:,} 行；0 test 预测。",
        f"- Macro-F1 相对原数值复算最大误差：{audit['max_macro_f1_error']:.3g}。",
        "- ID/顺序/真值/分数/argmax 与原始 CSV、split manifest 和保全预测文件逐项核对。", "",
    ]
    for suite in dict.fromkeys(r["suite"] for r in result["aggregate"]):
        lines += [f"## {suite}", "",
                  "| 预算 | 方法 | Macro-F1 均值±SD | 电池 Recall | 电池 FPR | 电池 AP | 充电器 Recall | 充电器 FPR |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        selected = [r for r in result["aggregate"] if r["suite"] == suite]
        for r in sorted(selected, key=lambda x: (int(x["tag"][:-1]), x["method"])):
            f1 = r["macro_f1"]
            b, c = r["battery"], r["charger"]
            lines.append(
                f"| {r['tag']} | {r['method']} | {100*f1['mean']:.2f}±{100*f1['sd']:.2f} | "
                f"{100*b['recall']['mean']:.2f} | {100*b['fpr']['mean']:.2f} | "
                f"{100*b['average_precision']['mean']:.2f} | {100*c['recall']['mean']:.2f} | {100*c['fpr']['mean']:.2f} |"
            )
    lines += ["", "## 范围与未完成事项", "",
              "- 不改写旧预测、旧指标或原冻结协议；当前输出是另存的语义纠正版。",
              "- 仅覆盖上列来源；旧 test 和已归档的错误版本不参加当前比较。排除记录与理由见协议。",
              "- decision_function 分数不作为概率：保留 SVC.predict 的类别和原始逐类分数，用于混淆指标与 AP。",
              "- 旧类别 2 专用二分类器是充电器-vs-rest，不能仅重命名为类别 1 电池检测器；本轮未重训。",
              "- 不选择新的报警阈值，不重新评价旧 test；旧校准阈值不能直接改称电池阈值。",
              "- AP 使用各自 score_1；Rocket 分数为 decision_function 经 softmax，未校准。",
              "- 四 seed 划分重叠、validation 反复使用，不构成独立外部确认。",
              "- 映射依据为用户本次确认；未收到原始 codebook，不从曲线形态推断故障机制。", ""]
    return "\n".join(lines)


def run(protocol_path: Path) -> Path:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["semantics_version"] != SEMANTICS_VERSION or protocol["class_names_zh"] != LABEL_NAMES_ZH:
        raise ValueError("Semantic mapping/version mismatch")
    if protocol["battery_label"] != BATTERY_LABEL or protocol["charger_label"] != CHARGER_LABEL:
        raise ValueError("Positive class mismatch")
    if any(protocol[k] for k in ("test_evaluated", "change_numeric_labels", "retrain", "select_new_thresholds")):
        raise ValueError("Only post-hoc validation semantics correction is authorized")
    if protocol["seeds"] != [42, 44, 45, 46]:
        raise ValueError("Only the four formal seeds are allowed")
    output = Path(protocol["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    dataset = Path(protocol["dataset"])
    if sha256_file(dataset) != protocol["dataset_sha256"]:
        raise ValueError("Dataset SHA mismatch")
    fingerprints = {str(protocol_path): sha256_file(protocol_path), str(dataset): protocol["dataset_sha256"]}
    expected = {}
    for seed in protocol["seeds"]:
        suffix = "pilot_seed42" if seed == 42 else f"expansion_seed{seed}"
        path = Path(protocol["split_manifest_dir"]) / f"e01_20261017_v2_{suffix}" / "split_manifest.json"
        split = json.loads(path.read_text(encoding="utf-8"))
        if split["protocol"]["dataset_sha256"] != protocol["dataset_sha256"] or split["protocol"]["random_state"] != seed:
            raise ValueError(f"Split fingerprint/seed mismatch: {path}")
        expected[seed] = [str(v) for v in split["splits"]["validation"]]
        if len(expected[seed]) != 3510 or len(set(expected[seed])) != 3510:
            raise ValueError(f"Unexpected validation split size: {path}")
        fingerprints[str(path)] = sha256_file(path)
    truths = load_truths(dataset, set().union(*(set(ids) for ids in expected.values())))
    conditions = []
    excluded_inputs = []
    for source in protocol["inputs"]:
        root = Path(source["root"])
        data_root = Path(source.get("data_root", root))
        manifest_path = root / "preservation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprints[str(manifest_path)] = sha256_file(manifest_path)
        files = manifest["files"]
        prediction_names = []
        for name in sorted(files):
            if Path(name).name != "predictions.csv":
                continue
            included = not source.get("include_regex") or re.search(source["include_regex"], name)
            excluded = source.get("exclude_regex") and re.search(source["exclude_regex"], name)
            if not included or excluded:
                excluded_inputs.append({"source": source["name"], "path": name,
                                        "reason": source["selection_reason"]})
            else:
                prediction_names.append(name)
        if len(prediction_names) != source["expected_conditions"]:
            raise ValueError(f"Condition count mismatch: {source['name']}")
        for name in prediction_names:
            path = data_root / name
            if sha256_file(path) != files[name]:
                raise ValueError(f"Preserved prediction SHA mismatch: {path}")
            seed_matches = re.findall(r"(?:^|[/_])seed(\d+)(?:/|$)", name)
            tag_matches = re.findall(r"(?:^|/)(\d+p)(?:/|$)", name)
            if len(seed_matches) != 1 or len(tag_matches) != 1:
                raise ValueError(f"Ambiguous condition path: {path}")
            seed = int(seed_matches[0])
            if seed not in expected:
                raise ValueError(f"Unexpected seed: {path}")
            metrics_path = path.with_name("metrics.json")
            relative_metrics = str(Path(name).with_name("metrics.json"))
            if sha256_file(metrics_path) != files[relative_metrics]:
                raise ValueError(f"Preserved metric SHA mismatch: {metrics_path}")
            original = json.loads(metrics_path.read_text(encoding="utf-8"))
            score_type = original.get("score_type", "probability")
            if score_type == "decision_function" and not source.get("allow_decision_scores"):
                raise ValueError(f"Decision scores not authorized for this source: {path}")
            y, pred, scores = read_validation(path, expected[seed], truths, score_type=score_type)
            rec = evaluate(y, pred, scores)
            original_metrics = original.get("validation", original)
            error = abs(rec["macro_f1"] - float(original_metrics["macro_f1"]))
            if error > 1e-10:
                raise ValueError(f"Original Macro-F1 mismatch: {path}: {error}")
            fields = source.get("method_fields")
            method = "/".join(str(original[key]) for key in fields) if fields else path.parent.name
            rec.update({"suite": source["name"], "seed": seed, "tag": tag_matches[0],
                        "method": method, "score_type": score_type,
                        "source_predictions": str(path),
                        "source_sha256": files[name], "source_metrics_sha256": files[relative_metrics],
                        "macro_f1_error": error, "test_evaluated": False})
            conditions.append(rec)
        print(f"Audited {source['name']}: {len(prediction_names)} conditions", flush=True)
    result = {
        "semantics_version": SEMANTICS_VERSION, "class_names_zh": LABEL_NAMES_ZH,
        "battery_label": BATTERY_LABEL, "charger_label": CHARGER_LABEL,
        "test_evaluated": False, "retrained": False, "thresholds_selected": False,
        "input_fingerprints": fingerprints, "excluded_inputs": excluded_inputs, "conditions": conditions,
        "aggregate": aggregate(conditions, protocol["seeds"]),
        "audit": {"conditions": len(conditions), "rows": sum(r["rows"] for r in conditions),
                  "test_rows": 0, "max_macro_f1_error": max(r["macro_f1_error"] for r in conditions)},
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "summary.json", result)
    (output / "summary.md").write_text(markdown(result), encoding="utf-8")
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    (output / "correction_source.py").write_bytes(Path(__file__).read_bytes())
    import tcn_moment.label_semantics as semantics
    (output / "label_semantics_source.py").write_bytes(Path(semantics.__file__).read_bytes())
    atomic_write_json(output / "preservation_manifest.json", {
        "files": {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()},
        "note": "New correction outputs only; all source predictions and metrics preserved unchanged.",
    })
    print(json.dumps(result["audit"], ensure_ascii=False), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT)
    args = parser.parse_args()
    run(args.protocol)
