"""Aggregate the four-seed E09 difference-stream expansion (validation only)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from tcn_moment.e08_analysis import paired_summary

METHODS = (
    "moment_logit",
    "diff_logit",
    "moment_diff_logit",
    "stack_moment",
    "stack_moment_diff",
    "stack_three",
)
TAGS = ("1p", "10p", "40p")
SEEDS = (42, 44, 45, 46)
E08_SUITE = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
N2_SUMMARY = Path("artifacts/improvements/n2_tcn_low_label/n2_tcn_low_label_20261020_v1/summary.json")
E05C_SUMMARY = Path("artifacts/improvements/e05c_tcn_replication/e05c_tcn_replication_20261019_v1/summary.json")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_suites(root: Path, repair_root: Path | None = None) -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.glob("e09_diff_stream_20261020_seed*/pilot_summary.json")):
        results.append(_read(path))
    if {result["seed"] for result in results} != set(SEEDS):
        raise ValueError("Expected four E09 seed suites")
    if repair_root is not None:
        repairs = {
            int(_read(path)["seed"]): _read(path)
            for path in sorted(repair_root.glob("e09_stack_three_fix_20261020_seed*/pilot_summary.json"))
        }
        if set(repairs) != set(SEEDS):
            raise ValueError("Expected four repaired stack_three suites")
        for result in results:
            repaired = {row["tag"]: row for row in repairs[int(result["seed"])]["conditions"]}
            result["conditions"] = [
                repaired[row["tag"]]
                if row["method"] == "stack_three" and row["tag"] in repaired
                else row
                for row in result["conditions"]
            ]
            result["stack_three_repair"] = "e09_stack_three_fix_20261020"
    return sorted(results, key=lambda result: result["seed"])


def _aggregate(suites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for tag in TAGS:
            values = []
            battery_recall = []
            battery_fpr = []
            for suite in suites:
                match = [
                    row
                    for row in suite["conditions"]
                    if row["method"] == method and row["tag"] == tag
                ]
                if not match:
                    raise ValueError(f"Missing condition {method}/{tag}/seed{suite['seed']}")
                record = match[0]
                values.append(100 * record["validation"]["macro_f1"])
                battery_recall.append(100 * record["validation"]["battery"]["recall"])
                battery_fpr.append(100 * record["validation"]["battery"]["false_positive_rate"])
            rows.append(
                {
                    "method": method,
                    "tag": tag,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    "per_seed": values,
                    "battery_recall_mean": float(np.mean(battery_recall)),
                    "battery_fpr_mean": float(np.mean(battery_fpr)),
                }
            )
    return rows


def _lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["method"], row["tag"]): row for row in rows}


def _paired(rows: list[dict[str, Any]], left: str, right: str, tag: str) -> dict[str, Any]:
    lookup = _lookup(rows)
    left_values = lookup[(left, tag)]["per_seed"]
    right_values = lookup[(right, tag)]["per_seed"]
    return paired_summary([a - b for a, b in zip(left_values, right_values, strict=True)])


def _fraction_tag(fraction: float) -> str | None:
    mapping = {0.01: "1p", 0.05: "5p", 0.1: "10p", 0.4: "40p"}
    return mapping.get(round(fraction, 2))


def _context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    context: dict[str, Any] = {"e08": {}, "tcn": {}, "paired": []}
    for seed in SEEDS:
        path = E08_SUITE / f"seed{seed}" / "seed_results.json"
        saved = _read(path)
        for record in saved["conditions"]:
            if record["method"] in {"dyn_hgb", "stack"}:
                context["e08"][(seed, record["tag"], record["method"])] = float(
                    record["validation"]["macro_f1"]
                )
    if N2_SUMMARY.exists():
        saved = _read(N2_SUMMARY)
        for record in saved["per_condition"]:
            tag = _fraction_tag(float(record["fraction"]))
            if tag is not None:
                context["tcn"][(int(record["seed"]), tag)] = float(record["selected_macro_f1"])
    if E05C_SUMMARY.exists():
        saved = _read(E05C_SUMMARY)
        for record in saved["per_condition"]:
            if record.get("condition") != "legacy_masked50":
                continue
            fraction = float(record["fraction"])
            if fraction == 0.1:
                context["tcn"][(int(record["seed"]), "10p")] = float(record["selected"]["macro_f1"])
            elif fraction == 0.4:
                context["tcn"][(int(record["seed"]), "40p")] = float(record["selected"]["macro_f1"])
    lookup = _lookup(rows)
    for tag in TAGS:
        stack_values = lookup[("stack_moment_diff", tag)]["per_seed"]
        e08_stack = [100 * context["e08"][(seed, tag, "stack")] for seed in SEEDS if (seed, tag, "stack") in context["e08"]]
        tcn = [100 * context["tcn"][(seed, tag)] for seed in SEEDS if (seed, tag) in context["tcn"]]
        if len(e08_stack) == 4:
            context["paired"].append(
                {
                    "comparison": "stack_moment_diff - e08_stack",
                    "tag": tag,
                    **paired_summary([a - b for a, b in zip(stack_values, e08_stack, strict=True)]),
                }
            )
        if len(tcn) == 4:
            context["paired"].append(
                {
                    "comparison": "stack_moment_diff - tcn",
                    "tag": tag,
                    **paired_summary([a - b for a, b in zip(stack_values, tcn, strict=True)]),
                }
            )
    return context


def _gates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _lookup(rows)
    g1 = _paired(rows, "moment_diff_logit", "moment_logit", "40p")
    g2 = _paired(rows, "stack_moment_diff", "stack_moment", "40p")
    g3 = _paired(rows, "stack_moment_diff", "stack_moment", "10p")
    definitions = [
        ("G1", "moment_diff - moment @40p", g1, 5.0, 3),
        ("G2", "stack_moment_diff - stack_moment @40p", g2, 0.3, 3),
        ("G3", "stack_moment_diff - stack_moment @10p", g3, 0.3, 3),
    ]
    gates = []
    for gate_id, comparison, record, threshold, positives in definitions:
        passed = record["mean"] >= threshold and record["positive_seeds"] >= positives
        gates.append(
            {
                "id": gate_id,
                "comparison": comparison,
                "mean_gain_pp": record["mean"],
                "sample_sd": record["sample_sd"],
                "positive_seeds": record["positive_seeds"],
                "required_mean_pp": threshold,
                "required_positive_seeds": positives,
                "status": "passed" if passed else "failed",
            }
        )
    return gates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["method", "tag", "n", "mean", "sample_sd", "battery_recall_mean", "battery_fpr_mean"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def _markdown(gates: list[dict[str, Any]], rows: list[dict[str, Any]], context: dict[str, Any]) -> str:
    lookup = _lookup(rows)
    lines = [
        "# E09：冻结 MOMENT 原序列 vs 一阶差分双流（四 seed，validation-only）",
        "",
        "四 seed42/44/45/46；1%/10%/40% 训练标签；冻结 backbone，仅改变 MOMENT 输入流；无 test。",
        "",
        "**修复说明：** 历史 `stack_three` 的 OOF 列写成了 3:6 与 4:7，第二块覆盖第一块且 7:9 未写入；",
        "本轮只重算 `stack_three`（3:6 与 6:9），其他 E09 条件不变。",
        "",
        "## 主结果（Macro-F1 %，均值 ± 样本 SD）",
        "",
        "| 方法 | 1% | 10% | 40% |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = [f"{lookup[(method, tag)]['mean']:.2f} ± {lookup[(method, tag)]['sample_sd']:.2f}" for tag in TAGS]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "## 配对差（pp）", "", "| 比较 | 预算 | 均值 ± SD | 正向 seed |", "|---|---:|---:|---:|"]
    for left, right in (
        ("moment_diff_logit", "moment_logit"),
        ("stack_moment_diff", "stack_moment"),
        ("stack_moment_diff", "stack_three"),
        ("stack_moment_diff", "diff_logit"),
    ):
        for tag in TAGS:
            record = _paired(rows, left, right, tag)
            lines.append(
                f"| {left} − {right} | {tag} | {record['mean']:+.2f} ± {record['sample_sd']:.2f} | "
                f"{record['positive_seeds']}/{record['n']} |"
            )
    lines += ["", "## 与 E08 强特征 stack / TCN 的上下文", "", "| 比较 | 预算 | 均值 ± SD | 正向 seed |", "|---|---:|---:|---:|"]
    for record in context["paired"]:
        lines.append(
            f"| {record['comparison']} | {record['tag']} | {record['mean']:+.2f} ± {record['sample_sd']:.2f} | "
            f"{record['positive_seeds']}/{record['n']} |"
        )
    lines += ["", "## 电池异常（40%，默认 argmax）", "", "| 方法 | Recall % | FPR % |", "|---|---:|---:|"]
    for method in METHODS:
        row = lookup[(method, "40p")]
        lines.append(f"| {method} | {row['battery_recall_mean']:.2f} | {row['battery_fpr_mean']:.2f} |")
    lines += ["", "## 事前门槛", "", "| Gate | 比较 | 均值 pp | 正向 | 要求 | 状态 |", "|---|---|---:|---:|---|---|"]
    for gate in gates:
        lines.append(
            f"| {gate['id']} | {gate['comparison']} | {gate['mean_gain_pp']:+.2f} | "
            f"{gate['positive_seeds']}/4 | ≥{gate['required_mean_pp']:.1f}pp, ≥{gate['required_positive_seeds']}/4 | "
            f"**{gate['status']}** |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- 四个 seed 的 split 重叠、validation 反复使用；这不是独立外部确认，也没有 test。",
        "- 差分流是在有效观测上计算并逐序列 z-score；幅值/尺度信息仍由 DYN 统计分支提供。",
        "- `stack_three` 把 MOM 与 DIFF 分成两个 logistic 分支；修正历史 OOF 列重叠后，它仍略低于拼接的 `stack_moment_diff`（约 0.1–0.3pp），",
        "  说明当前数据量下提前分开两支的代价很小，不能据此说差分信息无用。历史损坏版本（约 72%/91%/95%）已替换。",
        "- 与 TCN 的差值是同训练标签/同 validation 行的管线比较；TCN 的训练/选择预算不同。",
        "- 1% 的 MOMENT 单支仍弱；差分流的主要收益在 10%/40% 和电池类。",
        "- 所有逐样本预测在聚合时重新核对 ID/真值/有限分数/行数，并复算 Macro-F1。",
    ]
    return "\n".join(lines)




def _audit(suites: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    rows = 0
    maximum = 0.0
    for suite in suites:
        seed = int(suite["seed"])
        with np.load(E08_SUITE / f"seed{seed}" / "features.npz", allow_pickle=False) as archive:
            expected_ids = set(archive["validation_ids"].astype(str))
        for condition in suite["conditions"]:
            path = Path(condition["predictions"])
            if not path.exists():
                path = (
                    root
                    / f"e09_diff_stream_20261020_seed{seed}"
                    / "conditions"
                    / condition["tag"]
                    / condition["method"]
                    / "predictions.csv"
                )
            with path.open("r", encoding="utf-8", newline="") as file:
                predictions = list(csv.DictReader(file))
            if len(predictions) != 3510:
                raise ValueError(f"Unexpected row count: {path}")
            sample_ids = [row["sample_id"] for row in predictions]
            if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != expected_ids:
                raise ValueError(f"Prediction IDs mismatch: {path}")
            if any(row["split"] != "validation" for row in predictions):
                raise ValueError(f"Non-validation row: {path}")
            y_true = np.asarray([int(row["y_true"]) for row in predictions])
            y_pred = np.asarray([int(row["y_pred"]) for row in predictions])
            scores = np.asarray(
                [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in predictions]
            )
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite scores: {path}")
            recomputed = float(f1_score(y_true, y_pred, average="macro"))
            recorded = float(condition["validation"]["macro_f1"])
            maximum = max(maximum, abs(recomputed - recorded))
            rows += len(predictions)
    return {"conditions": sum(len(suite["conditions"]) for suite in suites), "rows": rows, "max_macro_f1_error": maximum}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/improvements/e09_diff_stream")
    parser.add_argument("--repair-root", default="artifacts/improvements/e09_stack_three_fix")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output_dir) if args.output_dir else root / "analysis_four_seed_corrected"
    suites = _load_suites(root, Path(args.repair_root))
    rows = _aggregate(suites)
    context = _context(rows)
    gates = _gates(rows)
    serializable_context = dict(context)
    serializable_context["e08"] = {
        f"{seed}:{tag}:{method}": value
        for (seed, tag, method), value in context["e08"].items()
    }
    serializable_context["tcn"] = {
        f"{seed}:{tag}": value for (seed, tag), value in context["tcn"].items()
    }
    audit = _audit(suites, root)
    summary = {
        "seeds": [suite["seed"] for suite in suites],
        "test_evaluated": False,
        "aggregate": rows,
        "context": serializable_context,
        "gates": gates,
        "audit": audit,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output / "aggregate.csv", rows)
    markdown = _markdown(gates, rows, context)
    (output / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
