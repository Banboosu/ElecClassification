"""E05A: retrospective, stdlib-only audit of saved TCN versus E03T predictions.

Never loads a model, trains, predicts, selects thresholds, or changes E03T gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tarfile
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_PROTOCOL = Path("configs/experiments/improvements/e05a_historical_tcn_audit_protocol.json")
SEEDS = [42, 44, 45, 46]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metrics(rows):
    if not rows:
        raise ValueError("Missing prediction rows.")
    cm = [[0] * 3 for _ in range(3)]
    for row in rows:
        y, pred = int(row["true_index"]), int(row["predicted_index"])
        if not 0 <= y < 3 or not 0 <= pred < 3:
            raise ValueError("Invalid class index.")
        cm[y][pred] += 1
    counts = [sum(r) for r in cm]
    if not all(counts):
        raise ValueError("Expected all three true classes.")
    per_class = []
    for i in range(3):
        predicted = sum(cm[j][i] for j in range(3))
        per_class.append(
            {
                "precision": cm[i][i] / predicted if predicted else 0,
                "recall": cm[i][i] / counts[i],
                "f1": 2 * cm[i][i] / (counts[i] + predicted),
            }
        )
    return {
        "accuracy": sum(cm[i][i] for i in range(3)) / len(rows),
        "macro_f1": statistics.mean(r["f1"] for r in per_class),
        "macro_precision": statistics.mean(r["precision"] for r in per_class),
        "macro_recall": statistics.mean(r["recall"] for r in per_class),
        "balanced_accuracy": statistics.mean(r["recall"] for r in per_class),
        "weighted_f1": sum(per_class[i]["f1"] * counts[i] for i in range(3)) / len(rows),
        "confusion_matrix": cm,
        "per_class": per_class,
        "battery_recall": per_class[2]["recall"],
        "battery_fpr": sum(cm[j][2] for j in (0, 1)) / sum(counts[:2]),
    }


def average_precision(rows, score_key):
    ranked = []
    for row in rows:
        score = float(row[score_key])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid saved battery probability.")
        ranked.append((score, int(row["true_index"]) == 2))
    ranked.sort(reverse=True)
    positives = sum(y for _, y in ranked)
    if not positives:
        raise ValueError("AP requires positive examples.")
    tp = seen = start = 0
    ap = 0.0
    while start < len(ranked):
        end = start
        added = 0
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            added += ranked[end][1]
            end += 1
        tp += added
        seen += end - start
        ap += added / positives * tp / seen
        start = end
    return ap


def match_rows(rows, reference):
    if len(rows) != len(reference) or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Missing/duplicated prediction IDs.")
    for row, expected in zip(rows, reference, strict=True):
        if row["sample_id"] != expected["sample_id"] or row["true_index"] != expected["true_index"]:
            raise ValueError("Prediction row order/true labels differ.")


def compare_metrics(recomputed, recorded):
    errors = []
    for key in (
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "weighted_f1",
    ):
        error = abs(recomputed[key] - recorded[key])
        if error > 1e-12:
            raise ValueError(f"Source prediction metric changed: {key}")
        errors.append(error)
    if recomputed["confusion_matrix"] != recorded["confusion_matrix"]:
        raise ValueError("Source confusion matrix differs.")
    return max(errors)


def describe(values):
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError("All four preexisting E03T seeds are required.")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 3.182446305284263 * sd / 2
    return {"mean": mean, "sample_sd": sd, "descriptive_ci95": [mean - half, mean + half]}


def validate_protocol(p):
    if (
        p["experiment_id"] != "E05A"
        or p["seeds"] != SEEDS
        or p["train_fraction"] != 0.4
        or not p["original_outcomes_already_known"]
        or any(
            p[key]
            for key in (
                "training_allowed",
                "model_inference_allowed",
                "threshold_tuning_allowed",
                "redefine_E03T_success_allowed",
                "new_confirmatory_claim_allowed",
                "automatic_followup_allowed",
            )
        )
    ):
        raise ValueError("This is only a retrospective audit, not another test evaluation.")


def audit(protocol_path):
    p = read(protocol_path)
    validate_protocol(p)
    root = Path(p["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "summary.json").exists():
        raise FileExistsError(
            "Preserve the completed retrospective audit; use verify, not overwrite."
        )
    final = Path(p["final_test_preservation"])
    old = Path(p["historical_root"])
    battery = Path(p["historical_predictions"])
    parents = Path(p["parent_preservation"])
    f = read(final / "frozen_protocol.json")
    summary = read(final / "summary.json")
    if (
        f["protocol"]["seeds"] != SEEDS
        or summary["audit_status"] != "passed"
        or f["file_sha256"].get("data/raw/最新多.csv") != p["dataset_sha256"]
    ):
        raise ValueError("Wrong E03T inputs.")
    if digest(p["historical_archive"]) != p["historical_archive_sha256"]:
        raise ValueError("Historical archive checksum changed.")
    files = {
        Path(protocol_path),
        Path(__file__),
        Path("tests/test_historical_tcn.py"),
        Path(p["historical_archive"]),
        final / "frozen_protocol.json",
        final / "summary.json",
        final / "preservation_manifest.json",
    }
    source_files = []
    for seed in SEEDS:
        run_name = f"tcn_40_percent_thesis_few_shot_v1_seed{seed}"
        case = old / "artifacts/tcn_few_shot" / run_name
        source_files.extend(
            case / name for name in ("metrics.json", "resolved_config.json", "status.json")
        )
        files.update(source_files[-3:])
        files.add(battery / f"predictions/{run_name}__fraction_0p4.csv")
        files.add(battery / f"run_results/{run_name}.json")
        files.update(
            final / f"seed{seed}/fusion_nn" / name
            for name in ("test_predictions.csv", "metrics.json")
        )
        parent_nn = parents / f["selected_models"][str(seed)]["fusion_nn"]["directory"]
        files.update(parent_nn / name for name in ("validation_predictions.csv", "metrics.json"))
        split = (
            parents
            / Path(f["selected_models"][str(seed)]["fusion_svm"]["directory"]).parent
            / "split_manifest.json"
        )
        files.add(split)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Retrospective input capture: both historical and E03T outcomes were known, NOT a prospective outcome-blind freeze.",
        "protocol": p,
        "file_sha256": {str(file): digest(file) for file in sorted(files)},
    }
    (root / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Read only the required known members; never extract an archive into the workspace.
    with tarfile.open(p["historical_archive"], "r:gz") as archive:
        for path in source_files:
            member = archive.extractfile(str(path.relative_to(old)))
            if member is None or hashlib.sha256(member.read()).hexdigest() != digest(path):
                raise ValueError("Unpacked historical metadata differs from the original archive.")
    rows = []
    max_error = 0.0
    max_ap_error = 0.0
    for seed in SEEDS:
        run_name = f"tcn_40_percent_thesis_few_shot_v1_seed{seed}"
        case = old / "artifacts/tcn_few_shot" / run_name
        m = read(case / "metrics.json")
        config = read(case / "resolved_config.json")
        source_data = m["data"]
        selected = f["selected_models"][str(seed)]["fusion_nn"]
        if (
            read(case / "status.json")["status"] != "completed"
            or source_data["dataset_sha256"] != p["dataset_sha256"]
            or source_data["classes"] != f["protocol"]["classes"]
            or source_data["train_subset"]["requested_fraction"] != 0.4
            or source_data["train_subset"]["selected_train_count"] != 9827
            or source_data["train_subset"]["sample_ids_sha256"] != selected["train_ids_sha256"]
        ):
            raise ValueError("Historical training-budget/data/classes/subset mismatch.")
        split_path = (
            parents
            / Path(f["selected_models"][str(seed)]["fusion_svm"]["directory"]).parent
            / "split_manifest.json"
        )
        split = read(split_path)
        for key in (
            "id_column",
            "sequence_column",
            "label_column",
            "invalid_labels",
            "min_length",
            "validation_size",
            "test_size",
            "random_state",
        ):
            if config["data"][key] != split["protocol"][key]:
                raise ValueError(f"Different historical data/split protocol: {key}")
        if (
            config["data"]["max_length"] != 1024
            or config["data"]["normalize"] != "none"
            or source_data["normalization"] != "none"
        ):
            raise ValueError("Unexpected historical TCN preprocessing.")
        if source_data["split"] != {"train": 9827, "validation": 3510, "test": 7020}:
            raise ValueError("Historical split counts mismatch.")
        predictions = read_rows(battery / f"predictions/{run_name}__fraction_0p4.csv")
        if {r["split"] for r in predictions} != {"validation", "test"}:
            raise ValueError("Unexpected prediction split scope.")
        historical = {
            s: [r for r in predictions if r["split"] == s] for s in ("validation", "test")
        }
        parent_nn = parents / selected["directory"]
        current = {
            "validation": read_rows(parent_nn / "validation_predictions.csv"),
            "test": read_rows(final / f"seed{seed}/fusion_nn/test_predictions.csv"),
        }
        ranking = read(battery / f"run_results/{run_name}.json")["results"]
        if (
            len(ranking) != 1
            or ranking[0]["train_fraction"] != 0.4
            or ranking[0]["seed"] != seed
            or ranking[0]["model"] != "TCN"
            or ranking[0]["score_type"] != "softmax_probability"
            or ranking[0]["critical_index"] != 2
        ):
            raise ValueError("Wrong historical saved-score provenance.")
        for part in ("validation", "test"):
            match_rows(historical[part], current[part])
            if [r["sample_id"] for r in historical[part]] != split["splits"][part]:
                raise ValueError("Actual prediction IDs do not match the split manifest.")
            for row in historical[part]:
                if (
                    row["true_label"] != row["true_index"]
                    or row["predicted_label"] != row["predicted_index"]
                ):
                    raise ValueError("Historical label encoding mismatch.")
            a = metrics(historical[part])
            b = metrics(current[part])
            if part == "test":
                max_error = max(max_error, compare_metrics(a, m["test_metrics"]))
                new_m = read(final / f"seed{seed}/fusion_nn/metrics.json")["test_metrics"]
            else:
                error = abs(a["macro_f1"] - m["best_validation_macro_f1"])
                if error > 1e-12:
                    raise ValueError("Historical validation selection score not reproduced.")
                max_error = max(max_error, error)
                new_m = read(parent_nn / "metrics.json")["validation_metrics"]
            max_error = max(max_error, compare_metrics(b, new_m))
            old_ap = average_precision(historical[part], "battery_score")
            ap_error = abs(old_ap - ranking[0]["ranking"][part]["average_precision"])
            if ap_error > 1e-12:
                raise ValueError("Saved TCN battery scores do not reproduce historical AP.")
            max_ap_error = max(max_ap_error, ap_error)
            rows.append(
                {
                    "seed": seed,
                    "split": part,
                    "tcn_macro_f1_percent": a["macro_f1"] * 100,
                    "fusion_nn_macro_f1_percent": b["macro_f1"] * 100,
                    "fusion_minus_tcn_pp": (b["macro_f1"] - a["macro_f1"]) * 100,
                    "tcn_accuracy_percent": a["accuracy"] * 100,
                    "fusion_accuracy_percent": b["accuracy"] * 100,
                    "tcn_battery_recall_percent": a["battery_recall"] * 100,
                    "fusion_battery_recall_percent": b["battery_recall"] * 100,
                    "tcn_battery_fpr_percent": a["battery_fpr"] * 100,
                    "fusion_battery_fpr_percent": b["battery_fpr"] * 100,
                    "tcn_battery_ap_percent": old_ap * 100,
                    "fusion_battery_ap_percent": average_precision(current[part], "probability_2")
                    * 100,
                    "historical_training_epochs": len(m["history"]),
                    "historical_configured_max_epochs": config["tcn_training"]["epochs"],
                    "subset_sha256": selected["train_ids_sha256"],
                }
            )
    aggregates = {}
    for part in ("validation", "test"):
        values = [r for r in rows if r["split"] == part]
        aggregates[part] = {
            key: describe([r[key] for r in values])
            for key in values[0]
            if key
            not in {
                "seed",
                "split",
                "subset_sha256",
                "historical_training_epochs",
                "historical_configured_max_epochs",
            }
        }
        aggregates[part]["fusion_strictly_positive_seeds"] = sum(
            r["fusion_minus_tcn_pp"] > 0 for r in values
        )
    output = {
        "audit_status": "passed",
        "scope": "retrospective historical same-label-budget pipeline comparison",
        "seeds": SEEDS,
        "comparability": {
            "dataset_classes_filter_split_parameters_match": True,
            "training_subset_fingerprints_match": True,
            "old_training_ID_arrays_available": False,
            "explicit_validation_test_ID_order_and_truth_match": True,
            "same_normalization": False,
            "same_training_selection_compute_budget": False,
            "original_TCN_checkpoint_hash_verified": False,
            "cross_hardware_cost_comparable": False,
        },
        "audited_historical_predictions": 4 * (3510 + 7020),
        "audited_fusion_predictions": 4 * (3510 + 7020),
        "maximum_source_metric_abs_error": max_error,
        "maximum_historical_AP_abs_error": max_ap_error,
        "per_seed": rows,
        "aggregates": aggregates,
        "original_E03T_confirmation_passed": summary["test_confirmation_passed"],
        "E03T_success_rules_changed": False,
        "training_performed": False,
        "model_inference_performed": False,
        "thresholds_selected": False,
        "original_outcomes_already_known": True,
        "limitation": "Historical metadata and saved predictions agree; original TCN checkpoint/source/environment bytes are not available in the metadata archive. This is not a newly controlled/retrained experiment or independent test confirmation.",
    }
    with (root / "per_seed_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    write_report(output, root)
    print(
        json.dumps(
            {"audit_status": "passed", "test": aggregates["test"], "scope": output["scope"]},
            indent=2,
        )
    )
    return output


def write_report(result, root):
    t = result["aggregates"]["test"]
    v = result["aggregates"]["validation"]
    a = t["tcn_macro_f1_percent"]
    b = t["fusion_nn_macro_f1_percent"]
    d = t["fusion_minus_tcn_pp"]
    lines = [
        "# E05A：历史40%预算TCN与E03T的可比性审计",
        "",
        "日期：2026-10-19。事后复用既有预测；没有训练、重复test推理或阈值选择，只重新核算保存预测。",
        "",
        "## 为什么补这项",
        "",
        "E03T预先固定的7方法对照没有包含TCN。此前提醒不能与历史完整标签TCN直接排名是正确的，但历史研究实际上还保存了40%预算TCN，不能遗漏这一更相关的对照。本报告不回改E03T门槛，也不把事后新增参照说成预先注册。",
        "",
        "## 核验与限制",
        "",
        "- 原20260728 metadata压缩包SHA b8367081...符合已记录指纹，12个metrics/config/status文件逐字节与压缩包内版本相同。",
        "- 四seed42/44/45/46的数据SHA、类别、过滤规则、split参数一致；9,827个训练样本的指纹全部一致。旧train ID数组未随metadata包保存，因此这是指纹核对，不虚称读取了旧数组。",
        "- 显式核对每seed3,510 validation、7,020 test的顺序/ID/真值。历史TCN和新融合各42,120行保存预测经审计；TCN的test混淆矩阵/指标、validation最佳F1和保存电池概率AP与原记录一致。",
        f"- 独立混淆指标最大误差{result['maximum_source_metric_abs_error']:.2e}，历史AP最大误差{result['maximum_historical_AP_abs_error']:.2e}。",
        "- TCN吃原始未归一化序列；MOMENT走z-score序列并旁路原始统计。两者拥有原始信息，但预处理/结构/预训练/选择预算并不相同，只能比较这些既定管线，不能孤立归因于架构或预训练。",
        "- TCN配置最多50轮，融合头最多200轮；训练成本/硬件不相同。旧TCN模型权重、完整源码及环境字节不在metadata包内，未冒充已核对原TCN checkpoint哈希或同机严格复现实验。",
        "",
        "## 同四seed的test结果",
        "",
        "| Seed | 历史TCN F1 % | 增益融合NN F1 % | 融合−TCN pp |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        if row["split"] == "test":
            lines.append(
                f"| {row['seed']} | {row['tcn_macro_f1_percent']:.4f} | {row['fusion_nn_macro_f1_percent']:.4f} | {row['fusion_minus_tcn_pp']:+.4f} |"
            )
    lines += [
        "",
        f"历史TCN：**{a['mean']:.4f}% ± {a['sample_sd']:.4f}%**；增益融合NN：**{b['mean']:.4f}% ± {b['sample_sd']:.4f}%**（样本SD）。",
        f"融合−TCN平均 **{d['mean']:+.4f}pp**，{t['fusion_strictly_positive_seeds']}/4严格正向；描述性95%配对区间 **[{d['descriptive_ci95'][0]:+.4f}, {d['descriptive_ci95'][1]:+.4f}]pp**。",
        f"validation侧融合−TCN为 **{v['fusion_minus_tcn_pp']['mean']:+.4f}pp**。未按test重挑任何模型。",
        "",
        "## 结论",
        "",
        "E03T仍然通过相对预先固定融合SVM和仅MOMENT的两条规则，但这不是在所有时序方法中最优的证明。补充同标签预算TCN后，TCN均值更高，融合在这四个seed上方差更低；配对区间跨0，不能把均值差写成总体显著优劣，也不能宣称融合普遍超过TCN。",
        "应将论文表述收紧为：原始统计旁路与增益头修复了冻结MOMENT管线的部分缺陷，并在既定40%预算下超过所列SVM/统计参照；在现有历史同预算TCN对照面前，没有证明其综合优越性。",
        "",
        "该比较在已有test结果之后提出，属于探索性补充；重叠split与历史模型选择限制保留。下一步不应在这份已消费的test上追分；若继续研究，应先获取新的设备/时间外推数据或独立协议，并保留TCN强参照。当前CSV没有可核实的设备/站点/时间分组字段，不能从唯一sample ID猜组。",
        "",
        f"机器可读证据：`{root}/summary.json`、`per_seed_comparison.csv`、`input_manifest.json`。原E03T与历史输出均不改写。",
    ]
    Path("docs/experiment_records/e05a_historical_tcn_audit_20261019.md").write_text(
        "\n".join(lines) + "\n"
    )


def verify(protocol_path):
    p = read(protocol_path)
    validate_protocol(p)
    root = Path(p["output_dir"])
    m = read(root / "input_manifest.json")
    if m["protocol"] != p:
        raise ValueError("Audit scope changed.")
    for file, expected in m["file_sha256"].items():
        if digest(file) != expected:
            raise ValueError(f"Retrospective source changed: {file}")
    if read(root / "summary.json")["audit_status"] != "passed":
        raise ValueError("Incomplete audit.")
    print("Retrospective E05A inputs verified; no inference performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "verify"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    globals()[args.action](args.protocol)


if __name__ == "__main__":
    main()
