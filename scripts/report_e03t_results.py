"""Render audited fixed-model E03T outputs; no model inference or selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    source = args.analysis / "summary.json"
    data = json.loads(source.read_text())
    status = json.loads((args.analysis / "suite_status.json").read_text())
    frozen = json.loads((args.analysis / "frozen_protocol.json").read_text())
    p = frozen["protocol"]
    if (
        data["audit_status"] != "passed"
        or status["status"] != "completed"
        or data["protocol_sha256"] != frozen["protocol_sha256"]
        or data["conditions"] != 28
        or data["audited_prediction_rows"] != 196560
        or data["pilot_seed_included"]
        or data["training_performed"]
    ):
        raise ValueError("Only a completed, audited, fixed E03T suite can be reported.")
    labels = {
        "fusion_nn": "增益融合NN",
        "statistics_nn": "统计NN",
        "moment_nn": "仅MOMENT NN",
        "fusion_svm": "父融合SVM",
        "matched_statistics_svm": "匹配带宽统计SVM",
        "statistics_svm": "原统计SVM",
        "statistics_rf": "原统计RF",
    }
    primary = data["primary_rule"]
    ablation = data["moment_ablation_rule"]
    primary_result = "通过" if primary["passed"] else "未通过"
    ablation_result = "通过" if ablation["passed"] else "未通过"
    confirmation = "通过" if data["test_confirmation_passed"] else "未通过"
    lines = [
        "# E03T：40%标签预算最终固定模型test结果",
        "",
        "日期：2026-10-19。用户明确批准后，先冻结最终方法/对照，再执行本阶段一次正式test；零重训、零调参。",
        "",
        "## 协议与审计",
        "",
        f"- 冻结时间 `{frozen['frozen_at']}`；协议SHA-256 `{data['protocol_sha256']}`。",
        f"- {len(frozen['file_sha256'])}个冻结输入文件，28个既有模型；四seed42/44/45/46，pilot43不加入。",
        "- 每seed选定训练标签9,827（40%），validation3,510，test7,020。NN加载E03E按validation选中的best checkpoint；SVM/RF加载既有CV-refitted pipeline。",
        "- 新RTX4080SUPER上四seed全部通过validation-only恢复及256行新特征probe，之后才编码任何test。原V100产物/指标/源码保持不变。",
        "- 28项全部完成；保存并审计196,560行预测。逐条件检查模型/cache/预测哈希、test ID和CSV原始真值、概率argmax、混淆矩阵及全部嵌套指标。",
        "",
        "## 全部固定方法",
        "",
        "| 方法 | Macro-F1 %（均值±样本SD） | Accuracy %（均值±样本SD） |",
        "|---|---:|---:|",
    ]
    for method in p["methods"]:
        a = data["aggregates"][method]["macro_f1_percent"]
        b = data["aggregates"][method]["accuracy_percent"]
        lines.append(
            f"| {labels[method]} | {a['mean']:.4f} ± {a['sample_sd']:.4f} | {b['mean']:.4f} ± {b['sample_sd']:.4f} |"
        )
    lines += [
        "",
        "## 预先固定的主比较",
        "",
        "| Seed | 融合NN F1 % | 父融合SVM F1 % | 配对增量pp | 融合best epoch |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in p["seeds"]:
        row = data["per_seed_f1_percent"][str(seed)]
        epoch = frozen["selected_models"][str(seed)]["fusion_nn"]["best_epoch"]
        lines.append(
            f"| {seed} | {row['fusion_nn']:.4f} | {row['fusion_svm']:.4f} | {row['fusion_nn'] - row['fusion_svm']:+.4f} | {epoch} |"
        )
    ci = primary["descriptive_ci95"]
    ai = ablation["descriptive_ci95"]
    lines += [
        "",
        f"- 融合NN−预先固定父融合SVM：均值 **{primary['mean']:+.4f}pp**，{primary['strictly_positive_seeds']}/4严格正向；描述性95%配对t区间 **[{ci[0]:+.4f}, {ci[1]:+.4f}]pp**。≥0.5pp且≥3/4正向的主规则：**{primary_result}**。",
        f"- 融合NN−仅MOMENT NN：均值 **{ablation['mean']:+.4f}pp**，{ablation['strictly_positive_seeds']}/4正向；描述性区间 **[{ai[0]:+.4f}, {ai[1]:+.4f}]pp**。≥0.25pp且≥3/4正向的消融规则：**{ablation_result}**。",
        f"- 两条同时满足的本阶段test确认规则：**{confirmation}**。不因test结果替换参照、pool43、改门槛或重选checkpoint。E03E的validation成功及此前失败门控均保持原判。",
        "",
        "## 其余固定配对比较",
        "",
        "| 融合NN减参照 | 平均差pp | 描述性95%区间pp |",
        "|---|---:|---:|",
    ]
    for method, item in data["fusion_minus_fixed_comparators"].items():
        ci = item["descriptive_ci95"]
        lines.append(f"| {labels[method]} | {item['mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += [
        "",
        "## 类别与历史类别2（充电器）指标",
        "",
        "| 方法 | 正常F1 % | 电池异常F1 % | 充电器故障F1 % | 充电器Recall % | 充电器FPR % | 充电器AP % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in p["methods"]:
        rows = [r for r in data["per_run"] if r["method"] == method]

        def mean(key, group=rows):
            return statistics.mean(r[key] for r in group)

        ap = (
            "不适用"
            if rows[0]["battery_ap_percent"] is None
            else f"{mean('battery_ap_percent'):.4f}"
        )
        lines.append(
            f"| {labels[method]} | {mean('class_0_f1_percent'):.4f} | {mean('class_1_f1_percent'):.4f} | {mean('class_2_f1_percent'):.4f} | {mean('battery_recall_percent'):.4f} | {mean('battery_fpr_percent'):.4f} | {ap} |"
        )
    lines += [
        "",
        "本格式器仅读取旧E03T汇总，旧battery_*键实际指类别2充电器，不是电池。使用默认argmax/predict，没有为充电器故障搜索阈值或校准SVM概率。AP仅对有原生概率输出的NN/RF报告；SVM未临时开启probability或用decision值冒充概率。Recall/FPR不是固定高召回安全运行点，不构成部署保证。",
        "",
        "## 成本与结论边界",
        "",
        f"- 四seed的test两种精度路径（AMP-FP16与FP32）骨干提取计时合计 **{data['extraction_seconds_both_precisions_total'] / 60:.2f}分钟**。不含CSV解析、技术关卡和全部审计，不能直接当线上延迟；每方法推理含加载的计时见per_run_metrics.csv。",
        "- 本阶段只评价固定的40%标签预算，不是完整标签训练结果；不能据此与历史全标签TCN95.99%/完全微调95.37%直接排名。",
        "- 数据集test在历史研究中已被使用，split跨seed重叠，validation也反复复用。此次是明确冻结后的阶段性test，不是全项目从未看过的全新数据；区间仅描述，不作独立外部验证或显著性保证。",
        "- GPU由V100更换为RTX4080SUPER；保留旧训练结果，只以预先规定的技术一致性关卡接受新机推理，未宣称跨卡逐位重现。耗时不与V100直接合并。",
        "- NN和SVM既有选择预算不同；差异不应单独归因于预训练、模型结构或优化算法。",
        "- 本阶段结束即停止；没有自动LoRA、追加seed/epoch、再测试或远程关机。",
        "",
        "## 证据",
        "",
        f"机器可读源：`{args.analysis}/summary.json`；SHA-256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`。",
        "各seed下包含test_features.npz/manifest及7方法的test_predictions.csv/metrics.json；冻结清单含来源模型、best epoch和依赖哈希。原28个模型及公共骨干仍保存在此前完整保全中。",
        "推理与审计入口：`python -m tcn_moment.final_test`。本报告仅渲染审计结果，没有重新推理、选择或修改冻结输入。",
    ]
    independent = json.loads((args.analysis / "independent_prediction_audit.json").read_text())
    if (
        independent["status"] != "passed"
        or independent["summary_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
    ):
        raise ValueError("Independent audit is missing or references another summary.")
    lines += [
        "",
        f"独立stdlib审计同样通过196,560行；由混淆矩阵直接计算的指标最大差{independent['maximum_confusion_derived_metric_abs_error']:.2e}，16项原生概率输出的分组同分AP最大差{independent['maximum_grouped_tie_AP_abs_error']:.2e}。",
    ]
    preservation = args.analysis / "preservation_verified.json"
    if preservation.exists():
        backup = json.loads(preservation.read_text())
        if backup["status"] != "passed":
            raise ValueError("Local preservation failed.")
        lines += [
            "",
            f"本地备份`{args.analysis}/`已核验{backup['files_verified']}文件、{backup['bytes_verified']:,} bytes；{backup['preserved_parent_models_verified']}/28来源模型及公共基座与此前完整保全一致，零失败。",
        ]
    if data["test_confirmation_passed"]:
        matched = data["fusion_minus_fixed_comparators"]["matched_statistics_svm"]["mean"]
        lines += [
            "",
            "## 可用于论文的结论",
            "",
            f"在本数据集40%训练标签预算下，冻结MOMENT表征与原始功率统计的增益融合头，正式test Macro-F1为{data['aggregates']['fusion_nn']['macro_f1_percent']['mean']:.2f}%，相对固定融合SVM净增{primary['mean']:.3f}pp，相对匹配带宽统计SVM净增{matched:.3f}pp。前者是小幅但四seed方向一致的改进，不应夸大。仅MOMENT分支明显较弱，结论是联合系统在该协议下有效，不是MOMENT单独优于统计，也不是完整标签下胜过TCN。",
        ]
    dest = Path("docs/experiment_records/e03t_final_test_results_20261019.md")
    dest.write_text("\n".join(lines) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
