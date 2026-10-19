"""Render a completed E02P validation-only pooling pilot; no fitting/inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis
    source = root / "summary.json"
    s = json.loads(source.read_text())
    f = json.loads((root / "frozen_protocol.json").read_text())
    status = json.loads((root / "suite_status.json").read_text())
    if (
        s["audit_status"] != "passed"
        or status["status"] != "completed"
        or s["test_evaluated"]
        or s["conditions"] != 6
    ):
        raise ValueError("Require completed validation-only pilot.")
    p = f["protocol"]
    d = s["pilot_decision"]
    selected = d["selected_pooling"]
    eligible = "通过" if d["eligible_for_separately_frozen_replication"] else "未通过"
    lines = [
        "# E02P：冻结表征池化与长度诊断结果（validation-only）",
        "",
        "日期：2026-10-19。只用现有数据，探索seed43、40%标签；没有test推理或encoder监督训练。",
        "",
        "## 协议与技术对照",
        "",
        f"- {f['frozen_at']}在新真实池化结果前冻结{len(f['file_sha256'])}输入；协议SHA `{f['protocol_sha256']}`。",
        "- 三种池化×仅MOMENT/原始统计融合，共6条件；各9827 train、3510 validation，5C×3gamma×5折，折内标准化，FP32骨干和float64核距离。",
        "- mean与mean+max/thirds来自同一批未适配patch token；完整有效patch掩码不改。多段输入维数增加，不能声称所有模型复杂度完全相同。",
        "",
        "| 池化 | 分支 | 维数（不含统计） | Validation Macro-F1 % | Accuracy % |",
        "|---|---|---:|---:|---:|",
    ]
    for row in s["per_condition"]:
        lines.append(
            f"| {row['pooling']} | {row['branch']} | {p['dimensions'][row['pooling']]} | {row['macro_f1_percent']:.4f} | {row['accuracy_percent']:.4f} |"
        )
    lines += [
        "",
        "mean两对照均通过原best params、validation逐行预测、75个fold分数的预设兼容性检查；新mean特征亦通过旧FP32缓存核对。",
        "",
        "## 固定pilot门槛",
        "",
        f"候选按预定顺序选择 **{selected}/fusion**，validation Macro-F1 **{d['selected_validation_f1'] * 100:.4f}%**。",
        f"- 相对本轮mean融合SVM **{d['over_mean_fusion_pp']:+.4f}pp**，门槛≥0.5pp。",
        f"- 相对既有最强非适配管线 **{d['over_strongest_prior_pp']:+.4f}pp**，门槛≥0.25pp。",
        f"- 两条同时满足的复制建议规则：**{eligible}**。不自动扩seed、改模型头、LoRA、加预算或test。",
        "",
        "| 既有参照（同seed validation） | Macro-F1 % |",
        "|---|---:|",
    ]
    for name, value in s["reference_validation_f1"].items():
        lines.append(f"| {name} | {value * 100:.4f} |")
    lines += [
        "",
        f"所选候选相对历史同预算TCN validation差 **{(d['selected_validation_f1'] - s['reference_validation_f1']['historical_tcn']) * 100:+.4f}pp**；不据单seed、不同管线选择预算和历史权重未重核这一上下文宣称普遍优于TCN。",
        "",
        "## 长度分层",
        "",
        f"仅选定train的1/3、2/3线性分位点：**{s['length_edges']}**。组0≤首点，组1介于两点（含右边界），组2>第二点。",
        "",
        "| 池化 | 分支 | 长度组 | N | 真值类别计数0/1/2 | Macro-F1 % |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in s["per_condition"]:
        for group in row["length_strata"]:
            score = (
                "空组" if group["metrics"] is None else f"{group['metrics']['macro_f1'] * 100:.4f}"
            )
            lines.append(
                f"| {row['pooling']} | {row['branch']} | {group['bin']} | {group['n']} | {group['class_counts']} | {score} |"
            )
    lines += [
        "",
        "各组使用固定3类别Macro-F1；如缺类别或样本少需结合计数解释，不据分组分数再选方法。",
        "",
        "## 边界与padding",
        "",
        f"固定合成有效长度{s['padding_diagnostics']['valid_lengths']}，包含短、常量及非整patch尾部。比较512/1024 padding及掩码外junk。差异：`{s['padding_diagnostics']['max_abs_feature_differences']}`。",
        f"全部在描述性容差1e-4内：**{s['padding_diagnostics']['all_within_descriptive_tolerance']}**。",
        "- thirds按完整有效patch的时间排名分段，不按padding长度切三段；空段补零，不复制观测。max也排除invalid token。",
        "- 不完整尾部不参与池化，并不代表对编码器无作用：外部z-score/内部RevIN仍可能读取这些有效原始点；未声称整条管线对尾部不变。",
        "",
        "## 证据与限制",
        "",
        "- 6条件全部保留，21,060行validation预测、混淆指标、CV/模型/cache哈希与长度组通过审计。所有模型仅在train标签上拟合，validation用于候选比较，未输出新test指标。",
        "- 这是已知历史/E03T test之后的validation探索；seed43也被多次用于方案发现，不是独立确认。不得据此修改E03T已固定结果或声称新外部泛化。",
        "- 旧TCN只作validation上下文；原E05A关于整体优越性未证明的限制不回改。",
        f"源summary SHA `{hashlib.sha256(source.read_bytes()).hexdigest()}`；机器可读源`{root}/summary.json`。缓存/模型/预测及冻结源码快照均保留。",
    ]
    manifest = json.loads((root / "prepared/manifest.json").read_text())
    lines += [
        f"- 特征准备/统计/合成诊断合计计时{manifest['extraction_seconds_including_loading_statistics_and_probes'] / 60:.2f}分钟，6次CV/拟合/validation预测合计{sum(r['training_seconds'] for r in s['per_condition']) / 60:.2f}分钟；这些不是线上延迟。"
    ]
    out = Path("docs/experiment_records/e02_pooling_pilot_results_20261019.md")
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
