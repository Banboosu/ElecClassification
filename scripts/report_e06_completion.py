"""Render audited E06 saved results; never train models or select new candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "original_tcn": "Original TCN",
    "tcn_only": "TCN + matched residual head",
    "tcn_moment": "TCN + pretrained MOMENT",
    "tcn_random": "TCN + random MOMENT",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.analysis / "summary.json").read_text())
    proof = json.loads(args.audit.read_text())
    if summary["audit_status"] != "passed" or proof["status"] != "passed" or summary["test_evaluated"]:
        raise ValueError("Require complete independently audited validation results")
    frozen = json.loads((args.analysis / "frozen_protocol.json").read_text())
    p = frozen["protocol"]
    lines = ["# E06：强TCN与冻结MOMENT互补结果", "",
             "本轮为已知历史test后的train/validation探索，不是新test或外部确认。",
             "固定10%训练池标签（每seed2456条）、validation3510条、seed42/44/45/46。",
             "TCN与MOMENT均冻结，只训练相同形状的残差头。原始TCN另作固定强参照。", "",
             "## 四seed结果", "", "单位%，均值±样本SD。", "",
             "| 方法 | validation Macro-F1 |", "|---|---:|"]
    names = list(LABELS)
    for name in names:
        a = summary["aggregate"][name]
        lines.append(f"| {LABELS[name]} | {a['macro_f1_mean_percent']:.4f} ± {a['macro_f1_sd_percent']:.4f} |")
    lines += ["", "## 原先冻结的三个成功门槛", "",
              "| MOMENT融合减参照 | 均值pp | 正向seed | 描述性95%区间pp | 达到门槛 |",
              "|---|---:|---:|---|---|"]
    for name, row in summary["comparisons"].items():
        ci = row["descriptive_95pct_interval_pp"]
        lines.append(f"| {LABELS[name]} | {row['mean_pp']:+.4f} | {row['positive_seeds']}/4 | "
                     f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {'是' if row['passed'] else '否'} |")
    passed = summary["all_success_rules_passed"]
    lines += ["", f"**三条全部通过：{'是' if passed else '否'}。**",
              "仅在三个预设规则均满足时，称达到本轮探索性实用增量标准；否则不改变门槛、参照或seed。", "",
              "### 如何解释", "",
              "必须区分相对原TCN的总增益与相对同形TCN头的净增益：前者也包含重新训练残差头的作用，不能全部归给MOMENT。",
              "本轮实用门槛是否通过只由上表三个固定比较决定；方向一致的小幅增量可如实报告，但不替代幅度门槛，也不构成独立外部确认。", "",
              "## 逐seed与类别/电池指标", "",
              "| seed | 方法 | Macro-F1 % | Accuracy % | 电池AP % | 电池Recall % | 电池FPR % | selected epoch |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["rows"]:
        m = row["metrics"]
        values = [m[key] * 100 for key in ("macro_f1", "accuracy", "battery_AP", "battery_recall", "battery_FPR")]
        cells = " | ".join(f"{value:.4f}" for value in values)
        lines.append(f"| {row['seed']} | {LABELS[row['condition']]} | {cells} | {row.get('selected_epoch', 'parent')} |")
    lines += ["", "## 设计、计时与边界", "",
              "- 同形1152→128→3残差头，输出零初始化，epoch0恢复原TCN且允许选中；每条件固定100轮、batch128、AdamW1e-3、WD1e-4、FP32且关闭TF32。",
              "- TCN-only的1024维旁路为零，参数数目相同但有效信息/自由度不同；随机MOMENT控制补充检验预训练价值。",
              "- 每分支scaler只在2456条所选训练样本拟合；父TCN已用validation选模型，新头又用同validation选轮数。存在重复选择/开发偏差，不能称独立泛化确认。",
              "- 训练标签2456与开发/选模标签3510分别报告，不能宣称全流程总标注仅2456条。",
              "- 预训练MOMENT和seed44/45/46随机表征复用历史缓存；seed42随机控制在本机补提取。不是统一端到端训练/推理成本比较。",
              "- 区间只描述四个重叠split；电池指标使用默认argmax，不调阈值，不作安全保证。",
              f"- 冻结时间：{frozen['frozen_at']}；协议SHA：`{frozen['protocol_sha256']}`。",
              f"- 运行host/GPU：`{frozen['runtime']['host']}` / `{frozen['runtime']['gpu']}`。",
              "", "## 完整性与保全", "",
              f"- 独立标准库审计{proof['head_training_epochs_audited']}轮训练、{proof['prediction_rows_audited']}行保存预测；原CSV真值、有序split ID、概率、混淆指标、AP、选轮和三条门槛核对通过。",
              f"- 最大指标误差{proof['maximum_independent_metric_error']:.3g}，AP误差{proof['maximum_independent_AP_error']:.3g}。",
              "- 本地结果前冻结文件与终态冻结字节一致；模型/特征/scaler/预测/源码快照通过输出manifest核对。",
              "- 保全根：`artifacts/imports/e06_tcn_moment_complement_20261019/`；完整结果在`terminal_artifacts/`，逐文件传输核验见`preservation_verified.json`。",
              "- 本轮结束，不自动追加实验、访问test、关机或启用pi-wake。", "",
              "![E06 four-seed validation comparison](../figures/e06_tcn_moment_complement_20261019.svg)"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#999999"]
    for i, (name, color) in enumerate(zip(names, colors)):
        a = summary["aggregate"][name]
        axes[0].errorbar(i, a["macro_f1_mean_percent"], yerr=a["macro_f1_sd_percent"],
                         fmt="o", color=color, capsize=5)
        axes[0].scatter([i] * 4, a["per_seed_percent"], color=color, alpha=.5, s=22)
    axes[0].set_xticks(range(4), ["Original\nTCN", "TCN\nhead", "TCN +\nMOMENT", "TCN +\nrandom"])
    axes[0].set_ylabel("Validation Macro-F1 (%)")
    axes[0].set_title("Four seeds: mean and sample SD")
    for name, row in summary["comparisons"].items():
        axes[1].plot(p["seeds"], row["per_seed_pp"], "o-", label=f"minus {LABELS[name]}")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].axhline(.5, color="grey", linestyle=":", linewidth=.8)
    axes[1].set_xticks(p["seeds"])
    axes[1].set_ylabel("Paired Macro-F1 gain (pp)")
    axes[1].set_xlabel("Seed")
    axes[1].set_title("Pretrained MOMENT fusion gain")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure.with_suffix(".svg"))
    figure.savefig(args.figure.with_suffix(".png"), dpi=300)
    plt.close(figure)
    print(args.report)


if __name__ == "__main__":
    main()
