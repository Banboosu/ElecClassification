"""Render independently audited E07 validation results; no training or model inference."""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from audit_e05c_saved_outputs import digest, read

LABELS = {"frozen_fusion": "继续冻结融合", "lora8_fusion": "LoRA8融合"}


def stats(values):
    return {"mean": st.mean(values), "sample_sd": st.stdev(values), "per_seed": values}


def paired(values):
    result = stats(values)
    half = 3.182446305284263 * result["sample_sd"] / math.sqrt(len(values))
    return {**result, "positive_seeds": sum(v > 0 for v in values),
            "descriptive_ci95": [result["mean"] - half, result["mean"] + half]}


def analyze(base):
    root = base / "terminal_artifacts"
    summary = read(root / "summary.json")
    wanted = digest(root / "summary.json")
    for filename in ("independent_output_audit.json", "independent_reference_audit.json", "checkpoint_audit.json"):
        proof = read(base / filename)
        if proof["status"] != "passed" or proof["summary_sha256"] != wanted or proof["test_evaluated"]:
            raise ValueError("Require matching complete validation-only audit proofs")
    preservation = read(base / "preservation_verified.json")
    if preservation["status"] != "completed_verified" or preservation["sha256_mismatches"]:
        raise ValueError("Require verified full terminal preservation")
    frozen = read(root / "frozen_protocol.json")
    if digest(root / "frozen_protocol.json") != digest(base / "startup/frozen_protocol.json"):
        raise ValueError("Original pre-outcome freeze changed")
    if (summary["audit_status"] != "passed" or summary["test_evaluated"]
        or read(root / "suite_status.json")["status"] != "completed"):
        raise ValueError("Incomplete or nonvalidation suite")
    seeds = frozen["protocol"]["seeds"]
    rows = {(r["seed"], r["condition"]): r for r in summary["per_condition"]}
    endpoint = {c: stats([100 * rows[seed, c]["endpoint_macro_f1"] for seed in seeds]) for c in LABELS}
    endpoint_gain = paired([100 * (rows[seed, "lora8_fusion"]["endpoint_macro_f1"]
                                  - rows[seed, "frozen_fusion"]["endpoint_macro_f1"]) for seed in seeds])
    reference = {name: stats([100 * summary["references"][str(seed)][name]["macro_f1"] for seed in seeds])
                 for name in summary["references"][str(seeds[0])]}
    class_battery = {}
    cost = {}
    for c in LABELS:
        values = {key: stats([100 * rows[seed, c]["selected"][key] for seed in seeds])
                  for key in ("accuracy", "battery_recall", "battery_FPR", "battery_AP")}
        values["class_f1"] = [stats([100 * rows[seed, c]["selected"]["class_f1"][i] for seed in seeds]) for i in range(3)]
        class_battery[c] = values
        cost[c] = {"additional30_epoch_minutes": stats([rows[seed, c]["training_seconds"] / 60 for seed in seeds]),
                   "peak_allocated_memory_mib": [read(root / f"seed{seed}" / c / "metrics.json")["peak_allocated_memory_mib"] for seed in seeds],
                   "parameters": rows[seeds[0], c]["parameters"]}
    derived = {"summary_sha256": wanted, "endpoint_macro_f1_percent": endpoint, "endpoint_gain_pp": endpoint_gain,
               "reference_macro_f1_percent": reference, "selected_class_battery_percent": class_battery,
               "cost": cost, "training_time_ratio_lora_over_frozen": cost["lora8_fusion"]["additional30_epoch_minutes"]["mean"] / cost["frozen_fusion"]["additional30_epoch_minutes"]["mean"],
               "test_evaluated": False, "inference_performed": False}
    return summary, frozen, derived


def render(base, report, figure=None):
    s, f, d = analyze(base)
    root = base / "terminal_artifacts"
    p = f["protocol"]
    seeds = p["seeds"]
    records = {(r["seed"], r["condition"]): r for r in s["per_condition"]}
    audit = read(base / "independent_output_audit.json")
    reference_audit = read(base / "independent_reference_audit.json")
    preservation = read(base / "preservation_verified.json")
    state = read(root / "suite_status.json")
    tz = ZoneInfo("Asia/Shanghai")
    start = datetime.fromisoformat(read(root / "started.json")["started_at"]).astimezone(tz)
    end = datetime.fromisoformat(state["updated_at"]).astimezone(tz)
    e = d["endpoint_gain_pp"]
    main = s["continued_frozen_rule"]
    passed = s["replication_rule_passed"]
    lines = ["# E07：强融合头LoRA精简复制结果（2026-10-20）", "",
             f"**8条件/240轮全部完成；原定实用增量门槛{'通过' if passed else '未通过'}。**",
             f"所选模型LoRA相对继续冻结的validation Macro-F1平均{main['mean_pp']:+.4f}pp，{main['positive_seeds']}/4 seed正向；原门槛为平均≥+0.5pp且至少3/4正向。",
             "这是已有强MOMENT融合管线上的小幅适配增量，不等于超过强TCN，也不是独立test确认。", "",
             "## 范围与执行", "",
             "- 固定seed42/44/45/46，排除提出假设的pilot seed43；每seed9827个训练标签（40%训练池）和3510个开发/选择validation标签，分别披露。",
             "- 两处理从同seed E03E best-within-200融合头及normalizer出发；均在线运行MOMENT编码器、重置optimizer/scheduler并追加30轮，无早停或rank/LR搜索。",
             "- LoRA仅24个encoder block的q/v，rank8/alpha16/dropout0.05；物理batch64、有效128、AdamW LR1e-4/WD0.01、固定FP16 scale1024。epoch0–30按原validation规则选择，epoch30端点另报。",
             f"- 套件起止：{start:%Y-%m-%d %H:%M:%S}—{end:%Y-%m-%d %H:%M:%S}（Asia/Shanghai），墙钟{state['wall_seconds']:.3f}秒（{state['wall_seconds']/3600:.3f}小时），未触及8小时上限。",
             "- 没有追加test、自动重试、扩大预算、下一实验、pi-wake或关机。", "",
             "## 主结果：validation所选模型", "", "均值±样本SD，单位%。TCN是事前固定的同训练标签上下文，训练/选择预算不同。", "",
             "| 方法 | 四seed Macro-F1 % |", "|---|---:|"]
    parent = d["reference_macro_f1_percent"]["parent_fusion_nn"]
    tcn = d["reference_macro_f1_percent"]["strong_tcn"]
    lines.append(f"| 原E03E强融合头（追加训练前） | {parent['mean']:.4f} ± {parent['sample_sd']:.4f} |")
    for c, label in LABELS.items():
        a = s["aggregate"][c]
        lines.append(f"| {label}，追加30轮 | {a['macro_f1_mean_percent']:.4f} ± {a['macro_f1_sample_sd_percent']:.4f} |")
    lines += [f"| E05C同40%强TCN（legacy masked，上下文） | {tcn['mean']:.4f} ± {tcn['sample_sd']:.4f} |", "",
              "| seed | 原融合头 % | 继续冻结 % | LoRA8 % | LoRA−冻结 pp | 选中epoch 冻结/LoRA |", "|---:|---:|---:|---:|---:|---:|"]
    for i, seed in enumerate(seeds):
        a, b = records[seed, "frozen_fusion"], records[seed, "lora8_fusion"]
        lines.append(f"| {seed} | {100*s['references'][str(seed)]['parent_fusion_nn']['macro_f1']:.4f} | {100*a['selected']['macro_f1']:.4f} | {100*b['selected']['macro_f1']:.4f} | {main['per_seed_pp'][i]:+.4f} | {a['best_epoch']}/{b['best_epoch']} |")
    total_gain = s["aggregate"]["lora8_fusion"]["macro_f1_mean_percent"] - parent["mean"]
    lines += ["", f"LoRA相对旧父头的总差为{total_gain:+.4f}pp，但继续训练冻结头本身也有收益，不能把这整个差值归给编码器适配；匹配继续训练控制后的净增量是{main['mean_pp']:+.4f}pp。", "",
              "## 原先冻结的两条判定", "", "两项均须平均≥+0.5pp且至少3/4 seed严格正向；未按结果改门槛或删seed。", "",
              "| LoRA减参照 | 平均pp | 正向seed | 描述性95%配对t区间pp | 达标 |", "|---|---:|---:|---|---|"]
    for key, label in (("continued_frozen_rule", "继续冻结融合"), ("strongest_nonadapted_rule", "固定集合内逐seed最强非适配")):
        row = s[key]
        ci = row["descriptive_ci95_pp"]
        lines.append(f"| {label} | {row['mean_pp']:+.4f} | {row['positive_seeds']}/4 | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {'是' if row['rule_passed'] else '否'} |")
    lines += ["", "本次固定非适配集合的最大值恰好均来自继续冻结融合，所以两项数值相同，不是两份独立证据。集合包含本轮冻结、父融合/统计/仅MOMENT NN、融合SVM、统计RF/SVM及匹配统计SVM，未事后增删。",
              f"LoRA−强TCN平均{s['lora_minus_strong_tcn_context']['mean_pp']:+.4f}pp，{s['lora_minus_strong_tcn_context']['positive_seeds']}/4正向。强TCN不是适配门槛，但总体参照仍明显更高。",
              "配对区间仅描述四个重叠随机split；主差区间跨0，不能写成显著或独立确认的普遍改善。", "",
              "## 预先要求的epoch30端点", "", "端点只作敏感性报告，不替换原所选模型门槛，也不据此另选轮数。", "",
              "| seed | 冻结端点 % | LoRA端点 % | 差pp |", "|---:|---:|---:|---:|"]
    for i, seed in enumerate(seeds):
        lines.append(f"| {seed} | {d['endpoint_macro_f1_percent']['frozen_fusion']['per_seed'][i]:.4f} | {d['endpoint_macro_f1_percent']['lora8_fusion']['per_seed'][i]:.4f} | {e['per_seed'][i]:+.4f} |")
    lines.append("")
    for c, label in LABELS.items():
        value = d["endpoint_macro_f1_percent"][c]
        lines.append(f"- {label}端点均值±SD：{value['mean']:.4f} ± {value['sample_sd']:.4f}%。")
    lines += [f"- 端点平均差{e['mean']:+.4f}pp，只有{e['positive_seeds']}/4正向；因此主表4/4正向指validation所选模型，不能说所有训练终点均有收益。", "",
              "## 类别与电池异常指标", "", "均为所选模型、默认argmax；不调阈值。单位%，下表为四seed均值。", "",
              "| 指标 | 继续冻结 | LoRA8 | 差pp |", "|---|---:|---:|---:|"]
    for name, key, index in (("正常F1", "class_f1", 0), ("充电器故障F1", "class_f1", 1), ("电池异常F1", "class_f1", 2),
                              ("Accuracy", "accuracy", None), ("电池AP", "battery_AP", None),
                              ("电池Recall", "battery_recall", None), ("电池FPR", "battery_FPR", None)):
        values = []
        for c in LABELS:
            metric = d["selected_class_battery_percent"][c][key]
            values.append((metric[index] if index is not None else metric)["mean"])
        lines.append(f"| {name} | {values[0]:.4f} | {values[1]:.4f} | {values[1]-values[0]:+.4f} |")
    lines += ["", "电池平均AP略升、FPR降低，但Recall下降；各seed并非一致改善，不构成电池安全或部署优势。", "",
              "### 逐seed电池指标与类别F1", "",
              "| seed | 条件 | 正常F1 % | 充电器F1 % | 电池F1 % | 电池AP % | 电池Recall % | 电池FPR % |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in s["per_condition"]:
        m = row["selected"]
        cells = " | ".join(f"{100*v:.4f}" for v in [*m["class_f1"], m["battery_AP"], m["battery_recall"], m["battery_FPR"]])
        lines.append(f"| {row['seed']} | {LABELS[row['condition']]} | {cells} |")
    lines += ["", "## 追加训练成本", "",
              f"同一记录环境：`{f['runtime']['host']}` / `{f['runtime']['gpu']}`，PyTorch `{f['runtime']['torch_build']}`。", "",
              "| 条件 | 每seed追加30轮均时 min | 可训练参数 | 管线总参数 | 峰值allocated MiB |", "|---|---:|---:|---:|---:|"]
    for c, label in LABELS.items():
        cost = d["cost"][c]
        param = cost["parameters"]
        lines.append(f"| {label} | {cost['additional30_epoch_minutes']['mean']:.3f} | {param['trainable_parameters']:,} | {param['total_parameters']:,} | {max(cost['peak_allocated_memory_mib']):.1f} |")
    lines += ["", f"本次LoRA逐epoch累计时间约为冻结在线控制的{d['training_time_ratio_lora_over_frozen']:.2f}倍。",
              "逐epoch时间含validation，不含历史父头训练、预训练、前置合成测试/冻结及完整部署开销；套件墙钟另含准备、迁移、审计和写盘。冻结控制也在线执行encoder，不等于缓存头训练成本。峰值为PyTorch allocated而非设备总显存或端到端内存。",
              "LoRA可训练参数不到百万，但完整推理仍依赖约3.42亿参数骨干/头管线；公共基础权重经哈希核验，不包含在仅learned-state的checkpoint体积里。没有开展统一端到端推理速度基准。", "",
              "## 完整性、迁移与保全", "",
              "- 结果前167项回归通过；四seed真实强头batch64合成检查、A/B梯度、冻结参数不变、learned-state重载一致均通过。",
              "- 任何真实更新前，四seed迁移已通过：seed42相对旧父预测1条变化、F1漂移0.026751pp，其他seed0条；未放宽原≤1条/≤0.05pp容差。每seed两处理epoch0概率完全相同。",
              f"- 原协议SHA：`{f['protocol_sha256']}`；终态冻结文件与本地结果前副本字节一致，397个冻结输入及运行环境终态复核通过。",
              f"- 独立标准库审计{audit['epochs_audited']}轮、{audit['prediction_rows_audited']}行初始/最佳/端点预测：原CSV真值、有序ID、概率、混淆指标、AP、选轮与原门槛核对通过；最大指标误差{audit['maximum_independent_metric_error']:.3g}，AP误差{audit['maximum_independent_AP_error']:.3g}。",
              f"- 32份事前固定参照预测另保全并重算，共{reference_audit['fixed_reference_prediction_rows']}行；与冻结哈希和指标一致，另核对8组所选模型类别F1。",
              "- CPU读取16个best/last checkpoint；最佳状态与已提交记录一致，全部30epoch、每条件2310次optimizer更新（合计18480次）、固定scale1024、零跳步、optimizer/scaler/CPU/CUDA/shuffle RNG均核对。没有重新运行模型推理。",
              f"- 完整终态套件与执行日志保全{preservation['file_count']}文件/{preservation['total_bytes']} bytes，逐SHA零不匹配；固定参照另32文件/{reference_audit['fixed_reference_bytes']} bytes。",
              f"- 本地根：`{base}`，原科学输出在`terminal_artifacts/`；核对证据见`independent_output_audit.json`、`checkpoint_audit.json`、`independent_reference_audit.json`、`preservation_verified.json`。", "",
              "## 可用表述与停止边界", "",
              "> 在已知test之后的四seed验证集探索中，从已优化的MOMENT统计融合头出发，rank8 LoRA使validation所选模型Macro-F1由92.231%提高到92.678%，四seed配对差均为正，但平均+0.448pp未达到事先设定的+0.5pp实用增量门槛。该结果表明本设置下存在小幅适配收益，不足以主张强适配优势或总体优于TCN。", "",
              "- 父头已在同一validation选过200轮，本次再选epoch0–30；validation与旧test历史已被消费，不是新的独立盲测或外部泛化确认。",
              "- split相互重叠，四seed不是四个独立设备/人群；9827训练标签之外还有3510个开发/选模标签及历史方案开发成本，不宣称总标签效率。",
              "- 本轮没有新的rank-matched仅MOMENT条件，也没有随机骨干适配对照；不能把增量解释为统计分支×适配的交互或新的预训练因果优势。",
              "- LoRA额外dropout使两处理随机轨迹不同；匹配的是起始头、数据和追加更新机会，不是相同计算量。",
              "- E03F seed43 pilot门槛通过仍保留；E07复制幅度门槛未通过，E03P及E01C/E06原失败记录均不更改。",
              "- 建议停止继续扩大LoRA参数搜索，按已有统计互补主线成稿，将本轮作为小幅收益与成本取舍的补充；不自动启动下一阶段。", "",
              "协议：[E07结果前方案](e07_fusion_lora_replication_protocol_20261019.md)。"]
    if figure:
        lines += ["", f"![E07 four-seed validation results](../figures/{figure.with_suffix('.svg').name})"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    (base / "completion_analysis.json").write_text(json.dumps(d, indent=2) + "\n")
    if figure:
        plot(s, d, seeds, figure)
    return report


def plot(s, d, seeds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "E07-fixed-results"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    groups = [("Parent\nfusion", d["reference_macro_f1_percent"]["parent_fusion_nn"]),
              ("Continued\nfrozen", stats(s["aggregate"]["frozen_fusion"]["per_seed_percent"])),
              ("LoRA8\nfusion", stats(s["aggregate"]["lora8_fusion"]["per_seed_percent"])),
              ("Strong TCN\ncontext", d["reference_macro_f1_percent"]["strong_tcn"])]
    for i, (_, a) in enumerate(groups):
        color = ("#999999", "#0072B2", "#D55E00", "#009E73")[i]
        axes[0].errorbar(i, a["mean"], yerr=a["sample_sd"], fmt="o", capsize=5, color=color)
        axes[0].scatter([i] * 4, a["per_seed"], color=color, alpha=.5, s=20)
        axes[0].annotate(f"{a['mean']:.3f}", (i, max(a["mean"] + a["sample_sd"], *a["per_seed"]) + .15), ha="center", fontsize=8)
    axes[0].set_xticks(range(4), [label for label, _ in groups])
    axes[0].set_ylabel("Validation Macro-F1 (%)")
    axes[0].set_title("Selected models: mean and sample SD")
    axes[0].set_ylim(90.8, 98.6)
    positions = list(range(4))
    main = s["continued_frozen_rule"]
    endpoint = d["endpoint_gain_pp"]
    axes[1].plot(positions, main["per_seed_pp"], "o-", color="#D55E00", label=f"Selected: mean {main['mean_pp']:+.3f} pp")
    axes[1].plot(positions, endpoint["per_seed"], "s--", color="#0072B2", label=f"Epoch30: mean {endpoint['mean']:+.3f} pp")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].axhline(.5, color="grey", linestyle=":", label="Mean target +0.5 pp (not per-seed)")
    axes[1].set_xticks(positions, seeds)
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("LoRA minus continued frozen (pp)")
    axes[1].set_title("Predeclared replication gates: " + ("met" if s["replication_rule_passed"] else "not met"))
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle("E07: validation-only post-test exploration; overlapping splits", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".svg"), metadata={"Date": None})
    figure.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserved-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    print(render(args.preserved_root, args.report, args.figure))


if __name__ == "__main__":
    main()
