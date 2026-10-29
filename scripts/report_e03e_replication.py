"""Posthoc E03E tables/curves from preserved validation outputs; no training/selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics as st
from pathlib import Path

SEEDS = (42, 44, 45, 46)
SUITE = 'e03e_gain_replication_20261018_v1'


def read(p):
    return json.loads(p.read_text())


def stats(values):
    mean = st.mean(values)
    sd = st.stdev(values)
    half = 3.182446305284263 * sd / math.sqrt(len(values))
    return dict(mean=mean, sample_sd=sd, descriptive_ci95=[mean-half, mean+half])


def average_precision(rows):
    # Non-interpolated AP, grouping exact score ties (sklearn convention).
    ordered = sorted(((float(r['probability_2']), int(r['true_index']) == 2) for r in rows), reverse=True)
    positives = sum(y for _, y in ordered)
    tp = seen = 0
    ap = 0.0
    i = 0
    while i < len(ordered):
        j = i
        group_tp = 0
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            group_tp += ordered[j][1]
            j += 1
        seen += j-i
        tp += group_tp
        ap += group_tp / positives * tp / seen
        i = j
    return ap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('artifacts/imports/autodl_preservation_20261018'))
    args = parser.parse_args()
    root = args.base/'artifacts/improvements/e03e_gain_replication'
    analysis = root/f'{SUITE}_analysis'
    summary = read(analysis/'summary.json')
    assert summary['audit_status'] == 'passed' and not summary['test_evaluated'] and not summary['pilot_seed_included']
    assert [r['seed'] for r in summary['per_seed']] == list(SEEDS)
    output = Path('artifacts/analysis/e03e_replication_20261018')
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    rows = []
    histories = {}
    parent_classes = {str(k): [] for k in range(3)}
    for seed in SEEDS:
        for branch in ('statistics', 'moment', 'fusion'):
            case = root/f'{SUITE}_seed{seed}_{branch}'
            m = read(case/'metrics.json')
            h = read(case/'history.json')
            histories[seed, branch] = h
            for name in ('metrics.json','history.json','validation_predictions.csv'):
                p = case/name
                inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            assert len(h) == 200 and all(r['grad_scaler_scale'] == 1024 and r['skipped_optimizer_steps'] == 0 for r in h)
            assert not m['test_evaluated'] and not m['encoder_trainable']
            with (case/'validation_predictions.csv').open(newline='') as f:
                predictions = list(csv.DictReader(f))
            assert len(predictions) == 3510
            cm = m['validation_metrics']['confusion_matrix']
            tp = cm[2][2]
            fp = sum(cm[k][2] for k in (0, 1))
            negatives = sum(sum(cm[k]) for k in (0, 1))
            row = dict(seed=seed,branch=branch,macro_f1_percent=m['validation_metrics']['macro_f1']*100,
                       accuracy_percent=m['validation_metrics']['accuracy']*100,best_epoch=m['best_epoch'],
                       endpoint_f1_percent=m['endpoint_metrics']['macro_f1']*100,
                       training_seconds=m['training_seconds'], invocation_seconds=m['current_invocation_training_seconds'],
                       peak_head_memory_mib=m['peak_allocated_memory_mib'],
                       battery_ap_percent=average_precision(predictions)*100,
                       battery_recall_percent=100*tp/sum(cm[2]), battery_fpr_percent=100*fp/negatives)
            row.update({f'class_{k}_f1_percent':m['validation_metrics']['classification_report'][str(k)]['f1-score']*100 for k in range(3)})
            rows.append(row)
        parent = args.base/'artifacts/improvements'
        if seed == 42:
            parent = parent/'e01/e01_20261017_v2_pilot_seed42'
        else:
            parent = parent/f'e01b_high_label/e01b_high_label_20261017_v1_seed{seed}'
        p = parent/'fusion_stats_svm_fraction_40/metrics.json'
        pm = read(p)['validation_metrics']
        for k in range(3):
            parent_classes[str(k)].append(pm['classification_report'][str(k)]['f1-score']*100)
    aggregates = {b:{k:stats([r[k] for r in rows if r['branch']==b]) for k in rows[0] if k not in ('seed','branch')} for b in ('statistics','moment','fusion')}
    endpoints = [r['fusion_endpoint_f1_percent']-r['strongest_reference_f1_percent'] for r in summary['per_seed']]
    derived = dict(source_summary=summary,head_aggregates=aggregates,per_run=rows,
                   endpoint_minus_strongest_descriptive=stats(endpoints),
                   parent_fusion_svm_class_f1_percent={k:stats(v) for k,v in parent_classes.items()},
                   total_head_training_seconds=sum(r['training_seconds'] for r in rows),
                   source_file_sha256=inputs,
                   note='All four-seed intervals descriptive only; overlapping splits, reused validation; no test or new candidate selection.')
    (output/'derived_results.json').write_text(json.dumps(derived,indent=2)+'\n')
    with (output/'per_run_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Standalone SVG: four panels, no plotting dependency or remote environment mutation.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="730" viewBox="0 0 1040 730">',
         '<rect width="1040" height="730" fill="white"/>',
         '<g font-family="sans-serif" font-size="12">',
         '<text x="520" y="24" text-anchor="middle">E03E: validation Macro-F1, 40% labels; pilot43 excluded</text>']
    colors={'statistics':'#009E73','moment':'#0072B2','fusion':'#D55E00'}
    for idx,seed in enumerate(SEEDS):
        x=65+(idx%2)*515;y=65+(idx//2)*320;w=420;hh=235
        def point(epoch,score):
            return x+(epoch-1)/199*w,y+hh-(score-40)/60*hh
        svg.append(f'<text x="{x+w/2}" y="{y-14}" text-anchor="middle">seed {seed}</text>')
        for value in (40,50,60,70,80,90,100):
            yy=point(1,value)[1]
            svg.append(f'<path d="M{x},{yy}h{w}" stroke="#ddd"/><text x="{x-8}" y="{yy+4}" text-anchor="end">{value}</text>')
        for epoch in (1,30,100,200):
            xx=point(epoch,40)[0]
            svg.append(f'<text x="{xx}" y="{y+hh+18}" text-anchor="middle">{epoch}</text>')
        ref=next(r['strongest_reference_f1_percent'] for r in summary['per_seed'] if r['seed']==seed)
        yy=point(1,ref)[1]
        svg.append(f'<path d="M{x},{yy}h{w}" stroke="#666" stroke-dasharray="5 3"/>')
        for branch,color in colors.items():
            coords=' '.join(f'{a:.2f},{b:.2f}' for a,b in (point(r['epoch'],r['validation_macro_f1']*100) for r in histories[seed,branch]))
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.1"/>')
        svg.append(f'<text x="{x+w/2}" y="{y+hh+37}" text-anchor="middle">Epoch (best within 200 is primary)</text>')
    for idx,(label,color) in enumerate(list(colors.items())+[('strongest reference','#666')]):
        xx=90+idx*235
        svg.append(f'<path d="M{xx},704h24" stroke="{color}" stroke-width="2"/><text x="{xx+30}" y="708">{html.escape(label)}</text>')
    svg.append('</g></svg>')
    figure=Path('docs/figures/e03e_gain_replication_20261018.svg');figure.write_text('\n'.join(svg)+'\n')
    f=aggregates['fusion'];primary=summary['primary_rule'];side=summary['moment_ablation_rule']
    lines=['# E03E 增益融合头四种子复制结果（validation-only）','',
           '日期：2026-10-18。按预先冻结协议完成，不修改门槛，不自动执行 test、LoRA 或扩大预算。','',
           '## 协议与审计','',
           '- seed42/44/45/46，40%训练标签；pilot43严格排除，未加入均值或成功判定。',
           '- 每seed 9,827个训练标签 + 3,510个validation标签；未适配MOMENT缓存，统计/仅MOMENT/融合三个增益头，各200轮，共12训练。',
           '- 208个冻结文件及依赖核验通过；协议SHA-256：`f34f3fbedfbd910be32752e694640b2edd8d96c9dcf03ea7b659b591e357963d`。',
           '- 套件于10:46:53冻结、11:10:45完成（Asia/Shanghai，约23分52秒）。接管后再次执行原模块verify/summarize，42,120行NN预测、参照、2,400个epoch、共享初值及模型哈希通过；固定scale1024、零跳步，无监督encoder训练。',
           '- 训练前preflight为106 tests、6 subtests通过；此结果报告未改训练实现或重新调参。','',
           '## 主结果','',
           '以下均为validation Macro-F1（%），每个seed采用预算内最好checkpoint。最强参照取本轮两单分支NN及父统计RF/SVM、匹配统计SVM、父融合SVM的最大值；本轮四seed均为父融合SVM。','',
           '| Seed | 统计NN | MOMENT NN | 融合NN | 最强参照 | 配对增量pp | 融合200轮端点 | 最好epoch | Accuracy % |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['per_seed']:
        item=next(v for v in rows if v['seed']==r['seed'] and v['branch']=='fusion')
        lines.append(f"| {r['seed']} | {r['statistics_f1_percent']:.2f} | {r['moment_f1_percent']:.2f} | {r['fusion_f1_percent']:.2f} | {r['strongest_reference_f1_percent']:.2f} | {r['fusion_minus_strongest_pp']:+.3f} | {r['fusion_endpoint_f1_percent']:.2f} | {item['best_epoch']} | {item['accuracy_percent']:.2f} |")
    lines+=['',f"融合Macro-F1均值 **{f['macro_f1_percent']['mean']:.2f}% ± {f['macro_f1_percent']['sample_sd']:.2f}%**（样本SD），Accuracy **{f['accuracy_percent']['mean']:.2f}% ± {f['accuracy_percent']['sample_sd']:.2f}%**。",
            f"相对最强参照平均 **+{primary['mean_difference_pp']:.3f}pp**，**4/4正向**；描述性95%配对t区间 **[{primary['ci95_lower_pp']:+.3f}, {primary['ci95_upper_pp']:+.3f}]pp**。均值≥0.5pp且≥3/4正向，主规则通过。",
            f"相对仅MOMENT平均 **+{side['mean_difference_pp']:.3f}pp**，4/4正向，描述性区间 **[{side['ci95_lower_pp']:+.3f}, {side['ci95_upper_pp']:+.3f}]pp**，0.25pp消融规则通过；因此整体复制规则通过。",
            f"200轮固定端点相对最强参照平均 **{st.mean(endpoints):+.3f}pp**、{sum(v>0 for v in endpoints)}/4正向；这是描述性结果，不能代替预算内最好模型的主规则。",'',
            '## 类别与历史类别2（充电器）边界','',
            '| 类别 | 融合F1均值 % | 父融合SVM F1均值 % | 配对均值差pp |','|---|---:|---:|---:|']
    for k,label in enumerate(('0：正常','1：电池异常','2：充电器故障')):
        nn=f[f'class_{k}_f1_percent']['mean'];parent=st.mean(parent_classes[str(k)])
        lines.append(f'| {label} | {nn:.2f} | {parent:.2f} | {nn-parent:+.2f} |')
    lines+=['',f"充电器故障（类别2，旧产物battery_*键）融合头的AP均值 **{f['battery_ap_percent']['mean']:.2f}%**，默认argmax Recall **{f['battery_recall_percent']['mean']:.2f}%**，one-vs-rest FPR **{f['battery_fpr_percent']['mean']:.2f}%**。AP由保存的validation概率重算，未选新阈值；不是固定高召回安全运行点，不等于部署保证。逐运行与各支路结果见CSV/JSON。",'',
            '## 成本、曲线与限制','',
            f"12个缓存头训练epoch计时合计 **{sum(r['training_seconds'] for r in rows)/60:.2f}分钟**；不包含骨干提取、SVM补控制、准备与审计，不可当作线上延迟。融合头135,555个参数，公共341M量级骨干依然是完整推理依赖，不可只报头大小。",
            '![E03E validation曲线](../figures/e03e_gain_replication_20261018.svg)','',
            '- 同数据集、重叠随机划分、反复复用validation；seed42用于过往探索，旧参照分数已知，不是独立外部确认。区间仅描述，不作显著性或总体泛化保证。',
            '- NN有200次epoch validation选择，旧SVM用CV选择，预算不等，不能把净差归因于单一架构优势。',
            '- seed44融合最好点在epoch200边界，seed46在191；至少两seed仍有末段最好点，不能称全部已充分收敛，也不据此自动增加轮数。',
            '- 这是40%预算下冻结表征+统计增益头的复制结果，不是完整标签test结果；不能与历史TCN/完全微调test直接排名。',
            '- E03P门控失败、E03D-v1数值事故及E01C整体实用增量规则失败保持原判；本轮未重测新头LoRA。','',
            '## 下一步待决定（不自动执行）','',
            '建议停止本轮探索，单独冻结最终比较清单与测试预算，再由用户明确批准一次正式test。候选包含本轮增益融合、同预算单分支、父融合SVM及匹配统计SVM；历史TCN/完全微调仅在协议可比时列为正式对照。若继续研究LoRA，应另立预算和新头公平参照，不回改本轮门槛。','',
            '完整保全与关机证据以 `artifacts/imports/autodl_preservation_20261018/preservation_verified.json`、`shutdown_verification.json` 和无人值守runbook为准。',
            '本次最终保全2,727文件、4,714,447,031 bytes，remote/local SHA全部一致（含旧E01完整分类器/缓存、E03D失败现场、E03E及实体公共基座）。11:19:06远程AutoDL关机命令返回0，11:19:21一次SSH复验为Connection refused；已验证SSH不可达，未另查平台电源/计费API。',
            '派生AP已用远程sklearn对12运行交叉核对，最大差1.85e-13个百分点；生成器合成AP检查与SVG XML解析通过。最终本地报告/脚本/台账副本及SHA另见保全目录的completion_bundle。',
            '机器可读派生：`artifacts/analysis/e03e_replication_20261018/derived_results.json`、`per_run_metrics.csv`；原始summary位于保全目录中，不覆盖原结果。']
    report=Path('docs/experiment_records/e03e_gain_replication_results_20261018.md');report.write_text('\n'.join(lines)+'\n')
    print(report);print(figure);print(json.dumps({b:{k:v['mean'] for k,v in values.items()} for b,values in aggregates.items()},indent=2))


if __name__ == '__main__':
    main()
