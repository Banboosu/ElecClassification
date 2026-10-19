"""Render verified E05C completion evidence without inference or refitting."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preservation',type=Path,required=True)
    args=parser.parse_args()
    saved=args.preservation
    root=saved/'terminal_artifacts'
    s=read(root/'summary.json')
    proof=read(root/'preservation_verified.json')
    audit=read(saved/'independent_output_audit.json')
    observation=read(saved/'backup_instance_observation.json')
    if s['audit_status']!='passed' or s['conditions']!=36 or s['test_evaluated'] or proof['status']!='passed' or audit['status']!='passed':
        raise ValueError('Require complete preserved and independently audited validation suite')
    lines=['# E05C / N1：TCN四seed复核与长度诊断结果','',
           '2026-10-19；套件于15:42:45（Asia/Shanghai）完成。seed42/44/45/46，pilot43不进入复核均值。所有结果为validation，既有test已知后的探索，无新test评价。','',
           '## 核心结果','',
           '| 训练标签预算 | 当前残差 masked F1 % | Legacy masked F1 % | Legacy unmasked F1 % |',
           '|---:|---:|---:|---:|']
    for fraction in (.1,.4,1.):
        cells=[next(r for r in s['aggregate'] if r['fraction']==fraction and r['condition']==c)
               for c in ('current_fixed50','legacy_masked50','legacy_unmasked50')]
        values=[f"{r['macro_f1_mean_percent']:.4f} ± {r['macro_f1_sample_sd_percent']:.4f}" for r in cells]
        lines.append(f"| {100*fraction:g}% | {' | '.join(values)} |")
    lines+=['','表中为四seed均值±样本SD；原始raw输入、统一split/嵌套子集、AdamW、固定50轮和FP32。按min_delta0.0001选selected，真峰值与端点另存，不按本轮结果换模型。','',
            '| 预算 | 配对差 | 均值pp | 正向seed | 描述性95%区间pp |','|---:|---|---:|---:|---|']
    for r in s['paired_differences']:
        lo,hi=r['descriptive_ci95_pp']
        lines.append(f"| {100*r['fraction']:g}% | {r['comparison']} | {r['mean_pp']:+.4f} | {r['positive_seeds']}/4 | [{lo:+.4f}, {hi:+.4f}] |")
    lines+=['','所有区间为重叠随机split上的描述性配对t区间，不能当独立外部实验的总体置信保证。','',
            '## 判断与后续选择','',
            '- legacy masked相对当前残差结构在三个预算均4/4正向，平均+24.4453/+5.5631/+1.7592pp；说明原低标签TCN参照配置没有充分代表这份数据上的TCN潜力。不是原Keras完整重现，也不是所有TCN架构的性能上限。',
            '- unmasked−masked均值为+0.1403/−0.0142/−0.1805pp，正向分别2/4、2/4、1/4，描述性区间均跨0。没有支持全padding池化稳定更强，seed43的+8.78pp不能概括到四seed均值。',
            '- 后续建议以legacy masked为主要TCN参照：其多seed表现强，且已有固定权重诊断显示对额外padding不敏感；unmasked完整保留为池化对照，不删除或改写任何原结果。这个主参照建议是在本轮结果后提出的，不冒充原E03T预先声明方法。',
            '- 尚未补1%/5%新TCN曲线、学习率/dropout有界搜索或受限validation标签实验，不能宣称低标签优势问题已全部解决。下一步仅建议N2/N3，不自动启动。',
            '- 40% legacy masked validation97.7790%是强参照。历史E03E同四seed融合validation约92.04%，仅作已知管线上下文；需要逐行/选择预算核对才能形成新的成对比较，不与91.516%的历史test跨split排名。','',
            '## 逐seed与电池指标','',
            '| 预算 | seed | 条件 | Accuracy % | Macro-F1 % | 电池AP % | 电池Recall % | 电池FPR % | selected epoch |',
            '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in s['per_condition']:
        m=r['selected']
        lines.append(f"| {100*r['fraction']:g}% | {r['seed']} | {r['condition']} | {100*m['accuracy']:.4f} | {100*m['macro_f1']:.4f} | {100*m['battery_AP']:.4f} | {100*m['battery_recall']:.4f} | {100*m['battery_FPR']:.4f} | {r['selected_epoch']} |")
    lines+=['','电池为原标签2，默认argmax；AP由完整保存概率计算，没有调阈值。不把Recall单独提高写成安全保证。各类F1、peak/endpoint完整保存于summary。','',
            '## Early7轨迹回放','',
            '| 预算 | 四seed回放停止epoch | 固定50轮selected−回放selected，均值pp |','|---:|---|---:|']
    for fraction in (.1,.4,1.):
        rows=[r for r in s['per_condition'] if r['fraction']==fraction and r['condition']=='current_fixed50']
        stops=[r['early7_replay']['stopped_epoch'] for r in rows]
        delta=statistics.mean(100*(r['selected']['macro_f1']-r['early7_replay']['macro_f1']) for r in rows)
        lines.append(f'| {100*fraction:g}% | {stops} | {delta:+.4f} |')
    lines+=['','只沿原轨迹回放patience7，不新增训练。延长当前模型的平均收益远小于结构差，不把早停视作唯一问题。','',
            '## 长度与padding诊断','',
            'seed43已有权重的18组padding及4树控制见[独立诊断报告](e05c_length_padding_diagnostics_20261019.md)。仅length浅树10%/100% F1为43.2441/36.3936%，15统计浅树为75.5304/76.4047%；两树具有相同候选/折数，但不是最强SVM或去长度消融。',
            'masked在1024/1280/1536保持输出不变，unmasked增至1536分别改变100/33条预测。说明padding依赖，不证明长度是唯一机制或泄漏；本轮没有length旁路训练。','',
            '## 执行加速、硬件与计时边界','',
            '原实现完成seed42后，独立resident/async执行层从已提交checkpoint恢复；不改模型/数据/batch/精度/优化器/选模。合成和9827行train回放的权重、AdamW、所有指标/概率及RNG逐位一致；真实单轮回放速度比1.292/1.335/1.307倍。不是整套运行或部署延迟保证。',
            '基准期间SIGSTOP操作等待计入少数原epoch_seconds；不据这些长epoch估计纯计算速度，也不改写其原日志。整体套件06:15:33至07:42:44 UTC约87.2分钟，包含诊断、暂停、重启、已完成条件哈希审计、保存及训练；不当作纯训练计时。',
            f"**备份时实例已变化**：冻结运行记录为`{observation['frozen_training_runtime']['host']}` / {observation['frozen_training_runtime']['gpu']}；本轮保全观察为`{observation['observed_backup_host']}` / {observation['observed_backup_gpu']}。切换发生时刻未经独立核实，因此不认证所有训练计时来自同一物理GPU。", 
            '首次备份因旧hostname保护拒绝，随后显式指定新hostname。原协议/执行冻结文件与此前本地副本逐字节一致，模型及所有保存输出通过SHA；本轮不加载模型推理，不因备份实例不同重新计算旧结果。','',
            '## 独立审计与保全','',
            f"- 标准库独立复核{audit['epochs_audited']}epoch及{audit['prediction_rows_audited']}行预测，原CSV真值、有序ID、3类/电池混淆指标、AP、selected/peak、逐seed均值/SD/差值区间与SHA均通过；最大指标误差{audit['maximum_independent_metric_error']:.2e}、AP误差{audit['maximum_independent_AP_error']:.2e}。",
            f"- 终态{proof['files_verified']}文件/{proof['bytes_verified']}bytes，逐SHA核验零失败，含原冻结源码、36条件模型/optimizer/RNG/history/预测、诊断及加速修订快照。",
            f"- 本地根：`{root}`；`preservation_verified.json`、`{saved}/independent_output_audit.json`及`backup_instance_observation.json`为证据。",
            '- 本轮没有新test、阈值选择、自动N2/LoRA、pi-wake或关机；检查时GPU计算进程为空。','',
            '![Four-seed validation comparison](../figures/e05c_tcn_replication_20261019.svg)']
    out=Path('docs/experiment_records/e05c_tcn_replication_results_20261019.md')
    out.write_text('\n'.join(lines)+'\n')
    print(out)


if __name__=='__main__':main()
