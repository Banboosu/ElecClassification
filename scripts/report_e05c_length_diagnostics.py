"""Independently check/render saved N1 length/padding diagnostics; no inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score(rows):
    matrix = [[0] * 3 for _ in range(3)]
    for row in rows:
        y, pred = int(row['y_true']), int(row['y_pred'])
        probabilities = [float(row[f'p{i}']) for i in range(3)]
        if (not 0 <= y < 3 or not 0 <= pred < 3
            or not all(0 <= p <= 1 for p in probabilities)
            or abs(sum(probabilities) - 1) > 2e-7
            or probabilities.index(max(probabilities)) != pred):
            raise ValueError('Invalid diagnostic prediction')
        matrix[y][pred] += 1
    f1 = statistics.mean(2 * matrix[i][i] / (sum(matrix[i]) + sum(matrix[j][i] for j in range(3)))
                         if sum(matrix[i]) + sum(matrix[j][i] for j in range(3)) else 0 for i in range(3))
    return matrix, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, required=True)
    args = parser.parse_args()
    r = args.diagnostics
    s = read(r / 'summary.json')
    if s['test_evaluated'] or s['seed'] != 43 or s['replication_mean_includes_this_seed']:
        raise ValueError('Unexpected diagnostic scope')
    if len(s['padding']) != 18 or len(s['tree_controls']) != 4:
        raise ValueError('Incomplete diagnostics')
    for name, expected in s['output_sha256'].items():
        if digest(r / name) != expected:
            raise ValueError(f'Diagnostic bytes changed: {name}')
    files = {}
    count, max_error = 0, 0.
    for item in s['padding'] + s['tree_controls']:
        with (r / (item['stem'] + '.csv')).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 3510 or len(set(row['sample_id'] for row in rows)) != 3510:
            raise ValueError('Missing or duplicate sample IDs')
        identity = [(row['sample_id'], row['y_true']) for row in rows]
        if files and identity != next(iter(files.values()))[0]:
            raise ValueError('Diagnostic ordered IDs or truth differ')
        matrix, f1 = score(rows)
        if matrix != item['metrics']['confusion_matrix']:
            raise ValueError('Independent confusion differs')
        max_error = max(max_error, abs(f1 - item['metrics']['macro_f1']))
        files[item['stem']] = (identity, [int(row['y_pred']) for row in rows])
        count += len(rows)
    for item in s['padding']:
        base = f"fraction{round(100*item['fraction']):03d}_{item['condition']}_padding1024"
        changed = sum(a != b for a, b in zip(files[item['stem']][1], files[base][1], strict=True))
        if changed != item['changed_predictions']:
            raise ValueError('Padding changed prediction count differs')
    if max_error > 1e-12:
        raise ValueError('Independent F1 mismatch')
    proof = {'status':'passed', 'diagnostic_prediction_rows':count, 'test_evaluated':False,
             'inference_performed':False, 'maximum_independent_f1_error':max_error,
             'source_summary_sha256':digest(r/'summary.json')}
    (r/'independent_saved_output_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    lines = ['# N1：seed43长度与padding诊断（四seed复核前的独立分析）','',
             '只使用已有TCN权重和train/validation；不重新训练TCN，不读取test预测。pilot43不进入四seed复核均值。','',
             '| 标签预算 | 浅树输入 | 最佳树深 | Validation Macro-F1 % |',
             '|---:|---|---:|---:|']
    for row in s['tree_controls']:
        lines.append(f"| {100*row['fraction']:g}% | {row['features']} | {row['best_params']['max_depth']} | {100*row['metrics']['macro_f1']:.4f} |")
    lines += ['', '两控制同4候选×5折，15统计含length；浅树是机制诊断，不等价于已优化统计SVM。length与类别可能相关，但仅length的这些分数不足以说明TCN高分只由长度造成，也不能作为因果或泄漏判断。','',
              '| 预算 | TCN | Padding | 预测改变数（相对1024） | F1 % | 最大logit差 |',
              '|---:|---|---:|---:|---:|---:|']
    for row in s['padding']:
        lines.append(f"| {100*row['fraction']:g}% | {row['condition']} | {row['width']} | {row['changed_predictions']} | {100*row['metrics']['macro_f1']:.4f} | {row['max_logit_difference']:.4g} |")
    lines += ['', '所有有效序列完全相同，只增加尾部无效零。masked模型在本次1024/1280/1536比较中logit与预测不变；unmasked模型对padding长度敏感。只证明依赖这种输入表示，不能证明长度是性能收益的唯一机制，也不能用新padding分数重选模型。','',
              '1024恢复已通过保存checkpoint/预测核对，允许概率绝对漂移1e-5且argmax零差异，容差在诊断前固定；实际漂移记录于机器summary。长度分层阈值来自选定train，逐类长度分布与全部分组指标一并保留。','',
              f"独立标准库核验{count}行CSV预测、混淆/指标及变化计数，F1最大误差{max_error:.2e}；所有原始输出文件SHA通过。四seed36次训练的完成状态与结果另行报告，不把本诊断写成复制成功。",'',
              '![Padding sensitivity](../figures/e05c_length_padding_20261019.svg)','',
              f'证据：`{r}/summary.json`、`independent_saved_output_audit.json`及CSV/NPZ/model/CV文件。']
    out = Path('docs/experiment_records/e05c_length_padding_diagnostics_20261019.md')
    out.write_text('\n'.join(lines)+'\n')
    print(json.dumps(proof,indent=2))
    print(out)


if __name__ == '__main__':main()
