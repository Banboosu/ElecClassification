"""Independent stdlib-only E05C saved-output audit; no inference or training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

SEEDS = (42, 44, 45, 46)
CONDITIONS = ('current_fixed50', 'legacy_masked50', 'legacy_unmasked50')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metrics(cm):
    n = sum(map(sum, cm))
    support = [sum(row) for row in cm]
    predicted = [sum(row[i] for row in cm) for i in range(3)]
    precision = [cm[i][i]/predicted[i] if predicted[i] else 0. for i in range(3)]
    recall = [cm[i][i]/support[i] if support[i] else 0. for i in range(3)]
    f1 = [2*cm[i][i]/(support[i]+predicted[i]) if support[i]+predicted[i] else 0. for i in range(3)]
    return {'accuracy':sum(cm[i][i] for i in range(3))/n,
            'macro_precision':statistics.mean(precision), 'macro_recall':statistics.mean(recall),
            'balanced_accuracy':statistics.mean(recall), 'macro_f1':statistics.mean(f1),
            'weighted_precision':sum(p*s/n for p,s in zip(precision,support)),
            'weighted_recall':sum(p*s/n for p,s in zip(recall,support)),
            'weighted_f1':sum(p*s/n for p,s in zip(f1,support)), 'class_f1':f1}


def average_precision(rows):
    ordered = sorted(((float(r['p2']), int(r['y_true']) == 2) for r in rows), reverse=True)
    positives = sum(label for _,label in ordered)
    total = correct = 0
    ap = 0.
    i = 0
    while i < len(ordered):
        j = i
        group_correct = 0
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            group_correct += ordered[j][1]
            j += 1
        total += j-i
        correct += group_correct
        ap += group_correct / positives * correct / total
        i = j
    return ap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis
    summary = read(root/'summary.json')
    frozen = read(root/'frozen_protocol.json')
    if (summary['audit_status'] != 'passed' or summary['test_evaluated']
        or summary['seed43_in_replication_mean'] or summary['conditions'] != 36):
        raise ValueError('Require completed36-condition validation-only suite')
    # Require the original pre-outcome local frozen files, not only remote claims.
    local_start = root.parent/'startup/frozen_protocol.json'
    local_execution = Path('artifacts/analysis/tcn_resident_speed_20261019/execution_revision/frozen_execution.json')
    if digest(local_start) != digest(root/'frozen_protocol.json'):
        raise ValueError('Original pre-outcome protocol bytes changed')
    if digest(local_execution) != digest(root/'execution_revisions/resident_async_v1/frozen_execution.json'):
        raise ValueError('Original execution revision changed')
    for name,want in read(root/'output_manifest.json').items():
        if digest(root/name) != want:
            raise ValueError(f'Output manifest differs: {name}')
    csv_path = Path(frozen['protocol']['parent_protocol'])
    parent = read(root/'source_snapshot'/csv_path)
    raw = Path(parent['dataset'])
    if digest(raw) != parent['dataset_sha256']:
        raise ValueError('Raw dataset changed')
    with raw.open(newline='', encoding='utf-8') as stream:
        truth = {r['charging_station_id']:r['InsertedColumn'] for r in csv.DictReader(stream, delimiter=' ')}
    max_error = max_ap_error = 0.
    rows_audited = epochs = 0
    saved_results = {}
    def compare(cm, expected):
        nonlocal max_error
        calculated = metrics(cm)
        for key,value in calculated.items():
            if key not in expected: continue
            errors = [abs(a-b) for a,b in zip(value,expected[key])] if isinstance(value,list) else [abs(value-expected[key])]
            max_error = max(max_error,*errors)
        return calculated
    for seed in SEEDS:
        stem = f'e01_20261017_v2_{"pilot" if seed==42 else "expansion"}_seed{seed}'
        split_path = Path('artifacts/imports/e01_validation_20261017/runs')/stem/'split_manifest.json'
        remote_split = frozen['protocol']['split_manifests'][str(seed)]
        if digest(split_path) != frozen['file_sha256'][remote_split]:
            raise ValueError('Original split byte mismatch')
        split = read(split_path)
        ids = split['splits']['validation']
        for fraction in (.1,.4,1.):
            pair_hashes = []
            for condition in CONDITIONS:
                d = root/f'seed{seed}'/f'fraction{round(100*fraction):03d}_{condition}'
                result, history = read(d/'result.json'), read(d/'history.json')
                if result['test_evaluated'] or result['protocol_sha256'] != frozen['protocol_sha256']:
                    raise ValueError('Case protocol mismatch')
                if result['condition'] != condition or result['fraction'] != fraction:
                    raise ValueError('Case identity mismatch')
                for name,want in result['output_sha256'].items():
                    if digest(d/name) != want: raise ValueError('Case bytes mismatch')
                if [h['epoch'] for h in history] != list(range(1,51)):
                    raise ValueError('Incomplete trajectory')
                count = {0.1:2456,0.4:9827,1.:24569}[fraction]
                if result['train_count'] != count or result['optimizer_steps'] != 50*math.ceil(count/32):
                    raise ValueError('Training budget mismatch')
                chosen, selected_epoch = -1., 0
                for h in history:
                    if h['optimizer_steps'] != math.ceil(count/32):raise ValueError('Update count mismatch')
                    if not all(math.isfinite(h[k]) for k in ('train_loss','val_loss','macro_f1')):
                        raise ValueError('Nonfinite history')
                    compare(h['confusion_matrix'],h)
                    if h['macro_f1'] > chosen + .0001:chosen,selected_epoch = h['macro_f1'],h['epoch']
                peak = max(history,key=lambda h:h['macro_f1'])
                if (chosen != result['selected_macro_f1'] or selected_epoch != result['selected_epoch']
                    or peak['macro_f1'] != result['peak_macro_f1'] or peak['epoch'] != result['peak_epoch']):
                    raise ValueError('Checkpoint selection mismatch')
                for suffix,expected in (('selected',chosen),('peak',peak['macro_f1']),('endpoint',history[-1]['macro_f1'])):
                    with (d/f'validation_{suffix}.csv').open(newline='') as stream:
                        rows = list(csv.DictReader(stream))
                    if [r['sample_id'] for r in rows] != ids or any(truth[r['sample_id']] != r['y_true'] for r in rows):
                        raise ValueError('Ordered validation IDs or raw truth mismatch')
                    cm = [[0]*3 for _ in range(3)]
                    for row in rows:
                        y,pred = int(row['y_true']),int(row['y_pred'])
                        p = [float(row[f'p{i}']) for i in range(3)]
                        if not all(math.isfinite(x) and 0<=x<=1 for x in p) or abs(sum(p)-1)>2e-7 or p.index(max(p))!=pred:
                            raise ValueError('Invalid probabilities')
                        cm[y][pred] += 1
                    m = metrics(cm)
                    max_error = max(max_error,abs(m['macro_f1']-expected))
                    sr = next(r for r in summary['per_condition'] if r['seed']==seed and r['fraction']==fraction and r['condition']==condition)
                    if sr[suffix]['confusion_matrix'] != cm:raise ValueError('Summary confusion mismatch')
                    compare(cm,sr[suffix])
                    ap = average_precision(rows)
                    max_ap_error = max(max_ap_error,abs(ap-sr[suffix]['battery_AP']))
                    max_error = max(max_error,abs(cm[2][2]/sum(cm[2])-sr[suffix]['battery_recall']),
                                    abs((cm[0][2]+cm[1][2])/(sum(cm[0])+sum(cm[1]))-sr[suffix]['battery_FPR']))
                    rows_audited += len(rows)
                saved_results[seed,fraction,condition] = chosen
                pair_hashes.append(result['initial_model_sha256'])
                epochs += len(history)
            if pair_hashes[1] != pair_hashes[2]:raise ValueError('Legacy initial weights differ')
    for row in summary['aggregate']:
        scores = [100*saved_results[s,row['fraction'],row['condition']] for s in SEEDS]
        max_error = max(max_error,abs(statistics.mean(scores)-row['macro_f1_mean_percent']),
                        abs(statistics.stdev(scores)-row['macro_f1_sample_sd_percent']))
    for row in summary['paired_differences']:
        a,b = row['comparison'].split('-')
        differences = [100*(saved_results[s,row['fraction'],a]-saved_results[s,row['fraction'],b]) for s in SEEDS]
        mean,sd = statistics.mean(differences),statistics.stdev(differences)
        ci = [mean-3.182446305284263*sd/2,mean+3.182446305284263*sd/2]
        max_error = max(max_error,abs(mean-row['mean_pp']),*(abs(x-y) for x,y in zip(ci,row['descriptive_ci95_pp'])))
        if differences!=row['per_seed_pp'] or sum(d>0 for d in differences)!=row['positive_seeds']:
            raise ValueError('Paired seed difference mismatch')
    if epochs!=1800 or rows_audited!=379080 or max_error>1e-12 or max_ap_error>1e-12:
        raise ValueError('Incomplete/inconsistent saved outputs')
    proof={'status':'passed','epochs_audited':epochs,'prediction_rows_audited':rows_audited,
           'maximum_independent_metric_error':max_error,'maximum_independent_AP_error':max_ap_error,
           'test_evaluated':False,'inference_performed':False,'summary_sha256':digest(root/'summary.json'),
           'original_protocol_and_execution_freeze_bytes_verified':True}
    (root.parent/'independent_output_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
