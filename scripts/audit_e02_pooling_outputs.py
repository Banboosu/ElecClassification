"""Independent confusion/CV/length audit; reads saved E02P outputs only."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np


def confusion_metrics(rows):
    matrix = [[0]*3 for _ in range(3)]
    for row in rows:
        y,pred=int(row['true_index']),int(row['predicted_index'])
        if not 0<=y<3 or not 0<=pred<3:raise ValueError('Invalid class')
        matrix[y][pred]+=1
    if not rows:return matrix,None,None
    f1=statistics.mean(2*matrix[i][i]/(sum(matrix[i])+sum(matrix[j][i] for j in range(3))) if sum(matrix[i])+sum(matrix[j][i] for j in range(3)) else 0 for i in range(3))
    return matrix,f1,sum(matrix[i][i] for i in range(3))/len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analysis',type=Path,required=True);args=parser.parse_args()
    root=args.analysis;f=json.loads((root/'frozen_protocol.json').read_text());p=f['protocol'];summary=json.loads((root/'summary.json').read_text())
    if summary['audit_status']!='passed' or summary['test_evaluated']:raise ValueError('Not a completed validation-only audit')
    manifest=json.loads((root/'prepared/manifest.json').read_text())
    with np.load(root/'prepared/train.npz',allow_pickle=False) as data:edges=np.quantile(data['lengths'],[1/3,2/3],method='linear')
    if edges.tolist()!=manifest['length_edges']:raise ValueError('Length edges not train-derived')
    with np.load(root/'prepared/validation.npz',allow_pickle=False) as data:ids=data['sample_ids'];truth=data['labels'];groups=np.searchsorted(edges,data['lengths'],side='left')
    row_count=fold_scores=0;max_error=0.
    for pool in p['poolings']:
        for branch in p['branches']:
            case=root/f'{pool}_{branch}';m=json.loads((case/'metrics.json').read_text())
            with (case/'validation_predictions.csv').open(newline='') as handle:rows=list(csv.DictReader(handle))
            if [r['sample_id'] for r in rows]!=ids.tolist() or [int(r['true_index']) for r in rows]!=truth.tolist():raise ValueError('ID/target mismatch')
            matrix,f1,accuracy=confusion_metrics(rows)
            if matrix!=m['validation_metrics']['confusion_matrix']:raise ValueError('Confusion mismatch')
            errors=[abs(f1-m['validation_metrics']['macro_f1']),abs(accuracy-m['validation_metrics']['accuracy'])]
            for index,bin_record in enumerate(m['length_strata']):
                subset=[r for r,g in zip(rows,groups,strict=True) if g==index]
                cm,bf1,ba=confusion_metrics(subset)
                if len(subset)!=bin_record['n']:raise ValueError('Length-bin count mismatch')
                if subset:
                    if cm!=bin_record['metrics']['confusion_matrix']:raise ValueError('Length-bin confusion mismatch')
                    errors.extend([abs(bf1-bin_record['metrics']['macro_f1']),abs(ba-bin_record['metrics']['accuracy'])])
                elif bin_record['metrics'] is not None:raise ValueError('Empty group not disclosed')
            with (case/'cv_results.csv').open(newline='') as handle:cv=list(csv.DictReader(handle))
            if len(cv)!=15:raise ValueError('Wrong candidate budget')
            for candidate in cv:
                scores=[float(candidate[f'split{i}_test_score']) for i in range(5)]
                if not all(np.isfinite(scores)) or not all(0<=v<=1 for v in scores):raise ValueError('Invalid fold scores')
                errors.append(abs(statistics.mean(scores)-float(candidate['mean_test_score'])));fold_scores+=5
            best=max(cv,key=lambda candidate:float(candidate['mean_test_score']))
            if ast.literal_eval(best['params'])!=m['best_params']:raise ValueError('Selected CV parameters changed')
            errors.append(abs(float(best['mean_test_score'])-m['cv_score']))
            max_error=max(max_error,*errors);row_count+=len(rows)
    if row_count!=21060 or fold_scores!=450 or max_error>1e-12:raise ValueError('Incomplete/inconsistent pooling results')
    report={'status':'passed','audited_prediction_rows':row_count,'cv_fold_scores':fold_scores,'maximum_independent_metric_error':max_error,
            'length_bins_source':'selected training only','test_evaluated':False,'inference_performed':False,
            'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()}
    (root/'independent_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
