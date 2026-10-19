"""Replay one discarded train epoch from saved checkpoints; never evaluate test."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from tcn_moment import replicate_tcn as replication
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.tcn_resident import make_resident_loader
from tcn_moment.train_tcn import make_loader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('resident_benchmark','scripts/benchmark_tcn_resident.py')
    bench=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    p=replication.protocol(replication.DEFAULT)
    cp=replication.seed_config(p,42)
    data=pilot.load_train_validation(cp)
    indices=replication.subset(data,.4,42)
    d=data['train']
    x,mask,y=d['x'][indices],d['mask'][indices],d['y'][indices]
    rows=[]
    for condition in replication.CONDITIONS:
        directory=pilot.root(cp)/pilot.case_name(.4,condition)
        result=pilot.read(directory/'result.json')
        checkpoint=directory/'checkpoint_latest.pt'
        before=sha256_file(checkpoint)
        if result['epochs']!=50 or before!=result['output_sha256']['checkpoint_latest.pt']:
            raise ValueError('Require a completed, unchanged reference checkpoint')
        a=bench.run(condition,make_loader,x,mask,y,1,checkpoint=checkpoint)
        b=bench.run(condition,make_resident_loader,x,mask,y,1,asynchronous=True,checkpoint=checkpoint)
        checks=bench.compare(a,b)
        if sha256_file(checkpoint)!=before:raise ValueError('Replay changed the source checkpoint')
        row={'condition':condition,'checks':checks,'checkpoint_sha256':before,
             'baseline_seconds':a['seconds'][0],'accelerated_seconds':b['seconds'][0],
             'speedup':a['seconds'][0]/b['seconds'][0]}
        rows.append(row)
        atomic_write_json(args.output,{'status':'passed' if all(all(r['checks'].values()) for r in rows) else 'failed',
            'conditions':rows,'train_rows':len(indices),'validation_or_test_rows_evaluated':0,
            'evaluation_note':'Only the first259 selected training rows are used for diagnostic epoch metrics.',
            'weights_discarded':True,'source_checkpoints_unchanged':True})
        print(row,flush=True)
        if not all(checks.values()):raise AssertionError('Real train replay differs; refuse switch')


if __name__=='__main__':main()
