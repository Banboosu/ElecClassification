"""Run E05C with an independently verified execution-only acceleration revision."""
from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import time
from pathlib import Path

from tcn_moment import replicate_tcn as original
from tcn_moment import tcn_fast_backend
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.io_utils import atomic_write_json, sha256_file

EVIDENCE = Path('artifacts/analysis/tcn_resident_speed_20261019')


def revision_root(p):return original.root(p)/'execution_revisions/resident_async_v1'


def freeze(path):
    parent=original.verify(path)
    p=parent['protocol'];r=revision_root(p)
    r.mkdir(parents=True,exist_ok=True)
    if (r/'frozen_execution.json').exists():return verify(path)
    files=[Path('src/tcn_moment')/name for name in
           ('tcn_resident.py','tcn_async.py','tcn_fast_backend.py','replicate_tcn_fast.py')]
    files += [Path('tests/test_tcn_fast.py'),Path('scripts/benchmark_tcn_resident.py'),
              Path('scripts/check_tcn_fast_replay.py'),
              Path('docs/experiment_records/tcn_speed_revision_20261019.md')]
    for name in ('synthetic_equivalence_v2.json','real_train_replay.json'):
        file=EVIDENCE/name;report=pilot.read(file)
        if report['status']!='passed' or len(report['conditions'])!=3 or any(not all(c['checks'].values()) for c in report['conditions']):
            raise ValueError('Full numerical equivalence evidence required')
        files.append(file)
    f={'recorded_at':pilot.now(),'parent_protocol_sha256':parent['protocol_sha256'],
       'backend':'resident_async_v1','changes_training_protocol':False,'test_evaluated':False,
       'file_sha256':{str(file):sha256_file(file) for file in files},
       'packages':{key:importlib.metadata.version(key) for key in original.PACKAGES}}
    for file in files:
        dest=r/'snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    atomic_write_json(r/'frozen_execution.json',f)
    return f


def verify(path):
    parent=original.verify(path)
    f=pilot.read(revision_root(parent['protocol'])/'frozen_execution.json')
    if f['parent_protocol_sha256']!=parent['protocol_sha256'] or f['changes_training_protocol']:
        raise ValueError('Execution revision protocol mismatch')
    for file,want in f['file_sha256'].items():
        if sha256_file(Path(file))!=want:raise ValueError(f'Acceleration source/evidence changed: {file}')
    if f['packages']!={key:importlib.metadata.version(key) for key in original.PACKAGES}:
        raise ValueError('Dependencies changed')
    return f


def run(path):
    verify(path)
    p=original.protocol(path)
    capture=revision_root(p)/f'before_resume_{time.time_ns()}'
    capture.mkdir()
    r=original.root(p)
    for file in r.glob('seed*/fraction*/*'):
        if file.is_file() and not (file.parent/'result.json').exists():
            dest=capture/file.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    if (r/'suite_status.json').exists():shutil.copy2(r/'suite_status.json',capture/'suite_status.json')
    # Existing frozen code is imported unchanged. The wrapper supplies only the
    # independently hashed execution implementation in this new process.
    original.pilot.run_case=tcn_fast_backend.run_case
    original.run(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','verify','run'))
    parser.add_argument('--protocol',type=Path,default=original.DEFAULT)
    args=parser.parse_args();globals()[args.action](args.protocol)


if __name__=='__main__':main()
