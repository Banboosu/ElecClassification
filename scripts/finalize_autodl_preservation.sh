#!/usr/bin/env bash
# Local final copy + remote/local SHA audit. Does not start experiments or shut down.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
base=artifacts/imports/autodl_preservation_20261018
mkdir -p "$base/artifacts" "$base/remote_project"
exec 9>"$base/final_backup.lock"
flock -n 9 || exit 1
stage=final_artifacts
status() {
 python3 - "$base/final_backup_status.json" "$1" "$stage" <<'PY'
import json,sys
from pathlib import Path
from datetime import datetime,timezone
p=Path(sys.argv[1]); tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps(dict(status=sys.argv[2],stage=sys.argv[3],updated_at=datetime.now(timezone.utc).isoformat()),indent=2));tmp.replace(p)
PY
}
trap 'status failed || true' ERR
status running
rsync -aLz --partial --timeout=90 -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
 --exclude='*.lock' --exclude='*.tmp' --exclude='/imports/***' \
 autodl:/root/autodl-tmp/ElecClassification/artifacts/ "$base/artifacts/"
stage=remote_project_snapshot
status running
rsync -aLz --partial --timeout=90 -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
 --exclude='/artifacts/***' --exclude='/.venv/***' --exclude='/.git/***' --exclude='/.pi/***' \
 --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.ruff_cache/' \
 autodl:/root/autodl-tmp/ElecClassification/ "$base/remote_project/"
stage=remote_manifest_and_local_verification
status running
python3 - "$base" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
base=Path(sys.argv[1])
remote=r'''
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
root=Path('/root/autodl-tmp/ElecClassification')
files={}
for p in sorted(root.rglob('*')):
 rel=p.relative_to(root);parts=rel.parts
 if parts[0] in ('.venv','.git','.pi') or any(x in parts for x in ('__pycache__','.pytest_cache','.ruff_cache')):continue
 if parts[0]=='artifacts':
  if len(parts)>1 and parts[1]=='imports':continue
  if p.name.endswith(('.lock','.tmp')):continue
  dest=rel.as_posix()
 else:dest='remote_project/'+rel.as_posix()
 if not p.is_file():continue
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 files[dest]={'sha256':h,'bytes':p.stat().st_size}
p=Path('/root/autodl-tmp/.huggingface/hub/models--AutonLab--MOMENT-1-large/snapshots/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc/model.safetensors')
with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
assert p.stat().st_size==1385501256 and h=='a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572'
files['public_model/model.safetensors']={'sha256':h,'bytes':p.stat().st_size}
print(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'remote_root':str(root),'exclusions':['artifacts/imports','*.lock and *.tmp in artifacts','.venv','.git','.pi','__pycache__','.pytest_cache','.ruff_cache'],'files':files},indent=2))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','autodl','/root/autodl-tmp/ElecClassification/.venv/bin/python -'],input=remote,text=True,capture_output=True,check=True,timeout=300)
manifest=json.loads(r.stdout)
(base/'remote_sha256_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
local={};failures=[]
for name,expected in manifest['files'].items():
 p=base/name
 if not p.is_file() or p.is_symlink():failures.append({'path':name,'error':'missing file or symlink'});continue
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 actual={'sha256':h,'bytes':p.stat().st_size};local[name]=actual
 if actual!=expected:failures.append({'path':name,'expected':expected,'actual':actual})
(base/'local_sha256_manifest.json').write_text(json.dumps(local,indent=2)+'\n')
report={'status':'passed' if not failures else 'failed','verified_at':datetime.now(timezone.utc).isoformat(),'files_verified':len(local),'total_bytes':sum(x['bytes'] for x in local.values()),'failures':failures,'public_model_dereferenced':not (base/'public_model/model.safetensors').is_symlink(),'remote_manifest_sha256':hashlib.sha256((base/'remote_sha256_manifest.json').read_bytes()).hexdigest(),'local_manifest_sha256':hashlib.sha256((base/'local_sha256_manifest.json').read_bytes()).hexdigest(),'scope':'All remote project artifacts including E03E and failed E03D v1; remote source/config/docs/tests/data snapshot; public MOMENT checkpoint. Excludes only stated transient/cache directories and artifacts/imports.','shutdown_performed':False}
(base/'preservation_verified.json').write_text(json.dumps(report,indent=2)+'\n')
if failures:raise RuntimeError(f'{len(failures)} backup mismatches')
print(json.dumps(report,indent=2))
PY
stage=final_preservation_verified
status completed
