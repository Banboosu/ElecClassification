#!/usr/bin/env bash
# Local-only, read-only remote backup. Never shuts down; final verification is separate.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
base="artifacts/imports/autodl_preservation_20261018"
mkdir -p "$base/artifacts" "$base/public_model" artifacts/logs
stage="static_artifacts"
write_status() {
 python3 - "$base/static_backup_status.json" "$1" "$stage" <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,sys
p=Path(sys.argv[1]);tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps({'status':sys.argv[2],'stage':sys.argv[3],'updated_at':datetime.now(timezone.utc).isoformat(),'final_preservation_verified':False},indent=2));tmp.replace(p)
PY
}
trap 'rc=$?; write_status failed || true; exit "$rc"' ERR
write_status running
rsync -az --partial --timeout=90 -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
 --exclude='*.lock' --exclude='*.tmp' --exclude='/imports/***' --exclude='/improvements/e03e_gain_replication/***' \
 autodl:/root/autodl-tmp/ElecClassification/artifacts/ "$base/artifacts/"
stage="public_model_dereferenced"
write_status running
rsync -aL --partial --timeout=90 -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
 autodl:/root/autodl-tmp/.huggingface/hub/models--AutonLab--MOMENT-1-large/snapshots/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc/model.safetensors \
 "$base/public_model/model.safetensors"
python3 - "$base/public_model/model.safetensors" <<'PY'
from pathlib import Path
import hashlib,sys
p=Path(sys.argv[1])
if p.is_symlink() or p.stat().st_size!=1385501256:raise RuntimeError('Missing or incorrect public checkpoint bytes')
with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
if actual!='a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572':raise RuntimeError('Public weight digest mismatch')
print('Public model byte count and SHA256 verified')
PY
stage="static_copy_complete_await_final_delta_and_manifest"
write_status copied
