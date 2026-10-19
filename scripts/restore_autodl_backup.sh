#!/usr/bin/env bash
# Local migration helper. No deletion, experiments, test evaluation, or shutdown.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
base="${PRESERVATION_ROOT:-artifacts/imports/autodl_preservation_20261018}"
record="${LOCAL_DEPLOYMENT_RECORD:-artifacts/deployments/autodl_20261019}"
expected_host="${RESTORE_EXPECTED_HOST:?Set the exact replacement hostname after inspection}"
remote=/root/autodl-tmp/ElecClassification
revision=ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc
cache="/root/autodl-tmp/.huggingface/hub/models--AutonLab--MOMENT-1-large"
mkdir -p "$record"
stage=check_replacement_host
status() {
 python3 - "$record/transfer_status.json" "$1" "$stage" <<'PY'
import json,sys
from pathlib import Path
from datetime import datetime,timezone
p=Path(sys.argv[1]);tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps({'status':sys.argv[2],'stage':sys.argv[3],'updated_at':datetime.now(timezone.utc).isoformat(),'training_started':False},indent=2)+'\n');tmp.replace(p)
PY
}
trap 'rc=$?; status failed || true; exit "$rc"' ERR
status running
actual="$(ssh -o BatchMode=yes -o ConnectTimeout=15 autodl hostname)"
[[ "$actual" == "$expected_host" ]] || { echo "SSH alias points at an unexpected server"; exit 1; }
[[ -f "$base/remote_sha256_manifest.json" && -f "$base/public_model/model.safetensors" ]]
stage=restore_artifacts
status running
rsync -az --partial --partial-dir=.rsync-partial --timeout=120 -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
 "$base/artifacts/" "autodl:$remote/artifacts/"
stage=restore_public_checkpoint
status running
ssh -o BatchMode=yes autodl "mkdir -p '$cache/snapshots/$revision' '$cache/refs'"
rsync -aL --partial --partial-dir=.rsync-partial --timeout=120 -e 'ssh -o BatchMode=yes -o ConnectTimeout=15' \
 "$base/public_model/model.safetensors" "autodl:$cache/snapshots/$revision/model.safetensors"
ssh -o BatchMode=yes autodl "printf '%s' '$revision' > '$cache/refs/main'"
stage=remote_full_manifest_verification
status running
rsync -az "$base/remote_sha256_manifest.json" "autodl:$remote/artifacts/deployments/autodl_20261019/original_archive_manifest.json"
rsync -az scripts/verify_autodl_restoration.py "autodl:$remote/scripts/"
ssh -o BatchMode=yes autodl "/root/miniconda3/bin/python '$remote/scripts/verify_autodl_restoration.py' --manifest '$remote/artifacts/deployments/autodl_20261019/original_archive_manifest.json' --root '$remote' --public-model '$cache/snapshots/$revision/model.safetensors' --output '$remote/artifacts/deployments/autodl_20261019/archive_restoration_verified.json'"
rsync -az "autodl:$remote/artifacts/deployments/autodl_20261019/archive_restoration_verified.json" "$record/"
stage=archived_restore_verified
status completed
