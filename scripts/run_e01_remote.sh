#!/usr/bin/env bash
# E01: pilot on seed 42, then unconditional baseline audit + gated fusion on 43-46.
# No stage in this script computes test metrics. Never pass --evaluate-test here.
set -Eeuo pipefail
export PATH="/root/.local/bin:/root/miniconda3/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/.uv-python}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Public official weights are pre-cached and checksummed; no per-seed network requests.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."
mkdir -p artifacts/logs artifacts/improvements
suite="${E01_SUITE_NAME:-e01_20261017_v2}"
cache="artifacts/improvements/${suite}_features"
log="artifacts/logs/${suite}.log"
exec 9>"artifacts/improvements/${suite}.lock"
flock -n 9 || { echo "Suite $suite is already running"; exit 1; }
exec > >(tee -a "$log") 2>&1
status="artifacts/improvements/${suite}_suite_status.json"
write_status() {
  uv run --frozen python -c '
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix(".tmp")
t.write_text(json.dumps({"status":sys.argv[2],"stage":sys.argv[3],
"updated_at":datetime.now(timezone.utc).isoformat(),"test_evaluated":False},indent=2))
t.replace(p)
' "$status" "$1" "$stage"
}
stage="preflight"
trap 'rc=$?; trap - ERR; write_status failed || true; echo "E01 failed: stage=$stage exit=$rc"; exit "$rc"' ERR
trap 'write_status interrupted || true; exit 130' INT TERM
write_status running
uv run --frozen moment-check-environment --require-cuda
uv run --frozen --with pytest python -m pytest tests -q

stage="pilot_seed42"
write_status running
pilot="artifacts/improvements/e01/${suite}_pilot_seed42"
if [[ -e "$pilot/status.json" ]]; then
  uv run --frozen python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="completed", "Incomplete pilot: preserve it and use a new E01_SUITE_NAME"' "$pilot/status.json"
else
  uv run --frozen python -m tcn_moment.train_feature_fusion \
    --run-name "${suite}_pilot" --seeds 42 --n-jobs 2 --cache-dir "$cache"
fi

stage="freeze_validation_gate"
write_status running
plan="artifacts/improvements/${suite}_expansion_plan.json"
uv run --frozen python -c '
import json,sys
from pathlib import Path
source=Path(sys.argv[1]); plan=json.loads(source.read_text())
plan.update({"pilot_run":str(source.parent),"seeds":[43,44,45,46],
"test_evaluated":False,"note":"Baselines expand regardless of winner; fusion only by pilot validation gate."})
p=Path(sys.argv[2])
if p.exists():
    assert json.loads(p.read_text())==plan, "Existing frozen plan differs"
else:
    p.write_text(json.dumps(plan,indent=2))
print(json.dumps(plan,indent=2))
' "$pilot/validation_gate.json" "$plan"
mapfile -t models < <(uv run --frozen python -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["models_to_expand"]))' "$plan")
[[ "${#models[@]}" -ge 4 ]]

for seed in 43 44 45 46; do
  stage="validation_expansion_seed${seed}"
  write_status running
  run="artifacts/improvements/e01/${suite}_expansion_seed${seed}"
  if [[ -e "$run/status.json" ]]; then
    uv run --frozen python -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="completed", "Incomplete expansion: preserve it and use a new E01_SUITE_NAME"' "$run/status.json"
    continue
  fi
  uv run --frozen python -m tcn_moment.train_feature_fusion \
    --run-name "${suite}_expansion" --seeds "$seed" --models "${models[@]}" --n-jobs 2 \
    --cache-dir "$cache"
done
stage="validation_complete_test_still_locked"
write_status completed
echo "E01 validation suite completed. Test evaluation requires a separate frozen-plan decision."
