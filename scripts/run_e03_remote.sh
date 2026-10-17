#!/usr/bin/env bash
# Seven-condition, single-seed validation pilot. Does not expand or evaluate test.
set -Eeuo pipefail
export PATH="/root/.local/bin:/root/miniconda3/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/.uv-python}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
protocol="${E03_PROTOCOL:-configs/experiments/improvements/e03_peft_pilot_protocol.json}"
mapfile -t settings < <(uv run --frozen python -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p["suite"]);print(p["output_dir"])' "$protocol")
suite="${settings[0]}"
plan="${settings[1]}/${suite}_analysis"
mkdir -p "$plan" artifacts/logs
exec 9>"$plan/suite.lock"
flock -n 9 || { echo "E03 pilot is already running"; exit 1; }
exec > >(tee -a "artifacts/logs/${suite}.log") 2>&1
stage="preflight"
child=""
write_status() {
  .venv/bin/python -c '
import sys
from pathlib import Path
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.experiment import utc_now
atomic_write_json(Path(sys.argv[1]),{"status":sys.argv[2],"stage":sys.argv[3],"updated_at":utc_now(),"test_evaluated":False,"automatic_replication_started":False})
' "$plan/suite_status.json" "$1" "$stage"
}
trap 'rc=$?; trap - ERR; write_status failed || true; echo "E03 failed stage=$stage exit=$rc"; exit "$rc"' ERR
stop() {
  trap - ERR INT TERM
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM "$child" || true
    wait "$child" || true
  fi
  write_status interrupted || true
  exit 130
}
trap stop INT TERM
write_status running
uv run --frozen --with pytest python -m pytest tests -q
stage="freeze_pilot_protocol"
write_status running
uv run --frozen python -m tcn_moment.analyze_e03 freeze --protocol "$protocol"
mapfile -t conditions < <(.venv/bin/python -c 'import json,sys;print("\n".join(c["name"] for c in json.load(open(sys.argv[1]))["conditions"]))' "$protocol")
for condition in "${conditions[@]}"; do
  stage="$condition"
  write_status running
  .venv/bin/python -m tcn_moment.analyze_e03 verify --protocol "$protocol"
  run="${settings[1]}/${suite}_${condition}_seed43_fraction40"
  args=(--protocol "$protocol" --condition "$condition")
  if [[ -d "$run" ]]; then args+=(--resume); fi
  # Preflight uv --frozen already checked this exact locked .venv. A direct Python
  # child gives TERM a known target and allows clean epoch-boundary recovery.
  .venv/bin/python -m tcn_moment.train_e03 "${args[@]}" &
  child=$!
  wait "$child"
  child=""
done
stage="audit_pilot"
write_status running
.venv/bin/python -m tcn_moment.analyze_e03 summarize --protocol "$protocol"
stage="completed_pilot_only_no_replication_no_test"
write_status completed
echo "E03 pilot complete. No automatic replication, full finetuning or test. Keep all seven outcomes."
