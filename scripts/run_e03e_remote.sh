#!/usr/bin/env bash
# Four replication seeds, three fixed gain heads each; no LoRA/test.
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
protocol="${E03E_PROTOCOL:-configs/experiments/improvements/e03e_gain_replication_protocol.json}"
mapfile -t settings < <(.venv/bin/python -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p["suite"]);print(p["output_dir"])' "$protocol")
suite="${settings[0]}"
plan="${settings[1]}/${suite}_analysis"
mkdir -p "$plan" artifacts/logs
exec 9>"$plan/suite.lock"
flock -n 9 || { echo "E03E already running"; exit 1; }
exec > >(tee -a "artifacts/logs/${suite}.log") 2>&1
stage="preflight"
child=""
status() {
 .venv/bin/python -c 'import sys;from pathlib import Path;from tcn_moment.io_utils import atomic_write_json;from tcn_moment.experiment import utc_now;atomic_write_json(Path(sys.argv[1]),{"status":sys.argv[2],"stage":sys.argv[3],"updated_at":utc_now(),"test_evaluated":False,"supervised_encoder_training":False})' "$plan/suite_status.json" "$1" "$stage"
}
trap 'rc=$?; trap - ERR; status failed || true; echo "E03E failed stage=$stage exit=$rc"; exit "$rc"' ERR
stop() {
 trap - ERR INT TERM
 if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then kill -TERM "$child" || true; wait "$child" || true; fi
 status interrupted || true
 exit 130
}
trap stop INT TERM
status running
uv run --frozen --with pytest python -m pytest tests -q
stage="freeze_replication"
status running
.venv/bin/python -m tcn_moment.replicate_gain_heads freeze --protocol "$protocol"
for seed in 42 44 45 46; do
 stage="prepare_seed${seed}_and_missing_strong_control"
 status running
 .venv/bin/python -m tcn_moment.replicate_gain_heads prepare --protocol "$protocol" --seed "$seed" &
 child=$!
 wait "$child"
 child=""
 for branch in statistics moment fusion; do
  stage="seed${seed}_${branch}"
  status running
  .venv/bin/python -m tcn_moment.replicate_gain_heads verify --protocol "$protocol"
  args=(run --protocol "$protocol" --seed "$seed" --branch "$branch")
  if [[ -d "${settings[1]}/${suite}_seed${seed}_${branch}" ]]; then args+=(--resume); fi
  .venv/bin/python -m tcn_moment.replicate_gain_heads "${args[@]}" &
  child=$!
  wait "$child"
  child=""
 done
done
stage="audit_replication"
status running
.venv/bin/python -m tcn_moment.replicate_gain_heads summarize --protocol "$protocol"
stage="completed_no_automatic_test_lora_or_expansion"
status completed
echo "E03E complete. Pilot43 excluded; preserve all outcomes. No automatic next stage."
