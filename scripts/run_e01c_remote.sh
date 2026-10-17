#!/usr/bin/env bash
# Reference-bandwidth/statistics and random-encoder controls; no test evaluation.
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
protocol="${E01C_PROTOCOL:-configs/experiments/improvements/e01c_attribution_protocol.json}"
mapfile -t settings < <(uv run --frozen python -c '
import json,sys
p=json.load(open(sys.argv[1]));print(p["suite"]);print(p["output_dir"])
' "$protocol")
suite="${settings[0]}"
base="${settings[1]}"
plan="${base}/${suite}_analysis"
mkdir -p "$plan" artifacts/logs
exec 9>"$plan/suite.lock"
flock -n 9 || { echo "E01C is already running"; exit 1; }
exec > >(tee -a "artifacts/logs/${suite}.log") 2>&1
stage="preflight"
write_status() {
  uv run --frozen python -c '
import json,sys
from pathlib import Path
from datetime import datetime,timezone
p=Path(sys.argv[1]);tmp=p.with_suffix(".tmp")
tmp.write_text(json.dumps({"status":sys.argv[2],"stage":sys.argv[3],
"updated_at":datetime.now(timezone.utc).isoformat(),"test_evaluated":False},indent=2));tmp.replace(p)
' "$plan/suite_status.json" "$1" "$stage"
}
trap 'rc=$?; trap - ERR; write_status failed || true; echo "E01C failed stage=$stage exit=$rc"; exit "$rc"' ERR
trap 'write_status interrupted || true; exit 130' INT TERM
write_status running
uv run --frozen --with pytest python -m pytest tests -q
stage="freeze_control_protocol"
write_status running
uv run --frozen python -m tcn_moment.analyze_e01c freeze --protocol "$protocol"
for seed in 43 44 45 46; do
  stage="controls_seed${seed}"
  write_status running
  uv run --frozen python -m tcn_moment.analyze_e01c verify --protocol "$protocol"
  run="$base/${suite}_seed${seed}"
  if [[ -f "$run/status.json" ]]; then
    uv run --frozen python -m tcn_moment.train_e01c --protocol "$protocol" --seed "$seed" --resume
  else
    uv run --frozen python -m tcn_moment.train_e01c --protocol "$protocol" --seed "$seed"
  fi
done
stage="audit_attribution_controls"
write_status running
uv run --frozen python -m tcn_moment.analyze_e01c summarize --protocol "$protocol"
stage="completed_test_still_locked"
write_status completed
echo "E01C complete. Keep separate conclusions for bandwidth and pretrained-vs-random controls. No automatic next training/test."
