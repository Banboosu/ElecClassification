#!/usr/bin/env bash
# Separate, prospectively frozen high-label validation replication. Test stays locked.
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
protocol="${E01B_PROTOCOL:-configs/experiments/improvements/e01b_high_label_protocol.json}"
mapfile -t settings < <(uv run --frozen python -c '
import json,sys
p=json.load(open(sys.argv[1]))
for key in ("suite","output_dir","cache_dir","training_config"):print(p[key])
' "$protocol")
suite="${settings[0]}"
base="${settings[1]}"
cache="${settings[2]}"
config="${settings[3]}"
plan_dir="${base}/${suite}_analysis"
mkdir -p "$plan_dir" artifacts/logs
exec 9>"${plan_dir}/suite.lock"
flock -n 9 || { echo "E01B suite already running"; exit 1; }
exec > >(tee -a "artifacts/logs/${suite}.log") 2>&1
stage="preflight"
write_status() {
  uv run --frozen python -c '
import json,sys
from pathlib import Path
from datetime import datetime,timezone
p=Path(sys.argv[1]);temporary=p.with_suffix(".tmp")
temporary.write_text(json.dumps({"status":sys.argv[2],"stage":sys.argv[3],
"updated_at":datetime.now(timezone.utc).isoformat(),"test_evaluated":False},indent=2))
temporary.replace(p)
' "$plan_dir/suite_status.json" "$1" "$stage"
}
trap 'rc=$?; trap - ERR; write_status failed || true; echo "E01B failed stage=$stage exit=$rc"; exit "$rc"' ERR
trap 'write_status interrupted || true; exit 130' INT TERM
write_status running
uv run --frozen --with pytest python -m pytest tests -q
stage="freeze_protocol_before_replications"
write_status running
uv run --frozen python -m tcn_moment.analyze_e01b freeze --protocol "$protocol"
for seed in 43 44 45 46; do
  stage="replication_seed${seed}"
  write_status running
  uv run --frozen python -m tcn_moment.analyze_e01b verify --protocol "$protocol"
  run="${base}/${suite}_seed${seed}"
  if [[ -f "$run/status.json" ]]; then
    if uv run --frozen python -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["status"]=="completed" else 1)' "$run/status.json"; then
      echo "Skipping completed $run"
    else
      uv run --frozen python -m tcn_moment.run_feature_fusion_fast \
        --resume-run "$run" --cache-dir "$cache" --n-jobs 5 --config "$config"
    fi
  else
    uv run --frozen python -m tcn_moment.run_feature_fusion_fast \
      --run-name "$suite" --output-dir "$base" --cache-dir "$cache" \
      --config "$config" --seeds "$seed" --models fusion_stats_svm \
      --fractions 0.2 0.4 --n-jobs 5
  fi
  # Generic trainer emits the old low-label gate for any input. It is NOT the
  # E01B rule; retain it with an explicit non-applicable name rather than use it.
  if [[ -f "$run/validation_gate.json" ]]; then
    mv "$run/validation_gate.json" "$run/legacy_low_label_gate_not_applicable.json"
  fi
done
stage="audit_and_evaluate_frozen_validation_rule"
write_status running
uv run --frozen python -m tcn_moment.analyze_e01b summarize --protocol "$protocol"
stage="completed_test_still_locked"
write_status completed
echo "E01B complete, including negative outcomes if any. Test remains locked; no automatic next training."
