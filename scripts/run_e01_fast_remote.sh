#!/usr/bin/env bash
# Execution-only continuation of E01; no test metrics and no search-space changes.
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
mkdir -p artifacts/logs artifacts/improvements
suite="${E01_SUITE_NAME:-e01_20261017_v2}"
cache="artifacts/improvements/${suite}_features"
status="artifacts/improvements/${suite}_suite_status.json"
report="artifacts/analysis/e01_fast_svm_equivalence_20261017.json"
exec 9>"artifacts/improvements/${suite}.lock"
flock -n 9 || { echo "Original/accelerated suite $suite is still running; refusing duplicate writer"; exit 1; }
exec > >(tee -a "artifacts/logs/${suite}.log") 2>&1
stage="acceleration_preflight"
write_status() {
  uv run --frozen python -c '
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]); temporary=p.with_suffix(".tmp")
temporary.write_text(json.dumps({"status":sys.argv[2],"stage":sys.argv[3],
"updated_at":datetime.now(timezone.utc).isoformat(),"test_evaluated":False,
"execution_backend":"fold_local_precomputed_rbf_float64","n_jobs":5},indent=2))
temporary.replace(p)
' "$status" "$1" "$stage"
}
trap 'rc=$?; trap - ERR; write_status failed || true; echo "Fast E01 failed stage=$stage exit=$rc"; exit "$rc"' ERR
trap 'write_status interrupted || true; exit 130' INT TERM
write_status running
echo "=== Switching to verified float64 precomputed-kernel CV; preserve all completed conditions ==="
uv run --frozen --with pytest python -m pytest tests -q

continue_run() {
  local run="$1"
  shift
  if [[ -f "$run/status.json" ]]; then
    if uv run --frozen python -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["status"]=="completed" else 1)' "$run/status.json"; then
      echo "Skipping completed $run"
    else
      uv run --frozen python -m tcn_moment.run_feature_fusion_fast \
        --resume-run "$run" --cache-dir "$cache" --n-jobs 5 --equivalence-report "$report"
    fi
  else
    uv run --frozen python -m tcn_moment.run_feature_fusion_fast \
      --cache-dir "$cache" --n-jobs 5 --equivalence-report "$report" "$@"
  fi
}
stage="pilot_accelerated_resume"
write_status running
pilot="artifacts/improvements/e01/${suite}_pilot_seed42"
continue_run "$pilot" --run-name "${suite}_pilot" --seeds 42

stage="freeze_validation_gate"
write_status running
plan="artifacts/improvements/${suite}_expansion_plan.json"
uv run --frozen python -c '
import json,sys
from pathlib import Path
source=Path(sys.argv[1]); plan=json.loads(source.read_text())
plan.update({"pilot_run":str(source.parent),"seeds":[43,44,45,46],"test_evaluated":False,
"note":"Baselines expand regardless of winner; fusion only by pilot validation gate."})
p=Path(sys.argv[2])
if p.exists():
    assert json.loads(p.read_text())==plan, "Existing frozen plan differs"
else:
    p.write_text(json.dumps(plan,indent=2))
print(json.dumps(plan,indent=2))
' "$pilot/validation_gate.json" "$plan"
mapfile -t models < <(uv run --frozen python -c 'import json,sys;print("\n".join(json.load(open(sys.argv[1]))["models_to_expand"]))' "$plan")
[[ "${#models[@]}" -ge 4 ]]
for seed in 43 44 45 46; do
  stage="validation_expansion_seed${seed}_accelerated"
  write_status running
  continue_run "artifacts/improvements/e01/${suite}_expansion_seed${seed}" \
    --run-name "${suite}_expansion" --seeds "$seed" --models "${models[@]}"
done
stage="validation_complete_test_still_locked"
write_status completed
echo "Accelerated E01 validation suite completed; test evaluation is still locked."
