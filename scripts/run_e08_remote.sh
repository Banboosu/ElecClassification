#!/usr/bin/env bash
# E08 only: strong dynamics-feature baselines + frozen MOMENT stacking.
# Validation only. No test evaluation, external watcher, or automatic next stage.
#
# The four seeds are independent except for the frozen protocol file.  Run them in
# parallel with 4 OpenMP threads each (the container quota is 16 vCPU).  Set
# E08_THREADS=1 E08_PARALLEL=0 to fall back to the original serial behaviour.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

THREADS="${E08_THREADS:-4}"
PARALLEL="${E08_PARALLEL:-1}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

exec 9>artifacts/e08_launcher.lock
flock -n 9

SUITE_NAME="${1:-e08_strong_features_20261020_v1}"
CACHE_DIR="${2:-artifacts/improvements/e01_20261017_v2_features}"

.venv/bin/python -m unittest discover -s tests -p "test_dynamics_features.py"
.venv/bin/python -m unittest discover -s tests -p "test_e08_analysis.py"
.venv/bin/python -m tcn_moment.run_e08 freeze --suite-name "$SUITE_NAME"

SEEDS=(42 44 45 46)
run_seed() {
  local seed="$1"
  .venv/bin/python -m tcn_moment.run_e08 run \
    --seed "$seed" \
    --suite-name "$SUITE_NAME" \
    --moment-cache-dir "$CACHE_DIR"
}

if [ "$PARALLEL" = "1" ]; then
  pids=()
  for seed in "${SEEDS[@]}"; do
    run_seed "$seed" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  [ "$status" -eq 0 ] || { echo "At least one seed failed; inspect status.json files." >&2; exit 1; }
else
  for seed in "${SEEDS[@]}"; do
    run_seed "$seed"
  done
fi

.venv/bin/python scripts/analyze_e08_results.py \
  --suite-dir "artifacts/improvements/e08_strong_features/$SUITE_NAME" \
  --tcn-mean 97.7790 \
  --tcn-sd 0.2277
