#!/usr/bin/env bash
# E12 only: cross-fitted TCN + DYN + MOMENT/ΔP stacking at 40% labels. Validation only.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec 9>artifacts/e12_launcher.lock
flock -n 9

.venv/bin/python -m unittest discover -s tests -p "test_e12_tcn_stack.py"
.venv/bin/python -m tcn_moment.run_e12_tcn_stack freeze

pids=()
for seed in 42 44 45 46; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    .venv/bin/python -m tcn_moment.run_e12_tcn_stack run --seed "$seed" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[ "$status" -eq 0 ] || { echo "At least one seed failed." >&2; exit 1; }
.venv/bin/python scripts/analyze_e12_results.py
