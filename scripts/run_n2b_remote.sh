#!/usr/bin/env bash
# N2b only: bounded LR sensitivity at 1%/5%. Validation only; no test.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec 9>artifacts/n2b_launcher.lock
flock -n 9

.venv/bin/python -m unittest discover -s tests -p "test_n2b_lr_sweep.py"
.venv/bin/python -m tcn_moment.run_n2b_lr_sweep freeze

pids=()
for seed in 42 44 45 46; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    .venv/bin/python -m tcn_moment.run_n2b_lr_sweep run --seed "$seed" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[ "$status" -eq 0 ] || echo "At least one seed failed; audit will record failures." >&2
.venv/bin/python -m tcn_moment.run_n2b_lr_sweep audit
