#!/usr/bin/env bash
# E09 expansion: 4 seeds x 1%/10%/40%, validation only; no test.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec 9>artifacts/e09_expand_launcher.lock
flock -n 9

.venv/bin/python -m unittest discover -s tests -p "test_e09_diff_stream.py"
for seed in 42 44 45 46; do
  protocol="configs/experiments/improvements/e09_diff_stream_seed${seed}_protocol.json"
  .venv/bin/python -m tcn_moment.run_e09_diff_stream freeze --protocol "$protocol"
  .venv/bin/python -m tcn_moment.run_e09_diff_stream prepare --protocol "$protocol"
  .venv/bin/python -m tcn_moment.run_e09_diff_stream run --protocol "$protocol"
done
.venv/bin/python scripts/analyze_e09_expansion.py
