#!/usr/bin/env bash
# E09 pilot only: seed42, 10%/40%, validation only; no test.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec 9>artifacts/e09_launcher.lock
flock -n 9

.venv/bin/python -m unittest discover -s tests -p "test_e09_diff_stream.py"
.venv/bin/python -m tcn_moment.run_e09_diff_stream freeze
.venv/bin/python -m tcn_moment.run_e09_diff_stream prepare
.venv/bin/python -m tcn_moment.run_e09_diff_stream run
