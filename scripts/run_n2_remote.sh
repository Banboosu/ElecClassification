#!/usr/bin/env bash
# N2 only: legacy masked TCN at 1%/5% labels. Validation only; no test.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec 9>artifacts/n2_launcher.lock
flock -n 9

.venv/bin/python -m unittest discover -s tests -p "test_n2_tcn_low_label.py"
.venv/bin/python -m tcn_moment.run_n2_tcn_low_label freeze
.venv/bin/python -m tcn_moment.run_n2_tcn_low_label run
