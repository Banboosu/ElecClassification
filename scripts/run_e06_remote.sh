#!/usr/bin/env bash
# E06 only. No test, wake plugin, automatic next stage or shutdown.
set -euo pipefail
cd /root/autodl-tmp/ElecClassification
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec 9>artifacts/e06_complement_launcher.lock
flock -n 9
.venv/bin/python -m unittest discover -s tests -p test_tcn_moment_complement.py -v
.venv/bin/python -m unittest discover -s tests -p test_tcn_legacy_pilot.py -v
.venv/bin/python -m tcn_moment.tcn_moment_complement freeze
.venv/bin/python -m tcn_moment.tcn_moment_complement run
