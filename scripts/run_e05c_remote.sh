#!/usr/bin/env bash
# N1 only: fixed36 TCN runs and seed43 diagnostics; no pi-wake/test/next stage.
set -euo pipefail
cd /root/autodl-tmp/ElecClassification
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -m unittest discover -s tests -p test_tcn_legacy_pilot.py -v
.venv/bin/python -m unittest discover -s tests -p test_replicate_tcn.py -v
.venv/bin/python -m tcn_moment.replicate_tcn freeze
.venv/bin/python -m tcn_moment.replicate_tcn run
