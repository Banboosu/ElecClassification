#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/ElecClassification
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
.venv/bin/python -m unittest discover -s tests -p test_tcn_legacy_pilot.py -v
.venv/bin/python -m tcn_moment.tcn_legacy_pilot freeze
.venv/bin/python -m tcn_moment.tcn_legacy_pilot run
