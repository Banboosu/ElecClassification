#!/usr/bin/env bash
# Exploratory E02 pooling; no test inference, encoder training or next-stage launch.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
.venv/bin/python -m tcn_moment.check_environment --require-cuda
.venv/bin/python -m tcn_moment.run_pooling_pilot freeze
.venv/bin/python -m tcn_moment.run_pooling_pilot run
