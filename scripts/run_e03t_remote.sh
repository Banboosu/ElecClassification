#!/usr/bin/env bash
# Explicitly authorized fixed test, zero fitting. No automatic next stage/shutdown.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
protocol=configs/experiments/improvements/e03t_final_test_protocol.json
.venv/bin/python -m tcn_moment.check_environment --require-cuda
.venv/bin/python -m tcn_moment.final_test freeze --protocol "$protocol"
.venv/bin/python -m tcn_moment.final_test run --protocol "$protocol"
