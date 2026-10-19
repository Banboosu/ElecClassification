#!/usr/bin/env bash
# Compact E07 only: 8 cases/240 epochs. No test, external watcher, or shutdown.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -m tcn_moment.replicate_fusion_lora coordinate "$@"
