#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

mkdir -p artifacts/logs

uv run moment-imputation \
  --config configs/experiments/moment_imputation_zero_shot.yaml \
  --run-name moment_imputation_zero_shot_thesis_v2 \
  2>&1 | tee artifacts/logs/moment_imputation_zero_shot_v2_20260803.log
