#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/logs

uv run moment-retrieval \
  --config configs/experiments/moment_retrieval_zero_shot.yaml \
  --run-name moment_retrieval_zero_shot_thesis_v1 \
  2>&1 | tee artifacts/logs/moment_retrieval_zero_shot_v1_20260803.log
