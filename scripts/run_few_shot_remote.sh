#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

mkdir -p artifacts/logs

suite_name="thesis_few_shot_v1"
seeds=(42 43 44 45 46)

uv run experiment-suite \
  --model tcn \
  --configs \
    configs/experiments/few_shot/tcn_01_percent.yaml \
    configs/experiments/few_shot/tcn_05_percent.yaml \
    configs/experiments/few_shot/tcn_10_percent.yaml \
    configs/experiments/few_shot/tcn_20_percent.yaml \
    configs/experiments/few_shot/tcn_40_percent.yaml \
  --seeds "${seeds[@]}" \
  --suite-name "${suite_name}" \
  2>&1 | tee artifacts/logs/few_shot_tcn_20260728.log

uv run experiment-suite \
  --model moment-svm-few-shot \
  --configs configs/experiments/few_shot/moment_svm.yaml \
  --seeds "${seeds[@]}" \
  --suite-name "${suite_name}" \
  2>&1 | tee artifacts/logs/few_shot_moment_svm_20260728.log
