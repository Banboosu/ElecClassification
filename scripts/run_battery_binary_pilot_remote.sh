#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

mkdir -p artifacts/logs

seed=42
tags=(01 05 10 full)
configs=(
  configs/experiments/few_shot/tcn_01_percent.yaml
  configs/experiments/few_shot/tcn_05_percent.yaml
  configs/experiments/few_shot/tcn_10_percent.yaml
  configs/experiments/normalization_none.yaml
)

{
  echo "Battery binary pilot started at $(date --iso-8601=seconds)"

  for index in "${!tags[@]}"; do
    tag="${tags[$index]}"
    config="${configs[$index]}"
    uv run battery-binary \
      --model stats \
      --config "${config}" \
      --seed "${seed}" \
      --run-name "battery_binary_stats_${tag}_pilot_v1_seed${seed}"
  done

  for index in "${!tags[@]}"; do
    tag="${tags[$index]}"
    config="${configs[$index]}"
    uv run battery-binary \
      --model tcn \
      --config "${config}" \
      --seed "${seed}" \
      --run-name "battery_binary_tcn_${tag}_pilot_v1_seed${seed}"
  done

  uv run battery-binary \
    --model moment-svm \
    --config configs/experiments/battery_binary/moment_svm.yaml \
    --seed "${seed}" \
    --run-name "battery_binary_moment_svm_pilot_v1_seed${seed}"

  echo "Battery binary pilot finished at $(date --iso-8601=seconds)"
} 2>&1 | tee artifacts/logs/battery_binary_pilot_20260804.log
