#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

mkdir -p artifacts/logs

seeds=(43 44 45 46)
tags=(01 05 10 full)
configs=(
  configs/experiments/few_shot/tcn_01_percent.yaml
  configs/experiments/few_shot/tcn_05_percent.yaml
  configs/experiments/few_shot/tcn_10_percent.yaml
  configs/experiments/normalization_none.yaml
)

require_new_or_completed() {
  local run_dir="$1"
  if [[ -f "${run_dir}/status.json" ]] \
    && grep -q '"status": "completed"' "${run_dir}/status.json"; then
    echo "Skipping completed run: ${run_dir}"
    return 1
  fi
  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
    exit 1
  fi
  return 0
}

{
  echo "Battery binary five-seed expansion started at $(date --iso-8601=seconds)"
  echo "Validation gate: expand STATISTICAL and MOMENT_RBF_SVM; do not expand TCN."

  for seed in "${seeds[@]}"; do
    for index in "${!tags[@]}"; do
      tag="${tags[$index]}"
      config="${configs[$index]}"
      run_name="battery_binary_stats_${tag}_thesis_v1_seed${seed}"
      run_dir="artifacts/battery_binary_stats/${run_name}"
      if require_new_or_completed "${run_dir}"; then
        uv run battery-binary \
          --model stats \
          --config "${config}" \
          --seed "${seed}" \
          --run-name "${run_name}"
      fi
    done
  done

  for seed in "${seeds[@]}"; do
    run_name="battery_binary_moment_svm_thesis_v1_seed${seed}"
    run_dir="artifacts/battery_binary_moment_svm/${run_name}"
    if require_new_or_completed "${run_dir}"; then
      uv run battery-binary \
        --model moment-svm \
        --config configs/experiments/battery_binary/moment_svm.yaml \
        --seed "${seed}" \
        --run-name "${run_name}"
    fi
  done

  echo "Battery binary five-seed expansion finished at $(date --iso-8601=seconds)"
} 2>&1 | tee artifacts/logs/battery_binary_expansion_20260804.log
