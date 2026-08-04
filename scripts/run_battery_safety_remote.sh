#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

mkdir -p artifacts/logs

run_dirs=(
  artifacts/tcn/normalization_none_thesis_tcn_norm_v2_seed{42..46}
  artifacts/moment/moment_full_finetune_thesis_moment_strategy_v2_v100_seed{42..46}
  artifacts/moment_svm/moment_svm_rbf_paper_v1_seed{42..46}
  artifacts/tcn_few_shot/tcn_{01,05,10,20,40}_percent_thesis_few_shot_v1_seed{42..46}
  artifacts/moment_svm_few_shot/moment_svm_thesis_few_shot_v1_seed{42..46}
)

uv run battery-safety-evaluate \
  --run-dirs "${run_dirs[@]}" \
  --output-dir artifacts/battery_safety_thesis_v1 \
  --critical-label 2 \
  --target-recalls 0.95 0.98 0.99 \
  --device cuda \
  2>&1 | tee artifacts/logs/battery_safety_20260804.log
