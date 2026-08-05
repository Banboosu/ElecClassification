#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

mkdir -p artifacts/logs

for seed in 42 43 44 45 46; do
  run_name="moment_svm_random_m01_random_encoder_v1_seed${seed}"
  run_dir="artifacts/moment_svm_pretraining_ablation/${run_name}"
  if [[ -f "${run_dir}/metrics.json" ]] \
    && [[ "$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${run_dir}/status.json")" == "completed" ]]; then
    echo "Skipping completed ${run_name}"
    continue
  fi
  uv run moment-svm-few-shot \
    --config configs/experiments/pretraining_ablation/moment_svm_random.yaml \
    --seed "${seed}" \
    --run-name "${run_name}"
done 2>&1 | tee artifacts/logs/m01_pretraining_ablation_random_20260805.log

pretrained_runs=()
random_runs=()
for seed in 42 43 44 45 46; do
  pretrained_runs+=(
    "artifacts/moment_svm_few_shot/moment_svm_thesis_few_shot_v1_seed${seed}"
  )
  random_runs+=(
    "artifacts/moment_svm_pretraining_ablation/moment_svm_random_m01_random_encoder_v1_seed${seed}"
  )
done

uv run python scripts/analyze_m01_pretraining_ablation.py \
  --pretrained-runs "${pretrained_runs[@]}" \
  --random-runs "${random_runs[@]}" \
  --output-dir artifacts/analysis/m01_pretraining_ablation \
  2>&1 | tee artifacts/logs/m01_pretraining_ablation_analysis_20260805.log
