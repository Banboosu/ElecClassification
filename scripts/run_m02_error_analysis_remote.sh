#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

output_dir="artifacts/m02_error_analysis_20260805"
log_path="artifacts/logs/m02_error_analysis_20260805.log"
status_path="${output_dir}/remote_status.txt"
mkdir -p "${output_dir}" artifacts/logs

write_status() {
  exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    state="COMPLETED"
  else
    state="FAILED"
  fi
  printf '%s\nexit_code=%s\nfinished_at=%s\n' \
    "${state}" "${exit_code}" "$(date --iso-8601=seconds)" > "${status_path}"
}
trap write_status EXIT
printf 'RUNNING\nstarted_at=%s\n' "$(date --iso-8601=seconds)" > "${status_path}"

tcn_runs=(
  artifacts/tcn/normalization_none_thesis_tcn_norm_v2_seed{42..46}
)
moment_runs=(
  artifacts/moment/moment_full_finetune_thesis_moment_strategy_v2_v100_seed{42..46}
)
svm_runs=(
  artifacts/moment_svm_few_shot/moment_svm_thesis_few_shot_v1_seed{42..46}
)

uv run python -m tcn_moment.evaluate_error_analysis \
  --tcn-run-dirs "${tcn_runs[@]}" \
  --moment-run-dirs "${moment_runs[@]}" \
  --svm-run-dirs "${svm_runs[@]}" \
  --low-label-fractions 0.01 0.05 0.10 \
  --output-dir "${output_dir}" \
  --critical-label 2 \
  --canonical-seed 42 \
  --length-bins 4 \
  --high-confidence-quantile 0.90 \
  --device cuda \
  2>&1 | tee "${log_path}"
