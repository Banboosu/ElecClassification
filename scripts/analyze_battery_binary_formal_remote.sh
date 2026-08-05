#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

run_dirs=(
  artifacts/battery_binary_stats/battery_binary_stats_{01,05,10,full}_pilot_v1_seed42
  artifacts/battery_binary_stats/battery_binary_stats_{01,05,10,full}_thesis_v1_seed{43..46}
  artifacts/battery_binary_moment_svm/battery_binary_moment_svm_pilot_v1_seed42
  artifacts/battery_binary_moment_svm/battery_binary_moment_svm_thesis_v1_seed{43..46}
)

uv run battery-binary-summarize \
  --run-dirs "${run_dirs[@]}" \
  --baseline-csv artifacts/battery_safety_thesis_v1/per_seed_metrics.csv \
  --gate-decision artifacts/battery_binary_analysis/pilot_seed42/gate_decision.json \
  --output-dir artifacts/battery_binary_analysis/formal_five_seed
