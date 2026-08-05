#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

stats_run_dirs=(
  artifacts/battery_binary_stats/battery_binary_stats_{01,05,10,full}_pilot_v1_seed42
  artifacts/battery_binary_stats/battery_binary_stats_{01,05,10,full}_thesis_v1_seed{43..46}
)

uv run python scripts/analyze_battery_binary_results.py \
  --formal-dir artifacts/battery_binary_analysis/formal_five_seed \
  --baseline-dir artifacts/battery_safety_thesis_v1 \
  --stats-run-dirs "${stats_run_dirs[@]}" \
  --figure-dir docs/figures
