#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/root/.local/bin:${PATH}"

output_dir="artifacts/analysis/m03_protocol_audit_20260805"
log_path="artifacts/logs/m03_protocol_audit_20260805.log"
mkdir -p "${output_dir}" artifacts/logs

uv run python -m tcn_moment.audit_protocol \
  --project-root . \
  --output-dir "${output_dir}" \
  2>&1 | tee "${log_path}"
