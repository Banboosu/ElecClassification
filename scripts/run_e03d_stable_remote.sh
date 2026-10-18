#!/usr/bin/env bash
# Cached heads only. No Transformer execution, LoRA training, or test evaluation.
set -Eeuo pipefail
export PATH="/root/.local/bin:/root/miniconda3/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/.uv-python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
protocol="${E03D_STABLE_PROTOCOL:-configs/experiments/improvements/e03d_head_diagnostic_stable_protocol.json}"
mapfile -t settings < <(.venv/bin/python -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p["suite"]);print(p["output_dir"])' "$protocol")
suite="${settings[0]}"
plan="${settings[1]}/${suite}_analysis"
mkdir -p "$plan" artifacts/logs
exec 9>"$plan/suite.lock"
flock -n 9 || { echo "E03D already running"; exit 1; }
exec > >(tee -a "artifacts/logs/${suite}.log") 2>&1
stage="preflight"
child=""
status() {
 .venv/bin/python -c 'import sys;from pathlib import Path;from tcn_moment.io_utils import atomic_write_json;from tcn_moment.experiment import utc_now;atomic_write_json(Path(sys.argv[1]),{"status":sys.argv[2],"stage":sys.argv[3],"updated_at":utc_now(),"test_evaluated":False,"encoder_executed":False})' "$plan/suite_status.json" "$1" "$stage"
}
trap 'rc=$?; trap - ERR; status failed || true; echo "E03D failed stage=$stage exit=$rc"; exit "$rc"' ERR
stop() {
 trap - ERR INT TERM
 if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then kill -TERM "$child" || true; wait "$child" || true; fi
 status interrupted || true
 exit 130
}
trap stop INT TERM
status running
uv run --frozen --with pytest python -m pytest tests -q
stage="freeze_diagnostic"
status running
.venv/bin/python -m tcn_moment.run_e03d_stable freeze --protocol "$protocol"
stage="prepare_cached_features_and_raw_statistics"
status running
.venv/bin/python -m tcn_moment.run_e03d_stable prepare --protocol "$protocol"
mapfile -t names < <(.venv/bin/python -c 'import sys;from pathlib import Path;from tcn_moment.run_e03d_stable import read_protocol,cases;print("\n".join(c["name"] for c in cases(read_protocol(Path(sys.argv[1])))))' "$protocol")
for name in "${names[@]}"; do
 stage="$name"
 status running
 .venv/bin/python -m tcn_moment.run_e03d_stable verify --protocol "$protocol"
 args=(run --protocol "$protocol" --condition "$name")
 if [[ -d "${settings[1]}/${suite}_${name}" ]]; then args+=(--resume); fi
 .venv/bin/python -m tcn_moment.run_e03d_stable "${args[@]}" &
 child=$!
 wait "$child"
 child=""
done
stage="audit_diagnostics"
status running
.venv/bin/python -m tcn_moment.run_e03d_stable summarize --protocol "$protocol"
stage="completed_no_automatic_lora_replication_or_test"
status completed
echo "E03D complete. Keep original E03P failure; no automatic next experiment."
