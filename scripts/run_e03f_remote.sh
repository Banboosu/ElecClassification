#!/usr/bin/env bash
# Four bounded strong-head warm-start treatments; NO test/replication/shutdown.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
protocol=configs/experiments/improvements/e03f_strong_head_lora_protocol.json
mapfile -t settings < <(.venv/bin/python -c 'import json,sys;p=json.load(open(sys.argv[1]));print(p["suite"]);print(p["output_dir"])' "$protocol")
suite="${settings[0]}"
plan="${settings[1]}/${suite}_analysis"
mkdir -p "$plan" artifacts/logs
exec 9>"$plan/suite.lock"
flock -n 9 || { echo 'E03F already running'; exit 1; }
exec > >(tee -a "artifacts/logs/${suite}.log") 2>&1
stage=preflight
child=""
status() {
 .venv/bin/python -c 'import sys;from pathlib import Path;from tcn_moment.io_utils import atomic_write_json;from tcn_moment.experiment import utc_now;atomic_write_json(Path(sys.argv[1]),{"status":sys.argv[2],"stage":sys.argv[3],"updated_at":utc_now(),"test_evaluated":False,"prior_test_results_known":True,"automatic_next_stage":False})' "$plan/suite_status.json" "$1" "$stage"
}
trap 'rc=$?; trap - ERR; status failed || true; echo "E03F failed stage=$stage exit=$rc"; exit "$rc"' ERR
stop() {
 trap - ERR INT TERM
 if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then kill -TERM "$child" || true; wait "$child" || true; fi
 status interrupted || true
 exit 130
}
trap stop INT TERM
status running
uv run --frozen --with pytest python -m pytest tests -q
if [[ ! -f "$plan/frozen_protocol.json" ]]; then
 stage=synthetic_strong_head_batch64_smoke
 status running
 .venv/bin/python -m tcn_moment.strong_head_lora smoke --protocol "$protocol" &
 child=$!; wait "$child"; child=""
 stage=freeze_before_new_training
 status running
 .venv/bin/python -m tcn_moment.strong_head_lora freeze --protocol "$protocol"
fi
stage=prepare_train_and_validation_only
status running
.venv/bin/python -m tcn_moment.strong_head_lora prepare --protocol "$protocol" &
child=$!; wait "$child"; child=""
for condition in frozen_fusion lora8_fusion frozen_moment lora8_moment; do
 stage="$condition"
 status running
 args=(run --protocol "$protocol" --condition "$condition")
 if [[ -d "${settings[1]}/${suite}_${condition}" ]]; then args+=(--resume); fi
 .venv/bin/python -m tcn_moment.strong_head_lora "${args[@]}" &
 child=$!; wait "$child"; child=""
done
stage=audit_all_four_conditions
status running
.venv/bin/python -m tcn_moment.strong_head_lora summarize --protocol "$protocol"
stage=completed_no_test_or_automatic_expansion
status completed
echo 'E03F complete. Preserve negative results; no automatic next experiment or shutdown.'
