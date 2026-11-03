#!/usr/bin/env bash
# Run the TODO1102-2C noise-robustness sweep with one process per seed.
# MultiRocket's fit loop is serial, so seed-level parallelism is where the
# wall-clock win comes from; the numba transform is parallel per process.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export NUMBA_THREADING_LAYER=omp

OUT="${1:-artifacts/analysis/todo1102_20261102/rocket_noise}"
THREADS="${ROCKET_THREADS:-24}"
mkdir -p "$OUT"

pids=()
for seed in 42 44 45 46; do
  env OMP_NUM_THREADS="$THREADS" NUMBA_NUM_THREADS="$THREADS" \
    .venv/bin/python scripts/todo1102_rocket_noise.py \
      --output "$OUT" --seeds "$seed" > "$OUT/seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[ "$status" -eq 0 ] || { echo "At least one seed failed; see $OUT/seed*.log" >&2; exit 1; }

.venv/bin/python scripts/todo1102_rocket_noise.py --output "$OUT" --merge-only
