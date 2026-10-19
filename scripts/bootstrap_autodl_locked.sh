#!/usr/bin/env bash
# Install the archived lock in an isolated venv. Never modifies the base conda env.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
record="${DEPLOYMENT_RECORD:-artifacts/deployments/autodl_20261019}"
mkdir -p "$record"
trap 'rc=$?; printf "{\"status\":\"failed\",\"exit_code\":%s}\n" "$rc" > "$record/install_status.json"; exit "$rc"' ERR
printf '{"status":"running","python":"3.11.17","lock_modified":false}\n' > "$record/install_status.json"
# This provider-managed file may contain proxy credentials: never print it/env.
if [[ -f /etc/network_turbo ]]; then
 set +u
 source /etc/network_turbo >/dev/null 2>&1
 set -u
fi
uv python install 3.11.17 --mirror "${UV_PYTHON_INSTALL_MIRROR:-https://github.com/astral-sh/python-build-standalone/releases/download}"
uv sync --frozen --python 3.11.17
.venv/bin/python -m tcn_moment.check_environment --require-cuda > "$record/environment.json"
.venv/bin/python - <<'PY'
import torch
x = torch.arange(16, device='cuda', dtype=torch.float64).reshape(4,4)
y = x @ x.T
assert torch.isfinite(y).all()
assert torch.cuda.get_device_capability() == (8,9), 'Unexpected replacement hardware'
print('Locked CUDA environment and float64 matrix smoke passed.')
PY
printf '{"status":"completed","python":"3.11.17","lock_modified":false,"training_started":false}\n' > "$record/install_status.json"
