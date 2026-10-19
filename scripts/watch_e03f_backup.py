"""Local operational watcher: terminal E03F-v2 -> backup/hash only, never train/test/shutdown.

Requires this machine and SSH to stay available. Times out after 8h without
changing server state. A failed/interrupted suite is also preserved as-is.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_HOST = "autodl-container-f0c94cb49a-92794602"
PROJECT = "/root/autodl-tmp/ElecClassification"
RELATIVE = "artifacts/improvements/e03f_strong_head_lora"
SUITE = "e03f_strong_head_lora_20261019_v2"
DESTINATION = Path("artifacts/imports/e03f_strong_head_lora_20261019")


def ssh_python(code):
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "autodl",
            "/root/miniconda3/bin/python -",
        ],
        input=code,
        text=True,
        capture_output=True,
        timeout=180,
        check=True,
    )
    return json.loads(result.stdout)


def status(value, **details):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    report = {
        "status": value,
        "updated_at": datetime.now(UTC).isoformat(),
        "training_started_by_watcher": False,
        "test_evaluated": False,
        "shutdown_performed": False,
        **details,
    }
    temp = DESTINATION / "watcher_status.tmp"
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(DESTINATION / "watcher_status.json")
    print(json.dumps(report), flush=True)


def main():
    started = time.monotonic()
    status("waiting", suite=SUITE)
    while time.monotonic() - started < 8 * 3600:
        try:
            state = ssh_python(f"""
import json,platform
from pathlib import Path
p=Path({(PROJECT + "/" + RELATIVE + "/" + SUITE + "_analysis/suite_status.json")!r})
print(json.dumps({{'host':platform.node(),'suite':json.loads(p.read_text())}}))
""")
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
            status("waiting_ssh_retry", error_type=type(error).__name__)
            time.sleep(120)
            continue
        if state["host"] != EXPECTED_HOST:
            raise RuntimeError("autodl alias points to a different instance; refusing backup")
        if state["suite"]["status"] in {"completed", "failed", "interrupted"}:
            break
        time.sleep(120)
    else:
        status("watch_timeout", boundary="No server changes; manually inspect and back up later")
        return
    status("copying", suite_status=state["suite"])
    manifest = ssh_python(f"""
import hashlib,json,platform
from pathlib import Path
assert platform.node()=={EXPECTED_HOST!r}
root=Path({(PROJECT + "/" + RELATIVE)!r})
files={{}}
for p in sorted(root.rglob('*')):
 if p.is_symlink():raise ValueError('Unexpected symlink in E03F artifacts')
 if p.is_file():
  with p.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
  files[str(p.relative_to(root))]={{'sha256':digest,'bytes':p.stat().st_size}}
print(json.dumps({{'files':files,'scope':'entire E03F v1/v2 artifact root; public backbone and parents are in earlier preservation'}}))
""")
    (DESTINATION / "terminal_remote_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    bundle = DESTINATION / "terminal_artifacts"
    bundle.mkdir(exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-az",
            "--partial",
            "--partial-dir=.rsync-partial",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=15",
            f"autodl:{PROJECT}/{RELATIVE}/",
            str(bundle) + "/",
        ],
        check=True,
    )
    failures = []
    for relative, expected in manifest["files"].items():
        path = bundle / relative
        if not path.is_file() or path.is_symlink():
            failures.append(relative)
            continue
        with path.open("rb") as h:
            digest = hashlib.file_digest(h, "sha256").hexdigest()
        if {"sha256": digest, "bytes": path.stat().st_size()} != expected:
            failures.append(relative)
    status(
        "verified" if not failures else "verification_failed",
        suite_status=state["suite"],
        files_verified=len(manifest["files"]),
        bytes_verified=sum(v["bytes"] for v in manifest["files"].values()),
        failures=failures,
    )
    if failures:
        raise RuntimeError("Backup mismatch")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("failed", error_type=type(error).__name__, error=str(error))
        raise
