"""E05B terminal observer: preserve/report only; never train/test/expand/shutdown.

Runs independently of the interactive turn, bound to its originating pi session.
A 180-second timer checks the existing suite; no new agent session is created.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = "/root/autodl-tmp/ElecClassification"
RELATIVE = "artifacts/improvements/e05b_tcn_legacy_pilot/e05b_tcn_legacy_pilot_20261019_v1"
HOST = "autodl-container-f0c94cb49a-92794602"
DEST = Path("artifacts/imports/e05b_tcn_legacy_pilot_20261019")
SESSION = os.environ.get("PI_SESSION_FILE")


def status(value, **details):
    report = {"status": value, "updated_at": datetime.now(timezone.utc).isoformat(),
              "owner_session_file": SESSION, "poll_timer_seconds": 180,
              "training_started_by_watcher": False, "test_evaluated": False,
              "automatic_expansion": False, "shutdown_performed": False, **details}
    temporary = DEST / "watcher_status.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(DEST / "watcher_status.json")
    print(json.dumps(report), flush=True)


def state():
    code = f"""
import json,platform
from pathlib import Path
if platform.node() != {HOST!r}: raise ValueError('AutoDL alias changed')
r=Path({(PROJECT + '/' + RELATIVE)!r})
s=json.loads((r/'suite_status.json').read_text())
progress={{p.parent.name:json.loads(p.read_text()) for p in r.glob('fraction*/status.json')}}
print(json.dumps({{'suite':s,'conditions':progress}}))
"""
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "autodl",
                             "/root/miniconda3/bin/python -"], input=code, text=True,
                            capture_output=True, check=True, timeout=30)
    return json.loads(result.stdout)


def rsync(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=12",
                    f"autodl:{PROJECT}/{source}", str(destination)], check=True, timeout=120)


def finish(remote):
    terminal = remote["suite"]["status"]
    status("preserving", remote=remote)
    subprocess.run([sys.executable, "scripts/preserve_completed_autodl_tree.py",
                    "--remote-root", RELATIVE, "--destination", str(DEST / "terminal_artifacts"),
                    "--terminal-status", terminal], check=True, timeout=300)
    rsync("artifacts/improvements/e05b_tcn_legacy_pilot/runner.log", DEST / "runner.log")
    if terminal != "completed":
        status("terminal_failure_preserved", remote=remote)
        return
    root = DEST / "terminal_artifacts"
    summary = json.loads((root / "summary.json").read_text())
    if summary["audit_status"] != "passed" or summary["test_evaluated"] or summary["conditions"] != 8:
        raise ValueError("Terminal suite lacks the complete validation-only audit")
    doc = Path("docs/experiment_records/e05b_tcn_legacy_pilot_results_20261019.md")
    rsync(str(doc), doc)
    plot_error = None
    try:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "autodl",
                        f"cd {PROJECT} && .venv/bin/python scripts/plot_e05b_tcn_legacy.py --analysis {RELATIVE}"],
                       check=True, timeout=45)
        figure = Path("docs/figures/e05b_tcn_legacy_pilot_20261019.svg")
        rsync(str(figure), figure)
    except (subprocess.SubprocessError, OSError) as error:
        plot_error = str(error)
    with doc.open("a") as stream:
        stream.write(f"\n本地终态归档：`{root}`；`preservation_verified.json`记录逐文件SHA核验。\n")
        if plot_error is None:
            stream.write("\n![Validation learning curves](../figures/e05b_tcn_legacy_pilot_20261019.svg)\n")
    status("completed_verified", remote=remote, conditions=8,
           epochs_audited=summary["epochs_audited"],
           prediction_rows_audited=summary["prediction_rows_audited"],
           report=str(doc), plot_error=plot_error)


def main():
    if not SESSION:
        raise RuntimeError("Observer must belong to the originating pi session")
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        while time.monotonic() - started < 8.5 * 3600:
            try:
                remote = state()
                if remote["suite"]["status"] in {"completed", "failed", "interrupted"}:
                    finish(remote)
                    return
                status("waiting", remote=remote)
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                status("retrying_operation", error=str(error))
            time.sleep(180)
        status("watch_timeout", boundary="No training/test/expansion/shutdown performed")


if __name__ == "__main__":
    main()
