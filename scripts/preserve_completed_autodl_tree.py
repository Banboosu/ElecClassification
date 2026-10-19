"""Preserve a terminal AutoDL artifact tree; no training, inference or shutdown."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_HOST = "autodl-container-f0c94cb49a-92794602"
PROJECT = "/root/autodl-tmp/ElecClassification/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--expected-host", default=EXPECTED_HOST,
                        help="Explicit expected instance hostname; record migrations separately.")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--terminal-status", choices=("completed", "failed", "interrupted"), default="completed")
    args = parser.parse_args()
    if not args.remote_root.startswith("artifacts/improvements/") or ".." in Path(args.remote_root).parts:
        raise ValueError("Require a project artifact directory")
    code = f"""
import hashlib,json,platform
from pathlib import Path
if platform.node() != {args.expected_host!r}: raise ValueError('Unexpected AutoDL instance')
r=Path({(PROJECT + args.remote_root)!r})
s=json.loads((r/'suite_status.json').read_text())
if s['status'] != {args.terminal_status!r}: raise ValueError('Suite terminal state does not match request')
files={{}}
for p in sorted(r.rglob('*')):
 if p.is_symlink(): raise ValueError('Unexpected artifact symlink')
 if p.is_file():
  with p.open('rb') as h: digest=hashlib.file_digest(h,'sha256').hexdigest()
  files[str(p.relative_to(r))]={{'sha256':digest,'bytes':p.stat().st_size}}
print(json.dumps({{'host':platform.node(),'status':s,'files':files}}))
"""
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                             "autodl", "/root/miniconda3/bin/python -"],
                            input=code, text=True, capture_output=True, check=True, timeout=60)
    manifest = json.loads(result.stdout)
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "preservation_remote_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["rsync", "-az", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=12",
                    f"autodl:{PROJECT}{args.remote_root}/", str(destination) + "/"], check=True)
    failures = []
    for relative, expected in manifest["files"].items():
        file = destination / relative
        if not file.is_file() or file.is_symlink():
            failures.append(relative)
            continue
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if {"sha256": digest, "bytes": file.stat().st_size} != expected:
            failures.append(relative)
    proof = {"status": "passed" if not failures else "failed",
             "verified_at": datetime.now(timezone.utc).isoformat(),
             "remote_root": args.remote_root, "verified_remote_host": manifest["host"],
             "files_verified": len(manifest["files"]),
             "bytes_verified": sum(f["bytes"] for f in manifest["files"].values()),
             "failures": failures, "test_inference_performed": False, "shutdown_performed": False}
    (destination / "preservation_verified.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))
    if failures:
        raise RuntimeError("Artifact preservation mismatch")


if __name__ == "__main__":
    main()
