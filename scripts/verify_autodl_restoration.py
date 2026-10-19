"""Verify an exact archived restore without loading checkpoints or running models."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--public-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    failures, checked, total = [], 0, 0
    for name, expected in manifest["files"].items():
        if name.startswith("remote_project/"):
            relative = Path(name.removeprefix("remote_project/"))
            target = args.root / relative
        elif name.startswith("artifacts/"):
            relative = Path(name)
            target = args.root / relative
        elif name == "public_model/model.safetensors":
            relative = Path(name)
            target = args.public_model
        else:
            failures.append({"path": name, "error": "unrecognized manifest prefix"})
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        if not target.is_file() or target.is_symlink():
            failures.append({"path": name, "error": "missing file or unexpected symlink"})
            continue
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        actual = {"bytes": target.stat().st_size, "sha256": digest}
        if actual != expected:
            failures.append({"path": name, "expected": expected, "actual": actual})
        checked += 1
        total += actual["bytes"]
        if checked % 250 == 0:
            print(f"verified {checked}/{len(manifest['files'])} archived files", flush=True)
    report = {
        "status": "passed" if not failures else "failed",
        "verified_at": datetime.now(UTC).isoformat(),
        "files_verified": checked,
        "bytes_verified": total,
        "failures": failures,
        "archived_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "scope": "Exact archived project snapshot, experimental artifacts and public checkpoint; extra deployment files excluded.",
        "training_started": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
