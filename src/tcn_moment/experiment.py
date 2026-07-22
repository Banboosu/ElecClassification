from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tcn_moment.io_utils import atomic_write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _git_state() -> tuple[dict[str, Any], bytes | None]:
    metadata: dict[str, Any] = {
        "git_commit": None,
        "git_dirty": None,
        "git_diff_sha256": None,
        "git_status": [],
    }
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        metadata["git_commit"] = commit_result.stdout.decode("utf-8").strip()
    except (OSError, subprocess.SubprocessError):
        return metadata, None

    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        status_lines = status_result.stdout.decode("utf-8").splitlines()
        metadata["git_status"] = status_lines
        metadata["git_dirty"] = bool(status_lines)
        if not status_lines:
            return metadata, None
        diff_result = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        diff = diff_result.stdout
        metadata["git_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        return metadata, diff
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return metadata, None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        versions = sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))
        return ", ".join(versions) or None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment(
    torch: Any,
    git_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    if git_metadata is None:
        git_metadata, _ = _git_state()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        **git_metadata,
        "packages": {
            "momentfm": _package_version("momentfm"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "torch": str(torch.__version__).split("+", maxsplit=1)[0],
            "torch_build": str(torch.__version__),
        },
        "cuda": {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": gpu_names,
            "nvidia_driver": _nvidia_driver(),
        },
    }


def _safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not cleaned:
        raise ValueError("run_name must contain at least one letter or number.")
    return cleaned


@dataclass
class RunContext:
    model_name: str
    run_name: str
    run_dir: Path
    status_path: Path
    started_at: str

    def set_status(self, status: str, **details: Any) -> None:
        updated_at = utc_now()
        payload = {
            "model": self.model_name,
            "run_name": self.run_name,
            "status": status,
            "started_at": self.started_at,
            "updated_at": updated_at,
            **_json_safe(details),
        }
        if status in {"completed", "interrupted", "failed"}:
            started = datetime.fromisoformat(self.started_at)
            finished = datetime.fromisoformat(updated_at)
            payload["finished_at"] = updated_at
            payload["duration_seconds"] = (finished - started).total_seconds()
        atomic_write_json(self.status_path, payload)


def prepare_run(
    *,
    model_name: str,
    base_output_dir: Path,
    config: Any,
    config_path: Path,
    torch: Any,
    run_name: str | None,
    resume_dir: Path | None,
) -> RunContext:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if resume_dir is not None:
        run_dir = resume_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
        selected_name = run_dir.name
        previous_status_path = run_dir / "status.json"
        previous_status = read_json(previous_status_path) if previous_status_path.exists() else {}
        started_at = str(previous_status.get("started_at", utc_now()))
    else:
        default_name = f"{model_name.lower()}_{timestamp}"
        selected_name = _safe_run_name(run_name or default_name)
        run_dir = (base_output_dir / selected_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        started_at = utc_now()

    context = RunContext(
        model_name=model_name,
        run_name=selected_name,
        run_dir=run_dir,
        status_path=run_dir / "status.json",
        started_at=started_at,
    )
    snapshot_name = f"config_resume_{timestamp}.yaml" if resume_dir is not None else "config.yaml"
    shutil.copy2(config_path, run_dir / snapshot_name)
    atomic_write_json(run_dir / "resolved_config.json", _json_safe(config))
    environment_name = (
        f"environment_resume_{timestamp}.json" if resume_dir is not None else "environment.json"
    )
    git_metadata, git_diff = _git_state()
    if git_diff is not None:
        git_diff_name = (
            f"git_diff_resume_{timestamp}.patch"
            if resume_dir is not None
            else "git_diff.patch"
        )
        (run_dir / git_diff_name).write_bytes(git_diff)
        git_metadata["git_diff_file"] = git_diff_name
    atomic_write_json(
        run_dir / environment_name,
        {
            "recorded_at": started_at,
            "command": sys.argv,
            "working_directory": os.getcwd(),
            **collect_environment(torch, git_metadata=git_metadata),
        },
    )
    context.set_status("running", resumed=resume_dir is not None)
    return context


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
