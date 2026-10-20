"""N2b: bounded learning-rate sensitivity for the legacy masked TCN at 1%/5%.

The fixed E05C configuration (LR 1e-3) is unstable at 1% labels for one seed.  This
sweep changes only the learning rate, keeps the structure/data/epochs/selection rule
fixed, and is validation-only.  It is a fairness check, not a hyperparameter search
for a headline result.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcn_moment import run_n2_tcn_low_label as n2
from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.data import stratified_train_subset_indices
from tcn_moment.io_utils import atomic_write_json, sha256_file

DEFAULT_PROTOCOL = Path("configs/experiments/improvements/n2b_lr_sweep_protocol.json")
PACKAGES = ("torch", "numpy", "pandas", "scikit-learn")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root(p: dict[str, Any]) -> Path:
    return Path(p["output_dir"]) / p["suite"]


def _validate(p: dict[str, Any]) -> None:
    if p.get("experiment_id") != "N2B":
        raise ValueError("Unexpected experiment id")
    if p.get("seeds") != [42, 44, 45, 46]:
        raise ValueError("Unexpected seeds")
    if p.get("train_fractions") != [0.01, 0.05]:
        raise ValueError("Unexpected fractions")
    if p.get("learning_rates") != [0.0003, 0.001, 0.003]:
        raise ValueError("Unexpected learning-rate grid")
    if p.get("condition") != "legacy_masked50":
        raise ValueError("Unexpected condition")
    if p.get("epochs") != 50 or p.get("batch_size") != 32 or p.get("evaluation_batch_size") != 64:
        raise ValueError("Unexpected training scope")
    if p.get("test_evaluated") or p.get("automatic_expansion") or p.get("automatic_shutdown"):
        raise ValueError("N2b must remain validation-only and non-expanding")
    if set(p["split_manifests"]) != {str(seed) for seed in p["seeds"]}:
        raise ValueError("Missing split manifests")


def _load(path: Path) -> dict[str, Any]:
    p = _read(path)
    _validate(p)
    return p


def _sources(protocol_path: Path) -> list[Path]:
    directory = Path(__file__).resolve().parent
    return [
        Path(__file__).resolve(),
        directory / "run_n2_tcn_low_label.py",
        directory / "tcn_legacy_pilot.py",
        directory / "data.py",
        directory / "train_tcn.py",
        directory / "training_utils.py",
        directory / "audit_protocol.py",
        directory / "io_utils.py",
        directory / "config.py",
        directory / "experiment.py",
        directory / "metrics.py",
        Path("tests/test_n2b_lr_sweep.py"),
        Path("scripts/run_n2b_remote.sh"),
        protocol_path,
        Path("legacy/改进TCN.py"),
    ]


def freeze(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    root = _root(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        return verify(protocol_path)
    if list(root.glob("lr*")):
        raise ValueError("Freeze must precede training")
    if sha256_file(Path(p["dataset"])) != p["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    for seed in p["seeds"]:
        manifest = _read(Path(p["split_manifests"][str(seed)]))
        if (
            manifest["protocol"]["random_state"] != seed
            or manifest["protocol"]["dataset_sha256"] != p["dataset_sha256"]
            or manifest["counts"]["train"] != 24569
            or manifest["counts"]["validation"] != 3510
        ):
            raise ValueError(f"Split mismatch for seed {seed}")
    checks = pilot.synthetic_check()
    pilot.configure()
    files = set(_sources(protocol_path))
    files.update(Path(p["split_manifests"][str(seed)]) for seed in p["seeds"])
    files.add(Path(p["dataset"]))
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Missing frozen input: {file}")
    frozen = {
        "frozen_at": _now(),
        "protocol": p,
        "protocol_sha256": sha256_file(protocol_path),
        "file_sha256": {str(file): sha256_file(file) for file in sorted(files)},
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "cuda": torch.version.cuda,
        },
        "synthetic_checks": checks,
        "test_evaluated": False,
        "prior_test_results_known": True,
        "new_training_outcomes_observed": False,
    }
    for file in sorted(files):
        if file.parts[0] in {"src", "configs", "tests", "scripts", "docs", "legacy"}:
            destination = root / "source_snapshot" / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    atomic_write_json(root / "frozen_protocol.json", frozen)
    print(f"Froze N2b: {root / 'frozen_protocol.json'}", flush=True)
    return frozen


def verify(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    frozen = _read(_root(p) / "frozen_protocol.json")
    if frozen["protocol"] != p or frozen["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Frozen protocol changed")
    for file, expected in frozen["file_sha256"].items():
        if sha256_file(Path(file)) != expected:
            raise ValueError(f"Frozen input changed: {file}")
    if frozen["packages"] != {name: importlib.metadata.version(name) for name in PACKAGES}:
        raise ValueError("Dependency versions changed")
    return frozen


def _case_dir(root: Path, learning_rate: float, seed: int, fraction: float) -> Path:
    return root / f"lr{learning_rate:g}" / f"seed{seed}" / pilot.case_name(fraction, "legacy_masked50")


def run_seed(protocol_path: Path, seed: int) -> None:
    frozen = verify(protocol_path)
    p = frozen["protocol"]
    if seed not in p["seeds"]:
        raise ValueError("Seed not in frozen protocol")
    root = _root(p)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"lr_sweep_seed{seed}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot.configure()
        if not torch.cuda.is_available():
            raise RuntimeError("N2b requires CUDA")
        base = dict(p)
        base["seed"] = seed
        base["split_manifest"] = p["split_manifests"][str(seed)]
        base["output_dir"] = str(root)
        data = pilot.load_train_validation(base)
        for learning_rate in p["learning_rates"]:
            for fraction in p["train_fractions"]:
                directory = _case_dir(root, learning_rate, seed, fraction)
                if (directory / "result.json").exists():
                    continue
                cp = dict(base)
                cp["learning_rate"] = float(learning_rate)
                cp["suite"] = f"lr{learning_rate:g}/seed{seed}"
                directory.parent.mkdir(parents=True, exist_ok=True)
                try:
                    pilot.run_case(cp, frozen, fraction, "legacy_masked50", data, time.monotonic() + 3600)
                except BaseException as error:
                    directory.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(
                        directory / "failure.json",
                        {
                            "learning_rate": learning_rate,
                            "seed": seed,
                            "fraction": fraction,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "test_evaluated": False,
                        },
                    )
                    print(
                        f"N2b lr={learning_rate:g} seed={seed} fraction={fraction} FAILED {error}",
                        flush=True,
                    )
                    raise


def audit(protocol_path: Path) -> dict[str, Any]:
    frozen = verify(protocol_path)
    p = frozen["protocol"]
    root = _root(p)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    prediction_rows = 0
    for seed in p["seeds"]:
        base = dict(p)
        base["seed"] = seed
        base["split_manifest"] = p["split_manifests"][str(seed)]
        data = pilot.load_train_validation(base)
        validation = data["validation"]
        expected = {
            fraction: stratified_train_subset_indices(
                data["train"]["y"], data["train"]["ids"], fraction, seed
            )
            for fraction in p["train_fractions"]
        }
        for learning_rate in p["learning_rates"]:
            for fraction in p["train_fractions"]:
                directory = _case_dir(root, learning_rate, seed, fraction)
                if (directory / "failure.json").exists():
                    failures.append(_read(directory / "failure.json"))
                    continue
                result = _read(directory / "result.json")
                history = _read(directory / "history.json")
                if result["test_evaluated"] or result["protocol_sha256"] != frozen["protocol_sha256"]:
                    raise ValueError("Unexpected result protocol")
                if len(history) != 50 or [row["epoch"] for row in history] != list(range(1, 51)):
                    raise ValueError("Incomplete history")
                if not np.array_equal(
                    np.load(directory / "train_ids.npy"), data["train"]["ids"][expected[fraction]]
                ):
                    raise ValueError("Training subset mismatch")
                selected_epoch, selected_f1 = n2._selected_epoch(history, p["early_stopping_min_delta"])
                if selected_epoch != result["selected_epoch"] or abs(selected_f1 - result["selected_macro_f1"]) > 1e-12:
                    raise ValueError("Selection mismatch")
                if max(row["macro_f1"] for row in history) != result["peak_macro_f1"]:
                    raise ValueError("Peak mismatch")
                for name, digest in result["output_sha256"].items():
                    if sha256_file(directory / name) != digest:
                        raise ValueError(f"Changed output: {directory / name}")
                for suffix, score in (
                    ("selected", result["selected_macro_f1"]),
                    ("peak", result["peak_macro_f1"]),
                    ("endpoint", history[-1]["macro_f1"]),
                ):
                    prediction_rows += n2._read_predictions(
                        directory / f"validation_{suffix}.csv", validation, score
                    )
                rows.append(
                    {
                        "seed": seed,
                        "fraction": fraction,
                        "learning_rate": learning_rate,
                        "selected_macro_f1": result["selected_macro_f1"],
                        "selected_epoch": result["selected_epoch"],
                        "peak_macro_f1": result["peak_macro_f1"],
                        "endpoint_macro_f1": history[-1]["macro_f1"],
                        "training_seconds": result["training_seconds"],
                    }
                )
    aggregate = []
    for fraction in p["train_fractions"]:
        for learning_rate in p["learning_rates"]:
            subset = [
                row
                for row in rows
                if row["fraction"] == fraction and row["learning_rate"] == learning_rate
            ]
            values = [100 * row["selected_macro_f1"] for row in subset]
            aggregate.append(
                {
                    "fraction": fraction,
                    "learning_rate": learning_rate,
                    "n": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "per_seed": values,
                }
            )
    baseline = {
        (row["fraction"], row["learning_rate"]): row["mean"]
        for row in aggregate
        if row["learning_rate"] == 0.001
    }
    for row in aggregate:
        base = baseline.get((row["fraction"], 0.001))
        row["delta_vs_lr_1e3_pp"] = (
            None if base is None or row["mean"] is None else row["mean"] - base
        )
    summary = {
        "audit_status": "passed" if not failures else "passed_with_failures",
        "recorded_at": _now(),
        "test_evaluated": False,
        "prior_test_results_known": True,
        "conditions": len(rows),
        "per_condition": rows,
        "failures": failures,
        "prediction_rows_audited": prediction_rows,
        "aggregate": aggregate,
        "limitations": (
            "Validation-only LR sensitivity; structure/data/epochs fixed; four overlapping splits; "
            "reused validation; no test or external validation. This is a fairness check, not a "
            "hyperparameter search for a headline result."
        ),
    }
    atomic_write_json(root / "summary.json", summary)
    _report(p, summary)
    return summary


def _report(p: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# N2b：legacy masked TCN 学习率敏感性（1%/5%，validation-only）",
        "",
        "固定 E05C 结构、raw 序列、1024 padding、50 轮、batch 32 和选模规则，只改变 AdamW 学习率；",
        "用于判断 N2 的 1% 训练坍塌是否由固定 LR 导致。没有 test。",
        "",
        "| 标签预算 | LR | Macro-F1 均值±SD % | 相对 LR 1e-3 | 逐 seed |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in summary["aggregate"]:
        per_seed = ", ".join(f"{value:.2f}" for value in row["per_seed"]) or "—"
        delta = "—" if row["delta_vs_lr_1e3_pp"] is None else f"{row['delta_vs_lr_1e3_pp']:+.2f}pp"
        sd = "—" if row["sample_sd"] is None else f"{row['sample_sd']:.2f}"
        mean_text = "—" if row["mean"] is None else f"{row['mean']:.2f}"
        lines.append(
            f"| {row['fraction']*100:g}% | {row['learning_rate']:g} | "
            f"{mean_text} ± {sd} | {delta} | {per_seed} |"
        )
    lines += [
        "",
        "## 逐 seed 选择 epoch",
        "",
        "| Seed | 预算 | LR 3e-4 | LR 1e-3 | LR 3e-3 |",
        "|---:|---:|---:|---:|---:|",
    ]
    condition_lookup = {
        (row["seed"], row["fraction"], row["learning_rate"]): row
        for row in summary["per_condition"]
    }
    for seed in p["seeds"]:
        for fraction in p["train_fractions"]:
            cells = []
            for learning_rate in p["learning_rates"]:
                row = condition_lookup.get((seed, fraction, learning_rate))
                cells.append("—" if row is None else f"{100*row['selected_macro_f1']:.2f}@{row['selected_epoch']}")
            lines.append(f"| {seed} | {fraction*100:g}% | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 判定与边界",
        "",
        "- 若某个 LR 在 1% 明显稳定并接近强特征基线，则 N2 的固定 LR 坍塌是优化/预算问题，不应当作架构上限。",
        "- 若所有 LR 仍远低于 1% 的 DYN+HGB / stack，则应把低标签优势归给强特征系统，而不是 TCN。",
        "- 四个 seed 的 split 相互重叠，validation 已反复使用；不能声称独立外部确认。",
        f"- 条件数 {summary['conditions']}，失败 {len(summary['failures'])}，审计预测行 {summary['prediction_rows_audited']}。",
        "",
        f"- 远端根：`{_root(p)}`；协议：`{DEFAULT_PROTOCOL}`；入口：`python -m tcn_moment.run_n2b_lr_sweep`。",
    ]
    Path("docs/experiment_records/n2b_lr_sensitivity_results_20261020.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "run", "audit"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol)
    elif args.action == "verify":
        verify(args.protocol)
    elif args.action == "run":
        if args.seed is None:
            parser.error("run requires --seed")
        run_seed(args.protocol, args.seed)
    else:
        audit(args.protocol)


if __name__ == "__main__":
    main()
