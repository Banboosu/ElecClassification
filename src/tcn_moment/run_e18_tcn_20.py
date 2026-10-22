"""E18: legacy masked TCN at 20% training labels (for the label-efficiency curve).

This extends E05C downwards without changing the completed 10%/40%/100% suite.
The runner reuses the frozen E05B/E05C training implementation (``tcn_legacy_pilot``)
but a new protocol and output directory. Validation only; no test is loaded or scored.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import math
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.data import stratified_train_subset_indices
from tcn_moment.io_utils import atomic_write_json, sha256_file

DEFAULT_PROTOCOL = Path("configs/experiments/improvements/e18_tcn_20_protocol.json")
DEFAULT_E05C_SUMMARY = Path(
    "artifacts/improvements/e05c_tcn_replication/e05c_tcn_replication_20261019_v1/summary.json"
)
DEFAULT_E08_SUMMARY = Path(
    "artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1/analysis/summary.json"
)
DEFAULT_E08_SUITE = Path(
    "artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1"
)
PACKAGES = ("torch", "numpy", "pandas", "scikit-learn")
CONDITION = "legacy_masked50"
FRACTIONS = (0.2,)
SEEDS = (42, 44, 45, 46)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root(p: dict[str, Any]) -> Path:
    return Path(p["output_dir"]) / p["suite"]


def _case(fraction: float) -> str:
    return pilot.case_name(fraction, CONDITION)


def _validate_protocol(p: dict[str, Any]) -> None:
    if p.get("experiment_id") != "E18":
        raise ValueError("Unexpected experiment id")
    if p.get("seeds") != list(SEEDS) or tuple(p.get("train_fractions", ())) != FRACTIONS:
        raise ValueError("E18 fixes seeds 42/44/45/46 and fraction 0.2")
    if p.get("condition") != CONDITION:
        raise ValueError("E18 uses only legacy_masked50")
    if (
        p.get("classes") != ["0", "1", "2"]
        or p.get("epochs") != 50
        or p.get("batch_size") != 32
        or p.get("evaluation_batch_size") != 64
        or p.get("normalization") != "none"
        or p.get("max_length") != 1024
        or p.get("optimizer") != "AdamW"
        or p.get("precision") != "fp32_tf32_disabled"
    ):
        raise ValueError("Unexpected training scope")
    if p.get("test_evaluated") or p.get("automatic_expansion") or p.get("automatic_shutdown"):
        raise ValueError("E18 must remain validation-only and non-expanding")
    if p.get("train_count") != 24569 or p.get("validation_count") != 3510:
        raise ValueError("Unexpected split sizes")
    if set(p["split_manifests"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("Missing split manifests")


def _load_protocol(path: Path) -> dict[str, Any]:
    p = _read(path)
    _validate_protocol(p)
    return p


def _source_files(protocol_path: Path) -> list[Path]:
    module_dir = Path(__file__).resolve().parent
    files = [
        Path(__file__).resolve(),
        module_dir / "tcn_legacy_pilot.py",
        module_dir / "data.py",
        module_dir / "train_tcn.py",
        module_dir / "training_utils.py",
        module_dir / "audit_protocol.py",
        module_dir / "io_utils.py",
        module_dir / "config.py",
        module_dir / "experiment.py",
        module_dir / "metrics.py",
        Path("tests/test_n2_tcn_low_label.py"),
        Path("scripts/run_n2_remote.sh"),
        protocol_path,
        Path("legacy/改进TCN.py"),
    ]
    return files


def freeze(protocol_path: Path) -> dict[str, Any]:
    p = _load_protocol(protocol_path)
    root = _root(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        return verify(protocol_path)
    if list(root.glob("fraction*")):
        raise ValueError("Freeze must precede training")
    dataset = Path(p["dataset"])
    if sha256_file(dataset) != p["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    for seed in SEEDS:
        manifest = _read(Path(p["split_manifests"][str(seed)]))
        if (
            manifest["protocol"]["random_state"] != seed
            or manifest["protocol"]["dataset_sha256"] != p["dataset_sha256"]
            or manifest["counts"]["train"] != p["train_count"]
            or manifest["counts"]["validation"] != p["validation_count"]
        ):
            raise ValueError(f"Split manifest mismatch for seed {seed}")
    checks = pilot.synthetic_check()
    pilot.configure()
    files = set(_source_files(protocol_path))
    files.update(Path(p["split_manifests"][str(seed)]) for seed in SEEDS)
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
    for seed in SEEDS:
        path = Path(p["split_manifests"][str(seed)])
        shutil.copy2(path, root / f"split_manifest_seed{seed}.json")
    atomic_write_json(root / "frozen_protocol.json", frozen)
    print(f"Froze E18: {root / 'frozen_protocol.json'}", flush=True)
    return frozen


def verify(protocol_path: Path) -> dict[str, Any]:
    p = _load_protocol(protocol_path)
    frozen = _read(_root(p) / "frozen_protocol.json")
    if frozen["protocol"] != p or frozen["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Frozen protocol changed")
    for file, expected in frozen["file_sha256"].items():
        if sha256_file(Path(file)) != expected:
            raise ValueError(f"Frozen input changed: {file}")
    if frozen["packages"] != {name: importlib.metadata.version(name) for name in PACKAGES}:
        raise ValueError("Dependency versions changed")
    return frozen


def _seed_protocol(p: dict[str, Any], seed: int) -> dict[str, Any]:
    result = dict(p)
    result["seed"] = seed
    result["split_manifest"] = p["split_manifests"][str(seed)]
    # Match E05C's per-seed layout: output_dir/suite/seed<seed>/fraction...
    result["output_dir"] = str(_root(p))
    result["suite"] = f"seed{seed}"
    return result


def _selected_epoch(history: list[dict[str, Any]], min_delta: float) -> tuple[int, float]:
    selected, selected_epoch = -1.0, 0
    for row in history:
        if row["macro_f1"] > selected + min_delta:
            selected, selected_epoch = row["macro_f1"], row["epoch"]
    return selected_epoch, selected


def _read_predictions(
    path: Path,
    validation: dict[str, Any],
    expected_macro_f1: float,
) -> int:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    probabilities = frame[["p0", "p1", "p2"]].to_numpy(dtype=np.float64)
    if (
        not np.array_equal(frame["sample_id"].to_numpy(), validation["ids"])
        or not np.array_equal(frame["y_true"].to_numpy(), validation["y"])
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(1), 1, atol=2e-7, rtol=0)
        or not np.array_equal(probabilities.argmax(1), frame["y_pred"].to_numpy())
    ):
        raise ValueError(f"Validation prediction mismatch: {path}")
    from tcn_moment.audit_protocol import metrics_from_confusion

    confusion = np.bincount(
        3 * validation["y"] + frame["y_pred"].to_numpy(), minlength=9
    ).reshape(3, 3).tolist()
    if abs(metrics_from_confusion(confusion)["macro_f1"] - expected_macro_f1) > 1e-12:
        raise ValueError(f"Independent Macro-F1 mismatch: {path}")
    return len(frame)


def audit(protocol_path: Path) -> dict[str, Any]:
    frozen = verify(protocol_path)
    p = frozen["protocol"]
    root = _root(p)
    data = {}
    rows: list[dict[str, Any]] = []
    prediction_rows = 0
    for seed in SEEDS:
        cp = _seed_protocol(p, seed)
        data[seed] = pilot.load_train_validation(cp)
        validation = data[seed]["validation"]
        for fraction in FRACTIONS:
            directory = pilot.root(cp) / _case(fraction)
            result = _read(directory / "result.json")
            history = _read(directory / "history.json")
            if result["test_evaluated"] or result["protocol_sha256"] != frozen["protocol_sha256"]:
                raise ValueError("Unexpected result protocol")
            if result["fraction"] != fraction or result["condition"] != CONDITION:
                raise ValueError("Result identity mismatch")
            if len(history) != p["epochs"] or [row["epoch"] for row in history] != list(range(1, p["epochs"] + 1)):
                raise ValueError("Incomplete epoch history")
            expected = stratified_train_subset_indices(
                data[seed]["train"]["y"], data[seed]["train"]["ids"], fraction, seed
            )
            if not np.array_equal(np.load(directory / "train_ids.npy"), data[seed]["train"]["ids"][expected]):
                raise ValueError("Training subset mismatch")
            if result["optimizer_steps"] != len(history) * math.ceil(len(expected) / p["batch_size"]):
                raise ValueError("Optimizer step count mismatch")
            selected_epoch, selected_f1 = _selected_epoch(history, p["early_stopping_min_delta"])
            if selected_epoch != result["selected_epoch"] or abs(selected_f1 - result["selected_macro_f1"]) > 1e-12:
                raise ValueError("Selection rule mismatch")
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
                prediction_rows += _read_predictions(
                    directory / f"validation_{suffix}.csv", validation, score
                )
            rows.append(
                {
                    "seed": seed,
                    "fraction": fraction,
                    "condition": CONDITION,
                    "train_count": result["train_count"],
                    "epochs": result["epochs"],
                    "selected_epoch": result["selected_epoch"],
                    "selected_macro_f1": result["selected_macro_f1"],
                    "peak_macro_f1": result["peak_macro_f1"],
                    "endpoint_macro_f1": history[-1]["macro_f1"],
                    "training_seconds": result["training_seconds"],
                    "total_parameters": result["total_parameters"],
                }
            )
    aggregate = []
    for fraction in FRACTIONS:
        subset = [row for row in rows if row["fraction"] == fraction]
        values = [row["selected_macro_f1"] * 100 for row in subset]
        aggregate.append(
            {
                "fraction": fraction,
                "n_seeds": len(subset),
                "macro_f1_mean": float(np.mean(values)),
                "macro_f1_sd": float(np.std(values, ddof=1)),
                "per_seed": values,
                "n_train": sorted({row["train_count"] for row in subset}),
            }
        )
    context = _context(rows)
    summary = {
        "audit_status": "passed",
        "recorded_at": _now(),
        "test_evaluated": False,
        "prior_test_results_known": True,
        "conditions": len(rows),
        "epochs_audited": sum(row["epochs"] for row in rows),
        "prediction_rows_audited": prediction_rows,
        "per_condition": rows,
        "aggregate": aggregate,
        "context": context,
        "limitations": (
            "Validation-only extension of E05C; seeds share overlapping splits; reused validation; "
            "no test and no independent external validation. TCN and E08 models have different "
            "training/selection budgets, so differences are pipeline comparisons, not single-factor effects."
        ),
    }
    atomic_write_json(root / "summary.json", summary)
    atomic_write_json(root / "suite_status.json", {"status": "completed", "updated_at": _now(), "test_evaluated": False})
    _report(p, summary)
    return summary


def _context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"e05c": None, "e08": None, "combined": []}
    e05c: dict[tuple[int, float], float] = {}
    if DEFAULT_E05C_SUMMARY.exists():
        saved = _read(DEFAULT_E05C_SUMMARY)
        for record in saved.get("per_condition", []):
            if record.get("condition") != CONDITION:
                continue
            e05c[(int(record["seed"]), float(record["fraction"]))] = float(record["selected"]["macro_f1"])
        result["e05c"] = {
            "path": str(DEFAULT_E05C_SUMMARY),
            "values": {f"{seed}:{fraction:g}": value for (seed, fraction), value in sorted(e05c.items())},
        }
    e08: dict[tuple[int, str], dict[str, float]] = {}
    if DEFAULT_E08_SUITE.exists():
        for seed in SEEDS:
            path = DEFAULT_E08_SUITE / f"seed{seed}" / "seed_results.json"
            if not path.exists():
                continue
            saved = _read(path)
            for record in saved["conditions"]:
                e08.setdefault((seed, record["tag"]), {})[record["method"]] = float(
                    record["validation"]["macro_f1"]
                )
        result["e08"] = {"suite": str(DEFAULT_E08_SUITE), "seeds": list(SEEDS)}
    tags = {0.01: "1p", 0.05: "5p", 0.10: "10p", 0.40: "40p", 1.00: "100p"}
    for fraction in (0.01, 0.05, 0.10, 0.40, 1.00):
        tcn_by_seed: dict[int, float] = {
            int(row["seed"]): float(row["selected_macro_f1"])
            for row in rows
            if row["fraction"] == fraction
        }
        tcn_by_seed.update(
            {
                seed: value
                for (seed, value_fraction), value in e05c.items()
                if value_fraction == fraction
            }
        )
        tag = tags[fraction]
        dyn_by_seed = {
            seed: e08[(seed, tag)]["dyn_hgb"]
            for seed in SEEDS
            if (seed, tag) in e08 and "dyn_hgb" in e08[(seed, tag)]
        }
        stack_by_seed = {
            seed: e08[(seed, tag)]["stack"]
            for seed in SEEDS
            if (seed, tag) in e08 and "stack" in e08[(seed, tag)]
        }
        common_dyn = sorted(set(tcn_by_seed) & set(dyn_by_seed))
        common_stack = sorted(set(tcn_by_seed) & set(stack_by_seed))
        diff_dyn = [100 * (tcn_by_seed[seed] - dyn_by_seed[seed]) for seed in common_dyn]
        diff_stack = [100 * (tcn_by_seed[seed] - stack_by_seed[seed]) for seed in common_stack]
        tcn_values = [100 * value for value in tcn_by_seed.values()]
        dyn_values = [100 * value for value in dyn_by_seed.values()]
        stack_values = [100 * value for value in stack_by_seed.values()]

        def _summary(values: list[float]) -> tuple[float | None, float | None, int]:
            if not values:
                return None, None, 0
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
            return mean, sd, int(np.sum(np.asarray(values) > 0))

        dyn_diff_mean, dyn_diff_sd, dyn_diff_positive = _summary(diff_dyn)
        stack_diff_mean, stack_diff_sd, stack_diff_positive = _summary(diff_stack)
        combined = {
            "fraction": fraction,
            "tcn_n": len(tcn_values),
            "tcn_mean": float(np.mean(tcn_values)) if tcn_values else None,
            "tcn_sd": float(np.std(tcn_values, ddof=1)) if len(tcn_values) > 1 else None,
            "dyn_hgb_mean": float(np.mean(dyn_values)) if dyn_values else None,
            "dyn_hgb_sd": float(np.std(dyn_values, ddof=1)) if len(dyn_values) > 1 else None,
            "stack_mean": float(np.mean(stack_values)) if stack_values else None,
            "stack_sd": float(np.std(stack_values, ddof=1)) if len(stack_values) > 1 else None,
            "tcn_minus_dyn_mean": dyn_diff_mean,
            "tcn_minus_dyn_sd": dyn_diff_sd,
            "tcn_minus_dyn_positive": dyn_diff_positive,
            "tcn_minus_stack_mean": stack_diff_mean,
            "tcn_minus_stack_sd": stack_diff_sd,
            "tcn_minus_stack_positive": stack_diff_positive,
        }
        result["combined"].append(combined)
    return result


def _report(p: dict[str, Any], summary: dict[str, Any]) -> None:
    rows = summary["per_condition"]
    lines = [
        "# E18：legacy masked TCN 的 20% 标签点（2026-10-20）",
        "",
        "**validation-only；无 test。** 为重绘 MOMENT vs TCN 的标签效率曲线，补 20% 标签点；",
        "结构与训练协议复用 E05C/N2 的 legacy masked TCN、raw 序列、1024 padding、AdamW/FP32、50 轮和验证集选模。",
        "",
        "| Seed | 20% Macro-F1 % | 选择 epoch |",
        "|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {100*row['selected_macro_f1']:.4f} | {row['selected_epoch']} |"
        )
    aggregate = summary["aggregate"][0]
    lines += [
        "",
        f"- 均值：**{aggregate['macro_f1_mean']:.2f} ± {aggregate['macro_f1_sd']:.2f}%**（{aggregate['n_seeds']} seeds）",
        f"- 条件数：{summary['conditions']}；审计 epoch：{summary['epochs_audited']}；预测行：{summary['prediction_rows_audited']}。",
        "- 四 seed split 重叠、validation 复用；不是独立盲测；不用于更新 test。",
        "",
        "## 证据",
        "",
        f"- 远端根：`{_root(p)}`",
        "- 汇总：`summary.json`；协议：`configs/experiments/improvements/e18_tcn_20_protocol.json`",
    ]
    Path("docs/experiment_records/e18_tcn_20_results_20261020.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(protocol_path: Path) -> None:
    frozen = verify(protocol_path)
    p = frozen["protocol"]
    root = _root(p)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "suite.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (root / "summary.json").exists():
                audit(protocol_path)
                return
            pilot.configure()
            if not torch.cuda.is_available():
                raise RuntimeError("E18 requires CUDA")
            started_at = _now()
            atomic_write_json(
                root / "suite_status.json",
                {"status": "running", "started_at": started_at, "updated_at": started_at, "test_evaluated": False},
            )
            deadline = time.monotonic() + p["wall_time_limit_hours"] * 3600
            for seed in SEEDS:
                cp = _seed_protocol(p, seed)
                pilot.root(cp).mkdir(parents=True, exist_ok=True)
                data = pilot.load_train_validation(cp)
                for fraction in FRACTIONS:
                    pilot.run_case(cp, frozen, fraction, CONDITION, data, deadline)
            audit(protocol_path)
        except BaseException as error:
            previous = _read(root / "suite_status.json") if (root / "suite_status.json").exists() else {}
            atomic_write_json(
                root / "suite_status.json",
                {
                    **previous,
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "updated_at": _now(),
                    "test_evaluated": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "run", "audit"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol)
    elif args.action == "verify":
        verify(args.protocol)
    elif args.action == "run":
        run(args.protocol)
    else:
        audit(args.protocol)


if __name__ == "__main__":
    main()
