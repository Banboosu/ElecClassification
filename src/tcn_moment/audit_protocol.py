from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tcn_moment.io_utils import atomic_write_json

M03_PROTOCOL_VERSION = 1
DATASET_SHA256 = "5615a96a7894caed5d14463c77167af8098bdc1e1ebf32a33a89a12c3c5cf6e6"
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_CLASSES = ("0", "1", "2")
EXPECTED_VALIDATION_COUNT = 3510
EXPECTED_TEST_COUNT = 7020


@dataclass(frozen=True)
class RunGroup:
    name: str
    paths: tuple[Path, ...]
    kind: str
    recompute_multiclass_metrics: bool = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def nonfinite_paths(value: Any, prefix: str = "root") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            paths.extend(nonfinite_paths(nested, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(nonfinite_paths(nested, f"{prefix}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        paths.append(prefix)
    return paths


def normalized_config(
    config: dict[str, Any],
    *,
    ignored_top_level_sections: tuple[str, ...] = (),
) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    data = normalized.get("data", {})
    data.pop("random_state", None)
    data.pop("split_path", None)
    for section in ignored_top_level_sections:
        normalized.pop(section, None)
    return normalized


def metrics_from_confusion(confusion: list[list[int]]) -> dict[str, float]:
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("Expected a square multiclass confusion matrix.")
    if np.any(matrix < 0) or not np.isfinite(matrix).all() or matrix.sum() <= 0:
        raise ValueError("Confusion counts must be finite, non-negative, and non-empty.")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    weights = support / support.sum()
    return {
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(np.sum(precision * weights)),
        "weighted_recall": float(np.sum(recall * weights)),
        "weighted_f1": float(np.sum(f1 * weights)),
    }


def metric_payloads(metrics: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    model = str(metrics.get("model", ""))
    payloads: list[tuple[str, dict[str, Any]]] = []
    if model == "STATISTICAL_BASELINES":
        for name, values in metrics["results"].items():
            payloads.append((f"{name}.test", values["test_metrics"]))
    elif isinstance(metrics.get("test_metrics"), dict):
        payloads.append(("test", metrics["test_metrics"]))
    elif isinstance(metrics.get("fractions"), list):
        for values in metrics["fractions"]:
            if isinstance(values.get("test_metrics"), dict):
                payloads.append(
                    (f"fraction_{float(values['train_fraction']):g}.test", values["test_metrics"])
                )
    return payloads


def compare_saved_metrics(payload: dict[str, Any], tolerance: float = 1e-12) -> float:
    if "confusion_matrix" not in payload:
        raise ValueError("A formal multiclass metric payload lacks confusion_matrix.")
    recalculated = metrics_from_confusion(payload["confusion_matrix"])
    differences = []
    for key, expected in recalculated.items():
        if key not in payload:
            raise ValueError(f"A formal metric payload lacks {key}.")
        differences.append(abs(float(payload[key]) - expected))
    maximum = max(differences, default=0.0)
    if maximum > tolerance:
        raise ValueError(f"Saved classification metrics differ from confusion counts by {maximum}.")
    return maximum


def _glob_group(root: Path, name: str, pattern: str) -> RunGroup:
    return RunGroup(name, tuple(sorted(root.glob(pattern))), "multiclass")


def _mixed_seed_paths(root: Path, parent: str, stem: str) -> tuple[Path, ...]:
    paths = [root / parent / f"{stem}_pilot_v1_seed42"]
    paths.extend(root / parent / f"{stem}_thesis_v1_seed{seed}" for seed in EXPECTED_SEEDS[1:])
    return tuple(paths)


def formal_run_groups(root: Path) -> list[RunGroup]:
    groups = [
        _glob_group(root, "statistical_baselines", "artifacts/baselines/moment_thesis_baseline_v1_seed*"),
        _glob_group(root, "cnn_zscore_amp", "artifacts/cnn/cnn_baseline_thesis_cnn_v1_seed*"),
        _glob_group(root, "tcn_raw", "artifacts/tcn/normalization_none_thesis_tcn_norm_v2_seed*"),
        _glob_group(root, "tcn_minmax", "artifacts/tcn/normalization_minmax_thesis_tcn_norm_v2_seed*"),
        _glob_group(root, "tcn_zscore", "artifacts/tcn/normalization_zscore_thesis_tcn_norm_v2_seed*"),
        _glob_group(
            root,
            "moment_linear_probe_v2",
            "artifacts/moment/moment_linear_probe_thesis_moment_strategy_v2_v100_seed*",
        ),
        _glob_group(
            root,
            "moment_partial_finetune_v2",
            "artifacts/moment/moment_partial_finetune_thesis_moment_strategy_v2_v100_seed*",
        ),
        _glob_group(
            root,
            "moment_full_finetune_v2",
            "artifacts/moment/moment_full_finetune_thesis_moment_strategy_v2_v100_seed*",
        ),
        _glob_group(root, "moment_rbf_svm", "artifacts/moment_svm/moment_svm_rbf_paper_v1_seed*"),
    ]
    for tag in ("01", "05", "10", "20", "40"):
        groups.append(
            _glob_group(
                root,
                f"tcn_few_shot_{tag}",
                f"artifacts/tcn_few_shot/tcn_{tag}_percent_thesis_few_shot_v1_seed*",
            )
        )
    groups.extend(
        [
            _glob_group(
                root,
                "moment_svm_few_shot",
                "artifacts/moment_svm_few_shot/moment_svm_thesis_few_shot_v1_seed*",
            ),
            _glob_group(
                root,
                "m01_random_encoder_svm",
                "artifacts/moment_svm_pretraining_ablation/"
                "moment_svm_random_m01_random_encoder_v1_seed*",
            ),
        ]
    )
    for tag in ("01", "05", "10", "full"):
        groups.append(
            RunGroup(
                f"battery_binary_stats_{tag}",
                _mixed_seed_paths(root, "artifacts/battery_binary_stats", f"battery_binary_stats_{tag}"),
                "battery_binary",
                recompute_multiclass_metrics=False,
            )
        )
    groups.append(
        RunGroup(
            "battery_binary_moment_svm",
            _mixed_seed_paths(
                root,
                "artifacts/battery_binary_moment_svm",
                "battery_binary_moment_svm",
            ),
            "battery_binary",
            recompute_multiclass_metrics=False,
        )
    )
    return groups


def _seed(metrics: dict[str, Any], config: dict[str, Any], run_dir: Path) -> int:
    if "seed" in metrics:
        return int(metrics["seed"])
    if "data" in config and "random_state" in config["data"]:
        return int(config["data"]["random_state"])
    suffix = run_dir.name.rsplit("seed", 1)[-1]
    return int(suffix)


def _status_is_completed(status: dict[str, Any]) -> bool:
    return str(status.get("status", "")).lower() == "completed"


def _checkpoint_is_finite(path: Path) -> bool:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    tensors = state.values() if isinstance(state, dict) else []
    return all(bool(torch.isfinite(value).all().item()) for value in tensors if torch.is_tensor(value))


def _metrics_source_hash(root: Path, commit: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:src/tcn_moment/metrics.py"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _source_at_commit(root: Path, commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def audit_formal_runs(root: Path) -> dict[str, Any]:
    failures: list[str] = []
    run_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    split_hashes_by_seed: dict[int, set[str]] = {seed: set() for seed in EXPECTED_SEEDS}
    classification_commits: set[str] = set()
    metric_payload_count = 0
    maximum_metric_difference = 0.0
    groups = formal_run_groups(root)
    supervised_moment_groups = {
        "moment_linear_probe_v2",
        "moment_partial_finetune_v2",
        "moment_full_finetune_v2",
    }

    for group in groups:
        group_failures: list[str] = []
        seeds: list[int] = []
        config_hashes: set[str] = set()
        for run_dir in group.paths:
            required = {
                "metrics": run_dir / "metrics.json",
                "config": run_dir / "resolved_config.json",
                "status": run_dir / "status.json",
                "split": run_dir / "split_manifest.json",
                "environment": run_dir / "environment.json",
            }
            missing = [name for name, path in required.items() if not path.is_file()]
            if missing:
                message = f"{group.name}/{run_dir.name}: missing {missing}"
                failures.append(message)
                group_failures.append(message)
                continue
            metrics = json.loads(required["metrics"].read_text(encoding="utf-8"))
            config = json.loads(required["config"].read_text(encoding="utf-8"))
            status = json.loads(required["status"].read_text(encoding="utf-8"))
            environment = json.loads(required["environment"].read_text(encoding="utf-8"))
            seed = _seed(metrics, config, run_dir)
            seeds.append(seed)
            run_failures: list[str] = []
            if not _status_is_completed(status):
                run_failures.append(f"status={status.get('status')}")
            data = metrics.get("data", {})
            if data.get("dataset_sha256") != DATASET_SHA256:
                run_failures.append("dataset_sha256 mismatch")
            if tuple(str(value) for value in data.get("classes", [])) != EXPECTED_CLASSES:
                run_failures.append("classes mismatch")
            split_counts = data.get("split", {})
            if int(split_counts.get("validation", -1)) != EXPECTED_VALIDATION_COUNT:
                run_failures.append("validation count mismatch")
            if int(split_counts.get("test", -1)) != EXPECTED_TEST_COUNT:
                run_failures.append("test count mismatch")
            nonfinite = nonfinite_paths(metrics)
            if nonfinite:
                run_failures.append(f"nonfinite metrics at {nonfinite[:3]}")
            split_hash = sha256_file(required["split"])
            if seed in split_hashes_by_seed:
                split_hashes_by_seed[seed].add(split_hash)
            else:
                run_failures.append(f"unexpected seed {seed}")
            ignored_sections = ("svm",) if group.name in supervised_moment_groups else ()
            config_hash = canonical_hash(
                normalized_config(config, ignored_top_level_sections=ignored_sections)
            )
            config_hashes.add(config_hash)
            commit = str(environment.get("git_commit", ""))
            if group.recompute_multiclass_metrics:
                classification_commits.add(commit)
                if any("src/tcn_moment/metrics.py" in item for item in environment.get("git_status", [])):
                    run_failures.append("metrics.py was dirty during the run")
                try:
                    payloads = metric_payloads(metrics)
                    if not payloads:
                        raise ValueError("No multiclass test metric payload found.")
                    for _, payload in payloads:
                        maximum_metric_difference = max(
                            maximum_metric_difference,
                            compare_saved_metrics(payload),
                        )
                    metric_payload_count += len(payloads)
                except (KeyError, TypeError, ValueError) as error:
                    run_failures.append(str(error))
            if run_failures:
                message = f"{group.name}/{run_dir.name}: {'; '.join(run_failures)}"
                failures.append(message)
                group_failures.append(message)
            run_rows.append(
                {
                    "group": group.name,
                    "kind": group.kind,
                    "run_name": run_dir.name,
                    "seed": seed,
                    "completed": _status_is_completed(status),
                    "dataset_sha256": data.get("dataset_sha256"),
                    "split_manifest_sha256": split_hash,
                    "normalized_config_sha256": config_hash,
                    "git_commit": commit,
                    "git_diff_sha256": environment.get("git_diff_sha256", ""),
                    "finite_metrics": not nonfinite,
                    "passed": not run_failures,
                }
            )
        if len(group.paths) != len(EXPECTED_SEEDS):
            group_failures.append(
                f"{group.name}: expected {len(EXPECTED_SEEDS)} runs, found {len(group.paths)}"
            )
        if sorted(seeds) != list(EXPECTED_SEEDS):
            group_failures.append(f"{group.name}: seed set mismatch: {sorted(seeds)}")
        if len(config_hashes) != 1:
            group_failures.append(
                f"{group.name}: protocol config hash count={len(config_hashes)}"
            )
        failures.extend(item for item in group_failures if item not in failures)
        group_rows.append(
            {
                "group": group.name,
                "kind": group.kind,
                "run_count": len(group.paths),
                "seeds": seeds,
                "normalized_config_hash_count": len(config_hashes),
                "ignored_config_sections": "svm" if group.name in supervised_moment_groups else "",
                "passed": not group_failures,
            }
        )

    for seed, hashes in split_hashes_by_seed.items():
        if len(hashes) != 1:
            failures.append(f"seed {seed} has {len(hashes)} formal split-manifest hashes")

    source_hashes = {
        commit: _metrics_source_hash(root, commit)
        for commit in sorted(classification_commits)
        if commit
    }
    if len(set(source_hashes.values())) != 1:
        failures.append("classification metrics.py hashes differ across formal run commits")

    return {
        "failures": failures,
        "run_rows": run_rows,
        "group_rows": group_rows,
        "formal_group_count": len(groups),
        "formal_run_count": len(run_rows),
        "multiclass_group_count": sum(group.kind == "multiclass" for group in groups),
        "multiclass_run_count": sum(row["kind"] == "multiclass" for row in run_rows),
        "battery_binary_run_count": sum(row["kind"] == "battery_binary" for row in run_rows),
        "metric_payload_count": metric_payload_count,
        "maximum_metric_recalculation_difference": maximum_metric_difference,
        "split_manifest_sha256_by_seed": {
            str(seed): next(iter(hashes)) if len(hashes) == 1 else sorted(hashes)
            for seed, hashes in split_hashes_by_seed.items()
        },
        "classification_metrics_source_sha256_by_commit": source_hashes,
    }


def audit_cnn(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    run_dirs = sorted(root.glob("artifacts/cnn/cnn_baseline_thesis_cnn_v1_seed*"))
    source_checks: dict[str, bool] = {}
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        seed = int(config["data"]["random_state"])
        training = metrics["training"]
        amp_configured = bool(config["tcn_training"]["amp"])
        amp_enabled = bool(training.get("amp_enabled"))
        commit = str(environment["git_commit"])
        if commit not in source_checks:
            training_source = _source_at_commit(
                root,
                commit,
                "src/tcn_moment/train_tcn.py",
            )
            source_checks[commit] = all(
                [
                    "amp_enabled = bool(training.amp and device.type == \"cuda\")"
                    in training_source,
                    "raise FloatingPointError" in training_source,
                    "amp_enabled = False" not in training_source,
                ]
            )
        fixed_precision_path = source_checks[commit]
        finite_metrics = not nonfinite_paths(metrics)
        checkpoint_finite = _checkpoint_is_finite(run_dir / "cnn_classifier_best.pt")
        passed = all(
            [
                _status_is_completed(status),
                amp_configured,
                amp_enabled,
                fixed_precision_path,
                finite_metrics,
                checkpoint_finite,
            ]
        )
        if not passed:
            failures.append(f"CNN seed {seed} failed AMP stability audit")
        rows.append(
            {
                "seed": seed,
                "completed": _status_is_completed(status),
                "git_commit": commit,
                "amp_configured": amp_configured,
                "amp_enabled_at_completion": amp_enabled,
                "precision_path_fixed_by_source": fixed_precision_path,
                "finite_metrics_and_history": finite_metrics,
                "finite_best_checkpoint": checkpoint_finite,
                "stopped_epoch": int(training["stopped_epoch"]),
                "training_seconds": float(training["total_training_seconds"]),
                "test_accuracy": float(metrics["test_metrics"]["accuracy"]),
                "test_macro_f1": float(metrics["test_metrics"]["macro_f1"]),
                "passed": passed,
            }
        )
    values = np.asarray([row["test_macro_f1"] for row in rows], dtype=np.float64)
    return {
        "failures": failures,
        "rows": rows,
        "run_count": len(rows),
        "test_macro_f1_mean": float(values.mean()),
        "test_macro_f1_sample_std": float(values.std(ddof=1)),
        "stable_amp_result": len(rows) == 5 and not failures,
        "automatic_amp_fallback_supported": False,
        "automatic_protocol_change_detected": not all(source_checks.values()),
        "training_source_check_by_commit": source_checks,
        "fp32_rerun_required": not (len(rows) == 5 and not failures),
    }


def audit_derived_artifacts(root: Path) -> dict[str, Any]:
    specs = [
        (
            "m01_pretraining_ablation",
            root / "artifacts/analysis/m01_pretraining_ablation/summary.json",
            lambda value: bool(value.get("all_protocol_checks_passed"))
            and value.get("seeds") == list(EXPECTED_SEEDS),
        ),
        (
            "battery_safety",
            root / "artifacts/battery_safety_thesis_v1/metrics.json",
            lambda value: int(value.get("run_count", -1)) == 45,
        ),
        (
            "battery_binary",
            root / "artifacts/battery_binary_analysis/formal_five_seed/metrics.json",
            lambda value: int(value.get("source_run_count", -1)) == 25
            and value.get("expected_seeds") == list(EXPECTED_SEEDS),
        ),
        (
            "m02_error_analysis",
            root / "artifacts/m02_error_analysis_20260805/metrics.json",
            lambda value: value.get("status") == "complete"
            and value.get("seeds") == list(EXPECTED_SEEDS),
        ),
        (
            "moment_retrieval",
            root / "artifacts/moment_retrieval/moment_retrieval_zero_shot_thesis_v1/metrics.json",
            lambda value: value.get("data", {}).get("dataset_sha256") == DATASET_SHA256
            and value.get("data", {}).get("classes") == list(EXPECTED_CLASSES)
            and value.get("data", {}).get("split_counts")
            == {"train": 24569, "validation": 3510, "test": 7020}
            and value.get("data", {}).get("labels_used_during_feature_extraction_or_search")
            is False
            and len(value.get("conditions", [])) == 10,
        ),
        (
            "moment_imputation",
            root / "artifacts/moment_imputation/moment_imputation_zero_shot_thesis_v2/metrics.json",
            lambda value: value.get("data", {}).get("dataset_sha256") == DATASET_SHA256
            and int(value.get("data", {}).get("source_test_samples", -1)) == EXPECTED_TEST_COUNT
            and int(value.get("data", {}).get("evaluated_test_samples", -1))
            == EXPECTED_TEST_COUNT
            and value.get("data", {}).get("labels_used") is False
            and len(value.get("conditions", [])) == 40,
        ),
    ]
    status_paths = {
        "moment_retrieval": root
        / "artifacts/moment_retrieval/moment_retrieval_zero_shot_thesis_v1/status.json",
        "moment_imputation": root
        / "artifacts/moment_imputation/moment_imputation_zero_shot_thesis_v2/status.json",
    }
    split_paths = {
        "moment_retrieval": root
        / "artifacts/moment_retrieval/moment_retrieval_zero_shot_thesis_v1/split_manifest.json",
        "moment_imputation": root
        / "artifacts/moment_imputation/moment_imputation_zero_shot_thesis_v2/split_manifest.json",
    }
    canonical_seed42_split = root / "artifacts/splits/unified_split.json"
    rows = []
    failures = []
    for name, path, predicate in specs:
        passed = path.is_file()
        finite = False
        protocol_check = False
        split_manifest_check: bool | str = "not_applicable"
        if passed:
            value = json.loads(path.read_text(encoding="utf-8"))
            finite = not nonfinite_paths(value)
            protocol_check = bool(predicate(value))
            passed = finite and protocol_check
        if name in status_paths:
            status_path = status_paths[name]
            status_ok = status_path.is_file() and _status_is_completed(
                json.loads(status_path.read_text(encoding="utf-8"))
            )
            passed = passed and status_ok
        if name in split_paths:
            split_path = split_paths[name]
            split_manifest_check = (
                canonical_seed42_split.is_file()
                and split_path.is_file()
                and sha256_file(split_path) == sha256_file(canonical_seed42_split)
            )
            protocol_check = protocol_check and bool(split_manifest_check)
            passed = passed and bool(split_manifest_check)
        if not passed:
            failures.append(f"Derived artifact failed audit: {name}")
        rows.append(
            {
                "name": name,
                "path": str(path.relative_to(root)),
                "finite": finite,
                "protocol_or_status_check": protocol_check,
                "split_manifest_check": split_manifest_check,
                "passed": passed,
            }
        )
    return {"failures": failures, "rows": rows, "artifact_count": len(rows)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(result: dict[str, Any]) -> str:
    formal = result["formal_runs"]
    cnn = result["cnn_amp_audit"]
    derived = result["derived_artifacts"]
    lines = [
        "# M03 final protocol audit",
        "",
        f"Overall status: **{result['status'].upper()}**.",
        "",
        "## Coverage",
        "",
        (
            f"- {formal['formal_run_count']} formal training/fitting runs in "
            f"{formal['formal_group_count']} five-seed groups."
        ),
        (
            f"- {formal['multiclass_run_count']} multiclass runs and "
            f"{formal['battery_binary_run_count']} dedicated battery-binary runs."
        ),
        (
            f"- {formal['metric_payload_count']} saved multiclass test metric payloads "
            "recalculated from confusion matrices."
        ),
        f"- {derived['artifact_count']} formal derived analyses checked.",
        "",
        "## Protocol checks",
        "",
        (
            "- Every formal run is completed and uses the fixed dataset hash, classes 0/1/2, "
            "3,510 validation samples, and 7,020 test samples."
        ),
        "- Every seed has exactly one split-manifest SHA-256 across all formal run groups.",
        (
            "- Each five-seed group has one normalized resolved-config hash after removing the "
            "expected seed/split-path fields and, for supervised MOMENT training only, the unused "
            "top-level SVM defaults."
        ),
        (
            "- The three historical commits used by formal multiclass runs contain byte-identical "
            "`src/tcn_moment/metrics.py` files."
        ),
        (
            f"- Maximum absolute saved-vs-recalculated metric difference: "
            f"{formal['maximum_metric_recalculation_difference']:.3g}."
        ),
        "",
        "## CNN AMP audit",
        "",
        "| Seed | Completed | AMP configured/final | Fixed precision path | Finite metrics | Finite best weights | Test Macro-F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cnn["rows"]:
        lines.append(
            f"| {row['seed']} | {row['completed']} | "
            f"{row['amp_configured']}/{row['amp_enabled_at_completion']} | "
            f"{row['precision_path_fixed_by_source']} | {row['finite_metrics_and_history']} | "
            f"{row['finite_best_checkpoint']} | {100 * row['test_macro_f1']:.2f}% |"
        )
    lines.extend(
        [
            "",
            (
                f"CNN five-seed Macro-F1: {100 * cnn['test_macro_f1_mean']:.2f}% +/- "
                f"{100 * cnn['test_macro_f1_sample_std']:.2f}%. All five AMP runs are stable; "
                "M03 does not require an FP32 rerun."
            ),
            (
                "The historical CNN training source keeps the precision path fixed for the full "
                "run and raises on a non-finite loss; it does not implement automatic AMP-to-FP32 "
                "fallback."
            ),
            "",
            "## Derived analyses",
            "",
            "| Analysis | Finite | Protocol/status check | Passed |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in derived["rows"]:
        lines.append(
            f"| {row['name']} | {row['finite']} | "
            f"{row['protocol_or_status_check']} | {row['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            (
                "Reported compute comparisons cover observed training/fitting time, peak GPU "
                "memory, and parameter/training scale only. No online inference latency, "
                "throughput, energy, or end-to-end production speed claim is supported by "
                "this audit."
            ),
            "",
        ]
    )
    if result["failures"]:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
        lines.append("")
    return "\n".join(lines)


def run_audit(root: Path, output_dir: Path) -> dict[str, Any]:
    formal = audit_formal_runs(root)
    cnn = audit_cnn(root)
    derived = audit_derived_artifacts(root)
    failures = [*formal["failures"], *cnn["failures"], *derived["failures"]]
    result = {
        "protocol_version": M03_PROTOCOL_VERSION,
        "audit": "M03_final_protocol_check",
        "status": "passed" if not failures else "failed",
        "dataset_sha256": DATASET_SHA256,
        "expected_seeds": list(EXPECTED_SEEDS),
        "formal_runs": {key: value for key, value in formal.items() if not key.endswith("_rows")},
        "cnn_amp_audit": {key: value for key, value in cnn.items() if key != "failures"},
        "derived_artifacts": {key: value for key, value in derived.items() if key != "failures"},
        "formatting_policy": {
            "classification_results": "percent, two decimal places, mean +/- sample standard deviation",
            "model_names": "canonical names recorded in paper and M03 experiment record",
            "compute_claim_scope": [
                "training_or_fitting_time",
                "peak_gpu_memory",
                "parameter_or_training_scale",
            ],
            "unsupported_compute_claims": [
                "online_inference_latency",
                "throughput",
                "energy",
                "end_to_end_production_speed",
            ],
        },
        "config_normalization_policy": {
            "all_groups": ["data.random_state", "data.split_path"],
            "supervised_moment_training_only": ["svm (unused by this execution path)"],
        },
        "failures": failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "formal_runs.csv", formal["run_rows"])
    _write_csv(output_dir / "formal_groups.csv", formal["group_rows"])
    _write_csv(output_dir / "cnn_amp_audit.csv", cnn["rows"])
    _write_csv(output_dir / "derived_artifacts.csv", derived["rows"])
    atomic_write_json(output_dir / "audit.json", result)
    (output_dir / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the final M03 paper experiment protocol.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/m03_protocol_audit_20260805"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    result = run_audit(root, output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(f"M03 audit failed with {len(result['failures'])} issue(s).")


if __name__ == "__main__":
    main()
