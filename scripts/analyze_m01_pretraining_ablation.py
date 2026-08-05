from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

from tcn_moment.io_utils import atomic_write_json


DEFAULT_FRACTIONS = (0.01, 0.05, 0.10)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction_map(payload: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(row["train_fraction"]): row for row in payload["fractions"]}


def _load_runs(
    paths: list[Path],
    expected_seeds: tuple[int, ...],
) -> dict[int, tuple[Path, dict[str, Any]]]:
    runs: dict[int, tuple[Path, dict[str, Any]]] = {}
    for run_dir in paths:
        metrics_path = run_dir / "metrics.json"
        status_path = run_dir / "status.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing status.json: {status_path}")
        status = _read_json(status_path)
        if status.get("status") != "completed":
            raise ValueError(f"Run is not completed: {run_dir} ({status.get('status')})")
        payload = _read_json(metrics_path)
        seed = int(payload["seed"])
        if seed in runs:
            raise ValueError(f"Duplicate seed {seed} in {paths}")
        runs[seed] = (run_dir, payload)
    if tuple(sorted(runs)) != expected_seeds:
        raise ValueError(
            f"Expected seeds {expected_seeds}, got {tuple(sorted(runs))}."
        )
    return runs


def _mean_interval(values: np.ndarray) -> dict[str, float | int]:
    count = int(len(values))
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if count > 1 else 0.0
    if count > 1:
        margin = float(t.ppf(0.975, count - 1) * standard_deviation / math.sqrt(count))
    else:
        margin = 0.0
    return {
        "count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci_95_lower": mean - margin,
        "ci_95_upper": mean + margin,
    }


def _selected_protocol(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    training = config["training"]
    svm = config["svm"]
    return {
        "data": config["data"],
        "model": {
            key: model[key]
            for key in (
                "model_id",
                "config_path",
                "num_channels",
                "freeze_backbone",
                "unfreeze_last_n_layers",
            )
        },
        "feature_extraction": {
            key: training[key]
            for key in (
                "feature_extraction_batch_size",
                "num_workers",
                "prefetch_factor",
                "device",
                "amp",
            )
        },
        "svm": {
            key: svm[key]
            for key in (
                "c_values",
                "gamma",
                "cv_folds",
                "max_samples",
                "n_jobs",
                "cache_size_mb",
                "max_iter",
            )
        },
    }


def analyze(
    pretrained_paths: list[Path],
    random_paths: list[Path],
    output_dir: Path,
    *,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    pretrained = _load_runs(pretrained_paths, seeds)
    random = _load_runs(random_paths, seeds)
    protocol_checks: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []

    for seed in seeds:
        pretrained_dir, pretrained_payload = pretrained[seed]
        random_dir, random_payload = random[seed]
        pretrained_fractions = _fraction_map(pretrained_payload)
        random_fractions = _fraction_map(random_payload)
        missing = [
            fraction
            for fraction in fractions
            if fraction not in pretrained_fractions or fraction not in random_fractions
        ]
        if missing:
            raise ValueError(f"Seed {seed} is missing fractions: {missing}")

        pretrained_config = _read_json(pretrained_dir / "resolved_config.json")
        random_config = _read_json(random_dir / "resolved_config.json")
        random_protocol = random_payload["protocol"]
        checks = {
            "seed": seed,
            "dataset_sha256_matches": (
                pretrained_payload["data"]["dataset_sha256"]
                == random_payload["data"]["dataset_sha256"]
            ),
            "split_manifest_matches": (
                _sha256(pretrained_dir / "split_manifest.json")
                == _sha256(random_dir / "split_manifest.json")
            ),
            "preprocessing_and_search_protocol_match": (
                _selected_protocol(pretrained_config)
                == _selected_protocol(random_config)
            ),
            "architecture_metadata_matches": all(
                pretrained_payload["protocol"].get(key)
                == random_payload["protocol"].get(key)
                for key in ("feature_dimension", "patch_len", "patch_stride")
            )
            and (
                pretrained_payload["execution"]["total_parameters"]
                == random_payload["execution"]["total_parameters"]
            ),
            "random_condition_did_not_load_checkpoint": (
                random_protocol.get("model_initialization") == "random"
                and random_protocol.get("pretrained_checkpoint_loaded") is False
                and random_protocol.get("initialization_seed") == seed
            ),
            "fraction_checks": [],
        }

        for fraction in fractions:
            pretrained_row = pretrained_fractions[fraction]
            random_row = random_fractions[fraction]
            tag = f"{fraction * 100:g}".replace(".", "p")
            pretrained_ids = np.load(
                pretrained_dir / f"train_subset_fraction_{tag}_sample_ids.npy"
            ).astype(str)
            random_ids = np.load(
                random_dir / f"train_subset_fraction_{tag}_sample_ids.npy"
            ).astype(str)
            fraction_check = {
                "train_fraction": fraction,
                "sample_ids_sha256_matches": (
                    pretrained_row["train_subset"]["sample_ids_sha256"]
                    == random_row["train_subset"]["sample_ids_sha256"]
                ),
                "sample_ids_exactly_match": bool(
                    np.array_equal(pretrained_ids, random_ids)
                ),
                "svm_search_matches": all(
                    pretrained_row["svm"].get(key) == random_row["svm"].get(key)
                    for key in ("kernel", "gamma", "c_values", "cv_folds")
                ),
            }
            checks["fraction_checks"].append(fraction_check)

            pretrained_macro_f1 = float(pretrained_row["test_metrics"]["macro_f1"])
            random_macro_f1 = float(random_row["test_metrics"]["macro_f1"])
            per_seed_rows.append(
                {
                    "train_fraction": fraction,
                    "seed": seed,
                    "pretrained_macro_f1": pretrained_macro_f1,
                    "random_macro_f1": random_macro_f1,
                    "paired_difference_pretrained_minus_random": (
                        pretrained_macro_f1 - random_macro_f1
                    ),
                    "pretrained_best_c": float(pretrained_row["svm"]["best_params"]["C"]),
                    "random_best_c": float(random_row["svm"]["best_params"]["C"]),
                }
            )
        protocol_checks.append(checks)

    failed_checks = []
    for checks in protocol_checks:
        for key, value in checks.items():
            if key in {"seed", "fraction_checks"}:
                continue
            if not value:
                failed_checks.append(f"seed={checks['seed']} {key}")
        for fraction_check in checks["fraction_checks"]:
            for key, value in fraction_check.items():
                if key == "train_fraction":
                    continue
                if not value:
                    failed_checks.append(
                        f"seed={checks['seed']} fraction={fraction_check['train_fraction']} {key}"
                    )
    if failed_checks:
        raise ValueError("M01 paired-protocol checks failed: " + "; ".join(failed_checks))

    per_seed = pd.DataFrame(per_seed_rows).sort_values(["train_fraction", "seed"])
    condition_summaries: list[dict[str, Any]] = []
    paired_summaries: list[dict[str, Any]] = []
    for fraction in fractions:
        selected = per_seed[per_seed["train_fraction"] == fraction]
        for condition, column in (
            ("pretrained", "pretrained_macro_f1"),
            ("random", "random_macro_f1"),
        ):
            condition_summaries.append(
                {
                    "train_fraction": fraction,
                    "condition": condition,
                    **_mean_interval(selected[column].to_numpy(dtype=np.float64)),
                }
            )
        paired_summaries.append(
            {
                "train_fraction": fraction,
                "difference": "pretrained_minus_random",
                **_mean_interval(
                    selected["paired_difference_pretrained_minus_random"].to_numpy(
                        dtype=np.float64
                    )
                ),
            }
        )

    condition_summary = pd.DataFrame(condition_summaries)
    paired_summary = pd.DataFrame(paired_summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_macro_f1.csv", index=False)
    condition_summary.to_csv(output_dir / "condition_summary.csv", index=False)
    paired_summary.to_csv(output_dir / "paired_summary.csv", index=False)
    atomic_write_json(
        output_dir / "protocol_checks.json",
        {"all_checks_passed": True, "checks": protocol_checks},
    )
    result = {
        "analysis": "M01_pretraining_attribution",
        "fractions": list(fractions),
        "seeds": list(seeds),
        "difference_direction": "pretrained_minus_random",
        "confidence_interval": "two-sided 95% Student t interval across five paired seeds",
        "all_protocol_checks_passed": True,
        "pretrained_runs": [str(path) for path in pretrained_paths],
        "random_runs": [str(path) for path in random_paths],
        "condition_summary": condition_summaries,
        "paired_summary": paired_summaries,
    }
    atomic_write_json(output_dir / "summary.json", result)
    print(condition_summary.to_string(index=False))
    print()
    print(paired_summary.to_string(index=False))
    print(f"Saved M01 analysis to {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the paired M01 pretraining ablation."
    )
    parser.add_argument("--pretrained-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--random-runs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/m01_pretraining_ablation"),
    )
    args = parser.parse_args()
    analyze(args.pretrained_runs, args.random_runs, args.output_dir)


if __name__ == "__main__":
    main()
