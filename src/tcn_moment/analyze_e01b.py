"""Freeze, verify and summarize the separate E01B validation replication."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import t

from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.experiment import utc_now
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.train_feature_fusion import BASELINE_NAMES, source_hashes

DEFAULT_PROTOCOL = "configs/experiments/improvements/e01b_high_label_protocol.json"


def budget_rule(differences_pp: list[float], rule: dict[str, Any]) -> dict[str, Any]:
    if len(differences_pp) != rule["number_of_replication_seeds"]:
        raise ValueError("Replication seed count differs from the frozen rule.")
    if not np.isfinite(differences_pp).all():
        raise ValueError("Non-finite replication difference.")
    mean = statistics.mean(differences_pp)
    positive = sum(value > 0 for value in differences_pp)
    half = (
        float(t.ppf(0.975, len(differences_pp) - 1))
        * statistics.stdev(differences_pp)
        / len(differences_pp) ** 0.5
    )
    return {
        "n_replication_seeds": len(differences_pp),
        "mean_difference_pp": mean,
        "positive_seeds": positive,
        "ci95_lower_pp": mean - half,
        "ci95_upper_pp": mean + half,
        "rule_passed": (
            mean >= rule["minimum_mean_difference_pp_per_fraction"] - 1e-12
            and positive >= rule["minimum_strictly_positive_seeds_per_fraction"]
        ),
    }


def read_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if (
        protocol["experiment_id"] != "E01B"
        or protocol["test_evaluated"]
        or protocol["replication_seeds"] != [43, 44, 45, 46]
        or protocol["discovery_seeds_excluded_from_success_rule"] != [42]
        or protocol["fractions"] != [0.2, 0.4]
        or protocol["models"] != ["fusion_stats_svm"]
        or protocol["baseline_models"] != list(BASELINE_NAMES)
    ):
        raise ValueError("Unexpected E01B scope or test evaluation requested.")
    if protocol["success_rule"] != {
        "minimum_mean_difference_pp_per_fraction": 1.0,
        "minimum_strictly_positive_seeds_per_fraction": 3,
        "require_both_fractions": True,
        "number_of_replication_seeds": 4,
    }:
        raise ValueError("Unexpected E01B replication success rule.")
    fixed = protocol["fixed_search"]
    if (
        fixed["cv_folds"] != 5
        or fixed["cv_shuffle"]
        or fixed["scoring"] != "macro_f1"
        or fixed["C"] != [0.1, 1.0, 10.0, 100.0, 1000.0]
        or fixed["gamma"] != ["scale", 0.05, 5.0]
        or fixed["statistics_branch_weight"] != 1.0
    ):
        raise ValueError("E01B search differs from the frozen E01 method.")
    return protocol


def analysis_directory(protocol: dict[str, Any]) -> Path:
    return Path(protocol["output_dir"]) / f"{protocol['suite']}_analysis"


def baseline_run(protocol: dict[str, Any], seed: int) -> Path:
    return Path(protocol["baseline_run_root"]) / protocol["baseline_run_pattern"].format(seed=seed)


def verify_frozen(directory: Path, protocol_path: Path) -> dict[str, Any]:
    frozen = json.loads((directory / "frozen_protocol.json").read_text())
    if (
        sha256_file(protocol_path) != frozen["protocol_sha256"]
        or read_protocol(protocol_path) != frozen["protocol"]
    ):
        raise ValueError("Protocol changed after freezing.")
    for name, digest in frozen["file_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Frozen source/config/baseline changed: {name}")
    return frozen


def freeze(protocol_path: Path) -> None:
    protocol = read_protocol(protocol_path)
    directory = analysis_directory(protocol)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "frozen_protocol.json").exists():
        verify_frozen(directory, protocol_path)
        print(f"Existing frozen E01B protocol verified: {directory}")
        return
    parent = json.loads(Path("artifacts/analysis/e01_validation_20261017/summary.json").read_text())
    if (
        parent["status"] != "passed"
        or parent["test_evaluated"]
        or parent["total_conditions"] != 110
    ):
        raise ValueError("Parent E01 audit has not passed.")
    files = [
        protocol_path,
        Path(protocol["training_config"]),
        Path("configs/moment.yaml"),
        Path("configs/models/moment-1-large.json"),
        Path("scripts/run_e01b_remote.sh"),
        Path(__file__),
        Path(__file__).with_name("fast_svm.py"),
        Path(__file__).with_name("run_feature_fusion_fast.py"),
    ]
    files.extend(Path(__file__).with_name(name) for name in source_hashes())
    for seed in protocol["replication_seeds"]:
        run = baseline_run(protocol, seed)
        if json.loads((run / "status.json").read_text())["status"] != "completed":
            raise ValueError(f"Parent baseline run incomplete: {seed}")
        files.extend(
            [
                run / "split_manifest.json",
                run / "experiment_plan.json",
                run / "resolved_config.json",
                Path(protocol["cache_dir"]) / f"seed{seed}" / "manifest.json",
            ]
        )
        for fraction in protocol["fractions"]:
            tag = f"{fraction * 100:g}"
            files.extend([run / f"train_ids_{tag}.npy", run / f"cv_folds_{tag}.npy"])
            for model in protocol["baseline_models"]:
                condition = run / f"{model}_fraction_{tag}"
                files.extend([condition / "metrics.json", condition / "validation_predictions.csv"])
    atomic_write_json(
        directory / "frozen_protocol.json",
        {
            "frozen_at": utc_now(),
            "protocol": protocol,
            "protocol_sha256": sha256_file(protocol_path),
            "file_sha256": {str(path): sha256_file(path) for path in files},
            "note": "Frozen before E01B seed43-46 fusion results; seed42 discovery is excluded from the success rule.",
        },
    )
    print(f"Frozen E01B protocol: {directory}; 8 conditions; test remains locked.")


def summarize(protocol_path: Path) -> None:
    protocol = read_protocol(protocol_path)
    directory = analysis_directory(protocol)
    frozen = verify_frozen(directory, protocol_path)
    rows = []
    for seed in protocol["replication_seeds"]:
        run = Path(protocol["output_dir"]) / f"{protocol['suite']}_seed{seed}"
        status = json.loads((run / "status.json").read_text())
        plan = json.loads((run / "experiment_plan.json").read_text())
        data = json.loads((run / "metrics.json").read_text())
        base = baseline_run(protocol, seed)
        baseline_plan = json.loads((base / "experiment_plan.json").read_text())
        if (
            status["status"] != "completed"
            or data["test_evaluated"]
            or plan["evaluate_test"]
            or plan["seed"] != seed
            or plan["models"] != protocol["models"]
            or plan["fractions"] != protocol["fractions"]
            or plan["source_hashes"] != baseline_plan["source_hashes"]
            or sha256_file(run / "split_manifest.json") != sha256_file(base / "split_manifest.json")
            or len(data["results"]) != 2
        ):
            raise ValueError(f"E01B run protocol/status mismatch: {run}")
        for fraction in protocol["fractions"]:
            tag = f"{fraction * 100:g}"
            selected = [row for row in data["results"] if row["train_fraction"] == fraction]
            if len(selected) != 1:
                raise ValueError("Missing/duplicate E01B condition.")
            row = selected[0]
            condition = run / f"fusion_stats_svm_fraction_{tag}"
            for filename in (f"train_ids_{tag}.npy", f"cv_folds_{tag}.npy"):
                np.testing.assert_array_equal(np.load(run / filename), np.load(base / filename))
            if row["model"] != "fusion_stats_svm" or row["test_evaluated"] or "test_metrics" in row:
                raise ValueError("Unexpected model or test metrics in E01B.")
            baseline_rows = []
            for name in protocol["baseline_models"]:
                base_condition = base / f"{name}_fraction_{tag}"
                baseline = json.loads((base_condition / "metrics.json").read_text())
                if (
                    row["train_ids_sha256"] != baseline["train_ids_sha256"]
                    or row["dataset_sha256"] != baseline["dataset_sha256"]
                ):
                    raise ValueError("Candidate and baseline train on different samples.")
                baseline_rows.append((name, baseline))
            best_name, best = max(
                baseline_rows, key=lambda item: item[1]["validation_metrics"]["macro_f1"]
            )
            with (condition / "validation_predictions.csv").open(newline="") as handle:
                predictions = list(csv.DictReader(handle))
            with (base / f"{best_name}_fraction_{tag}" / "validation_predictions.csv").open(
                newline=""
            ) as handle:
                reference = list(csv.DictReader(handle))
            if [(p["sample_id"], p["true_index"]) for p in predictions] != [
                (p["sample_id"], p["true_index"]) for p in reference
            ]:
                raise ValueError("Candidate and baseline validation IDs/labels differ.")
            confusion = [[0] * 3 for _ in range(3)]
            for prediction in predictions:
                confusion[int(prediction["true_index"])][int(prediction["predicted_index"])] += 1
            if confusion != row["validation_metrics"]["confusion_matrix"]:
                raise ValueError("Confusion matrix differs from saved predictions.")
            for key, value in metrics_from_confusion(confusion).items():
                if abs(value - row["validation_metrics"][key]) > 1e-12:
                    raise ValueError("Saved validation metrics differ from predictions.")
            model = joblib.load(condition / "classifier.joblib")
            if model.classes_.tolist() != [0, 1, 2] or model.n_features_in_ != 1039:
                raise ValueError("Classifier has wrong classes/features.")
            if any(model.get_params()[key] != value for key, value in row["best_params"].items()):
                raise ValueError("Classifier/selected parameter mismatch.")
            del model
            baseline_f1 = best["validation_metrics"]["macro_f1"] * 100
            fusion_f1 = row["validation_metrics"]["macro_f1"] * 100
            rows.append(
                {
                    "seed": seed,
                    "train_fraction": fraction,
                    "best_baseline": best_name,
                    "baseline_macro_f1_percent": baseline_f1,
                    "fusion_macro_f1_percent": fusion_f1,
                    "difference_pp": fusion_f1 - baseline_f1,
                    "fit_seconds": row["fit_seconds"],
                }
            )
    budgets = []
    for fraction in protocol["fractions"]:
        conditions = [row for row in rows if row["train_fraction"] == fraction]
        budgets.append(
            {
                "train_fraction": fraction,
                "baseline_mean_percent": statistics.mean(
                    row["baseline_macro_f1_percent"] for row in conditions
                ),
                "fusion_mean_percent": statistics.mean(
                    row["fusion_macro_f1_percent"] for row in conditions
                ),
                **budget_rule(
                    [row["difference_pp"] for row in conditions], protocol["success_rule"]
                ),
            }
        )
    result = {
        "audit_status": "passed",
        "test_evaluated": False,
        "recorded_at": utc_now(),
        "protocol_frozen_at": frozen["frozen_at"],
        "conditions": len(rows),
        "discovery_seed_included": False,
        "per_seed": rows,
        "per_budget": budgets,
        "overall_rule_passed": all(row["rule_passed"] for row in budgets),
        "limitations": protocol["limitations"],
    }
    atomic_write_json(directory / "summary.json", result)
    with (directory / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# E01B validation replication",
        "",
        "Only seeds 43-46; seed42 discovery excluded; test not evaluated.",
        "",
        "| Training labels | Best baseline mean | Fusion mean | Difference (pp) | Positive seeds | Rule passed |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in budgets:
        lines.append(
            f"| {row['train_fraction']:.0%} | {row['baseline_mean_percent']:.2f} | "
            f"{row['fusion_mean_percent']:.2f} | {row['mean_difference_pp']:+.2f} | "
            f"{row['positive_seeds']}/4 | {row['rule_passed']} |"
        )
    lines.extend(
        [
            "",
            f"Overall frozen rule passed: {result['overall_rule_passed']}",
            "",
            "Intervals in summary.json are descriptive; random splits overlap, no external-test confirmation.",
        ]
    )
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "summarize"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    path = Path(args.protocol)
    if args.action == "freeze":
        freeze(path)
    elif args.action == "verify":
        verify_frozen(analysis_directory(read_protocol(path)), path)
        print("E01B frozen protocol and parent baselines verified.")
    else:
        summarize(path)


if __name__ == "__main__":
    main()
