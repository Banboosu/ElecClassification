"""Prospectively freeze and audit the E01C reference-bandwidth attribution controls."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tcn_moment.analyze_e01b import budget_rule
from tcn_moment.analyze_e01b import verify_frozen as verify_parent
from tcn_moment.audit_protocol import metrics_from_confusion
from tcn_moment.config import load_config
from tcn_moment.controlled_svm import reference_scaler_digest, resolve_gamma
from tcn_moment.experiment import utc_now
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.train_feature_fusion import source_hashes

DEFAULT_PROTOCOL = "configs/experiments/improvements/e01c_attribution_protocol.json"
NEW_FILES = ("analyze_e01c.py", "train_e01c.py", "controlled_svm.py")
PACKAGES = ("momentfm", "transformers", "torch", "numpy", "scikit-learn", "joblib")


def read_protocol(path: Path) -> dict[str, Any]:
    p = json.loads(path.read_text())
    if (
        p["experiment_id"] != "E01C"
        or p["test_evaluated"]
        or p["seeds"] != [43, 44, 45, 46]
        or p["fractions"] != [0.2, 0.4]
        or p["conditions"] != ["statistics_matched_kernel", "random_fusion_matched_kernel"]
    ):
        raise ValueError("Unexpected attribution scope or test evaluation requested.")
    if p["fixed_search"] != {
        "C": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "gamma_tokens": ["reference_scale", 0.05, 5.0],
        "cv_folds": 5,
        "cv_shuffle": False,
        "scoring": "macro_f1",
        "reference_dim": 1024,
        "statistics_dim": 15,
        "statistics_branch_weight": 1.0,
    }:
        raise ValueError("Unexpected controlled C/gamma/geometry settings.")
    rules = p["decision_rules"]
    if (
        rules["required_seeds"] != 4
        or not rules["require_both_fractions_for_each_claim"]
        or rules["beyond_statistics"]
        != {"minimum_mean_difference_pp": 1.0, "minimum_positive_seeds": 3}
        or rules["pretrained_vs_random"]
        != {"minimum_mean_difference_pp": 0.5, "minimum_positive_seeds": 3}
    ):
        raise ValueError("Attribution thresholds differ from the stated plan.")
    return p


def plan_directory(p: dict[str, Any]) -> Path:
    return Path(p["output_dir"]) / f"{p['suite']}_analysis"


def reference_run(p: dict[str, Any], seed: int) -> Path:
    return Path(p["reference_run_root"]) / p["reference_run_pattern"].format(seed=seed)


def legacy_run(p: dict[str, Any], seed: int) -> Path:
    return Path(p["legacy_run_root"]) / p["legacy_run_pattern"].format(seed=seed)


def verify(protocol_path: Path) -> dict[str, Any]:
    p = read_protocol(protocol_path)
    frozen = json.loads((plan_directory(p) / "frozen_protocol.json").read_text())
    if frozen["protocol_sha256"] != sha256_file(protocol_path) or frozen["protocol"] != p:
        raise ValueError("E01C protocol changed after freezing.")
    for filename, digest in frozen["file_sha256"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError(f"Frozen attribution input changed: {filename}")
    if {name: importlib.metadata.version(name) for name in PACKAGES} != frozen["packages"]:
        raise ValueError("Core package versions changed during attribution experiments.")
    return frozen


def freeze(protocol_path: Path) -> None:
    p = read_protocol(protocol_path)
    directory = plan_directory(p)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "frozen_protocol.json").exists():
        verify(protocol_path)
        print("Existing E01C protocol verified.")
        return
    verify_parent(
        Path("artifacts/improvements/e01b_high_label/e01b_high_label_20261017_v1_analysis"),
        Path("configs/experiments/improvements/e01b_high_label_protocol.json"),
    )
    report = Path(p["reference_equivalence_report"])
    equivalence = json.loads(report.read_text())
    if (
        equivalence["status"] != "passed"
        or equivalence["test_evaluated"]
        or not equivalence["validation_predictions_identical"]
        or not equivalence["best_params_identical"]
        or not equivalence["reference_scaler_identical"]
        or equivalence["controlled_backend_sha256"]
        != sha256_file(Path(__file__).with_name("controlled_svm.py"))
    ):
        raise ValueError(
            "The exact reference pipeline must pass the real-data compatibility check first."
        )
    smoke_path = Path("artifacts/analysis/e01c_random_initialization_smoke_20261017.json")
    smoke = json.loads(smoke_path.read_text())
    if (
        smoke["status"] != "passed"
        or not smoke["standard_pretrained_loaders_blocked"]
        or not smoke["probe_differs_from_public_checkpoint"]
        or smoke["total_parameters"] != 341243395
        or smoke["test_evaluated"]
        or not smoke["finite_features"]
    ):
        raise ValueError("The real random model must pass guarded initialization/forward checks.")
    config = load_config(p["training_config"])
    files = [
        protocol_path,
        report,
        Path(p["training_config"]),
        Path("configs/moment.yaml"),
        config.model.config_path,
        config.data.csv_path,
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("artifacts/e01_model_weight_fingerprint.json"),
        Path("scripts/run_e01c_remote.sh"),
        Path("docs/experiment_records/e01c_attribution_protocol_20261017.md"),
        Path("scripts/benchmark_e01c_reference.py"),
        Path("scripts/check_e01c_random_model.py"),
        smoke_path,
        Path(__file__).with_name("fast_svm.py"),
    ]
    files.extend(Path(__file__).with_name(name) for name in [*NEW_FILES, *source_hashes()])
    import momentfm

    library = Path(momentfm.__file__).parent
    files.extend(
        library / name
        for name in ("models/moment.py", "models/layers/revin.py", "models/layers/embed.py")
    )
    for seed in p["seeds"]:
        parent, legacy = reference_run(p, seed), legacy_run(p, seed)
        if json.loads((parent / "status.json").read_text())["status"] != "completed":
            raise ValueError("Reference E01B run incomplete.")
        files.extend(
            [
                parent / "split_manifest.json",
                parent / "experiment_plan.json",
                Path(p["pretrained_cache"]) / f"seed{seed}" / "manifest.json",
            ]
        )
        for fraction in p["fractions"]:
            tag = f"{fraction * 100:g}"
            files.extend([parent / f"train_ids_{tag}.npy", parent / f"cv_folds_{tag}.npy"])
            condition = parent / f"fusion_stats_svm_fraction_{tag}"
            files.extend(
                condition / name
                for name in (
                    "metrics.json",
                    "classifier.joblib",
                    "cv_results.csv",
                    "validation_predictions.csv",
                )
            )
            for model in ("stats_rf", "stats_svm"):
                files.extend(
                    legacy / f"{model}_fraction_{tag}" / name
                    for name in ("metrics.json", "validation_predictions.csv")
                )
    atomic_write_json(
        directory / "frozen_protocol.json",
        {
            "frozen_at": utc_now(),
            "protocol": p,
            "protocol_sha256": sha256_file(protocol_path),
            "file_sha256": {str(path): sha256_file(path) for path in files},
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
            "reference_equivalence": equivalence,
            "random_initialization_smoke": smoke,
            "note": "Reference outcomes were known; the 16 control outcomes had not been inspected. Conditional attribution, not external confirmation.",
        },
    )
    print(
        f"Frozen E01C: {directory}; 16 new controls; exact shared fold-local bandwidth; test locked."
    )


def evaluate_differences(values: list[float], p: dict[str, Any], claim: str) -> dict[str, Any]:
    rule = p["decision_rules"][claim]
    return budget_rule(
        values,
        {
            "number_of_replication_seeds": p["decision_rules"]["required_seeds"],
            "minimum_mean_difference_pp_per_fraction": rule["minimum_mean_difference_pp"],
            "minimum_strictly_positive_seeds_per_fraction": rule["minimum_positive_seeds"],
        },
    )


def summarize(protocol_path: Path) -> None:
    frozen = verify(protocol_path)
    p, rows = frozen["protocol"], []
    for seed in p["seeds"]:
        run = Path(p["output_dir"]) / f"{p['suite']}_seed{seed}"
        payload = json.loads((run / "metrics.json").read_text())
        if (
            json.loads((run / "status.json").read_text())["status"] != "completed"
            or payload["test_evaluated"]
            or len(payload["results"]) != 4
        ):
            raise ValueError("Missing/incomplete attribution run.")
        random_cache = Path(p["random_cache"]) / f"seed{seed}"
        provenance = json.loads((random_cache / "pretrained_loading_guard.json").read_text())
        manifest = json.loads((random_cache / "manifest.json").read_text())
        if (
            not provenance["guard_enabled"]
            or provenance["manifest_sha256"] != sha256_file(random_cache / "manifest.json")
            or manifest["signature"]["initialization"] != "random"
            or manifest["signature"]["checkpoint"] != {"initialization": "random", "seed": seed}
            or set(manifest["splits"]) != {"train", "validation"}
        ):
            raise ValueError("Random encoder provenance/cache mismatch.")
        if provenance["source_sha256"] != sha256_file(Path(__file__).with_name("train_e01c.py")):
            raise ValueError("Random feature loading guard belongs to another source version.")
        parent = reference_run(p, seed)
        split_ids = json.loads((parent / "split_manifest.json").read_text())["splits"]
        for split in ("train", "validation"):
            path = random_cache / f"{split}.npz"
            if sha256_file(path) != manifest["splits"][split]["sha256"]:
                raise ValueError("Random feature file checksum mismatch.")
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["sample_ids"], split_ids[split])
                if (
                    data["features"].shape != (len(split_ids[split]), 1024)
                    or not np.isfinite(data["features"]).all()
                ):
                    raise ValueError("Invalid random features.")
        for fraction in p["fractions"]:
            tag = f"{fraction * 100:g}"
            reference = json.loads(
                (parent / f"fusion_stats_svm_fraction_{tag}/metrics.json").read_text()
            )
            reference_model = joblib.load(
                parent / f"fusion_stats_svm_fraction_{tag}/classifier.joblib"
            )
            expected_scaler_digest = reference_scaler_digest(reference_model.named_steps["scale"])
            del reference_model
            for filename in (f"train_ids_{tag}.npy", f"cv_folds_{tag}.npy"):
                np.testing.assert_array_equal(np.load(run / filename), np.load(parent / filename))
            evaluated, geometries = {}, []
            for name in p["conditions"]:
                matching = [
                    r
                    for r in payload["results"]
                    if r["condition"] == name and r["train_fraction"] == fraction
                ]
                if len(matching) != 1:
                    raise ValueError("Duplicate/missing control condition.")
                row = matching[0]
                if (
                    row["test_evaluated"]
                    or "test_metrics" in row
                    or row["seed"] != seed
                    or row["protocol_sha256"] != frozen["protocol_sha256"]
                    or row["train_ids_sha256"] != reference["train_ids_sha256"]
                ):
                    raise ValueError("Control/reference protocol mismatch.")
                condition = run / f"{name}_fraction_{tag}"
                with (condition / "validation_predictions.csv").open(newline="") as handle:
                    predictions = list(csv.DictReader(handle))
                with (parent / f"fusion_stats_svm_fraction_{tag}/validation_predictions.csv").open(
                    newline=""
                ) as handle:
                    target = list(csv.DictReader(handle))
                if [(v["sample_id"], v["true_index"]) for v in predictions] != [
                    (v["sample_id"], v["true_index"]) for v in target
                ]:
                    raise ValueError("Validation IDs/ground truth changed.")
                matrix = [[0] * 3 for _ in range(3)]
                for v in predictions:
                    true, predicted = int(v["true_index"]), int(v["predicted_index"])
                    if true not in range(3) or predicted not in range(3):
                        raise ValueError("Invalid predicted class.")
                    matrix[true][predicted] += 1
                if matrix != row["validation_metrics"]["confusion_matrix"]:
                    raise ValueError("Control confusion matrix differs from predictions.")
                for key, actual in metrics_from_confusion(matrix).items():
                    if abs(actual - row["validation_metrics"][key]) > 1e-12:
                        raise ValueError("Control metric differs from predictions.")
                geometry = json.loads((condition / "kernel_geometry.json").read_text())
                if geometry["full_reference_scaler_sha256"] != expected_scaler_digest:
                    raise ValueError(
                        "Controls did not use the original full-reference training scaler."
                    )
                if geometry["resolved_refit_gamma"] != resolve_gamma(
                    row["best_params"]["gamma_token"], geometry["full_reference_scale_gamma"]
                ):
                    raise ValueError("Final gamma token was not resolved against the reference.")
                with (condition / "cv_results.csv").open(newline="") as handle:
                    cv_rows = list(csv.DictReader(handle))
                selected = [
                    c for c in cv_rows if ast.literal_eval(c["params"]) == row["best_params"]
                ]
                if (
                    len(cv_rows) != 15
                    or len(selected) != 1
                    or abs(float(selected[0]["mean_test_score"]) - row["cv_score"]) > 1e-12
                    or abs(max(float(c["mean_test_score"]) for c in cv_rows) - row["cv_score"])
                    > 1e-12
                ):
                    raise ValueError("Controlled grid/selection score mismatch.")
                geometries.append(geometry)
                model = joblib.load(condition / "classifier.joblib")
                expected_dim = 15 if name == "statistics_matched_kernel" else 1039
                svc = model.named_steps["svc"]
                if (
                    model.n_features_in_ != expected_dim
                    or model.classes_.tolist() != [0, 1, 2]
                    or svc.C != row["best_params"]["C"]
                    or svc.gamma != geometry["resolved_refit_gamma"]
                    or svc.kernel != "rbf"
                ):
                    raise ValueError("Control classifier/refit gamma mismatch.")
                del model
                evaluated[name] = row["validation_metrics"]["macro_f1"] * 100
            for key in ("folds", "full_reference_scale_gamma", "full_reference_scaler_sha256"):
                if geometries[0][key] != geometries[1][key]:
                    raise ValueError(
                        "The two controls did not share exactly the same gamma reference."
                    )
            legacy_values = []
            for name in ("stats_rf", "stats_svm"):
                old = json.loads(
                    (legacy_run(p, seed) / f"{name}_fraction_{tag}/metrics.json").read_text()
                )
                legacy_values.append(old["validation_metrics"]["macro_f1"] * 100)
            ref_f1 = reference["validation_metrics"]["macro_f1"] * 100
            strongest_stats = max(*legacy_values, evaluated["statistics_matched_kernel"])
            rows.append(
                {
                    "seed": seed,
                    "train_fraction": fraction,
                    "pretrained_fusion_f1_percent": ref_f1,
                    "matched_statistics_f1_percent": evaluated["statistics_matched_kernel"],
                    "random_fusion_f1_percent": evaluated["random_fusion_matched_kernel"],
                    "strongest_statistics_f1_percent": strongest_stats,
                    "pretrained_minus_matched_stats_pp": ref_f1
                    - evaluated["statistics_matched_kernel"],
                    "pretrained_minus_strongest_stats_pp": ref_f1 - strongest_stats,
                    "pretrained_minus_random_pp": ref_f1
                    - evaluated["random_fusion_matched_kernel"],
                }
            )
    comparisons = []
    for fraction in p["fractions"]:
        values = [r for r in rows if r["train_fraction"] == fraction]
        for claim, column in (
            ("beyond_statistics", "pretrained_minus_strongest_stats_pp"),
            ("pretrained_vs_random", "pretrained_minus_random_pp"),
        ):
            comparisons.append(
                {
                    "train_fraction": fraction,
                    "claim": claim,
                    **evaluate_differences([r[column] for r in values], p, claim),
                }
            )
    result = {
        "audit_status": "passed",
        "test_evaluated": False,
        "recorded_at": utc_now(),
        "control_conditions": 16,
        "per_seed": rows,
        "comparisons": comparisons,
        "claim_rules_passed": {
            claim: all(r["rule_passed"] for r in comparisons if r["claim"] == claim)
            for claim in ("beyond_statistics", "pretrained_vs_random")
        },
        "limitations": p["limitations"],
    }
    directory = plan_directory(p)
    atomic_write_json(directory / "summary.json", result)
    with (directory / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# E01C attribution controls (validation only)",
        "",
        "| Budget | Claim | Mean difference (pp) | Positive seeds | Rule passed |",
        "|---:|---|---:|---:|:---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['train_fraction']:.0%} | {row['claim']} | {row['mean_difference_pp']:+.2f} | {row['positive_seeds']}/4 | {row['rule_passed']} |"
        )
    lines.extend(
        [
            "",
            "Claims remain conditional on the tested controls; random splits overlap; test not evaluated.",
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
        verify(path)
        print("E01C protocol, software and parent inputs verified.")
    else:
        summarize(path)


if __name__ == "__main__":
    main()
