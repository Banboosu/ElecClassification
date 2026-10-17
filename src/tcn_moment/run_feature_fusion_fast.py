"""Execution-only E01 acceleration; preserve the original feature/cache protocol.

Fresh runs dispatch through the unchanged experiment trainer. Incomplete runs
can resume at completed conditions, retaining every finished classifier and CV
artifact. A real-data equivalence report is required before enabling fast CV.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tcn_moment import train_feature_fusion as original
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices, train_subset_sha256
from tcn_moment.experiment import RunContext, utc_now
from tcn_moment.fast_svm import FastRBFGridSearchCV
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.metrics import classification_metrics
from tcn_moment.train_moment_svm import _atomic_joblib_dump
from tcn_moment.training_utils import seed_everything

BACKEND_FILES = ("fast_svm.py", "run_feature_fusion_fast.py")
DEFAULT_CONFIG = "configs/experiments/improvements/feature_fusion.yaml"
DEFAULT_REPORT = "artifacts/analysis/e01_fast_svm_equivalence_20261017.json"


def backend_record(report_path: Path, n_jobs: int) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    module = Path(__file__).parent
    if (
        report["status"] != "passed"
        or not report["best_params_identical"]
        or not report["validation_predictions_identical"]
        or report["test_evaluated"]
    ):
        raise ValueError("A passed validation-only equivalence check is required.")
    if report["source_sha256"] != sha256_file(module / "fast_svm.py"):
        raise ValueError("Fast backend changed after the equivalence check.")
    return {
        "recorded_at": utc_now(),
        "backend": "fold_local_precomputed_rbf_float64",
        "n_jobs": n_jobs,
        "kernel_device": "cuda",
        "test_evaluated": False,
        "equivalence_report": str(report_path),
        "equivalence_report_sha256": sha256_file(report_path),
        "source_hashes": {name: sha256_file(module / name) for name in BACKEND_FILES},
        "policy": "Same training samples, folds, scorer, C/gamma grid, and ordinary RBF refit. No search pruning.",
    }


def save_backend_snapshot(directory: Path, report_path: Path, n_jobs: int) -> None:
    record = backend_record(report_path, n_jobs)
    directory.mkdir(parents=True, exist_ok=False)
    for name in BACKEND_FILES:
        shutil.copy2(Path(__file__).parent / name, directory / name)
    shutil.copy2(report_path, directory / "equivalence_report.json")
    atomic_write_json(directory / "execution_backend.json", record)


def resume_run(args: argparse.Namespace) -> None:
    run = Path(args.resume_run).resolve()
    original_run_name = run.name.rsplit("_seed", 1)[0]
    processes = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in processes.stdout.splitlines():
        fields = line.split(None, 2)
        if (
            len(fields) == 3
            and fields[1] == "python"
            and "-m tcn_moment.train_feature_fusion " in fields[2]
            and f"--run-name {original_run_name} " in fields[2]
        ):
            raise RuntimeError(
                "Original writer is still running; stop its owning suite before resume."
            )
    lock = (run / "acceleration.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((run / "experiment_plan.json").read_text())
    if plan["evaluate_test"]:
        raise ValueError("Resumption is validation-only; test evaluation is not supported here.")
    if original.source_hashes() != plan["source_hashes"]:
        raise ValueError(
            "Original feature/experiment sources changed; refusing mixed-protocol resume."
        )
    config = with_random_seed(load_config(args.config), plan["seed"])
    import torch

    seed_everything(torch, plan["seed"])
    expected_config = json.loads((run / "resolved_config.json").read_text())
    if json.loads(json.dumps(asdict(config), default=str)) != expected_config:
        raise ValueError("Resolved configuration differs from the original run.")
    status = json.loads((run / "status.json").read_text())
    if status["status"] == "completed":
        print(f"Skipping completed {run}", flush=True)
        return
    context = RunContext(
        "FEATURE_FUSION_E01", run.name, run, run / "status.json", status["started_at"]
    )
    attempt = run / ("acceleration_" + time.strftime("%Y%m%d_%H%M%S"))
    save_backend_snapshot(attempt, Path(args.equivalence_report), args.n_jobs)
    if (run / "metrics_partial.json").exists():
        shutil.copy2(run / "metrics_partial.json", attempt / "metrics_partial_before_resume.json")
    context.set_status("running", resumed=True, execution_backend="fast_rbf", n_jobs=args.n_jobs)
    try:
        bundle = load_dataset(config.data)
        if sha256_file(bundle.split_path) != sha256_file(run / "split_manifest.json"):
            raise ValueError("Split manifest changed.")
        stats = original.raw_statistics(config, bundle)
        moment, _ = original.frozen_features(
            config,
            bundle,
            Path(args.cache_dir) / f"seed{plan['seed']}",
            evaluate_test=False,
        )
        names = [str(value) for value in bundle.label_encoder.classes_]
        results = []
        for fraction in plan["fractions"]:
            indices = stratified_train_subset_indices(
                bundle.y_train,
                bundle.ids_train,
                fraction,
                plan["seed"],
            )
            tag = f"{fraction * 100:g}".replace(".", "p")
            subset_labels = bundle.y_train[indices]
            if (
                len(indices) > config.svm.max_samples
                or np.bincount(subset_labels).min() < config.svm.cv_folds
            ):
                raise ValueError("Requested subset violates the original sample/CV limits.")
            ids_path = run / f"train_ids_{tag}.npy"
            if ids_path.exists():
                np.testing.assert_array_equal(np.load(ids_path), bundle.ids_train[indices])
            else:
                np.save(ids_path, bundle.ids_train[indices])
            subset_hash = train_subset_sha256(bundle.ids_train[indices])
            cv_path = run / f"cv_folds_{tag}.npy"
            assignments = np.full(len(indices), -1, dtype=np.int8)
            for fold, (_, held_out) in enumerate(
                StratifiedKFold(config.svm.cv_folds, shuffle=False).split(indices, subset_labels)
            ):
                assignments[held_out] = fold
            if cv_path.exists():
                np.testing.assert_array_equal(np.load(cv_path), assignments)
            else:
                np.save(cv_path, assignments)
            subset_path = run / f"train_subset_{tag}.json"
            if not subset_path.exists():
                atomic_write_json(
                    subset_path,
                    {
                        "requested_fraction": fraction,
                        "selected_count": len(indices),
                        "full_training_count": len(bundle.y_train),
                        "class_counts": np.bincount(subset_labels).tolist(),
                        "sample_ids_sha256": subset_hash,
                        "cv_folds": config.svm.cv_folds,
                        "cv_shuffle": False,
                    },
                )
            for name in plan["models"]:
                destination = run / f"{name}_fraction_{tag}"
                metric_path = destination / "metrics.json"
                if metric_path.exists():
                    row = json.loads(metric_path.read_text())
                    if (
                        row["model"] != name
                        or row["seed"] != plan["seed"]
                        or row["train_fraction"] != fraction
                        or row["train_ids_sha256"] != subset_hash
                        or row["test_evaluated"]
                        or row["dataset_sha256"] != bundle.dataset_sha256
                    ):
                        raise ValueError(f"Completed condition does not match: {destination}")
                    for artifact in (
                        "cv_results.csv",
                        "classifier.joblib",
                        "validation_predictions.csv",
                    ):
                        if not (destination / artifact).is_file():
                            raise ValueError(
                                f"Completed condition is missing {artifact}: {destination}"
                            )
                    results.append(row)
                    print(f"Preserved {name} fraction={fraction:g}", flush=True)
                    continue
                # Keep any interrupted, non-atomic per-condition outputs for auditing.
                if destination.exists() and any(destination.iterdir()):
                    shutil.move(str(destination), str(attempt / destination.name))
                destination.mkdir(exist_ok=True)
                train = original.assemble_features(name, moment["train"], stats["train"])
                validation = original.assemble_features(
                    name, moment["validation"], stats["validation"]
                )
                reference = original.build_search(
                    name,
                    train.shape[1],
                    moment["train"].shape[1],
                    config,
                    n_jobs=args.n_jobs,
                    seed=plan["seed"],
                )
                search = FastRBFGridSearchCV(**reference.get_params(deep=False))
                print(
                    f"Resuming {name} fraction={fraction:g} n={len(indices)} jobs={args.n_jobs}",
                    flush=True,
                )
                started = time.perf_counter()
                search.fit(train[indices], bundle.y_train[indices])
                fit_seconds = time.perf_counter() - started
                pd.DataFrame(search.cv_results_).to_csv(destination / "cv_results.csv", index=False)
                _atomic_joblib_dump(search.best_estimator_, destination / "classifier.joblib")
                predicted = search.predict(validation)
                row = {
                    "model": name,
                    "seed": plan["seed"],
                    "train_fraction": fraction,
                    "train_samples": len(indices),
                    "train_ids_sha256": subset_hash,
                    "dataset_sha256": bundle.dataset_sha256,
                    "split_sha256": sha256_file(bundle.split_path),
                    "best_params": search.best_params_,
                    "cv_score": float(search.best_score_),
                    "selection_metric": "accuracy" if name == "moment_paper_svm" else "macro_f1",
                    "fit_seconds": fit_seconds,
                    "feature_dimension": train.shape[1],
                    "test_evaluated": False,
                    "execution_backend": getattr(
                        search, "backend_timings_", {"n_jobs": args.n_jobs}
                    ),
                    "validation_metrics": classification_metrics(
                        bundle.y_val, predicted, names, include_details=True
                    ),
                }
                pd.DataFrame(
                    {
                        "sample_id": bundle.ids_val,
                        "true_index": bundle.y_val,
                        "predicted_index": predicted,
                        "true_label": [names[int(value)] for value in bundle.y_val],
                        "predicted_label": [names[int(value)] for value in predicted],
                    }
                ).to_csv(destination / "validation_predictions.csv", index=False)
                atomic_write_json(metric_path, row)
                results.append(row)
                atomic_write_json(run / "metrics_partial.json", {"results": results})
                print(
                    f"Completed {name} fraction={fraction:g} fit_seconds={fit_seconds:.1f} "
                    f"val_macro_f1={row['validation_metrics']['macro_f1']:.6f}",
                    flush=True,
                )
        atomic_write_json(
            run / "metrics.json",
            {
                "protocol_version": original.PROTOCOL_VERSION,
                "seed": plan["seed"],
                "results": results,
                "test_evaluated": False,
            },
        )
        atomic_write_json(run / "validation_gate.json", original.validation_gate(results))
        context.set_status("completed", test_evaluated=False, resumed=True)
    except BaseException as exc:
        context.set_status(
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--resume-run")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument("--equivalence-report", default=DEFAULT_REPORT)
    args, remaining = parser.parse_known_args()
    if not 1 <= args.n_jobs <= 5:
        raise ValueError("Use 1-5 jobs on the current 6-vCPU / 25-GiB server.")
    backend_record(Path(args.equivalence_report), args.n_jobs)
    if args.resume_run:
        if remaining:
            raise ValueError(f"Unexpected resume arguments: {remaining}")
        resume_run(args)
        return
    if "--evaluate-test" in remaining:
        raise ValueError("The accelerated exploration launcher is validation-only.")
    # Deliberate dependency injection only in this process. On-disk original
    # sources and feature cache signatures remain byte-for-byte unchanged.
    original.GridSearchCV = FastRBFGridSearchCV
    original_snapshot = original.save_source_snapshot

    def snapshot(run_dir):
        original_snapshot(run_dir)
        save_backend_snapshot(
            run_dir / "execution_backend", Path(args.equivalence_report), args.n_jobs
        )

    original.save_source_snapshot = snapshot
    sys.argv = [
        sys.argv[0],
        *remaining,
        "--config",
        args.config,
        "--cache-dir",
        args.cache_dir,
        "--n-jobs",
        str(args.n_jobs),
    ]
    original.main()


if __name__ == "__main__":
    main()
