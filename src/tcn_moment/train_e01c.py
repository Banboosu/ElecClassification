"""Run the two frozen E01C controls, without changing earlier experiment sources."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tcn_moment.analyze_e01c import (
    DEFAULT_PROTOCOL,
    NEW_FILES,
    plan_directory,
    reference_run,
    verify,
)
from tcn_moment.config import load_config, with_random_seed
from tcn_moment.controlled_svm import fit_controlled_rbf
from tcn_moment.data import (
    load_dataset,
    save_label_encoder,
    stratified_train_subset_indices,
    train_subset_sha256,
)
from tcn_moment.experiment import _json_safe, prepare_run
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.metrics import classification_metrics
from tcn_moment.train_feature_fusion import (
    cache_signature,
    frozen_features,
    raw_statistics,
    save_source_snapshot,
    validate_cache,
)
from tcn_moment.train_moment_svm import _atomic_joblib_dump
from tcn_moment.training_utils import seed_everything

GUARDED_APIS = [
    "MOMENTPipeline.from_pretrained",
    "T5Model.from_pretrained",
    "T5EncoderModel.from_pretrained",
]


@contextmanager
def forbid_pretrained_loading():
    from momentfm import MOMENTPipeline
    from transformers import T5EncoderModel, T5Model

    with ExitStack() as stack:
        for target in (MOMENTPipeline, T5Model, T5EncoderModel):
            stack.enter_context(
                patch.object(
                    target,
                    "from_pretrained",
                    side_effect=RuntimeError(
                        "Pretrained loading is forbidden in the random encoder control."
                    ),
                )
            )
        yield


def condition_arrays(pretrained: np.ndarray, random: np.ndarray, statistics: np.ndarray):
    if (
        pretrained.shape != random.shape
        or pretrained.ndim != 2
        or pretrained.shape[1] != 1024
        or statistics.shape != (len(pretrained), 15)
    ):
        raise ValueError("Control features have the wrong sample counts or dimensions.")
    return np.concatenate([pretrained, statistics], axis=1), {
        "statistics_matched_kernel": statistics,
        "random_fusion_matched_kernel": np.concatenate([random, statistics], axis=1),
    }


def guarded_random_features(config, bundle, cache_dir):
    if config.model.initialization != "random":
        raise ValueError("Random control requires direct random initialization.")
    with forbid_pretrained_loading():
        features, metadata = frozen_features(config, bundle, cache_dir, evaluate_test=False)
    manifest_path = cache_dir / "manifest.json"
    marker_path = cache_dir / "pretrained_loading_guard.json"
    expected = {
        "guard_enabled": True,
        "pretrained_checkpoint_loaded": False,
        "random_initialization_seed": config.data.random_state,
        "guarded_apis": GUARDED_APIS,
        "source_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": sha256_file(manifest_path),
    }
    if metadata["cache_hit"]:
        if not marker_path.exists() or json.loads(marker_path.read_text()) != expected:
            raise ValueError(
                "Random feature cache has no matching loading-guard provenance; quarantine it before retrying."
            )
    else:
        atomic_write_json(marker_path, expected)
    return features, metadata


def run(args):
    frozen = verify(Path(args.protocol))
    p, seed = frozen["protocol"], args.seed
    if seed not in p["seeds"]:
        raise ValueError("Seed is outside the frozen attribution protocol.")
    run_name = f"{p['suite']}_seed{seed}"
    run_dir = Path(p["output_dir"]) / run_name
    lock = (plan_directory(p) / f"seed{seed}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if run_dir.exists() and not args.resume:
        raise FileExistsError("Run exists; resume it explicitly instead of overwriting.")
    if args.resume and not run_dir.exists():
        raise FileNotFoundError("Cannot resume a missing run.")
    if args.resume and json.loads((run_dir / "status.json").read_text())["status"] == "completed":
        print(f"Skipping completed {run_dir}", flush=True)
        lock.close()
        return
    config_path = Path(p["training_config"])
    config = with_random_seed(load_config(config_path), seed)
    random_config = replace(config, model=replace(config.model, initialization="random"))
    import torch

    seed_everything(torch, seed)
    context = prepare_run(
        model_name="E01C_ATTRIBUTION",
        base_output_dir=Path(p["output_dir"]),
        config=config,
        config_path=config_path,
        torch=torch,
        run_name=run_name,
        resume_dir=run_dir if args.resume else None,
    )
    try:
        if not args.resume:
            save_source_snapshot(context.run_dir)
            for name in NEW_FILES:
                shutil.copy2(Path(__file__).with_name(name), context.run_dir / "source" / name)
            shutil.copy2(
                plan_directory(p) / "frozen_protocol.json", context.run_dir / "frozen_protocol.json"
            )
            atomic_write_json(
                context.run_dir / "encoder_configs.json",
                {
                    "pretrained_reference": _json_safe(config.model),
                    "random_control": _json_safe(random_config.model),
                },
            )
            atomic_write_json(
                context.run_dir / "experiment_plan.json",
                {
                    "seed": seed,
                    "protocol_sha256": frozen["protocol_sha256"],
                    "conditions": p["conditions"],
                    "fractions": p["fractions"],
                    "test_evaluated": False,
                },
            )
        elif (
            json.loads((context.run_dir / "experiment_plan.json").read_text())["protocol_sha256"]
            != frozen["protocol_sha256"]
        ):
            raise ValueError("Resume protocol differs from the original run.")
        print(f"Preparing E01C data seed={seed}; validation only", flush=True)
        bundle = load_dataset(config.data)
        parent = reference_run(p, seed)
        if sha256_file(bundle.split_path) != sha256_file(parent / "split_manifest.json"):
            raise ValueError("Parent/control data split mismatch.")
        shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
        save_label_encoder(bundle.label_encoder, context.run_dir)
        reference_cache = Path(p["pretrained_cache"]) / f"seed{seed}"
        pretrained = validate_cache(
            reference_cache,
            cache_signature(config, bundle),
            {"train": bundle.ids_train, "validation": bundle.ids_val},
        )
        if pretrained is None:
            raise ValueError(
                "Verified pretrained caches are required; do not re-extract/mix protocols."
            )
        stats = raw_statistics(config, bundle)
        print("Reference cache verified; extracting/reusing a guarded random encoder", flush=True)
        random, extraction = guarded_random_features(
            random_config,
            bundle,
            Path(p["random_cache"]) / f"seed{seed}",
        )
        atomic_write_json(
            context.run_dir / "feature_extraction.json",
            {
                "reference_cache_hit": True,
                "random": extraction,
                "pretrained_loading_guard_enabled": True,
                "random_pretrained_checkpoint_loaded": False,
            },
        )
        reference_train, train_controls = condition_arrays(
            pretrained["train"], random["train"], stats["train"]
        )
        _, validation_controls = condition_arrays(
            pretrained["validation"], random["validation"], stats["validation"]
        )
        class_names = [str(value) for value in bundle.label_encoder.classes_]
        results = []
        for fraction in p["fractions"]:
            tag = f"{fraction * 100:g}"
            indices = stratified_train_subset_indices(
                bundle.y_train, bundle.ids_train, fraction, seed
            )
            ids = bundle.ids_train[indices]
            np.testing.assert_array_equal(ids, np.load(parent / f"train_ids_{tag}.npy"))
            np.save(context.run_dir / f"train_ids_{tag}.npy", ids)
            shutil.copy2(parent / f"cv_folds_{tag}.npy", context.run_dir / f"cv_folds_{tag}.npy")
            for name in p["conditions"]:
                destination = context.run_dir / f"{name}_fraction_{tag}"
                metric_path = destination / "metrics.json"
                if metric_path.exists():
                    row = json.loads(metric_path.read_text())
                    if (
                        row["protocol_sha256"] != frozen["protocol_sha256"]
                        or row["seed"] != seed
                        or row["condition"] != name
                        or row["train_fraction"] != fraction
                        or row["train_ids_sha256"] != train_subset_sha256(ids)
                        or row["test_evaluated"]
                    ):
                        raise ValueError("Completed condition belongs to a different protocol.")
                    for required in (
                        "classifier.joblib",
                        "cv_results.csv",
                        "kernel_geometry.json",
                        "validation_predictions.csv",
                    ):
                        if not (destination / required).is_file():
                            raise ValueError("Incomplete completed-condition artifact set.")
                    results.append(row)
                    print(f"Preserved {name} fraction={fraction:g}", flush=True)
                    continue
                if destination.exists() and any(destination.iterdir()):
                    shutil.move(
                        str(destination),
                        str(
                            destination.with_name(
                                destination.name + f"_interrupted_{time.time_ns()}"
                            )
                        ),
                    )
                destination.mkdir(exist_ok=True)
                kind = "statistics" if name == "statistics_matched_kernel" else "fusion"
                print(
                    f"Starting {name} seed={seed} fraction={fraction:g} n={len(indices)}",
                    flush=True,
                )
                result = fit_controlled_rbf(
                    train_controls[name][indices],
                    bundle.y_train[indices],
                    reference_train[indices],
                    kind=kind,
                    reference_dim=1024,
                    c_values=p["fixed_search"]["C"],
                    gamma_tokens=p["fixed_search"]["gamma_tokens"],
                    n_jobs=p["execution"]["n_jobs"],
                    device=p["execution"]["kernel_device"],
                    cache_size_mb=config.svm.cache_size_mb,
                    max_iter=config.svm.max_iter,
                )
                pd.DataFrame(result.cv_results_).to_csv(destination / "cv_results.csv", index=False)
                atomic_write_json(destination / "kernel_geometry.json", result.geometry_)
                _atomic_joblib_dump(result.best_estimator_, destination / "classifier.joblib")
                predicted = result.predict(validation_controls[name])
                row = {
                    "condition": name,
                    "seed": seed,
                    "train_fraction": fraction,
                    "train_samples": len(indices),
                    "train_ids_sha256": train_subset_sha256(ids),
                    "dataset_sha256": bundle.dataset_sha256,
                    "protocol_sha256": frozen["protocol_sha256"],
                    "best_params": result.best_params_,
                    "cv_score": result.best_score_,
                    "fit_seconds": result.backend_timings_["total_seconds"],
                    "execution_backend": result.backend_timings_,
                    "test_evaluated": False,
                    "validation_metrics": classification_metrics(
                        bundle.y_val, predicted, class_names, include_details=True
                    ),
                }
                pd.DataFrame(
                    {
                        "sample_id": bundle.ids_val,
                        "true_index": bundle.y_val,
                        "predicted_index": predicted,
                        "true_label": [class_names[int(v)] for v in bundle.y_val],
                        "predicted_label": [class_names[int(v)] for v in predicted],
                    }
                ).to_csv(destination / "validation_predictions.csv", index=False)
                atomic_write_json(metric_path, row)
                results.append(row)
                atomic_write_json(context.run_dir / "metrics_partial.json", {"results": results})
                print(
                    f"Completed {name} fraction={fraction:g} val_macro_f1={row['validation_metrics']['macro_f1']:.6f} "
                    f"fit_seconds={row['fit_seconds']:.1f}",
                    flush=True,
                )
        atomic_write_json(
            context.run_dir / "metrics.json",
            {"seed": seed, "results": results, "test_evaluated": False},
        )
        context.set_status("completed", test_evaluated=False)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
