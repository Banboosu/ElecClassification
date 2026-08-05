from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcn_moment.config import ExperimentConfig, load_config, with_random_seed
from tcn_moment.data import (
    DatasetBundle,
    load_dataset,
    save_label_encoder,
    stratified_train_subset_indices,
    train_subset_sha256,
)
from tcn_moment.experiment import RunContext, prepare_run
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.metrics import classification_metrics
from tcn_moment.train_moment import MOMENT_PROTOCOL_VERSION, require_torch_and_moment
from tcn_moment.train_moment_svm import (
    SVM_PROTOCOL_VERSION,
    _atomic_joblib_dump,
    _cv_results,
    _data_record,
    _make_prediction_frame,
    build_paper_svm_search,
    extract_frozen_moment_features,
)
from tcn_moment.training_utils import seed_everything


FEW_SHOT_PROTOCOL_VERSION = 1


def _fraction_tag(fraction: float) -> str:
    return f"{fraction * 100:g}".replace(".", "p")


def _subset_record(
    bundle: DatasetBundle,
    indices: np.ndarray,
    fraction: float,
) -> dict[str, Any]:
    labels, counts = np.unique(bundle.y_train[indices], return_counts=True)
    sample_ids = bundle.ids_train[indices]
    return {
        "requested_fraction": fraction,
        "full_train_count": len(bundle.y_train),
        "selected_train_count": len(indices),
        "selected_fraction": len(indices) / len(bundle.y_train),
        "class_counts": {
            str(bundle.label_encoder.inverse_transform([int(label)])[0]): int(count)
            for label, count in zip(labels, counts, strict=True)
        },
        "sample_ids_sha256": train_subset_sha256(sample_ids),
    }


def _run(
    config: ExperimentConfig,
    context: RunContext,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    tqdm: Any,
    MOMENTPipeline: Any,
) -> None:
    if config.data.train_fraction != 1:
        raise ValueError(
            "The MOMENT few-shot runner extracts the full split once; "
            "data.train_fraction must be 1."
        )

    bundle = load_dataset(config.data)
    shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
    save_label_encoder(bundle.label_encoder, context.run_dir)
    extracted, extraction = extract_frozen_moment_features(
        config,
        bundle,
        torch,
        DataLoader,
        TensorDataset,
        tqdm,
        MOMENTPipeline,
    )

    class_names = [str(name) for name in bundle.label_encoder.classes_.tolist()]
    fraction_results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    total_svm_fit_seconds = 0.0

    for fraction in config.svm.few_shot_fractions:
        indices = stratified_train_subset_indices(
            bundle.y_train,
            bundle.ids_train,
            fraction,
            config.data.random_state,
        )
        if len(indices) > config.svm.max_samples:
            raise ValueError(
                f"Fraction {fraction:g} selects {len(indices)} samples, exceeding "
                f"svm.max_samples={config.svm.max_samples}. Reduce the largest fraction "
                "so the declared label budget is not silently resampled."
            )

        train_features = extracted["train"][0][indices]
        train_labels = extracted["train"][1][indices]
        train_ids = bundle.ids_train[indices]
        search = build_paper_svm_search(config)
        fit_started = time.perf_counter()
        search.fit(train_features, train_labels)
        fit_seconds = time.perf_counter() - fit_started
        total_svm_fit_seconds += fit_seconds
        classifier = search.best_estimator_

        validation_predictions = classifier.predict(extracted["validation"][0])
        test_predictions = classifier.predict(extracted["test"][0])
        validation_metrics = classification_metrics(
            extracted["validation"][1],
            validation_predictions,
            class_names,
            include_details=True,
        )
        test_metrics = classification_metrics(
            extracted["test"][1],
            test_predictions,
            class_names,
            include_details=True,
        )

        tag = _fraction_tag(fraction)
        _atomic_joblib_dump(
            classifier,
            context.run_dir / f"moment_rbf_svm_fraction_{tag}.joblib",
        )
        np.save(
            context.run_dir / f"train_subset_fraction_{tag}_sample_ids.npy",
            train_ids.astype(str),
        )
        for split, ids, labels, predictions in (
            (
                "validation",
                bundle.ids_val,
                extracted["validation"][1],
                validation_predictions,
            ),
            ("test", bundle.ids_test, extracted["test"][1], test_predictions),
        ):
            frame = _make_prediction_frame(
                split,
                ids,
                labels,
                predictions,
                class_names,
            )
            frame.insert(0, "train_fraction", fraction)
            prediction_frames.append(frame)

        subset = _subset_record(bundle, indices, fraction)
        fraction_result = {
            "train_fraction": fraction,
            "train_subset": subset,
            "svm": {
                "kernel": "rbf",
                "gamma": config.svm.gamma,
                "c_values": list(config.svm.c_values),
                "cv_folds": config.svm.cv_folds,
                "best_params": {
                    key: float(value) for key, value in search.best_params_.items()
                },
                "best_cross_validation_accuracy": float(search.best_score_),
                "number_of_support_vectors": int(classifier.n_support_.sum()),
                "support_vectors_per_class": classifier.n_support_.astype(int).tolist(),
                "cv_results": _cv_results(search),
            },
            "svm_fit_seconds": fit_seconds,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }
        fraction_results.append(fraction_result)
        print(
            f"fraction={fraction:g} samples={len(indices)} "
            f"best_C={search.best_params_['C']} "
            f"test_macro_f1={test_metrics['macro_f1']:.6f}"
        )
        atomic_write_json(
            context.run_dir / "metrics_partial.json",
            {
                "seed": config.data.random_state,
                "completed_fractions": fraction_results,
            },
        )

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        context.run_dir / "predictions.csv",
        index=False,
    )
    result = {
        "model": "MOMENT_RBF_SVM_FEW_SHOT",
        "run_name": context.run_name,
        "seed": config.data.random_state,
        "data": _data_record(bundle, config),
        "protocol": {
            "few_shot_protocol_version": FEW_SHOT_PROTOCOL_VERSION,
            "svm_protocol_version": SVM_PROTOCOL_VERSION,
            "moment_protocol_version": MOMENT_PROTOCOL_VERSION,
            "fractions": list(config.svm.few_shot_fractions),
            "nested_stratified_subsets": True,
            "subset_random_state": config.data.random_state,
            "same_split_and_subset_ids_for_all_models": True,
            "paper_aligned_downstream_classifier": True,
            "backbone_frozen": True,
            "model_initialization": extraction["model_initialization"],
            "pretrained_checkpoint_loaded": extraction[
                "pretrained_checkpoint_loaded"
            ],
            "initialization_seed": extraction["initialization_seed"],
            "features_extracted_once_per_seed": True,
            "validation_split_used_for_selection": False,
            "test_split_used_for_selection": False,
            "selection_metric": "5-fold cross-validation accuracy",
            "feature_dimension": extraction["feature_dimension"],
            "patch_len": extraction["patch_len"],
            "patch_stride": extraction["patch_stride"],
        },
        "execution": {
            **extraction,
            "total_svm_fit_seconds": total_svm_fit_seconds,
        },
        "fractions": fraction_results,
    }
    atomic_write_json(context.run_dir / "metrics.json", result)
    (context.run_dir / "metrics_partial.json").unlink(missing_ok=True)
    print(f"Saved artifacts to {context.run_dir}")


def train(
    config: ExperimentConfig,
    config_path: Path,
    *,
    run_name: str | None = None,
) -> None:
    if not config.model.freeze_backbone or config.model.unfreeze_last_n_layers != 0:
        raise ValueError("MOMENT RBF-SVM few-shot evaluation requires a frozen backbone.")
    torch, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()
    seed_everything(torch, config.data.random_state)
    context = prepare_run(
        model_name="MOMENT_RBF_SVM_FEW_SHOT",
        base_output_dir=config.training.output_dir,
        config=config,
        config_path=config_path,
        torch=torch,
        run_name=run_name,
        resume_dir=None,
    )
    try:
        _run(config, context, torch, DataLoader, TensorDataset, tqdm, MOMENTPipeline)
    except KeyboardInterrupt:
        context.set_status("interrupted", message="Restart this non-resumable run.")
        raise
    except BaseException as exc:
        context.set_status("failed", error_type=type(exc).__name__, message=str(exc))
        raise
    else:
        context.set_status("completed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen MOMENT representations on nested few-shot training subsets."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/few_shot/moment_svm.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--seed", type=int, help="Override random seed and split manifest.")
    parser.add_argument("--run-name", help="Unique output directory name.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.seed is not None:
        config = with_random_seed(config, args.seed)
    train(config, config_path, run_name=args.run_name)


if __name__ == "__main__":
    main()
