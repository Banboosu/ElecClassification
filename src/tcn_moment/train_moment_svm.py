from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from tcn_moment.config import ExperimentConfig, load_config, with_random_seed
from tcn_moment.data import DatasetBundle, load_dataset, save_label_encoder
from tcn_moment.experiment import RunContext, prepare_run
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.metrics import classification_metrics
from tcn_moment.train_moment import (
    MOMENT_PROTOCOL_VERSION,
    build_model,
    cache_features,
    make_loader,
    require_torch_and_moment,
    select_device,
    set_num_classes,
)
from tcn_moment.training_utils import seed_everything


SVM_PROTOCOL_VERSION = 1


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".joblib.tmp")
    os.close(descriptor)
    try:
        joblib.dump(value, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def select_paper_training_subset(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match MOMENT's official fit_svm helper: stratified sample, random_state=0."""
    if len(features) <= max_samples:
        return features, labels, sample_ids
    selected = train_test_split(
        features,
        labels,
        sample_ids,
        train_size=max_samples,
        random_state=0,
        stratify=labels,
    )
    return selected[0], selected[2], selected[4]


def build_paper_svm_search(config: ExperimentConfig) -> GridSearchCV:
    """Build the RBF-SVM grid used by MOMENT's official classification helper."""
    estimator = SVC(
        kernel="rbf",
        gamma=config.svm.gamma,
        degree=3,
        coef0=0,
        shrinking=True,
        tol=0.001,
        cache_size=config.svm.cache_size_mb,
        class_weight=None,
        verbose=False,
        max_iter=config.svm.max_iter,
        decision_function_shape="ovr",
    )
    return GridSearchCV(
        estimator,
        {"C": list(config.svm.c_values)},
        scoring="accuracy",
        cv=config.svm.cv_folds,
        n_jobs=config.svm.n_jobs,
        refit=True,
        return_train_score=False,
    )


def _cv_results(search: GridSearchCV) -> list[dict[str, float | int]]:
    results = search.cv_results_
    rows = []
    for index, params in enumerate(results["params"]):
        rows.append(
            {
                "C": float(params["C"]),
                "mean_validation_accuracy": float(results["mean_test_score"][index]),
                "std_validation_accuracy": float(results["std_test_score"][index]),
                "rank": int(results["rank_test_score"][index]),
                "mean_fit_seconds": float(results["mean_fit_time"][index]),
                "mean_score_seconds": float(results["mean_score_time"][index]),
            }
        )
    return rows


def _data_record(bundle: DatasetBundle, config: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset_sha256": bundle.dataset_sha256,
        "split_manifest": str(bundle.split_path),
        "split": bundle.split_counts,
        "classes": bundle.label_encoder.classes_.tolist(),
        "short_sequences_excluded": bundle.short_sequence_count,
        "invalid_label_counts": bundle.invalid_label_counts,
        "truncated_sequences": bundle.truncated_sequence_count,
        "max_length": config.data.max_length,
        "normalization": config.data.normalize,
    }


def _make_prediction_frame(
    split: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split,
            "sample_id": sample_ids.astype(str),
            "true_index": labels,
            "predicted_index": predictions,
            "true_label": [class_names[int(value)] for value in labels],
            "predicted_label": [class_names[int(value)] for value in predictions],
        }
    )


def _run(
    config: ExperimentConfig,
    context: RunContext,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    tqdm: Any,
    MOMENTPipeline: Any,
) -> None:
    bundle = load_dataset(config.data)
    shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
    save_label_encoder(bundle.label_encoder, context.run_dir)
    device = select_device(torch, config.training.device)
    model = build_model(config, MOMENTPipeline, bundle.num_classes)
    set_num_classes(model, bundle.num_classes)
    model.init()
    model.to(device)
    model.eval()

    pin_memory = device.type == "cuda"
    loader_args = (torch, DataLoader, TensorDataset)
    split_values = {
        "train": (bundle.x_train, bundle.mask_train, bundle.y_train),
        "validation": (bundle.x_val, bundle.mask_val, bundle.y_val),
        "test": (bundle.x_test, bundle.mask_test, bundle.y_test),
    }
    amp_enabled = bool(config.training.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    extracted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    extraction_seconds: dict[str, float] = {}
    for split, (x, mask, labels) in split_values.items():
        loader = make_loader(
            *loader_args,
            x,
            mask,
            labels,
            config.training.feature_extraction_batch_size,
            config.training.num_workers if split == "train" else 0,
            shuffle=False,
            pin_memory=pin_memory,
            persistent_workers=False,
            prefetch_factor=config.training.prefetch_factor,
        )
        started = time.perf_counter()
        features, cached_labels = cache_features(
            torch,
            model,
            loader,
            device,
            tqdm,
            f"extract {split} features",
            amp_enabled=amp_enabled,
        )
        extraction_seconds[split] = time.perf_counter() - started
        extracted[split] = (features.numpy(), cached_labels.numpy())

    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )
    patch_len = int(getattr(model, "patch_len", 0))
    patch_stride = int(getattr(model.config, "patch_stride_len", patch_len))
    pooled_feature_dim = int(extracted["train"][0].shape[1])
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_features, train_labels = extracted["train"]
    selected_features, selected_labels, selected_ids = select_paper_training_subset(
        train_features,
        train_labels,
        bundle.ids_train,
        config.svm.max_samples,
    )
    search = build_paper_svm_search(config)
    fit_started = time.perf_counter()
    search.fit(selected_features, selected_labels)
    fit_seconds = time.perf_counter() - fit_started
    classifier = search.best_estimator_

    class_names = [str(name) for name in bundle.label_encoder.classes_.tolist()]
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

    _atomic_joblib_dump(classifier, context.run_dir / "moment_rbf_svm.joblib")
    predictions = pd.concat(
        [
            _make_prediction_frame(
                "validation",
                bundle.ids_val,
                extracted["validation"][1],
                validation_predictions,
                class_names,
            ),
            _make_prediction_frame(
                "test",
                bundle.ids_test,
                extracted["test"][1],
                test_predictions,
                class_names,
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(context.run_dir / "predictions.csv", index=False)
    np.save(context.run_dir / "svm_training_sample_ids.npy", selected_ids.astype(str))

    result = {
        "model": "MOMENT_RBF_SVM",
        "run_name": context.run_name,
        "seed": config.data.random_state,
        "data": _data_record(bundle, config),
        "protocol": {
            "svm_protocol_version": SVM_PROTOCOL_VERSION,
            "moment_protocol_version": MOMENT_PROTOCOL_VERSION,
            "paper_aligned_downstream_classifier": True,
            "backbone_frozen": True,
            "mask_aware_pooling": True,
            "feature_dimension": pooled_feature_dim,
            "patch_len": patch_len,
            "patch_stride": patch_stride,
            "training_pool": "training split only",
            "training_samples_before_subsample": int(len(train_features)),
            "training_samples_after_subsample": int(len(selected_features)),
            "subsample_random_state": 0,
            "selection_metric": "5-fold cross-validation accuracy",
            "validation_split_used_for_selection": False,
            "test_split_used_for_selection": False,
        },
        "svm": {
            "kernel": "rbf",
            "gamma": config.svm.gamma,
            "c_values": list(config.svm.c_values),
            "cv_folds": config.svm.cv_folds,
            "n_jobs": config.svm.n_jobs,
            "cache_size_mb": config.svm.cache_size_mb,
            "best_params": {key: float(value) for key, value in search.best_params_.items()},
            "best_cross_validation_accuracy": float(search.best_score_),
            "number_of_support_vectors": int(classifier.n_support_.sum()),
            "support_vectors_per_class": classifier.n_support_.astype(int).tolist(),
            "cv_results": _cv_results(search),
        },
        "execution": {
            "device": str(device),
            "amp_enabled": amp_enabled,
            "feature_extraction_batch_size": config.training.feature_extraction_batch_size,
            "feature_extraction_seconds": extraction_seconds,
            "total_feature_extraction_seconds": float(sum(extraction_seconds.values())),
            "svm_fit_seconds": fit_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "total_parameters": total_parameters,
            "trainable_backbone_parameters": 0,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    atomic_write_json(context.run_dir / "metrics.json", result)
    print(f"Best SVM parameters: {search.best_params_}")
    print(f"Validation Macro-F1: {validation_metrics['macro_f1']:.6f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.6f}")
    print(f"Saved artifacts to {context.run_dir}")


def train(
    config: ExperimentConfig,
    config_path: Path,
    *,
    run_name: str | None = None,
) -> None:
    if not config.model.freeze_backbone or config.model.unfreeze_last_n_layers != 0:
        raise ValueError("MOMENT RBF-SVM requires a fully frozen backbone.")
    torch, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()
    seed_everything(torch, config.data.random_state)
    context = prepare_run(
        model_name="MOMENT_RBF_SVM",
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
        description="Evaluate frozen MOMENT representations with the paper's RBF-SVM protocol."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/moment_svm_rbf.yaml",
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
