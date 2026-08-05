from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from tqdm.auto import tqdm

from tcn_moment.config import ExperimentConfig, load_config, with_random_seed
from tcn_moment.data import (
    DatasetBundle,
    load_dataset,
    save_label_encoder,
    stratified_train_subset_indices,
)
from tcn_moment.evaluate_battery_safety import (
    evaluate_operating_points,
    ranking_metrics,
)
from tcn_moment.experiment import RunContext, prepare_run
from tcn_moment.io_utils import atomic_torch_save, atomic_write_json
from tcn_moment.train_baselines import (
    FEATURE_NAMES,
    _atomic_joblib_dump,
    extract_statistical_features,
)
from tcn_moment.train_moment_svm import (
    extract_frozen_moment_features,
    select_paper_training_subset,
)
from tcn_moment.train_tcn import TCNClassifier, make_loader, select_device
from tcn_moment.training_utils import seed_everything


BATTERY_BINARY_PROTOCOL_VERSION = 1
DEFAULT_POSITIVE_WEIGHTS = (1.0, 2.0, 4.0)
DEFAULT_TARGET_RECALLS = (0.95, 0.98, 0.99)


def critical_class_index(bundle: DatasetBundle, critical_label: str) -> int:
    classes = [str(value) for value in bundle.label_encoder.classes_.tolist()]
    if critical_label not in classes:
        raise ValueError(f"Critical label {critical_label!r} not found in {classes}.")
    return classes.index(critical_label)


def binary_targets(labels: np.ndarray, critical_index: int) -> np.ndarray:
    return (np.asarray(labels) == critical_index).astype(np.int64)


def select_best_candidate(
    candidates: list[dict[str, Any]],
    *,
    metric: str = "best_validation_average_precision",
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("At least one candidate is required.")
    return max(
        candidates,
        key=lambda item: (float(item[metric]), -float(item["positive_weight"])),
    )


def build_binary_svm_search(
    config: ExperimentConfig,
    positive_weights: tuple[float, ...],
) -> GridSearchCV:
    estimator = SVC(
        kernel="rbf",
        gamma=config.svm.gamma,
        cache_size=config.svm.cache_size_mb,
        max_iter=config.svm.max_iter,
        decision_function_shape="ovr",
    )
    class_weights = [{0: 1.0, 1: float(weight)} for weight in positive_weights]
    return GridSearchCV(
        estimator,
        {
            "C": list(config.svm.c_values),
            "class_weight": class_weights,
        },
        scoring="average_precision",
        cv=config.svm.cv_folds,
        n_jobs=config.svm.n_jobs,
        refit=True,
        return_train_score=False,
    )


def binary_svm_cv_results(search: GridSearchCV) -> list[dict[str, float | int]]:
    results = search.cv_results_
    rows = []
    for index, params in enumerate(results["params"]):
        rows.append(
            {
                "C": float(params["C"]),
                "positive_weight": float(params["class_weight"][1]),
                "mean_validation_average_precision": float(
                    results["mean_test_score"][index]
                ),
                "std_validation_average_precision": float(
                    results["std_test_score"][index]
                ),
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
        "train_subset": bundle.train_subset_record,
        "classes": bundle.label_encoder.classes_.tolist(),
        "normalization": config.data.normalize,
        "max_length": config.data.max_length,
    }


def _protocol_record(
    *,
    critical_label: str,
    positive_weights: tuple[float, ...],
    target_recalls: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "battery_binary_protocol_version": BATTERY_BINARY_PROTOCOL_VERSION,
        "task": f"label_{critical_label}_vs_rest",
        "critical_label": critical_label,
        "positive_weights": list(positive_weights),
        "model_selection_metric": "validation average precision",
        "threshold_selection_split": "validation",
        "target_recalls": list(target_recalls),
        "test_used_for_model_or_threshold_selection": False,
    }


def _prediction_frame(
    *,
    split: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    variant: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": variant,
            "split": split,
            "sample_id": sample_ids.astype(str),
            "battery_is_true": labels.astype(bool),
            "battery_is_default_prediction": predictions.astype(bool),
            "battery_score": scores,
        }
    )


def _evaluate_binary_outputs(
    *,
    validation_labels: np.ndarray,
    validation_predictions: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_predictions: np.ndarray,
    test_scores: np.ndarray,
    target_recalls: tuple[float, ...],
) -> dict[str, Any]:
    return evaluate_operating_points(
        validation_labels=validation_labels,
        validation_predictions=validation_predictions,
        validation_scores=validation_scores,
        test_labels=test_labels,
        test_predictions=test_predictions,
        test_scores=test_scores,
        critical_index=1,
        target_recalls=target_recalls,
        default_prediction_selection="model binary argmax; no threshold tuning",
    )


def _evaluate_tcn_loader(
    model: nn.Module,
    loader: Any,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    labels = []
    predictions = []
    scores = []
    with torch.inference_mode():
        for batch_x, batch_mask, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x, batch_mask)
            total_loss += float(loss_fn(logits, batch_y).cpu()) * len(batch_y)
            probabilities = torch.softmax(logits.float(), dim=1)
            labels.append(batch_y.cpu())
            predictions.append(torch.argmax(logits, dim=1).cpu())
            scores.append(probabilities[:, 1].cpu())
    label_values = torch.cat(labels).numpy()
    score_values = torch.cat(scores).numpy()
    return {
        "loss": total_loss / len(loader.dataset),
        "labels": label_values,
        "predictions": torch.cat(predictions).numpy(),
        "scores": score_values,
        **ranking_metrics(label_values, score_values, 1),
    }


def _tcn_candidate(
    *,
    config: ExperimentConfig,
    bundle: DatasetBundle,
    critical_index: int,
    positive_weight: float,
    context: RunContext,
) -> dict[str, Any]:
    seed_everything(torch, config.data.random_state)
    training = config.tcn_training
    generator = torch.Generator().manual_seed(config.data.random_state)
    train_labels = binary_targets(bundle.y_train, critical_index)
    validation_labels = binary_targets(bundle.y_val, critical_index)
    train_loader = make_loader(
        bundle.x_train,
        bundle.mask_train,
        train_labels,
        training.batch_size,
        training.num_workers,
        shuffle=True,
        generator=generator,
    )
    validation_loader = make_loader(
        bundle.x_val,
        bundle.mask_val,
        validation_labels,
        training.batch_size,
        0,
        shuffle=False,
    )
    device = select_device(training.device)
    model = TCNClassifier(
        2,
        config.tcn_model.channels,
        config.tcn_model.kernel_size,
        config.tcn_model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, positive_weight], dtype=torch.float32, device=device)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training.scheduler_factor,
        patience=training.scheduler_patience,
    )
    best_average_precision = -1.0
    epochs_without_improvement = 0
    history = []
    tag = f"{positive_weight:g}".replace(".", "p")
    best_path = context.run_dir / f"candidate_positive_weight_{tag}.pt"
    started = time.perf_counter()

    for epoch in range(1, training.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_mask, batch_y in tqdm(
            train_loader,
            desc=f"battery TCN w={positive_weight:g} epoch {epoch}",
            leave=False,
        ):
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x, batch_mask)
            loss = loss_fn(logits, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite binary TCN loss for positive weight {positive_weight:g}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training.gradient_clip_norm,
            )
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(batch_y)

        validation = _evaluate_tcn_loader(
            model,
            validation_loader,
            loss_fn,
            device,
        )
        average_precision = float(validation["average_precision"])
        scheduler.step(average_precision)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader.dataset),
                "validation_loss": validation["loss"],
                "validation_average_precision": average_precision,
                "validation_roc_auc": validation["roc_auc"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"weight={positive_weight:g} epoch={epoch} "
            f"val_ap={average_precision:.6f} val_auc={validation['roc_auc']:.6f}"
        )
        if average_precision > best_average_precision + training.early_stopping_min_delta:
            best_average_precision = average_precision
            epochs_without_improvement = 0
            atomic_torch_save(torch, model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= training.early_stopping_patience:
            break

    return {
        "positive_weight": positive_weight,
        "best_validation_average_precision": best_average_precision,
        "stopped_epoch": int(history[-1]["epoch"]),
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "weights_path": str(best_path),
    }


def _run_tcn(
    config: ExperimentConfig,
    context: RunContext,
    *,
    critical_label: str,
    positive_weights: tuple[float, ...],
    target_recalls: tuple[float, ...],
) -> None:
    bundle = load_dataset(config.data)
    critical_index = critical_class_index(bundle, critical_label)
    shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
    save_label_encoder(bundle.label_encoder, context.run_dir)
    candidates = [
        _tcn_candidate(
            config=config,
            bundle=bundle,
            critical_index=critical_index,
            positive_weight=weight,
            context=context,
        )
        for weight in positive_weights
    ]
    selected = select_best_candidate(candidates)
    selected_path = Path(str(selected["weights_path"]))
    final_path = context.run_dir / "battery_tcn_best.pt"
    shutil.copy2(selected_path, final_path)

    device = select_device(config.tcn_training.device)
    model = TCNClassifier(
        2,
        config.tcn_model.channels,
        config.tcn_model.kernel_size,
        config.tcn_model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(final_path, map_location=device, weights_only=True))
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, float(selected["positive_weight"])],
            dtype=torch.float32,
            device=device,
        )
    )
    validation_labels = binary_targets(bundle.y_val, critical_index)
    test_labels = binary_targets(bundle.y_test, critical_index)
    validation_loader = make_loader(
        bundle.x_val,
        bundle.mask_val,
        validation_labels,
        config.tcn_training.batch_size,
        0,
        shuffle=False,
    )
    test_loader = make_loader(
        bundle.x_test,
        bundle.mask_test,
        test_labels,
        config.tcn_training.batch_size,
        0,
        shuffle=False,
    )
    validation = _evaluate_tcn_loader(model, validation_loader, loss_fn, device)
    test = _evaluate_tcn_loader(model, test_loader, loss_fn, device)
    evaluation = _evaluate_binary_outputs(
        validation_labels=validation["labels"],
        validation_predictions=validation["predictions"],
        validation_scores=validation["scores"],
        test_labels=test["labels"],
        test_predictions=test["predictions"],
        test_scores=test["scores"],
        target_recalls=target_recalls,
    )
    predictions = pd.concat(
        [
            _prediction_frame(
                split="validation",
                sample_ids=bundle.ids_val,
                labels=validation["labels"],
                predictions=validation["predictions"],
                scores=validation["scores"],
                variant="battery_binary_tcn",
            ),
            _prediction_frame(
                split="test",
                sample_ids=bundle.ids_test,
                labels=test["labels"],
                predictions=test["predictions"],
                scores=test["scores"],
                variant="battery_binary_tcn",
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(context.run_dir / "predictions.csv", index=False)
    candidate_records = [
        {key: value for key, value in candidate.items() if key != "weights_path"}
        for candidate in candidates
    ]
    for candidate in candidates:
        Path(str(candidate["weights_path"])).unlink(missing_ok=True)
    atomic_write_json(
        context.run_dir / "metrics.json",
        {
            "model": "BATTERY_BINARY_TCN",
            "run_name": context.run_name,
            "seed": config.data.random_state,
            "data": _data_record(bundle, config),
            "protocol": _protocol_record(
                critical_label=critical_label,
                positive_weights=positive_weights,
                target_recalls=target_recalls,
            ),
            "candidates": candidate_records,
            "selected_positive_weight": selected["positive_weight"],
            "selected_validation_average_precision": selected[
                "best_validation_average_precision"
            ],
            **evaluation,
        },
    )


def _classical_candidates(
    *,
    estimator_name: str,
    seed: int,
    positive_weights: tuple[float, ...],
) -> list[tuple[float, Any]]:
    values = []
    for weight in positive_weights:
        class_weight = {0: 1.0, 1: float(weight)}
        if estimator_name == "logistic_regression":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                    class_weight=class_weight,
                ),
            )
        elif estimator_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                n_jobs=-1,
                class_weight=class_weight,
            )
        else:
            raise ValueError(f"Unsupported classical estimator: {estimator_name}")
        values.append((weight, model))
    return values


def _run_stats(
    config: ExperimentConfig,
    context: RunContext,
    *,
    critical_label: str,
    positive_weights: tuple[float, ...],
    target_recalls: tuple[float, ...],
) -> None:
    if config.data.normalize != "none":
        raise ValueError("Statistical battery baselines require data.normalize=none.")
    bundle = load_dataset(config.data)
    critical_index = critical_class_index(bundle, critical_label)
    shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
    save_label_encoder(bundle.label_encoder, context.run_dir)
    features = {
        "train": extract_statistical_features(bundle.x_train, bundle.mask_train),
        "validation": extract_statistical_features(bundle.x_val, bundle.mask_val),
        "test": extract_statistical_features(bundle.x_test, bundle.mask_test),
    }
    labels = {
        "train": binary_targets(bundle.y_train, critical_index),
        "validation": binary_targets(bundle.y_val, critical_index),
        "test": binary_targets(bundle.y_test, critical_index),
    }
    results = {}
    prediction_frames = []
    for estimator_name in ("logistic_regression", "random_forest"):
        candidates = []
        fitted = {}
        for weight, model in _classical_candidates(
            estimator_name=estimator_name,
            seed=config.data.random_state,
            positive_weights=positive_weights,
        ):
            started = time.perf_counter()
            model.fit(features["train"], labels["train"])
            validation_scores = model.predict_proba(features["validation"])[:, 1]
            validation_ap = ranking_metrics(
                labels["validation"],
                validation_scores,
                1,
            )["average_precision"]
            candidates.append(
                {
                    "positive_weight": weight,
                    "best_validation_average_precision": validation_ap,
                    "fit_seconds": time.perf_counter() - started,
                }
            )
            fitted[weight] = model
        selected = select_best_candidate(candidates)
        selected_model = fitted[float(selected["positive_weight"])]
        validation_predictions = selected_model.predict(features["validation"])
        test_predictions = selected_model.predict(features["test"])
        validation_scores = selected_model.predict_proba(features["validation"])[:, 1]
        test_scores = selected_model.predict_proba(features["test"])[:, 1]
        evaluation = _evaluate_binary_outputs(
            validation_labels=labels["validation"],
            validation_predictions=validation_predictions,
            validation_scores=validation_scores,
            test_labels=labels["test"],
            test_predictions=test_predictions,
            test_scores=test_scores,
            target_recalls=target_recalls,
        )
        model_path = context.run_dir / f"battery_{estimator_name}.joblib"
        _atomic_joblib_dump(selected_model, model_path)
        results[estimator_name] = {
            "candidates": candidates,
            "selected_positive_weight": selected["positive_weight"],
            **evaluation,
        }
        prediction_frames.extend(
            [
                _prediction_frame(
                    split="validation",
                    sample_ids=bundle.ids_val,
                    labels=labels["validation"],
                    predictions=validation_predictions,
                    scores=validation_scores,
                    variant=f"battery_binary_{estimator_name}",
                ),
                _prediction_frame(
                    split="test",
                    sample_ids=bundle.ids_test,
                    labels=labels["test"],
                    predictions=test_predictions,
                    scores=test_scores,
                    variant=f"battery_binary_{estimator_name}",
                ),
            ]
        )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        context.run_dir / "predictions.csv",
        index=False,
    )
    atomic_write_json(
        context.run_dir / "metrics.json",
        {
            "model": "BATTERY_BINARY_STATISTICAL",
            "run_name": context.run_name,
            "seed": config.data.random_state,
            "data": _data_record(bundle, config),
            "feature_names": FEATURE_NAMES,
            "protocol": _protocol_record(
                critical_label=critical_label,
                positive_weights=positive_weights,
                target_recalls=target_recalls,
            ),
            "results": results,
        },
    )


def _run_moment_svm(
    config: ExperimentConfig,
    context: RunContext,
    *,
    critical_label: str,
    positive_weights: tuple[float, ...],
    target_recalls: tuple[float, ...],
) -> None:
    if config.data.train_fraction != 1:
        raise ValueError("MOMENT binary SVM requires data.train_fraction=1.")
    from tcn_moment.train_moment import require_torch_and_moment

    moment_torch, DataLoader, TensorDataset, moment_tqdm, MOMENTPipeline = (
        require_torch_and_moment()
    )
    bundle = load_dataset(config.data)
    critical_index = critical_class_index(bundle, critical_label)
    shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
    save_label_encoder(bundle.label_encoder, context.run_dir)
    extracted, extraction = extract_frozen_moment_features(
        config,
        bundle,
        moment_torch,
        DataLoader,
        TensorDataset,
        moment_tqdm,
        MOMENTPipeline,
    )
    validation_binary = binary_targets(extracted["validation"][1], critical_index)
    test_binary = binary_targets(extracted["test"][1], critical_index)
    fraction_results = []
    prediction_frames = []
    for fraction in config.svm.few_shot_fractions:
        indices = stratified_train_subset_indices(
            bundle.y_train,
            bundle.ids_train,
            fraction,
            config.data.random_state,
        )
        train_features = extracted["train"][0][indices]
        train_original_labels = extracted["train"][1][indices]
        train_ids = bundle.ids_train[indices]
        if len(indices) > config.svm.max_samples:
            train_features, train_original_labels, train_ids = (
                select_paper_training_subset(
                    train_features,
                    train_original_labels,
                    train_ids,
                    config.svm.max_samples,
                )
            )
        train_binary = binary_targets(train_original_labels, critical_index)
        search = build_binary_svm_search(config, positive_weights)
        fit_started = time.perf_counter()
        search.fit(train_features, train_binary)
        fit_seconds = time.perf_counter() - fit_started
        classifier = search.best_estimator_
        validation_scores = classifier.decision_function(extracted["validation"][0])
        test_scores = classifier.decision_function(extracted["test"][0])
        validation_predictions = classifier.predict(extracted["validation"][0])
        test_predictions = classifier.predict(extracted["test"][0])
        evaluation = _evaluate_binary_outputs(
            validation_labels=validation_binary,
            validation_predictions=validation_predictions,
            validation_scores=validation_scores,
            test_labels=test_binary,
            test_predictions=test_predictions,
            test_scores=test_scores,
            target_recalls=target_recalls,
        )
        tag = f"{fraction * 100:g}".replace(".", "p")
        _atomic_joblib_dump(
            classifier,
            context.run_dir / f"battery_moment_svm_fraction_{tag}.joblib",
        )
        np.save(
            context.run_dir / f"train_subset_fraction_{tag}_sample_ids.npy",
            train_ids.astype(str),
        )
        fraction_results.append(
            {
                "train_fraction": fraction,
                "selected_train_samples": len(train_features),
                "fit_seconds": fit_seconds,
                "best_params": {
                    "C": float(search.best_params_["C"]),
                    "class_weight": {
                        str(key): float(value)
                        for key, value in search.best_params_["class_weight"].items()
                    },
                },
                "best_cross_validation_average_precision": float(search.best_score_),
                "cv_results": binary_svm_cv_results(search),
                **evaluation,
            }
        )
        prediction_frames.extend(
            [
                _prediction_frame(
                    split="validation",
                    sample_ids=bundle.ids_val,
                    labels=validation_binary,
                    predictions=validation_predictions,
                    scores=validation_scores,
                    variant=f"battery_binary_moment_svm_fraction_{fraction:g}",
                ),
                _prediction_frame(
                    split="test",
                    sample_ids=bundle.ids_test,
                    labels=test_binary,
                    predictions=test_predictions,
                    scores=test_scores,
                    variant=f"battery_binary_moment_svm_fraction_{fraction:g}",
                ),
            ]
        )
        print(
            f"fraction={fraction:g} best_C={search.best_params_['C']} "
            f"best_weight={search.best_params_['class_weight'][1]:g} "
            f"cv_ap={search.best_score_:.6f}"
        )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        context.run_dir / "predictions.csv",
        index=False,
    )
    atomic_write_json(
        context.run_dir / "metrics.json",
        {
            "model": "BATTERY_BINARY_MOMENT_RBF_SVM",
            "run_name": context.run_name,
            "seed": config.data.random_state,
            "data": _data_record(bundle, config),
            "protocol": {
                **_protocol_record(
                    critical_label=critical_label,
                    positive_weights=positive_weights,
                    target_recalls=target_recalls,
                ),
                "features_extracted_once_per_seed": True,
                "training_subsets_match_existing_three_class_few_shot_protocol": True,
                "svm_selection_metric": "training-only CV average precision",
            },
            "execution": extraction,
            "fractions": fraction_results,
        },
    )


def train(
    *,
    model_name: str,
    config: ExperimentConfig,
    config_path: Path,
    run_name: str | None,
    critical_label: str,
    positive_weights: tuple[float, ...],
    target_recalls: tuple[float, ...],
) -> None:
    output_dirs = {
        "tcn": Path("artifacts/battery_binary_tcn"),
        "stats": Path("artifacts/battery_binary_stats"),
        "moment-svm": Path("artifacts/battery_binary_moment_svm"),
    }
    effective_config = (
        replace(config, data=replace(config.data, normalize="none"))
        if model_name == "stats"
        else config
    )
    context = prepare_run(
        model_name=f"BATTERY_BINARY_{model_name.upper()}",
        base_output_dir=output_dirs[model_name],
        config=effective_config,
        config_path=config_path,
        torch=torch,
        run_name=run_name,
        resume_dir=None,
    )
    try:
        if model_name == "tcn":
            _run_tcn(
                config,
                context,
                critical_label=critical_label,
                positive_weights=positive_weights,
                target_recalls=target_recalls,
            )
        elif model_name == "stats":
            _run_stats(
                effective_config,
                context,
                critical_label=critical_label,
                positive_weights=positive_weights,
                target_recalls=target_recalls,
            )
        else:
            _run_moment_svm(
                effective_config,
                context,
                critical_label=critical_label,
                positive_weights=positive_weights,
                target_recalls=target_recalls,
            )
    except BaseException as exc:
        context.set_status("failed", error_type=type(exc).__name__, message=str(exc))
        raise
    else:
        context.set_status("completed")
        print(f"Saved dedicated battery binary results to {context.run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dedicated battery-vs-rest detectors.")
    parser.add_argument("--model", choices=("tcn", "stats", "moment-svm"), required=True)
    parser.add_argument("--config", default="configs/moment.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--critical-label", default="2")
    parser.add_argument(
        "--positive-weights",
        nargs="+",
        type=float,
        default=DEFAULT_POSITIVE_WEIGHTS,
    )
    parser.add_argument(
        "--target-recalls",
        nargs="+",
        type=float,
        default=DEFAULT_TARGET_RECALLS,
    )
    args = parser.parse_args()
    positive_weights = tuple(float(value) for value in args.positive_weights)
    target_recalls = tuple(float(value) for value in args.target_recalls)
    if any(value <= 0 for value in positive_weights):
        parser.error("--positive-weights must be positive.")
    if any(not 0 < value <= 1 for value in target_recalls):
        parser.error("--target-recalls must be in (0, 1].")
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.seed is not None:
        config = with_random_seed(config, args.seed)
    train(
        model_name=args.model,
        config=config,
        config_path=config_path,
        run_name=args.run_name,
        critical_label=str(args.critical_label),
        positive_weights=positive_weights,
        target_recalls=target_recalls,
    )


if __name__ == "__main__":
    main()
