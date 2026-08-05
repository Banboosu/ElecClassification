from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from tcn_moment.config import ExperimentConfig, load_config
from tcn_moment.data import DatasetBundle, load_dataset
from tcn_moment.io_utils import atomic_write_json


BATTERY_SAFETY_PROTOCOL_VERSION = 1


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics_from_predictions(
    y_true: np.ndarray,
    y_predicted_positive: np.ndarray,
    critical_index: int,
) -> dict[str, float | int]:
    """Calculate safety-oriented one-vs-rest metrics for a critical class."""
    true_positive_mask = np.asarray(y_true) == critical_index
    predicted_positive_mask = np.asarray(y_predicted_positive, dtype=bool)
    if true_positive_mask.shape != predicted_positive_mask.shape:
        raise ValueError("y_true and y_predicted_positive must have equal shapes.")

    tp = int(np.sum(true_positive_mask & predicted_positive_mask))
    fn = int(np.sum(true_positive_mask & ~predicted_positive_mask))
    fp = int(np.sum(~true_positive_mask & predicted_positive_mask))
    tn = int(np.sum(~true_positive_mask & ~predicted_positive_mask))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    negative_predictive_value = _safe_divide(tn, tn + fn)
    f2 = _safe_divide(5 * precision * recall, 4 * precision + recall)

    return {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "battery_precision": precision,
        "battery_recall": recall,
        "miss_rate": _safe_divide(fn, tp + fn),
        "specificity": specificity,
        "false_positive_rate": _safe_divide(fp, tn + fp),
        "negative_predictive_value": negative_predictive_value,
        "f2": f2,
        "binary_accuracy": _safe_divide(tp + tn, tp + fn + fp + tn),
        "misses_per_1000_battery": 1000 * _safe_divide(fn, tp + fn),
        "false_alarms_per_1000_non_battery": 1000 * _safe_divide(fp, tn + fp),
    }


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    critical_index: int,
) -> dict[str, float]:
    binary_true = (np.asarray(y_true) == critical_index).astype(np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if binary_true.shape != scores.shape:
        raise ValueError("y_true and scores must have equal shapes.")
    if not np.isfinite(scores).all():
        raise ValueError("Critical-class scores must all be finite.")
    return {
        "average_precision": float(average_precision_score(binary_true, scores)),
        "roc_auc": float(roc_auc_score(binary_true, scores)),
        "prevalence": float(binary_true.mean()),
    }


def threshold_for_target_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    critical_index: int,
    target_recall: float,
) -> float:
    """Choose the highest validation threshold that attains the requested recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in the interval (0, 1].")
    positive_scores = np.asarray(scores, dtype=np.float64)[
        np.asarray(y_true) == critical_index
    ]
    if not len(positive_scores):
        raise ValueError("The validation split has no critical-class examples.")
    if not np.isfinite(positive_scores).all():
        raise ValueError("Critical-class scores must all be finite.")

    required_true_positives = int(math.ceil(target_recall * len(positive_scores)))
    descending = np.sort(positive_scores)[::-1]
    return float(descending[required_true_positives - 1])


def threshold_for_max_fbeta(
    y_true: np.ndarray,
    scores: np.ndarray,
    critical_index: int,
    *,
    beta: float = 2.0,
) -> float:
    """Select a validation threshold maximizing F-beta, preferring recall on ties."""
    if beta <= 0:
        raise ValueError("beta must be positive.")
    binary_true = np.asarray(y_true) == critical_index
    scores = np.asarray(scores, dtype=np.float64)
    if binary_true.shape != scores.shape:
        raise ValueError("y_true and scores must have equal shapes.")
    if not binary_true.any():
        raise ValueError("The validation split has no critical-class examples.")
    if not np.isfinite(scores).all():
        raise ValueError("Critical-class scores must all be finite.")

    candidates = np.unique(scores)
    best: tuple[float, float, float, float] | None = None
    best_threshold = float(candidates[0])
    beta_squared = beta**2
    for threshold in candidates:
        metrics = binary_metrics_from_predictions(
            y_true,
            scores >= threshold,
            critical_index,
        )
        precision = float(metrics["battery_precision"])
        recall = float(metrics["battery_recall"])
        fbeta = _safe_divide(
            (1 + beta_squared) * precision * recall,
            beta_squared * precision + recall,
        )
        # Safety-oriented tie breaking: recall, then precision, then higher threshold.
        key = (fbeta, recall, precision, float(threshold))
        if best is None or key > best:
            best = key
            best_threshold = float(threshold)
    return best_threshold


def evaluate_operating_points(
    *,
    validation_labels: np.ndarray,
    validation_predictions: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_predictions: np.ndarray,
    test_scores: np.ndarray,
    critical_index: int,
    target_recalls: tuple[float, ...],
    default_prediction_selection: str = "model multiclass argmax; no threshold tuning",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ranking": {
            "validation": ranking_metrics(
                validation_labels, validation_scores, critical_index
            ),
            "test": ranking_metrics(test_labels, test_scores, critical_index),
        },
        "operating_points": {},
    }

    result["operating_points"]["argmax"] = {
        "selection": default_prediction_selection,
        "threshold": None,
        "validation": binary_metrics_from_predictions(
            validation_labels,
            validation_predictions == critical_index,
            critical_index,
        ),
        "test": binary_metrics_from_predictions(
            test_labels,
            test_predictions == critical_index,
            critical_index,
        ),
    }

    thresholds: list[tuple[str, float, str]] = [
        (
            "max_f2",
            threshold_for_max_fbeta(
                validation_labels,
                validation_scores,
                critical_index,
                beta=2.0,
            ),
            "maximize F2 on validation",
        )
    ]
    for target in target_recalls:
        thresholds.append(
            (
                f"recall_{target:.3f}".rstrip("0").rstrip("."),
                threshold_for_target_recall(
                    validation_labels,
                    validation_scores,
                    critical_index,
                    target,
                ),
                f"highest validation threshold with battery recall >= {target:g}",
            )
        )

    for name, threshold, selection in thresholds:
        result["operating_points"][name] = {
            "selection": selection,
            "threshold": threshold,
            "validation": binary_metrics_from_predictions(
                validation_labels,
                validation_scores >= threshold,
                critical_index,
            ),
            "test": binary_metrics_from_predictions(
                test_labels,
                test_scores >= threshold,
                critical_index,
            ),
        }
    return result


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _critical_index(bundle: DatasetBundle, critical_label: str) -> int:
    classes = [str(value) for value in bundle.label_encoder.classes_.tolist()]
    if critical_label not in classes:
        raise ValueError(f"Critical label {critical_label!r} not found in classes {classes}.")
    return classes.index(critical_label)


def _prediction_frame(
    *,
    split: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    critical_index: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split,
            "sample_id": sample_ids.astype(str),
            "true_index": labels.astype(int),
            "predicted_index": predictions.astype(int),
            "true_label": [class_names[int(value)] for value in labels],
            "predicted_label": [class_names[int(value)] for value in predictions],
            "battery_is_true": labels == critical_index,
            "battery_is_argmax": predictions == critical_index,
            "battery_score": scores,
        }
    )


def _load_run(run_dir: Path) -> tuple[ExperimentConfig, dict[str, Any]]:
    config_path = run_dir / "resolved_config.json"
    metrics_path = run_dir / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Run directory must contain resolved_config.json and metrics.json: {run_dir}"
        )
    config = load_config(config_path)
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    return config, metrics


def _infer_tcn_or_cnn(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    metrics: dict[str, Any],
    bundle: DatasetBundle,
    device_name: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import torch

    from tcn_moment.train_tcn import CNNClassifier, TCNClassifier, make_loader, select_device

    model_name = str(metrics["model"])
    model_class = TCNClassifier if model_name == "TCN" else CNNClassifier
    device = select_device(device_name)
    model = model_class(
        bundle.num_classes,
        config.tcn_model.channels,
        config.tcn_model.kernel_size,
        config.tcn_model.dropout,
    ).to(device)
    weights_path = run_dir / f"{model_name.lower()}_classifier_best.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    values = (
        ("validation", bundle.x_val, bundle.mask_val, bundle.y_val),
        ("test", bundle.x_test, bundle.mask_test, bundle.y_test),
    )
    with torch.inference_mode():
        for split, x, mask, labels in values:
            loader = make_loader(
                x,
                mask,
                labels,
                config.tcn_training.batch_size,
                0,
                shuffle=False,
            )
            probability_chunks = []
            prediction_chunks = []
            for batch_x, batch_mask, _ in loader:
                logits = model(batch_x.to(device), batch_mask.to(device))
                probabilities = torch.softmax(logits.float(), dim=1)
                probability_chunks.append(probabilities.cpu())
                prediction_chunks.append(torch.argmax(logits, dim=1).cpu())
            outputs[split] = (
                labels.copy(),
                torch.cat(prediction_chunks).numpy(),
                torch.cat(probability_chunks).numpy(),
            )
    return outputs


def _infer_moment(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    bundle: DatasetBundle,
    device_name: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from tcn_moment.train_moment import (
        build_model,
        configure_trainable_parameters,
        forward_logits,
        make_loader,
        require_torch_and_moment,
        select_device,
        set_num_classes,
    )
    from tcn_moment.training_utils import load_model_weights

    torch, DataLoader, TensorDataset, _, MOMENTPipeline = require_torch_and_moment()
    device = select_device(torch, device_name)
    model = build_model(config, MOMENTPipeline, bundle.num_classes)
    set_num_classes(model, bundle.num_classes)
    model.init()
    model.to(device)
    configure_trainable_parameters(model, config)
    load_model_weights(
        torch=torch,
        model=model,
        path=run_dir / "moment_classifier_best.pt",
        device=device,
    )
    model.eval()
    amp_enabled = bool(config.training.amp and device.type == "cuda")
    pin_memory = device.type == "cuda"

    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    values = (
        ("validation", bundle.x_val, bundle.mask_val, bundle.y_val),
        ("test", bundle.x_test, bundle.mask_test, bundle.y_test),
    )
    with torch.inference_mode():
        for split, x, mask, labels in values:
            loader = make_loader(
                torch,
                DataLoader,
                TensorDataset,
                x,
                mask,
                labels,
                config.training.evaluation_batch_size,
                0,
                shuffle=False,
                pin_memory=pin_memory,
                prefetch_factor=config.training.prefetch_factor,
            )
            probability_chunks = []
            prediction_chunks = []
            for batch_x, batch_mask, _ in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = forward_logits(model, batch_x, batch_mask)
                probabilities = torch.softmax(logits.float(), dim=1)
                probability_chunks.append(probabilities.cpu())
                prediction_chunks.append(torch.argmax(logits, dim=1).cpu())
            outputs[split] = (
                labels.copy(),
                torch.cat(prediction_chunks).numpy(),
                torch.cat(probability_chunks).numpy(),
            )
    return outputs


def _extract_moment_evaluation_features(
    config: ExperimentConfig,
    bundle: DatasetBundle,
    device_name: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tcn_moment.train_moment import (
        build_model,
        cache_features,
        make_loader,
        require_torch_and_moment,
        select_device,
        set_num_classes,
    )

    torch, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()
    device = select_device(torch, device_name)
    model = build_model(config, MOMENTPipeline, bundle.num_classes)
    set_num_classes(model, bundle.num_classes)
    model.init()
    model.to(device)
    model.eval()
    amp_enabled = bool(config.training.amp and device.type == "cuda")
    pin_memory = device.type == "cuda"
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    values = (
        ("validation", bundle.x_val, bundle.mask_val, bundle.y_val),
        ("test", bundle.x_test, bundle.mask_test, bundle.y_test),
    )
    for split, x, mask, labels in values:
        loader = make_loader(
            torch,
            DataLoader,
            TensorDataset,
            x,
            mask,
            labels,
            config.training.feature_extraction_batch_size,
            0,
            shuffle=False,
            pin_memory=pin_memory,
            prefetch_factor=config.training.prefetch_factor,
        )
        features, cached_labels = cache_features(
            torch,
            model,
            loader,
            device,
            tqdm,
            f"battery safety: {split} MOMENT features",
            amp_enabled=amp_enabled,
        )
        outputs[split] = (features.numpy(), cached_labels.numpy())
    return outputs


def _svm_outputs(
    classifier: Any,
    features: dict[str, tuple[np.ndarray, np.ndarray]],
    critical_index: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    matches = np.flatnonzero(np.asarray(classifier.classes_) == critical_index)
    if len(matches) != 1:
        raise ValueError(
            f"Critical index {critical_index} not found exactly once in SVM classes."
        )
    critical_column = int(matches[0])
    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split, (x, labels) in features.items():
        decisions = np.asarray(classifier.decision_function(x))
        if decisions.ndim != 2:
            raise ValueError("Expected a multiclass SVM decision matrix.")
        outputs[split] = (
            labels.copy(),
            np.asarray(classifier.predict(x), dtype=np.int64),
            decisions,
        )
    return outputs


def _model_name(metrics: dict[str, Any], config: ExperimentConfig) -> str:
    raw_name = str(metrics["model"])
    if raw_name != "MOMENT":
        return raw_name
    if not config.model.freeze_backbone:
        return "MOMENT_FULL_FINETUNE"
    if config.model.unfreeze_last_n_layers:
        return f"MOMENT_LAST_{config.model.unfreeze_last_n_layers}_LAYERS"
    return "MOMENT_LINEAR_PROBE"


def _evaluate_outputs(
    *,
    run_dir: Path,
    model_name: str,
    train_fraction: float,
    score_type: str,
    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    bundle: DatasetBundle,
    critical_label: str,
    target_recalls: tuple[float, ...],
    output_dir: Path,
    variant: str,
) -> dict[str, Any]:
    critical_index = _critical_index(bundle, critical_label)
    class_names = [str(value) for value in bundle.label_encoder.classes_.tolist()]
    validation_labels, validation_predictions, validation_matrix = outputs["validation"]
    test_labels, test_predictions, test_matrix = outputs["test"]
    validation_scores = validation_matrix[:, critical_index]
    test_scores = test_matrix[:, critical_index]
    evaluation = evaluate_operating_points(
        validation_labels=validation_labels,
        validation_predictions=validation_predictions,
        validation_scores=validation_scores,
        test_labels=test_labels,
        test_predictions=test_predictions,
        test_scores=test_scores,
        critical_index=critical_index,
        target_recalls=target_recalls,
    )

    prediction_frame = pd.concat(
        [
            _prediction_frame(
                split="validation",
                sample_ids=bundle.ids_val,
                labels=validation_labels,
                predictions=validation_predictions,
                scores=validation_scores,
                class_names=class_names,
                critical_index=critical_index,
            ),
            _prediction_frame(
                split="test",
                sample_ids=bundle.ids_test,
                labels=test_labels,
                predictions=test_predictions,
                scores=test_scores,
                class_names=class_names,
                critical_index=critical_index,
            ),
        ],
        ignore_index=True,
    )
    safe_variant = variant.replace(".", "p")
    prediction_path = output_dir / "predictions" / f"{run_dir.name}__{safe_variant}.csv"
    _atomic_write_csv(prediction_frame, prediction_path)
    return {
        "run_name": run_dir.name,
        "source_run_dir": str(run_dir),
        "model": model_name,
        "variant": variant,
        "seed": config_seed(bundle),
        "train_fraction": train_fraction,
        "critical_label": critical_label,
        "critical_index": critical_index,
        "score_type": score_type,
        "prediction_file": str(prediction_path),
        **evaluation,
    }


def config_seed(bundle: DatasetBundle) -> int:
    with bundle.split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)
    return int(split["protocol"]["random_state"])


def evaluate_run(
    *,
    run_dir: Path,
    output_dir: Path,
    critical_label: str,
    target_recalls: tuple[float, ...],
    device_name: str,
) -> list[dict[str, Any]]:
    config, source_metrics = _load_run(run_dir)
    bundle = load_dataset(config.data)
    raw_model_name = str(source_metrics["model"])
    model_name = _model_name(source_metrics, config)

    if raw_model_name in {"TCN", "CNN"}:
        outputs = _infer_tcn_or_cnn(
            run_dir=run_dir,
            config=config,
            metrics=source_metrics,
            bundle=bundle,
            device_name=device_name,
        )
        return [
            _evaluate_outputs(
                run_dir=run_dir,
                model_name=model_name,
                train_fraction=config.data.train_fraction,
                score_type="softmax_probability",
                outputs=outputs,
                bundle=bundle,
                critical_label=critical_label,
                target_recalls=target_recalls,
                output_dir=output_dir,
                variant=f"fraction_{config.data.train_fraction:g}",
            )
        ]

    if raw_model_name == "MOMENT":
        outputs = _infer_moment(
            run_dir=run_dir,
            config=config,
            bundle=bundle,
            device_name=device_name,
        )
        return [
            _evaluate_outputs(
                run_dir=run_dir,
                model_name=model_name,
                train_fraction=config.data.train_fraction,
                score_type="softmax_probability",
                outputs=outputs,
                bundle=bundle,
                critical_label=critical_label,
                target_recalls=target_recalls,
                output_dir=output_dir,
                variant=f"fraction_{config.data.train_fraction:g}",
            )
        ]

    if raw_model_name not in {"MOMENT_RBF_SVM", "MOMENT_RBF_SVM_FEW_SHOT"}:
        raise ValueError(f"Unsupported completed run model: {raw_model_name}")

    features = _extract_moment_evaluation_features(config, bundle, device_name)
    classifiers: list[tuple[str, float, Path]] = []
    if raw_model_name == "MOMENT_RBF_SVM":
        classifiers.append(("fraction_1", 1.0, run_dir / "moment_rbf_svm.joblib"))
    else:
        from tcn_moment.train_moment_svm_fewshot import _fraction_tag

        for fraction_result in source_metrics["fractions"]:
            fraction = float(fraction_result["train_fraction"])
            classifiers.append(
                (
                    f"fraction_{fraction:g}",
                    fraction,
                    run_dir / f"moment_rbf_svm_fraction_{_fraction_tag(fraction)}.joblib",
                )
            )

    results = []
    critical_index = _critical_index(bundle, critical_label)
    for variant, fraction, classifier_path in classifiers:
        classifier = joblib.load(classifier_path)
        outputs = _svm_outputs(classifier, features, critical_index)
        results.append(
            _evaluate_outputs(
                run_dir=run_dir,
                model_name=model_name,
                train_fraction=fraction,
                score_type="ovr_decision_function",
                outputs=outputs,
                bundle=bundle,
                critical_label=critical_label,
                target_recalls=target_recalls,
                output_dir=output_dir,
                variant=variant,
            )
        )
    return results


def _flat_rows(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for operating_point, values in result["operating_points"].items():
            row: dict[str, Any] = {
                "run_name": result["run_name"],
                "model": result["model"],
                "variant": result["variant"],
                "seed": result["seed"],
                "train_fraction": result["train_fraction"],
                "operating_point": operating_point,
                "threshold": values["threshold"],
                "validation_average_precision": result["ranking"]["validation"][
                    "average_precision"
                ],
                "validation_roc_auc": result["ranking"]["validation"]["roc_auc"],
                "test_average_precision": result["ranking"]["test"]["average_precision"],
                "test_roc_auc": result["ranking"]["test"]["roc_auc"],
            }
            for split in ("validation", "test"):
                row.update(
                    {
                        f"{split}_{key}": value
                        for key, value in values[split].items()
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    groups = ["model", "train_fraction", "operating_point"]
    metrics = [
        "threshold",
        "validation_battery_precision",
        "validation_battery_recall",
        "validation_false_positive_rate",
        "validation_f2",
        "test_battery_precision",
        "test_battery_recall",
        "test_miss_rate",
        "test_false_positive_rate",
        "test_f2",
        "test_average_precision",
        "test_roc_auc",
        "test_false_negatives",
        "test_false_positives",
    ]
    aggregate = rows.groupby(groups, dropna=False)[metrics].agg(["count", "mean", "std"])
    aggregate.columns = [f"{metric}_{statistic}" for metric, statistic in aggregate.columns]
    return aggregate.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate battery-abnormality miss/false-alarm trade-offs from completed runs. "
            "Thresholds are selected on validation data and applied unchanged to test data."
        )
    )
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/battery_safety"),
    )
    parser.add_argument("--critical-label", default="2")
    parser.add_argument(
        "--target-recalls",
        nargs="+",
        type=float,
        default=(0.95, 0.98, 0.99),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target_recalls = tuple(float(value) for value in args.target_recalls)
    for value in target_recalls:
        if not 0 < value <= 1:
            parser.error("Every --target-recalls value must be in (0, 1].")

    all_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        result_path = output_dir / "run_results" / f"{run_dir.name}.json"
        if result_path.is_file() and not args.overwrite:
            print(f"Reusing completed safety evaluation: {result_path}")
            with result_path.open("r", encoding="utf-8") as file:
                stored = json.load(file)
            all_results.extend(stored["results"])
            continue

        print(f"Evaluating battery safety for {run_dir}...")
        run_started = time.perf_counter()
        results = evaluate_run(
            run_dir=run_dir,
            output_dir=output_dir,
            critical_label=str(args.critical_label),
            target_recalls=target_recalls,
            device_name=str(args.device),
        )
        atomic_write_json(
            result_path,
            {
                "protocol_version": BATTERY_SAFETY_PROTOCOL_VERSION,
                "elapsed_seconds": time.perf_counter() - run_started,
                "results": results,
            },
        )
        all_results.extend(results)

    rows = _flat_rows(all_results)
    aggregate = _aggregate_rows(rows)
    _atomic_write_csv(rows, output_dir / "per_seed_metrics.csv")
    _atomic_write_csv(aggregate, output_dir / "aggregate_metrics.csv")
    atomic_write_json(
        output_dir / "metrics.json",
        {
            "protocol_version": BATTERY_SAFETY_PROTOCOL_VERSION,
            "critical_label": str(args.critical_label),
            "target_recalls": list(target_recalls),
            "threshold_selection_split": "validation",
            "final_evaluation_split": "test",
            "test_used_for_threshold_selection": False,
            "run_count": len(args.run_dirs),
            "result_count": len(all_results),
            "elapsed_seconds": time.perf_counter() - started,
            "per_seed_metrics_csv": str(output_dir / "per_seed_metrics.csv"),
            "aggregate_metrics_csv": str(output_dir / "aggregate_metrics.csv"),
        },
    )
    print(f"Saved battery safety evaluation to {output_dir}")


if __name__ == "__main__":
    main()
