from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tcn_moment.config import ExperimentConfig
from tcn_moment.data import DatasetBundle, load_dataset
from tcn_moment.evaluate_battery_safety import (
    _atomic_write_csv,
    _extract_moment_evaluation_features,
    _infer_moment,
    _infer_tcn_or_cnn,
    _load_run,
)
from tcn_moment.io_utils import atomic_write_json

plt.switch_backend("Agg")


ERROR_ANALYSIS_PROTOCOL_VERSION = 1
DEFAULT_LOW_LABEL_FRACTIONS = (0.01, 0.05, 0.10)
MODEL_ORDER = (
    "TCN_full",
    "MOMENT_full",
    "MOMENT_SVM_1pct",
    "MOMENT_SVM_5pct",
    "MOMENT_SVM_10pct",
)
MODEL_DISPLAY = {
    "TCN_full": "TCN (100%)",
    "MOMENT_full": "MOMENT full fine-tune (100%)",
    "MOMENT_SVM_1pct": "MOMENT-SVM (1%)",
    "MOMENT_SVM_5pct": "MOMENT-SVM (5%)",
    "MOMENT_SVM_10pct": "MOMENT-SVM (10%)",
}
CURVE_FEATURES = (
    "effective_length",
    "mean",
    "std",
    "value_range",
    "slope",
    "mean_abs_step",
    "max_abs_step",
    "turning_rate",
    "zero_fraction",
)
FEATURE_DISPLAY = {
    "effective_length": "有效长度",
    "mean": "平均功率",
    "std": "功率标准差",
    "value_range": "功率极差",
    "slope": "整体斜率",
    "mean_abs_step": "平均相邻变化",
    "max_abs_step": "最大相邻跳变",
    "turning_rate": "变化方向转折率",
    "zero_fraction": "零值比例",
}
ERROR_GROUP_DISPLAY = {
    "shared_error": "TCN 与 MOMENT 共同错误",
    "tcn_only_error": "仅 TCN 错误",
    "moment_only_error": "仅 MOMENT 错误",
    "both_correct": "两模型均正确",
}


def _softmax(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def normalize_class_scores(matrix: np.ndarray, score_type: str) -> np.ndarray:
    """Return row-normalized scores without presenting SVM scores as probabilities."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Expected a two-dimensional multiclass score matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Class scores must all be finite.")
    if score_type == "softmax_probability":
        row_sums = values.sum(axis=1)
        if np.any(values < -1e-7) or not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError("Softmax probability rows must be non-negative and sum to one.")
        return values
    if score_type == "ovr_decision_function":
        return _softmax(values)
    raise ValueError(f"Unsupported score type: {score_type}")


def make_length_bins(lengths: np.ndarray, n_bins: int = 4) -> tuple[list[float], list[str]]:
    """Create 3--5 deterministic quantile bins from effective sequence lengths."""
    if not 3 <= n_bins <= 5:
        raise ValueError("n_bins must be between 3 and 5.")
    values = np.asarray(lengths, dtype=np.int64)
    if not len(values) or np.any(values <= 0):
        raise ValueError("Effective lengths must be a non-empty positive array.")
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    interior = np.quantile(values, quantiles, method="nearest").astype(int)
    interior = np.unique(interior)
    if len(interior) < 2:
        raise ValueError("Sequence lengths do not support at least three distinct bins.")
    if len(interior) > 4:
        interior = interior[:4]
    edges: list[float] = [-math.inf, *[float(value) for value in interior], math.inf]
    labels = [f"<= {interior[0]}"]
    labels.extend(f"{left + 1}-{right}" for left, right in pairwise(interior))
    labels.append(f"> {interior[-1]}")
    return edges, labels


def assign_length_bins(
    lengths: Iterable[int], edges: list[float], labels: list[str]
) -> pd.Categorical:
    if len(edges) != len(labels) + 1:
        raise ValueError("Length-bin edges and labels are inconsistent.")
    return pd.cut(
        np.asarray(list(lengths), dtype=np.int64),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
        ordered=True,
    )


def curve_features(values: np.ndarray) -> dict[str, float]:
    sequence = np.asarray(values, dtype=np.float64)
    if sequence.ndim != 1 or not len(sequence):
        raise ValueError("A curve must be a non-empty one-dimensional array.")
    differences = np.diff(sequence)
    if len(sequence) > 1:
        normalized_time = np.linspace(0.0, 1.0, len(sequence))
        slope = float(np.polyfit(normalized_time, sequence, 1)[0])
    else:
        slope = 0.0
    if len(differences) > 1:
        signs = np.sign(differences)
        nonzero = signs[signs != 0]
        turning_rate = (
            float(np.mean(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0.0
        )
    else:
        turning_rate = 0.0
    return {
        "effective_length": float(len(sequence)),
        "mean": float(sequence.mean()),
        "std": float(sequence.std()),
        "value_range": float(sequence.max() - sequence.min()),
        "slope": slope,
        "mean_abs_step": float(np.mean(np.abs(differences))) if len(differences) else 0.0,
        "max_abs_step": float(np.max(np.abs(differences))) if len(differences) else 0.0,
        "turning_rate": turning_rate,
        "zero_fraction": float(np.mean(np.isclose(sequence, 0.0, atol=1e-8))),
    }


def _fraction_tag(fraction: float) -> str:
    percentage = fraction * 100
    if not math.isclose(percentage, round(percentage), abs_tol=1e-9):
        raise ValueError(f"M02 expects integer percentage fractions, got {fraction}.")
    return str(round(percentage))


def _model_key(family: str, fraction: float) -> str:
    if family == "TCN":
        return "TCN_full"
    if family == "MOMENT":
        return "MOMENT_full"
    if family == "MOMENT_SVM":
        return f"MOMENT_SVM_{_fraction_tag(fraction)}pct"
    raise ValueError(f"Unsupported family: {family}")


def _seed_from_config(config: ExperimentConfig) -> int:
    return int(config.data.random_state)


def _check_run_family(metrics: dict[str, Any], expected: set[str], run_dir: Path) -> str:
    family = str(metrics.get("model"))
    if family not in expected:
        raise ValueError(f"Unexpected model {family!r} in {run_dir}; expected {sorted(expected)}.")
    return family


def _class_names(bundle: DatasetBundle) -> list[str]:
    return [str(value) for value in bundle.label_encoder.classes_.tolist()]


def _ordered_svm_scores(classifier: Any, features: np.ndarray, num_classes: int) -> np.ndarray:
    decisions = np.asarray(classifier.decision_function(features), dtype=np.float64)
    if decisions.ndim != 2 or decisions.shape[1] != len(classifier.classes_):
        raise ValueError("Expected a multiclass SVM decision matrix aligned with classes_.")
    ordered = np.full((len(decisions), num_classes), np.nan, dtype=np.float64)
    for source_column, class_index in enumerate(np.asarray(classifier.classes_, dtype=int)):
        if not 0 <= int(class_index) < num_classes:
            raise ValueError(f"SVM class index is out of range: {class_index}")
        ordered[:, int(class_index)] = decisions[:, source_column]
    if not np.isfinite(ordered).all():
        raise ValueError("The SVM does not provide a finite score for every dataset class.")
    return ordered


def _prediction_frame(
    *,
    split: str,
    bundle: DatasetBundle,
    labels: np.ndarray,
    predictions: np.ndarray,
    native_scores: np.ndarray,
    score_type: str,
    family: str,
    train_fraction: float,
    run_dir: Path,
) -> pd.DataFrame:
    class_names = _class_names(bundle)
    normalized_scores = normalize_class_scores(native_scores, score_type)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if split == "validation":
        sample_ids, raw_lengths = bundle.ids_val, bundle.lengths_val
    elif split == "test":
        sample_ids, raw_lengths = bundle.ids_test, bundle.lengths_test
    else:
        raise ValueError(f"Unsupported split: {split}")
    if not (
        len(sample_ids)
        == len(raw_lengths)
        == len(labels)
        == len(predictions)
        == len(native_scores)
    ):
        raise ValueError("Prediction arrays do not align with the requested dataset split.")

    row_index = np.arange(len(predictions))
    predicted_normalized_score = normalized_scores[row_index, predictions]
    competitors = normalized_scores.copy()
    competitors[row_index, predictions] = -np.inf
    margins = predicted_normalized_score - np.max(competitors, axis=1)
    frame = pd.DataFrame(
        {
            "run_name": run_dir.name,
            "source_run_dir": str(run_dir),
            "model_family": family,
            "model_key": _model_key(family, train_fraction),
            "seed": _seed_from_config_from_bundle(bundle),
            "train_fraction": train_fraction,
            "split": split,
            "sample_id": sample_ids.astype(str),
            "raw_length": raw_lengths.astype(int),
            "effective_length": np.minimum(raw_lengths, bundle.x_test.shape[-1]).astype(int),
            "true_index": labels,
            "predicted_index": predictions,
            "true_label": [class_names[int(value)] for value in labels],
            "predicted_label": [class_names[int(value)] for value in predictions],
            "correct": labels == predictions,
            "score_type": score_type,
            "normalized_score_semantics": (
                "model_softmax_probability"
                if score_type == "softmax_probability"
                else "softmax_of_ovr_decision_scores_not_calibrated_probability"
            ),
            "prediction_confidence": predicted_normalized_score,
            "prediction_margin": margins,
        }
    )
    for class_index, class_name in enumerate(class_names):
        frame[f"class_{class_index}_label"] = class_name
        frame[f"class_{class_index}_score"] = native_scores[:, class_index]
        frame[f"class_{class_index}_normalized_score"] = normalized_scores[:, class_index]
    return frame


def _seed_from_config_from_bundle(bundle: DatasetBundle) -> int:
    with bundle.split_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return int(manifest["protocol"]["random_state"])


def _prediction_path(output_dir: Path, run_dir: Path, model_key: str) -> Path:
    return output_dir / "predictions" / f"{run_dir.name}__{model_key}.csv"


def _write_or_reuse_prediction(
    frame: pd.DataFrame, path: Path, *, overwrite: bool
) -> pd.DataFrame:
    if path.is_file() and not overwrite:
        print(f"Reusing per-sample predictions: {path}", flush=True)
        return pd.read_csv(path, dtype={"sample_id": str})
    _atomic_write_csv(frame, path)
    print(f"Saved per-sample predictions: {path}", flush=True)
    return frame


def evaluate_neural_run(
    *,
    run_dir: Path,
    family: str,
    output_dir: Path,
    device_name: str,
    overwrite: bool,
) -> tuple[pd.DataFrame, ExperimentConfig]:
    config, metrics = _load_run(run_dir)
    expected = {"TCN"} if family == "TCN" else {"MOMENT"}
    _check_run_family(metrics, expected, run_dir)
    model_key = _model_key(family, 1.0)
    path = _prediction_path(output_dir, run_dir, model_key)
    if path.is_file() and not overwrite:
        return pd.read_csv(path, dtype={"sample_id": str}), config

    bundle = load_dataset(config.data)
    if family == "TCN":
        outputs = _infer_tcn_or_cnn(
            run_dir=run_dir,
            config=config,
            metrics=metrics,
            bundle=bundle,
            device_name=device_name,
        )
    else:
        outputs = _infer_moment(
            run_dir=run_dir,
            config=config,
            bundle=bundle,
            device_name=device_name,
        )
    frames = []
    for split in ("validation", "test"):
        labels, predictions, scores = outputs[split]
        frames.append(
            _prediction_frame(
                split=split,
                bundle=bundle,
                labels=labels,
                predictions=predictions,
                native_scores=scores,
                score_type="softmax_probability",
                family=family,
                train_fraction=1.0,
                run_dir=run_dir,
            )
        )
    return _write_or_reuse_prediction(pd.concat(frames, ignore_index=True), path, overwrite=overwrite), config


def evaluate_svm_run(
    *,
    run_dir: Path,
    fractions: tuple[float, ...],
    output_dir: Path,
    device_name: str,
    overwrite: bool,
) -> tuple[list[pd.DataFrame], ExperimentConfig]:
    config, metrics = _load_run(run_dir)
    _check_run_family(metrics, {"MOMENT_RBF_SVM_FEW_SHOT"}, run_dir)
    requested: list[tuple[float, Path, str]] = []
    for fraction in fractions:
        key = _model_key("MOMENT_SVM", fraction)
        prediction_path = _prediction_path(output_dir, run_dir, key)
        classifier_path = run_dir / f"moment_rbf_svm_fraction_{_fraction_tag(fraction)}.joblib"
        if not classifier_path.is_file():
            raise FileNotFoundError(f"Missing completed SVM classifier: {classifier_path}")
        requested.append((fraction, prediction_path, key))
    if not overwrite and all(path.is_file() for _, path, _ in requested):
        return [
            pd.read_csv(path, dtype={"sample_id": str}) for _, path, _ in requested
        ], config

    bundle = load_dataset(config.data)
    features = _extract_moment_evaluation_features(config, bundle, device_name)
    frames: list[pd.DataFrame] = []
    for fraction, path, _ in requested:
        if path.is_file() and not overwrite:
            frames.append(pd.read_csv(path, dtype={"sample_id": str}))
            continue
        classifier_path = run_dir / f"moment_rbf_svm_fraction_{_fraction_tag(fraction)}.joblib"
        classifier = joblib.load(classifier_path)
        split_frames = []
        for split in ("validation", "test"):
            split_features, labels = features[split]
            scores = _ordered_svm_scores(classifier, split_features, bundle.num_classes)
            predictions = np.asarray(classifier.predict(split_features), dtype=np.int64)
            split_frames.append(
                _prediction_frame(
                    split=split,
                    bundle=bundle,
                    labels=labels,
                    predictions=predictions,
                    native_scores=scores,
                    score_type="ovr_decision_function",
                    family="MOMENT_SVM",
                    train_fraction=fraction,
                    run_dir=run_dir,
                )
            )
        frames.append(
            _write_or_reuse_prediction(
                pd.concat(split_frames, ignore_index=True), path, overwrite=overwrite
            )
        )
    return frames, config


def add_high_confidence_battery_flags(
    predictions: pd.DataFrame, critical_label: str, quantile: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < quantile < 1:
        raise ValueError("High-confidence quantile must be in (0, 1).")
    frame = predictions.copy()
    frame["battery_high_confidence_prediction"] = False
    rows: list[dict[str, Any]] = []
    for (model_key, seed), indices in frame.groupby(["model_key", "seed"]).groups.items():
        group = frame.loc[indices]
        predicted_battery = group["predicted_label"].astype(str) == critical_label
        battery_predictions = group.loc[predicted_battery, "prediction_margin"]
        threshold = (
            float(battery_predictions.quantile(quantile, interpolation="higher"))
            if len(battery_predictions)
            else math.inf
        )
        high = predicted_battery & (group["prediction_margin"] >= threshold)
        frame.loc[group.index[high], "battery_high_confidence_prediction"] = True
        true_battery = group["true_label"].astype(str) == critical_label
        false_negative = true_battery & ~predicted_battery
        false_positive = ~true_battery & predicted_battery
        high_confidence_false_positive = ~true_battery & high
        rows.append(
            {
                "model_key": model_key,
                "seed": int(seed),
                "battery_count": int(true_battery.sum()),
                "non_battery_count": int((~true_battery).sum()),
                "battery_predictions": int(predicted_battery.sum()),
                "false_negatives": int(false_negative.sum()),
                "false_negative_rate": float(false_negative.sum() / max(1, true_battery.sum())),
                "false_positives": int(false_positive.sum()),
                "false_positive_rate": float(false_positive.sum() / max(1, (~true_battery).sum())),
                "high_confidence_quantile": quantile,
                "high_confidence_margin_threshold": threshold,
                "high_confidence_false_positives": int(high_confidence_false_positive.sum()),
                "high_confidence_false_positives_per_1000_non_battery": float(
                    1000 * high_confidence_false_positive.sum() / max(1, (~true_battery).sum())
                ),
            }
        )
    return frame, pd.DataFrame(rows)


def summarize_battery(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "false_negatives",
        "false_negative_rate",
        "false_positives",
        "false_positive_rate",
        "high_confidence_margin_threshold",
        "high_confidence_false_positives",
        "high_confidence_false_positives_per_1000_non_battery",
    )
    aggregate = rows.groupby("model_key", sort=False)[list(metrics)].agg(["count", "mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def error_directions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    class_names = sorted(
        set(predictions["true_label"].astype(str))
        | set(predictions["predicted_label"].astype(str))
    )
    for (model_key, seed), group in predictions.groupby(["model_key", "seed"], sort=False):
        true_values = group["true_label"].astype(str)
        predicted_values = group["predicted_label"].astype(str)
        for true_label in class_names:
            true_group = group.loc[true_values == true_label]
            denominator = len(true_group)
            for predicted_label in class_names:
                if predicted_label == true_label:
                    continue
                count = int(
                    (
                        (true_values == true_label)
                        & (predicted_values == predicted_label)
                    ).sum()
                )
                rows.append(
                    {
                        "model_key": model_key,
                        "seed": int(seed),
                        "true_label": str(true_label),
                        "predicted_label": str(predicted_label),
                        "true_class_count": denominator,
                        "error_count": count,
                        "error_rate_within_true_class": (
                            float(count / denominator) if denominator else 0.0
                        ),
                    }
                )
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, per_seed
    groups = ["model_key", "true_label", "predicted_label"]
    aggregate = per_seed.groupby(groups, sort=False).agg(
        seed_count=("seed", "count"),
        total_error_count=("error_count", "sum"),
        mean_error_count=("error_count", "mean"),
        std_error_count=("error_count", "std"),
        mean_error_rate=("error_rate_within_true_class", "mean"),
        std_error_rate=("error_rate_within_true_class", "std"),
    ).reset_index()
    aggregate["direction_rank"] = (
        aggregate.groupby(["model_key", "true_label"])["mean_error_count"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return per_seed, aggregate.sort_values(
        ["model_key", "true_label", "direction_rank"], kind="stable"
    )


def confusion_tables(
    predictions: pd.DataFrame, class_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (model_key, seed), group in predictions.groupby(["model_key", "seed"], sort=False):
        true_values = group["true_label"].astype(str)
        predicted_values = group["predicted_label"].astype(str)
        for true_label in class_names:
            true_mask = true_values == true_label
            true_count = int(true_mask.sum())
            for predicted_label in class_names:
                count = int((true_mask & (predicted_values == predicted_label)).sum())
                rows.append(
                    {
                        "model_key": model_key,
                        "seed": int(seed),
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                        "true_class_count": true_count,
                        "row_fraction": float(count / true_count) if true_count else 0.0,
                    }
                )
    per_seed = pd.DataFrame(rows)
    aggregate = per_seed.groupby(
        ["model_key", "true_label", "predicted_label"], sort=False
    ).agg(
        seed_count=("seed", "count"),
        total_count=("count", "sum"),
        mean_count=("count", "mean"),
        std_count=("count", "std"),
        mean_row_fraction=("row_fraction", "mean"),
        std_row_fraction=("row_fraction", "std"),
    ).reset_index()
    return per_seed, aggregate


def length_stratified_metrics(
    predictions: pd.DataFrame, edges: list[float], labels: list[str], class_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.copy()
    frame["length_bin"] = assign_length_bins(frame["effective_length"], edges, labels)
    rows: list[dict[str, Any]] = []
    for (model_key, seed, length_bin), group in frame.groupby(
        ["model_key", "seed", "length_bin"], observed=True, sort=False
    ):
        rows.append(
            {
                "model_key": model_key,
                "seed": int(seed),
                "length_bin": str(length_bin),
                "sample_count": len(group),
                "macro_f1": float(
                    f1_score(
                        group["true_index"],
                        group["predicted_index"],
                        labels=list(range(class_count)),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    per_seed = pd.DataFrame(rows)
    per_seed["length_bin"] = pd.Categorical(
        per_seed["length_bin"], categories=labels, ordered=True
    )
    aggregate = per_seed.groupby(
        ["model_key", "length_bin"], observed=True, sort=False
    ).agg(
        seed_count=("seed", "count"),
        mean_sample_count=("sample_count", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    return per_seed, aggregate


def _test_rows(frame: pd.DataFrame, model_key: str, seed: int) -> pd.DataFrame:
    selected = frame.loc[
        (frame["model_key"] == model_key)
        & (frame["seed"].astype(int) == seed)
        & (frame["split"] == "test")
    ].copy()
    if selected["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate test sample IDs for {model_key}, seed {seed}.")
    return selected


def _paired_full_model_predictions(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    tcn = _test_rows(predictions, "TCN_full", seed).set_index("sample_id")
    moment = _test_rows(predictions, "MOMENT_full", seed).set_index("sample_id")
    if set(tcn.index) != set(moment.index):
        raise ValueError(f"TCN and MOMENT test IDs do not match for seed {seed}.")
    paired = pd.DataFrame(index=tcn.index)
    paired["true_label"] = tcn["true_label"].astype(str)
    if not np.array_equal(
        paired["true_label"].to_numpy(), moment.loc[paired.index, "true_label"].astype(str).to_numpy()
    ):
        raise ValueError(f"TCN and MOMENT labels do not match for seed {seed}.")
    paired["tcn_correct"] = tcn["correct"].astype(bool)
    paired["moment_correct"] = moment.loc[paired.index, "correct"].astype(bool)
    paired["tcn_prediction"] = tcn["predicted_label"].astype(str)
    paired["moment_prediction"] = moment.loc[paired.index, "predicted_label"].astype(str)
    paired["tcn_margin"] = tcn["prediction_margin"].astype(float)
    paired["moment_margin"] = moment.loc[paired.index, "prediction_margin"].astype(float)
    return paired.reset_index()


def _error_group(paired: pd.DataFrame) -> np.ndarray:
    tcn_correct = paired["tcn_correct"].to_numpy(dtype=bool)
    moment_correct = paired["moment_correct"].to_numpy(dtype=bool)
    return np.select(
        [
            ~tcn_correct & ~moment_correct,
            ~tcn_correct & moment_correct,
            tcn_correct & ~moment_correct,
        ],
        ["shared_error", "tcn_only_error", "moment_only_error"],
        default="both_correct",
    )


def build_curve_feature_rows(
    *,
    predictions: pd.DataFrame,
    tcn_configs: dict[int, ExperimentConfig],
    canonical_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    canonical_curves: dict[str, np.ndarray] = {}
    for seed, config in sorted(tcn_configs.items()):
        bundle = load_dataset(config.data)
        paired = _paired_full_model_predictions(predictions, seed).set_index("sample_id")
        paired["error_group"] = _error_group(paired.reset_index())
        if set(bundle.ids_test.astype(str)) != set(paired.index.astype(str)):
            raise ValueError(f"TCN dataset does not match collected predictions for seed {seed}.")
        for sample_id, raw_length, padded in zip(
            bundle.ids_test.astype(str), bundle.lengths_test, bundle.x_test, strict=True
        ):
            effective_length = min(int(raw_length), padded.shape[-1])
            sequence = np.asarray(padded[:effective_length], dtype=np.float64)
            paired_row = paired.loc[sample_id]
            row: dict[str, Any] = {
                "seed": seed,
                "sample_id": sample_id,
                "true_label": str(paired_row["true_label"]),
                "error_group": str(paired_row["error_group"]),
                "raw_length": int(raw_length),
            }
            row.update(curve_features(sequence))
            rows.append(row)
            if seed == canonical_seed:
                canonical_curves[sample_id] = sequence
    return pd.DataFrame(rows), canonical_curves


def summarize_curve_features(feature_rows: pd.DataFrame) -> pd.DataFrame:
    reference = feature_rows.loc[feature_rows["error_group"] == "both_correct"]
    if reference.empty:
        raise ValueError("No samples are correctly classified by both full-label models.")
    rows: list[dict[str, Any]] = []
    for group_name in ERROR_GROUP_DISPLAY:
        group = feature_rows.loc[feature_rows["error_group"] == group_name]
        for feature in CURVE_FEATURES:
            all_values = feature_rows[feature].to_numpy(dtype=float)
            q25, q75 = np.quantile(all_values, [0.25, 0.75])
            robust_scale = float(q75 - q25)
            group_median = float(group[feature].median()) if len(group) else math.nan
            reference_median = float(reference[feature].median())
            difference = group_median - reference_median
            rows.append(
                {
                    "error_group": group_name,
                    "feature": feature,
                    "sample_count": len(group),
                    "median": group_median,
                    "mean": float(group[feature].mean()) if len(group) else math.nan,
                    "reference_both_correct_median": reference_median,
                    "median_difference": difference,
                    "pooled_iqr": robust_scale,
                    "robust_standardized_difference": (
                        difference / robust_scale if robust_scale > 1e-12 else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def curve_characteristics_text(summary: pd.DataFrame) -> str:
    sentences = []
    for group_name in ("shared_error", "tcn_only_error", "moment_only_error"):
        group = summary.loc[summary["error_group"] == group_name].copy()
        sample_count = int(group["sample_count"].iloc[0]) if len(group) else 0
        if sample_count == 0:
            sentences.append(f"{ERROR_GROUP_DISPLAY[group_name]}未出现。")
            continue
        group["absolute_effect"] = group["robust_standardized_difference"].abs()
        top = group.sort_values(
            ["absolute_effect", "feature"], ascending=[False, True], kind="stable"
        ).head(2)
        descriptions = []
        for row in top.itertuples(index=False):
            direction = "更高" if row.median_difference >= 0 else "更低"
            descriptions.append(
                f"{FEATURE_DISPLAY[row.feature]}中位数{direction}"
                f"（{row.median:.3g} vs {row.reference_both_correct_median:.3g}）"
            )
        sentences.append(
            f"{ERROR_GROUP_DISPLAY[group_name]}共 {sample_count} 个五-seed (seed, sample) 单元，"
            + "，".join(descriptions)
            + "。"
        )
    return "".join(sentences)


def _merge_example_predictions(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    keys = ("TCN_full", "MOMENT_full", "MOMENT_SVM_10pct")
    merged: pd.DataFrame | None = None
    for key in keys:
        subset = _test_rows(predictions, key, seed)[
            [
                "sample_id",
                "true_label",
                "predicted_label",
                "correct",
                "prediction_margin",
                "battery_high_confidence_prediction",
            ]
        ].copy()
        suffix = key.replace("MOMENT_SVM_10pct", "svm10").replace("MOMENT_full", "moment").replace("TCN_full", "tcn")
        subset = subset.rename(
            columns={
                column: f"{column}_{suffix}"
                for column in subset.columns
                if column not in {"sample_id", "true_label"}
            }
        )
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(
                subset,
                on="sample_id",
                how="inner",
                validate="one_to_one",
                suffixes=("", "_candidate"),
            )
            candidate = "true_label_candidate"
            if candidate in merged:
                if not np.array_equal(
                    merged["true_label"].astype(str), merged[candidate].astype(str)
                ):
                    raise ValueError("True labels disagree among canonical-seed models.")
                merged = merged.drop(columns=candidate)
    if merged is None or merged.empty:
        raise ValueError("No aligned canonical-seed predictions were found.")
    return merged


def select_typical_examples(predictions: pd.DataFrame, seed: int, critical_label: str) -> pd.DataFrame:
    """Apply predeclared rules in order, excluding samples already selected by an earlier rule."""
    frame = _merge_example_predictions(predictions, seed)
    tcn_correct = frame["correct_tcn"].astype(bool)
    moment_correct = frame["correct_moment"].astype(bool)
    svm_correct = frame["correct_svm10"].astype(bool)
    true_battery = frame["true_label"].astype(str) == critical_label
    pred_columns = ["predicted_label_tcn", "predicted_label_moment", "predicted_label_svm10"]
    margin_columns = ["prediction_margin_tcn", "prediction_margin_moment", "prediction_margin_svm10"]
    missed_battery = pd.concat(
        [(frame[column].astype(str) != critical_label) for column in pred_columns], axis=1
    )
    high_fp = pd.concat(
        [
            (frame[pred].astype(str) == critical_label)
            & frame[high].astype(bool)
            for pred, high in zip(
                pred_columns,
                [
                    "battery_high_confidence_prediction_tcn",
                    "battery_high_confidence_prediction_moment",
                    "battery_high_confidence_prediction_svm10",
                ],
                strict=True,
            )
        ],
        axis=1,
    )
    mean_margin = frame[margin_columns].mean(axis=1)
    max_margin = frame[margin_columns].max(axis=1)
    rules: list[tuple[str, pd.Series, pd.Series, str]] = [
        (
            "battery_false_negative",
            true_battery & missed_battery.any(axis=1),
            missed_battery.sum(axis=1) * 10 + mean_margin,
            "true class is battery; rank by number of models missing it, then mean margin",
        ),
        (
            "battery_high_confidence_false_positive",
            ~true_battery & high_fp.any(axis=1),
            high_fp.sum(axis=1) * 10 + max_margin,
            "non-battery predicted as battery in a run-local top-decile margin set",
        ),
        (
            "shared_tcn_moment_error",
            ~tcn_correct & ~moment_correct,
            frame[["prediction_margin_tcn", "prediction_margin_moment"]].mean(axis=1),
            "TCN and full MOMENT both wrong; rank by their mean prediction margin",
        ),
        (
            "tcn_only_error",
            ~tcn_correct & moment_correct,
            frame["prediction_margin_tcn"],
            "TCN wrong and full MOMENT correct; rank by TCN wrong-prediction margin",
        ),
        (
            "moment_only_error",
            tcn_correct & ~moment_correct,
            frame["prediction_margin_moment"],
            "full MOMENT wrong and TCN correct; rank by MOMENT wrong-prediction margin",
        ),
        (
            "svm10_only_error",
            tcn_correct & moment_correct & ~svm_correct,
            frame["prediction_margin_svm10"],
            "10% MOMENT-SVM wrong and both full-label models correct; rank by SVM margin",
        ),
    ]
    selected_rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for rule_order, (rule_name, eligible, ranking, rule_text) in enumerate(rules, start=1):
        candidates = frame.loc[eligible].copy()
        candidates["selection_score"] = ranking.loc[candidates.index]
        candidates = candidates.loc[~candidates["sample_id"].astype(str).isin(used)]
        candidates = candidates.sort_values(
            ["selection_score", "sample_id"], ascending=[False, True], kind="stable"
        )
        if candidates.empty:
            continue
        row = candidates.iloc[0].to_dict()
        sample_id = str(row["sample_id"])
        used.add(sample_id)
        row.update(
            {
                "rule_order": rule_order,
                "selection_rule": rule_name,
                "selection_rule_definition": rule_text,
                "seed": seed,
            }
        )
        selected_rows.append(row)
    return pd.DataFrame(selected_rows)


def plot_typical_examples(
    selected: pd.DataFrame,
    curves: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[Path, Path]:
    if selected.empty:
        raise ValueError("The fixed selection rules did not yield any typical examples.")
    columns = 2
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.4 * rows), squeeze=False)
    color = "#2864DC"
    for axis, row in zip(axes.flat, selected.itertuples(index=False), strict=False):
        sample_id = str(row.sample_id)
        sequence = curves.get(sample_id)
        if sequence is None:
            raise ValueError(f"Missing canonical raw curve for selected sample {sample_id}.")
        token = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8]
        axis.plot(np.arange(len(sequence)), sequence, color=color, linewidth=1.0)
        axis.set_title(
            f"{row.selection_rule} | sample {token}\n"
            f"true={row.true_label}; TCN={row.predicted_label_tcn}; "
            f"MOMENT={row.predicted_label_moment}; SVM10={row.predicted_label_svm10}",
            fontsize=9,
        )
        axis.set_xlabel("Valid time-step index")
        axis.set_ylabel("Charging power")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    figure.suptitle("M02 deterministic typical-error curves (canonical seed 42)", fontsize=12)
    figure.tight_layout()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "typical_error_curves_seed42.png"
    svg_path = figure_dir / "typical_error_curves_seed42.svg"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


def _ordered_models(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model_key"] = pd.Categorical(result["model_key"], MODEL_ORDER, ordered=True)
    return result.sort_values("model_key", kind="stable")


def _format_mean_std(mean: float, std: float, *, percent: bool = False) -> str:
    scale = 100 if percent else 1
    return f"{scale * mean:.2f} +/- {scale * std:.2f}"


def write_summary(
    *,
    output_dir: Path,
    error_aggregate: pd.DataFrame,
    battery_aggregate: pd.DataFrame,
    length_aggregate: pd.DataFrame,
    curve_text: str,
    selected: pd.DataFrame,
    critical_label: str,
    high_confidence_quantile: float,
    length_labels: list[str],
) -> Path:
    lines = [
        "# M02 lightweight error analysis",
        "",
        "## Fixed protocol",
        "",
        "- Five completed seeds are evaluated without retraining; validation and test predictions are exported.",
        "- Neural class scores are model softmax probabilities. SVM native class scores are one-vs-rest decision values; normalized SVM scores are only softmax transforms for within-model ranking and are not calibrated probabilities.",
        f"- High-confidence battery predictions are the top {(1 - high_confidence_quantile) * 100:.0f}% prediction margins among samples predicted as class {critical_label}, separately within every model/seed run.",
        f"- Effective-length bins are fixed from all valid seed-42 inputs: {', '.join(length_labels)}.",
        "- Typical curves use canonical seed 42 and predeclared rules; rules are applied in order and previously selected samples are excluded.",
        "",
        "## Main error direction per true class",
        "",
        "| Model | True class | Main wrong prediction | Mean count/seed | Mean within-class rate |",
        "|---|---:|---:|---:|---:|",
    ]
    main_errors = error_aggregate.loc[error_aggregate["direction_rank"] == 1]
    for row in _ordered_models(main_errors).itertuples(index=False):
        direction = row.predicted_label if row.total_error_count else "none observed"
        lines.append(
            f"| {MODEL_DISPLAY[str(row.model_key)]} | {row.true_label} | {direction} | "
            f"{row.mean_error_count:.2f} | {100 * row.mean_error_rate:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"## Battery class ({critical_label}) misses and high-confidence false positives",
            "",
            "| Model | False negatives/seed | Miss rate | False positives/seed | FP rate | High-confidence FP/seed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _ordered_models(battery_aggregate).itertuples(index=False):
        lines.append(
            f"| {MODEL_DISPLAY[str(row.model_key)]} | "
            f"{_format_mean_std(row.false_negatives_mean, row.false_negatives_std)} | "
            f"{_format_mean_std(row.false_negative_rate_mean, row.false_negative_rate_std, percent=True)}% | "
            f"{_format_mean_std(row.false_positives_mean, row.false_positives_std)} | "
            f"{_format_mean_std(row.false_positive_rate_mean, row.false_positive_rate_std, percent=True)}% | "
            f"{_format_mean_std(row.high_confidence_false_positives_mean, row.high_confidence_false_positives_std)} |"
        )
    lines.extend(
        [
            "",
            "## Macro-F1 by effective length",
            "",
            "| Model | Length bin | Mean sample count | Macro-F1 |",
            "|---|---|---:|---:|",
        ]
    )
    ordered_lengths = _ordered_models(length_aggregate)
    ordered_lengths["length_bin"] = pd.Categorical(
        ordered_lengths["length_bin"], categories=length_labels, ordered=True
    )
    ordered_lengths = ordered_lengths.sort_values(
        ["model_key", "length_bin"], kind="stable"
    )
    for row in ordered_lengths.itertuples(index=False):
        lines.append(
            f"| {MODEL_DISPLAY[str(row.model_key)]} | {row.length_bin} | "
            f"{row.mean_sample_count:.1f} | "
            f"{_format_mean_std(row.macro_f1_mean, row.macro_f1_std, percent=True)}% |"
        )
    lines.extend(
        [
            "",
            "## Curve characteristics",
            "",
            curve_text,
            "",
            "The comparison pools five-seed `(seed, sample_id)` test units and uses the both-correct group as the descriptive reference. It is descriptive rather than a causal test.",
            "",
            "## Deterministically selected curves",
            "",
        ]
    )
    for row in selected.sort_values("rule_order").itertuples(index=False):
        token = hashlib.sha256(str(row.sample_id).encode("utf-8")).hexdigest()[:8]
        lines.append(
            f"- `{row.selection_rule}`: anonymized sample `{token}`; {row.selection_rule_definition}."
        )
    lines.extend(
        [
            "",
            "See `figures/typical_error_curves_seed42.png` and the SVG counterpart. Exact internal sample IDs and all scores are retained in `selected_typical_examples.csv` and `predictions/`.",
            "",
        ]
    )
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _validate_run_sets(
    tcn_dirs: list[Path], moment_dirs: list[Path], svm_dirs: list[Path]
) -> None:
    lengths = {len(tcn_dirs), len(moment_dirs), len(svm_dirs)}
    if len(lengths) != 1 or not tcn_dirs:
        raise ValueError("TCN, MOMENT, and SVM run lists must have the same non-zero length.")


def _write_protocol(
    output_dir: Path,
    *,
    fractions: tuple[float, ...],
    critical_label: str,
    canonical_seed: int,
    high_confidence_quantile: float,
    length_bin_count: int,
) -> None:
    atomic_write_json(
        output_dir / "protocol.json",
        {
            "protocol_version": ERROR_ANALYSIS_PROTOCOL_VERSION,
            "analysis_split": "test",
            "prediction_export_splits": ["validation", "test"],
            "low_label_fractions": list(fractions),
            "critical_label": critical_label,
            "canonical_plot_seed": canonical_seed,
            "length_bin_count_requested": length_bin_count,
            "length_definition": "min(raw parsed length, model max_length)",
            "length_bin_definition": "nearest empirical quantiles of all valid canonical-seed inputs",
            "high_confidence_quantile": high_confidence_quantile,
            "high_confidence_definition": "run-local prediction-margin quantile among predicted critical-class samples",
            "curve_selection_order": [
                "battery_false_negative",
                "battery_high_confidence_false_positive",
                "shared_tcn_moment_error",
                "tcn_only_error",
                "moment_only_error",
                "svm10_only_error",
            ],
            "duplicate_curve_policy": "exclude samples already selected by an earlier rule",
            "svm_score_note": "normalized SVM score is softmax(decision_function), not a calibrated probability",
        },
    )


def run_analysis(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fractions = tuple(float(value) for value in args.low_label_fractions)
    canonical_seed = int(args.canonical_seed)
    _validate_run_sets(args.tcn_run_dirs, args.moment_run_dirs, args.svm_run_dirs)
    _write_protocol(
        output_dir,
        fractions=fractions,
        critical_label=str(args.critical_label),
        canonical_seed=canonical_seed,
        high_confidence_quantile=float(args.high_confidence_quantile),
        length_bin_count=int(args.length_bins),
    )

    started = time.perf_counter()
    all_frames: list[pd.DataFrame] = []
    tcn_configs: dict[int, ExperimentConfig] = {}
    for family, run_dirs in (
        ("TCN", args.tcn_run_dirs),
        ("MOMENT", args.moment_run_dirs),
    ):
        for run_dir in run_dirs:
            run_dir = Path(run_dir).resolve()
            print(f"Evaluating {family} run: {run_dir}", flush=True)
            frame, config = evaluate_neural_run(
                run_dir=run_dir,
                family=family,
                output_dir=output_dir,
                device_name=str(args.device),
                overwrite=bool(args.overwrite),
            )
            all_frames.append(frame)
            if family == "TCN":
                tcn_configs[_seed_from_config(config)] = config
    for run_dir in args.svm_run_dirs:
        run_dir = Path(run_dir).resolve()
        print(f"Evaluating low-label MOMENT-SVM run: {run_dir}", flush=True)
        frames, _ = evaluate_svm_run(
            run_dir=run_dir,
            fractions=fractions,
            output_dir=output_dir,
            device_name=str(args.device),
            overwrite=bool(args.overwrite),
        )
        all_frames.extend(frames)

    all_predictions = pd.concat(all_frames, ignore_index=True)
    available_seeds = sorted(all_predictions["seed"].astype(int).unique().tolist())
    if canonical_seed not in available_seeds:
        raise ValueError(
            f"Canonical seed {canonical_seed} is absent; available seeds are {available_seeds}."
        )
    expected_keys = set(MODEL_ORDER)
    actual_keys = set(all_predictions["model_key"].astype(str).unique())
    if actual_keys != expected_keys:
        raise ValueError(f"Expected model variants {sorted(expected_keys)}, got {sorted(actual_keys)}.")

    test_predictions = all_predictions.loc[all_predictions["split"] == "test"].copy()
    test_predictions, battery_per_seed = add_high_confidence_battery_flags(
        test_predictions,
        str(args.critical_label),
        float(args.high_confidence_quantile),
    )
    _atomic_write_csv(test_predictions, output_dir / "all_test_predictions.csv")

    error_per_seed, error_aggregate = error_directions(test_predictions)
    battery_aggregate = summarize_battery(battery_per_seed)
    reference_config = tcn_configs[canonical_seed]
    reference_bundle = load_dataset(reference_config.data)
    class_names = _class_names(reference_bundle)
    confusion_per_seed, confusion_aggregate = confusion_tables(test_predictions, class_names)
    all_effective_lengths = np.minimum(
        np.concatenate(
            [
                reference_bundle.lengths_train,
                reference_bundle.lengths_val,
                reference_bundle.lengths_test,
            ]
        ),
        reference_config.data.max_length,
    )
    length_edges, length_labels = make_length_bins(all_effective_lengths, int(args.length_bins))
    length_per_seed, length_aggregate = length_stratified_metrics(
        test_predictions, length_edges, length_labels, reference_bundle.num_classes
    )
    curve_rows, canonical_curves = build_curve_feature_rows(
        predictions=test_predictions,
        tcn_configs=tcn_configs,
        canonical_seed=canonical_seed,
    )
    curve_summary = summarize_curve_features(curve_rows)
    curve_text = curve_characteristics_text(curve_summary)
    selected = select_typical_examples(
        test_predictions, canonical_seed, str(args.critical_label)
    )
    png_path, svg_path = plot_typical_examples(selected, canonical_curves, output_dir)

    tables = {
        "error_directions_per_seed.csv": error_per_seed,
        "error_directions_aggregate.csv": error_aggregate,
        "confusion_matrix_per_seed.csv": confusion_per_seed,
        "confusion_matrix_aggregate.csv": confusion_aggregate,
        "battery_errors_per_seed.csv": battery_per_seed,
        "battery_errors_aggregate.csv": battery_aggregate,
        "length_macro_f1_per_seed.csv": length_per_seed,
        "length_macro_f1_aggregate.csv": length_aggregate,
        "curve_feature_rows.csv": curve_rows,
        "curve_feature_summary.csv": curve_summary,
        "selected_typical_examples.csv": selected,
    }
    for name, frame in tables.items():
        _atomic_write_csv(frame, output_dir / name)
    summary_path = write_summary(
        output_dir=output_dir,
        error_aggregate=error_aggregate,
        battery_aggregate=battery_aggregate,
        length_aggregate=length_aggregate,
        curve_text=curve_text,
        selected=selected,
        critical_label=str(args.critical_label),
        high_confidence_quantile=float(args.high_confidence_quantile),
        length_labels=length_labels,
    )
    atomic_write_json(
        output_dir / "metrics.json",
        {
            "protocol_version": ERROR_ANALYSIS_PROTOCOL_VERSION,
            "status": "complete",
            "seeds": available_seeds,
            "model_variants": list(MODEL_ORDER),
            "low_label_fractions": list(fractions),
            "prediction_row_count_all_splits": len(all_predictions),
            "test_prediction_row_count": len(test_predictions),
            "length_bin_edges": [
                None if not np.isfinite(value) else value for value in length_edges
            ],
            "length_bin_labels": length_labels,
            "curve_characteristics": curve_text,
            "selected_curve_count": len(selected),
            "elapsed_seconds": time.perf_counter() - started,
            "summary": str(summary_path),
            "typical_curve_png": str(png_path),
            "typical_curve_svg": str(svg_path),
            "tables": {name: str(output_dir / name) for name in tables},
        },
    )
    print(f"M02 error analysis complete: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic M02 per-sample and lightweight error analysis."
    )
    parser.add_argument("--tcn-run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--moment-run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--svm-run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--low-label-fractions",
        nargs="+",
        type=float,
        default=DEFAULT_LOW_LABEL_FRACTIONS,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m02_error_analysis"))
    parser.add_argument("--critical-label", default="2")
    parser.add_argument("--canonical-seed", type=int, default=42)
    parser.add_argument("--length-bins", type=int, default=4)
    parser.add_argument("--high-confidence-quantile", type=float, default=0.90)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 3 <= args.length_bins <= 5:
        parser.error("--length-bins must be between 3 and 5.")
    if not 0 < args.high_confidence_quantile < 1:
        parser.error("--high-confidence-quantile must be in (0, 1).")
    if any(not 0 < fraction < 1 for fraction in args.low_label_fractions):
        parser.error("Every --low-label-fractions value must be in (0, 1).")
    run_analysis(args)


if __name__ == "__main__":
    main()
