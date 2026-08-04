from __future__ import annotations

import argparse
import hashlib
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tcn_moment.config import ExperimentConfig, load_config
from tcn_moment.data import DatasetBundle, load_dataset, save_label_encoder
from tcn_moment.evaluate_imputation import generate_observation_mask
from tcn_moment.experiment import RunContext, prepare_run
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.train_moment import (
    build_model,
    forward_features,
    require_torch_and_moment,
    select_device,
    set_num_classes,
)
from tcn_moment.training_utils import seed_everything


RETRIEVAL_PROTOCOL_VERSION = 1
RETRIEVAL_METHODS = ("moment", "raw_resampled", "statistical")
STATISTICAL_FEATURE_NAMES = (
    "valid_length_ratio",
    "visible_ratio",
    "mean",
    "std",
    "minimum",
    "q25",
    "median",
    "q75",
    "maximum",
    "first",
    "last",
    "rms",
    "slope",
    "z_first",
    "z_last",
    "z_q25",
    "z_q75",
    "mean_absolute_rate",
)


def l2_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [samples, dimensions].")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-12))


def cosine_topk(
    normalized_gallery: np.ndarray,
    normalized_queries: np.ndarray,
    k: int,
    batch_size: int,
    *,
    torch: Any | None = None,
    device: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return exact cosine nearest neighbours without materializing the full matrix."""
    gallery = np.asarray(normalized_gallery, dtype=np.float32)
    queries = np.asarray(normalized_queries, dtype=np.float32)
    if gallery.ndim != 2 or queries.ndim != 2 or gallery.shape[1] != queries.shape[1]:
        raise ValueError("gallery and query features must have matching dimensions.")
    if not 0 < k <= len(gallery) or batch_size <= 0:
        raise ValueError("k and batch_size must be positive and k cannot exceed gallery size.")

    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    started = time.perf_counter()
    if torch is not None and device is not None and device.type == "cuda":
        gallery_tensor = torch.from_numpy(gallery).to(device=device)
        with torch.inference_mode():
            for start in range(0, len(queries), batch_size):
                query_tensor = torch.from_numpy(queries[start : start + batch_size]).to(
                    device=device
                )
                scores = query_tensor @ gallery_tensor.T
                values, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
                all_indices.append(indices.cpu().numpy().astype(np.int32, copy=False))
                all_scores.append(values.cpu().numpy().astype(np.float32, copy=False))
        torch.cuda.synchronize(device)
        del gallery_tensor
    else:
        for start in range(0, len(queries), batch_size):
            scores = queries[start : start + batch_size] @ gallery.T
            candidate_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
            candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)
            order = np.argsort(-candidate_scores, axis=1, kind="stable")
            all_indices.append(
                np.take_along_axis(candidate_indices, order, axis=1).astype(np.int32)
            )
            all_scores.append(
                np.take_along_axis(candidate_scores, order, axis=1).astype(np.float32)
            )
    return (
        np.concatenate(all_indices),
        np.concatenate(all_scores),
        time.perf_counter() - started,
    )


def _visible_rows(
    values: np.ndarray,
    input_mask: np.ndarray,
    observation_mask: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if values.shape != input_mask.shape or values.shape != observation_mask.shape:
        raise ValueError("values, input_mask, and observation_mask must have equal shapes.")
    rows = []
    for row in range(len(values)):
        valid = input_mask[row].astype(bool)
        visible = observation_mask[row].astype(bool) & valid
        valid_positions = np.flatnonzero(valid)
        visible_positions = np.flatnonzero(visible)
        if len(valid_positions) < 2 or len(visible_positions) < 2:
            raise ValueError("Every sequence must contain at least two visible valid points.")
        rows.append((valid_positions, visible_positions, values[row, visible_positions]))
    return rows


def raw_resampled_features(
    values: np.ndarray,
    input_mask: np.ndarray,
    observation_mask: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Build scale-normalized shape features using visible points only."""
    if target_length < 2:
        raise ValueError("target_length must be at least 2.")
    features = np.empty((len(values), target_length), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, target_length)
    for row, (valid_positions, visible_positions, visible_values) in enumerate(
        _visible_rows(values, input_mask, observation_mask)
    ):
        valid_length = len(valid_positions)
        normalized_visible_positions = visible_positions / max(valid_length - 1, 1)
        filled = np.interp(
            np.linspace(0.0, 1.0, valid_length),
            normalized_visible_positions,
            visible_values.astype(np.float64),
        )
        mean = float(filled.mean())
        std = float(filled.std())
        normalized = (filled - mean) / std if std > 1e-8 else np.zeros_like(filled)
        features[row] = np.interp(
            target_positions,
            np.linspace(0.0, 1.0, valid_length),
            normalized,
        ).astype(np.float32)
    return features


def statistical_features(
    values: np.ndarray,
    input_mask: np.ndarray,
    observation_mask: np.ndarray,
) -> np.ndarray:
    """Extract label-free amplitude and shape descriptors from visible values only."""
    result = np.empty((len(values), len(STATISTICAL_FEATURE_NAMES)), dtype=np.float32)
    max_length = values.shape[1]
    for row, (valid_positions, visible_positions, visible_values) in enumerate(
        _visible_rows(values, input_mask, observation_mask)
    ):
        raw = visible_values.astype(np.float64)
        valid_length = len(valid_positions)
        positions = visible_positions.astype(np.float64) / max(valid_length - 1, 1)
        mean = float(raw.mean())
        std = float(raw.std())
        z = (raw - mean) / std if std > 1e-8 else np.zeros_like(raw)
        centered_positions = positions - positions.mean()
        position_variance = float(np.square(centered_positions).sum())
        slope = (
            float((centered_positions * (raw - mean)).sum() / position_variance)
            if position_variance > 1e-12
            else 0.0
        )
        position_steps = np.diff(positions)
        rates = np.diff(raw) / np.maximum(position_steps, 1e-12)
        q25, median, q75 = np.quantile(raw, [0.25, 0.5, 0.75])
        z_q25, z_q75 = np.quantile(z, [0.25, 0.75])
        result[row] = np.asarray(
            [
                valid_length / max_length,
                len(visible_positions) / valid_length,
                mean,
                std,
                float(raw.min()),
                q25,
                median,
                q75,
                float(raw.max()),
                float(raw[0]),
                float(raw[-1]),
                float(np.sqrt(np.square(raw).mean())),
                slope,
                float(z[0]),
                float(z[-1]),
                z_q25,
                z_q75,
                float(np.abs(rates).mean()) if len(rates) else 0.0,
            ],
            dtype=np.float32,
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite statistical retrieval feature detected.")
    return result


def retrieval_metrics(
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    gallery_lengths: np.ndarray,
    query_lengths: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[str, float]:
    if neighbor_indices.shape != neighbor_scores.shape:
        raise ValueError("neighbor indices and scores must have equal shapes.")
    if neighbor_indices.shape[0] != len(query_labels):
        raise ValueError("Every query must have a neighbour row.")
    classes = np.unique(np.concatenate([gallery_labels, query_labels]))
    retrieved_labels = gallery_labels[neighbor_indices]
    relevant = retrieved_labels == query_labels[:, None]
    metrics: dict[str, float] = {"mean_top1_cosine": float(neighbor_scores[:, 0].mean())}
    ranks = np.arange(1, neighbor_indices.shape[1] + 1, dtype=np.float64)
    discounts = 1.0 / np.log2(ranks + 1.0)

    for k in k_values:
        relevant_k = relevant[:, :k]
        per_query_precision = relevant_k.mean(axis=1)
        metrics[f"precision_at_{k}"] = float(per_query_precision.mean())
        class_precisions = []
        for label in classes:
            selected = query_labels == label
            value = float(per_query_precision[selected].mean())
            metrics[f"class_{int(label)}_precision_at_{k}"] = value
            class_precisions.append(value)
        metrics[f"macro_precision_at_{k}"] = float(np.mean(class_precisions))

        cumulative_precision = np.cumsum(relevant_k, axis=1) / np.arange(1, k + 1)
        relevant_totals = np.asarray(
            [(gallery_labels == label).sum() for label in query_labels],
            dtype=np.int64,
        )
        denominators = np.minimum(relevant_totals, k)
        average_precision = (
            (cumulative_precision * relevant_k).sum(axis=1)
            / np.maximum(denominators, 1)
        )
        metrics[f"map_at_{k}"] = float(average_precision.mean())

        dcg = (relevant_k * discounts[:k]).sum(axis=1)
        ideal = np.asarray(
            [discounts[: min(int(total), k)].sum() for total in relevant_totals]
        )
        metrics[f"ndcg_at_{k}"] = float((dcg / np.maximum(ideal, 1e-12)).mean())

        neighbor_lengths = gallery_lengths[neighbor_indices[:, :k]]
        relative_error = np.abs(neighbor_lengths - query_lengths[:, None]) / np.maximum(
            query_lengths[:, None], 1
        )
        metrics[f"mean_length_relative_error_at_{k}"] = float(relative_error.mean())
    return metrics


def neighbor_overlap(
    clean_indices: np.ndarray,
    perturbed_indices: np.ndarray,
    k_values: tuple[int, ...],
) -> dict[str, float]:
    if clean_indices.shape != perturbed_indices.shape:
        raise ValueError("Clean and perturbed neighbour arrays must have equal shapes.")
    result = {}
    for k in k_values:
        overlaps = [
            len(set(clean[:k].tolist()).intersection(changed[:k].tolist())) / k
            for clean, changed in zip(clean_indices, perturbed_indices, strict=True)
        ]
        result[f"clean_neighbor_overlap_at_{k}"] = float(np.mean(overlaps))
    return result


def paired_feature_cosine(clean: np.ndarray, perturbed: np.ndarray) -> float:
    if clean.shape != perturbed.shape:
        raise ValueError("Clean and perturbed feature arrays must have equal shapes.")
    return float((l2_normalize(clean) * l2_normalize(perturbed)).sum(axis=1).mean())


def _extract_moment_features(
    torch: Any,
    model: Any,
    device: Any,
    values: np.ndarray,
    observation_mask: np.ndarray,
    *,
    batch_size: int,
    amp_enabled: bool,
) -> tuple[np.ndarray, float, bool, int | None]:
    outputs: list[np.ndarray] = []
    active_amp = amp_enabled
    fallback_batch_start: int | None = None
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            batch_values = values[start:stop].copy()
            batch_observation = observation_mask[start:stop]
            batch_values[batch_observation == 0] = 0.0
            batch_x = torch.from_numpy(batch_values[:, None, :]).to(
                device=device, dtype=torch.float32
            )
            batch_mask = torch.from_numpy(batch_observation).to(
                device=device, dtype=torch.float32
            )
            with torch.amp.autocast(device_type=device.type, enabled=active_amp):
                features = forward_features(model, batch_x, batch_mask)
            if active_amp and not torch.isfinite(features).all():
                print(
                    "Warning: non-finite MOMENT retrieval features under AMP at "
                    f"batch start {start}; retrying in FP32 and disabling AMP."
                )
                active_amp = False
                fallback_batch_start = start
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    features = forward_features(model, batch_x, batch_mask)
            if not torch.isfinite(features).all():
                raise FloatingPointError(
                    f"Non-finite MOMENT retrieval features in FP32 at batch {start}."
                )
            outputs.append(features.float().cpu().numpy())
    return (
        np.concatenate(outputs).astype(np.float32, copy=False),
        time.perf_counter() - started,
        active_amp,
        fallback_batch_start,
    )


def _mask_sha256(sample_ids: np.ndarray, observation_mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    for sample_id, row in zip(sample_ids, observation_mask, strict=True):
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(row, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = {
        "condition",
        "pattern",
        "mask_rate",
        "mask_seed",
        "actual_mask_rate",
        "method",
        "mask_sha256",
        "feature_seconds",
        "search_seconds",
    }
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in metadata and isinstance(value, (float, int))
        }
    )
    summaries: list[dict[str, Any]] = []
    groups = sorted({(str(row["condition"]), str(row["method"])) for row in rows})
    for condition, method in groups:
        selected = [
            row
            for row in rows
            if row["condition"] == condition and row["method"] == method
        ]
        summary: dict[str, Any] = {
            "condition": condition,
            "method": method,
            "seeds": len(selected),
            "mask_rate": float(selected[0]["mask_rate"]),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in selected if metric in row]
            if not values:
                continue
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sample_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def _example_query_indices(labels: np.ndarray, sample_ids: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    classes = np.unique(labels)
    per_class = max(1, int(np.ceil(count / len(classes))))
    selected: list[int] = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        order = np.argsort(sample_ids[candidates].astype(str), kind="stable")
        selected.extend(candidates[order[:per_class]].tolist())
    return np.asarray(selected[:count], dtype=np.int64)


def _append_examples(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    method: str,
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    bundle: DatasetBundle,
    class_names: list[str],
    max_neighbors: int,
) -> None:
    for query_index in query_indices:
        query_label = int(bundle.y_test[query_index])
        for rank in range(min(max_neighbors, neighbor_indices.shape[1])):
            gallery_index = int(neighbor_indices[query_index, rank])
            gallery_label = int(bundle.y_train[gallery_index])
            rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "query_id": str(bundle.ids_test[query_index]),
                    "query_label": class_names[query_label],
                    "rank": rank + 1,
                    "neighbor_id": str(bundle.ids_train[gallery_index]),
                    "neighbor_label": class_names[gallery_label],
                    "same_label": query_label == gallery_label,
                    "cosine_similarity": float(neighbor_scores[query_index, rank]),
                }
            )


def _data_record(bundle: DatasetBundle, config: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset_sha256": bundle.dataset_sha256,
        "split_manifest": str(bundle.split_path),
        "gallery_split": "train",
        "query_split": "test",
        "gallery_samples": len(bundle.y_train),
        "query_samples": len(bundle.y_test),
        "validation_samples_unused": len(bundle.y_val),
        "labels_used_during_feature_extraction_or_search": False,
        "labels_used_for_post_hoc_evaluation_only": True,
        "split_counts": bundle.split_counts,
        "classes": bundle.label_encoder.classes_.tolist(),
        "max_length": config.data.max_length,
        "input_normalization": config.data.normalize,
    }


def _run(
    config: ExperimentConfig,
    context: RunContext,
    torch: Any,
    MOMENTPipeline: Any,
) -> None:
    if config.data.normalize != "none":
        raise ValueError(
            "Retrieval robustness evaluation requires data.normalize=none so hidden "
            "query values cannot influence external normalization statistics."
        )
    if not config.model.freeze_backbone or config.model.unfreeze_last_n_layers != 0:
        raise ValueError("Unsupervised retrieval requires a fully frozen MOMENT backbone.")

    wall_started = time.perf_counter()
    bundle = load_dataset(config.data)
    shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
    save_label_encoder(bundle.label_encoder, context.run_dir)
    np.save(context.run_dir / "gallery_sample_ids.npy", bundle.ids_train.astype(str))
    np.save(context.run_dir / "query_sample_ids.npy", bundle.ids_test.astype(str))

    device = select_device(torch, config.training.device)
    model = build_model(config, MOMENTPipeline, bundle.num_classes)
    set_num_classes(model, bundle.num_classes)
    model.init()
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    patch_len = int(getattr(model, "patch_len", 0))
    if patch_len <= 0:
        raise ValueError("Unable to determine MOMENT patch length.")

    capped_train_lengths = np.minimum(bundle.lengths_train, config.data.max_length)
    capped_test_lengths = np.minimum(bundle.lengths_test, config.data.max_length)
    if bool(
        (capped_test_lengths // patch_len < config.retrieval.min_complete_patches).any()
    ):
        raise ValueError("At least one test sequence has too few complete MOMENT patches.")
    train_mask = bundle.mask_train.astype(np.uint8, copy=False)
    test_mask = bundle.mask_test.astype(np.uint8, copy=False)
    amp_requested = bool(config.training.amp and device.type == "cuda")
    amp_enabled = amp_requested
    amp_fallback_events: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("Extracting clean frozen MOMENT gallery features...")
    gallery_moment, seconds, amp_enabled, fallback = _extract_moment_features(
        torch,
        model,
        device,
        bundle.x_train,
        train_mask,
        batch_size=config.training.feature_extraction_batch_size,
        amp_enabled=amp_enabled,
    )
    feature_seconds = seconds
    if fallback is not None:
        amp_fallback_events.append({"condition": "clean_gallery", "batch_start": fallback})

    print("Extracting clean frozen MOMENT query features...")
    clean_moment, seconds, amp_enabled, fallback = _extract_moment_features(
        torch,
        model,
        device,
        bundle.x_test,
        test_mask,
        batch_size=config.training.feature_extraction_batch_size,
        amp_enabled=amp_enabled,
    )
    feature_seconds += seconds
    if fallback is not None:
        amp_fallback_events.append({"condition": "clean_query", "batch_start": fallback})

    gallery_raw = raw_resampled_features(
        bundle.x_train,
        train_mask,
        train_mask,
        config.retrieval.raw_resample_length,
    )
    clean_raw = raw_resampled_features(
        bundle.x_test,
        test_mask,
        test_mask,
        config.retrieval.raw_resample_length,
    )
    gallery_statistics_unscaled = statistical_features(
        bundle.x_train, train_mask, train_mask
    )
    clean_statistics_unscaled = statistical_features(bundle.x_test, test_mask, test_mask)
    statistics_scaler = StandardScaler().fit(gallery_statistics_unscaled)
    gallery_statistics = statistics_scaler.transform(
        gallery_statistics_unscaled
    ).astype(np.float32)
    clean_statistics = statistics_scaler.transform(clean_statistics_unscaled).astype(
        np.float32
    )

    gallery_features = {
        "moment": gallery_moment,
        "raw_resampled": gallery_raw,
        "statistical": gallery_statistics,
    }
    clean_features = {
        "moment": clean_moment,
        "raw_resampled": clean_raw,
        "statistical": clean_statistics,
    }
    normalized_gallery = {
        method: l2_normalize(features) for method, features in gallery_features.items()
    }
    normalized_clean = {
        method: l2_normalize(features) for method, features in clean_features.items()
    }

    class_names = [str(value) for value in bundle.label_encoder.classes_.tolist()]
    max_k = max(config.retrieval.k_values)
    query_examples = _example_query_indices(
        bundle.y_test,
        bundle.ids_test,
        min(config.retrieval.example_query_count, len(bundle.y_test)),
    )
    condition_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    clean_neighbor_indices: dict[str, np.ndarray] = {}
    search_seconds = 0.0
    clean_archive: dict[str, np.ndarray] = {}

    for method in RETRIEVAL_METHODS:
        indices, scores, elapsed = cosine_topk(
            normalized_gallery[method],
            normalized_clean[method],
            max_k,
            config.retrieval.search_batch_size,
            torch=torch,
            device=device,
        )
        search_seconds += elapsed
        clean_neighbor_indices[method] = indices
        clean_archive[f"{method}_indices"] = indices
        clean_archive[f"{method}_scores"] = scores
        metrics = retrieval_metrics(
            indices,
            scores,
            bundle.y_train,
            bundle.y_test,
            capped_train_lengths,
            capped_test_lengths,
            config.retrieval.k_values,
        )
        condition_rows.append(
            {
                "condition": "clean",
                "pattern": "none",
                "mask_rate": 0.0,
                "mask_seed": -1,
                "actual_mask_rate": 0.0,
                "method": method,
                "mask_sha256": "none",
                "feature_seconds": 0.0,
                "search_seconds": elapsed,
                **metrics,
            }
        )
        _append_examples(
            example_rows,
            condition="clean",
            method=method,
            query_indices=query_examples,
            neighbor_indices=indices,
            neighbor_scores=scores,
            bundle=bundle,
            class_names=class_names,
            max_neighbors=max_k,
        )
    np.savez_compressed(context.run_dir / "neighbors_clean.npz", **clean_archive)

    completed_conditions: list[dict[str, Any]] = []
    complete_patch_points = int(
        sum((int(length) // patch_len) * patch_len for length in capped_test_lengths)
    )
    for pattern in config.retrieval.mask_patterns:
        for mask_seed in config.retrieval.mask_seeds:
            condition = f"{pattern}_rate{config.retrieval.query_mask_rate:g}"
            print(f"Evaluating {condition} seed={mask_seed}")
            observation_mask = generate_observation_mask(
                test_mask,
                capped_test_lengths,
                bundle.ids_test,
                mask_rate=config.retrieval.query_mask_rate,
                pattern=pattern,
                mask_seed=mask_seed,
                patch_len=patch_len,
                min_complete_patches=config.retrieval.min_complete_patches,
            )
            hidden = (test_mask == 1) & (observation_mask == 0)
            actual_mask_rate = float(hidden.sum() / complete_patch_points)
            masked_moment, elapsed_features, new_amp_enabled, fallback = (
                _extract_moment_features(
                    torch,
                    model,
                    device,
                    bundle.x_test,
                    observation_mask,
                    batch_size=config.training.feature_extraction_batch_size,
                    amp_enabled=amp_enabled,
                )
            )
            feature_seconds += elapsed_features
            if fallback is not None:
                amp_fallback_events.append(
                    {
                        "condition": condition,
                        "mask_seed": mask_seed,
                        "batch_start": fallback,
                    }
                )
            amp_enabled = new_amp_enabled
            masked_raw = raw_resampled_features(
                bundle.x_test,
                test_mask,
                observation_mask,
                config.retrieval.raw_resample_length,
            )
            masked_statistics = statistics_scaler.transform(
                statistical_features(bundle.x_test, test_mask, observation_mask)
            ).astype(np.float32)
            masked_features = {
                "moment": masked_moment,
                "raw_resampled": masked_raw,
                "statistical": masked_statistics,
            }
            archive: dict[str, np.ndarray] = {}
            for method in RETRIEVAL_METHODS:
                normalized_masked = l2_normalize(masked_features[method])
                indices, scores, elapsed = cosine_topk(
                    normalized_gallery[method],
                    normalized_masked,
                    max_k,
                    config.retrieval.search_batch_size,
                    torch=torch,
                    device=device,
                )
                search_seconds += elapsed
                archive[f"{method}_indices"] = indices
                archive[f"{method}_scores"] = scores
                metrics = retrieval_metrics(
                    indices,
                    scores,
                    bundle.y_train,
                    bundle.y_test,
                    capped_train_lengths,
                    capped_test_lengths,
                    config.retrieval.k_values,
                )
                stability = neighbor_overlap(
                    clean_neighbor_indices[method], indices, config.retrieval.k_values
                )
                condition_rows.append(
                    {
                        "condition": condition,
                        "pattern": pattern,
                        "mask_rate": config.retrieval.query_mask_rate,
                        "mask_seed": mask_seed,
                        "actual_mask_rate": actual_mask_rate,
                        "method": method,
                        "mask_sha256": _mask_sha256(bundle.ids_test, observation_mask),
                        "feature_seconds": elapsed_features if method == "moment" else 0.0,
                        "search_seconds": elapsed,
                        **metrics,
                        **stability,
                        "clean_query_feature_cosine": paired_feature_cosine(
                            clean_features[method], masked_features[method]
                        ),
                    }
                )
                if mask_seed == config.retrieval.mask_seeds[0]:
                    _append_examples(
                        example_rows,
                        condition=condition,
                        method=method,
                        query_indices=query_examples,
                        neighbor_indices=indices,
                        neighbor_scores=scores,
                        bundle=bundle,
                        class_names=class_names,
                        max_neighbors=max_k,
                    )
            archive["observation_mask"] = observation_mask.astype(np.uint8)
            tag = f"{pattern}_rate{config.retrieval.query_mask_rate:g}_seed{mask_seed}".replace(
                ".", "p"
            )
            np.savez_compressed(context.run_dir / f"neighbors_{tag}.npz", **archive)
            completed_conditions.append(
                {
                    "condition": condition,
                    "mask_seed": mask_seed,
                    "actual_mask_rate": actual_mask_rate,
                    "mask_sha256": _mask_sha256(bundle.ids_test, observation_mask),
                    "moment_feature_seconds": elapsed_features,
                }
            )
            atomic_write_json(
                context.run_dir / "metrics_partial.json",
                {
                    "protocol_version": RETRIEVAL_PROTOCOL_VERSION,
                    "completed_conditions": completed_conditions,
                },
            )

    summary = _summary_rows(condition_rows)
    pd.DataFrame(condition_rows).to_csv(
        context.run_dir / "condition_metrics.csv", index=False
    )
    pd.DataFrame(summary).to_csv(context.run_dir / "summary.csv", index=False)
    pd.DataFrame(example_rows).to_csv(
        context.run_dir / "example_neighbors.csv", index=False
    )

    gallery_counts = np.bincount(bundle.y_train, minlength=bundle.num_classes)
    gallery_priors = gallery_counts / gallery_counts.sum()
    chance_micro_precision = float(gallery_priors[bundle.y_test].mean())
    chance_macro_precision = float(np.mean(gallery_priors))
    wall_seconds = time.perf_counter() - wall_started
    execution = {
        "device": str(device),
        "amp_requested": amp_requested,
        "amp_enabled_at_end": amp_enabled,
        "amp_fallback_events": amp_fallback_events,
        "feature_extraction_batch_size": config.training.feature_extraction_batch_size,
        "search_batch_size": config.retrieval.search_batch_size,
        "feature_extraction_seconds": feature_seconds,
        "exact_search_seconds": search_seconds,
        "wall_seconds": wall_seconds,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "total_parameters": total_parameters,
        "trainable_parameters": 0,
    }
    result = {
        "model": "MOMENT_UNSUPERVISED_RETRIEVAL",
        "run_name": context.run_name,
        "data": _data_record(bundle, config),
        "protocol": {
            "retrieval_protocol_version": RETRIEVAL_PROTOCOL_VERSION,
            "backbone_frozen": True,
            "parameter_updates": False,
            "labels_used_during_search": False,
            "labels_used_for_post_hoc_metrics": True,
            "gallery_query_disjoint": True,
            "distance": "cosine",
            "exact_topk_search": True,
            "k_values": list(config.retrieval.k_values),
            "methods": list(RETRIEVAL_METHODS),
            "raw_baseline": {
                "resample_length": config.retrieval.raw_resample_length,
                "masked_query_fill": "linear interpolation from visible points only",
                "per_sequence_normalization": "z-score after visible-only interpolation",
            },
            "statistical_feature_names": list(STATISTICAL_FEATURE_NAMES),
            "statistical_scaler_fit": "unlabelled gallery features only",
            "query_mask_rate": config.retrieval.query_mask_rate,
            "mask_patterns": list(config.retrieval.mask_patterns),
            "mask_seeds": list(config.retrieval.mask_seeds),
            "primary_metric": f"macro_precision_at_{max_k}",
            "robustness_metric": f"clean_neighbor_overlap_at_{max_k}",
            "analytic_random_chance_micro_precision": chance_micro_precision,
            "analytic_random_chance_macro_precision": chance_macro_precision,
        },
        "execution": execution,
        "conditions": completed_conditions,
        "condition_metrics": condition_rows,
        "summary": summary,
    }
    atomic_write_json(context.run_dir / "metrics.json", result)
    (context.run_dir / "metrics_partial.json").unlink(missing_ok=True)
    print(f"Clean and masked retrieval evaluation completed in {wall_seconds:.2f}s")
    print(f"Saved artifacts to {context.run_dir}")


def evaluate(
    config: ExperimentConfig,
    config_path: Path,
    *,
    run_name: str | None = None,
) -> None:
    torch, _DataLoader, _TensorDataset, _tqdm, MOMENTPipeline = require_torch_and_moment()
    seed_everything(torch, config.data.random_state)
    context = prepare_run(
        model_name="MOMENT_UNSUPERVISED_RETRIEVAL",
        base_output_dir=config.training.output_dir,
        config=config,
        config_path=config_path,
        torch=torch,
        run_name=run_name,
        resume_dir=None,
    )
    try:
        _run(config, context, torch, MOMENTPipeline)
    except KeyboardInterrupt:
        context.set_status("interrupted", message="Restart this non-resumable evaluation.")
        raise
    except BaseException as exc:
        context.set_status("failed", error_type=type(exc).__name__, message=str(exc))
        raise
    else:
        context.set_status("completed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen MOMENT representations for label-free retrieval."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/moment_retrieval_zero_shot.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--run-name", help="Unique output directory name.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    evaluate(config, config_path, run_name=args.run_name)


if __name__ == "__main__":
    main()
