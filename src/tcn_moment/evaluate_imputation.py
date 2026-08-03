from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from tcn_moment.config import ExperimentConfig, load_config
from tcn_moment.data import DatasetBundle, load_dataset
from tcn_moment.experiment import RunContext, prepare_run
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.train_moment import require_torch_and_moment, select_device
from tcn_moment.training_utils import seed_everything


IMPUTATION_PROTOCOL_VERSION = 1
BASELINE_METHODS = ("mean", "forward_fill", "linear", "pchip")
ALL_METHODS = ("moment_zero_shot", *BASELINE_METHODS)


def _sample_seed(mask_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{mask_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _centered_block(center: int, size: int, total: int) -> np.ndarray:
    start = min(max(center - size // 2, 0), total - size)
    return np.arange(start, start + size, dtype=np.int64)


def generate_observation_mask(
    input_mask: np.ndarray,
    lengths: np.ndarray,
    sample_ids: np.ndarray,
    *,
    mask_rate: float,
    pattern: str,
    mask_seed: int,
    patch_len: int,
    min_complete_patches: int,
) -> np.ndarray:
    """Create patch-aligned masks; 1 means observed and 0 means hidden/padding."""
    if input_mask.ndim != 2 or len(input_mask) != len(lengths):
        raise ValueError("input_mask and lengths have incompatible shapes.")
    if len(sample_ids) != len(lengths):
        raise ValueError("sample_ids and lengths must have equal lengths.")
    if not 0 < mask_rate < 1:
        raise ValueError("mask_rate must be in (0, 1).")
    if pattern not in {"random_patches", "contiguous_block"}:
        raise ValueError(f"Unsupported mask pattern: {pattern}")
    if patch_len <= 0 or min_complete_patches < 2:
        raise ValueError("patch_len must be positive and min_complete_patches at least 2.")

    observation_mask = input_mask.astype(np.uint8, copy=True)
    sequence_length = input_mask.shape[1]
    for row, (length, sample_id) in enumerate(zip(lengths, sample_ids, strict=True)):
        valid_length = min(int(length), sequence_length)
        complete_patches = valid_length // patch_len
        if complete_patches < min_complete_patches:
            raise ValueError(
                f"Sample {sample_id} has only {complete_patches} complete patches."
            )
        masked_patch_count = max(
            1,
            min(complete_patches - 1, int(round(complete_patches * mask_rate))),
        )
        generator = np.random.default_rng(_sample_seed(mask_seed, str(sample_id)))
        if pattern == "random_patches":
            patch_indices = generator.permutation(complete_patches)[:masked_patch_count]
        else:
            center = int(generator.integers(0, complete_patches))
            patch_indices = _centered_block(
                center,
                masked_patch_count,
                complete_patches,
            )
        for patch_index in patch_indices:
            start = int(patch_index) * patch_len
            observation_mask[row, start : start + patch_len] = 0
    return observation_mask


def baseline_prediction(
    values: np.ndarray,
    visible: np.ndarray,
    method: str,
) -> np.ndarray:
    """Reconstruct a single valid (unpadded) sequence from visible values only."""
    values = np.asarray(values, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    observed_indices = np.flatnonzero(visible)
    if len(observed_indices) < 2:
        raise ValueError("At least two visible points are required for imputation.")
    observed_values = values[observed_indices]
    positions = np.arange(len(values), dtype=np.float64)

    if method == "mean":
        return np.full_like(values, float(observed_values.mean()))
    if method == "forward_fill":
        source_indices = np.maximum.accumulate(
            np.where(visible, np.arange(len(values)), -1)
        )
        source_indices[source_indices < 0] = observed_indices[0]
        return values[source_indices]
    if method == "linear":
        return np.interp(positions, observed_indices, observed_values)
    if method == "pchip":
        interpolator = PchipInterpolator(
            observed_indices.astype(np.float64),
            observed_values,
            extrapolate=True,
        )
        return np.asarray(interpolator(positions), dtype=np.float64)
    raise ValueError(f"Unsupported baseline method: {method}")


@dataclass
class ErrorAccumulator:
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    point_count: int = 0
    negative_prediction_count: int = 0
    sequence_mae: list[float] = field(default_factory=list)
    sequence_nrmse: list[float] = field(default_factory=list)
    unscaled_sequence_count: int = 0

    def update(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        full_valid_sequence: np.ndarray,
    ) -> None:
        target64 = np.asarray(target, dtype=np.float64)
        prediction64 = np.asarray(prediction, dtype=np.float64)
        if len(target64) == 0 or target64.shape != prediction64.shape:
            raise ValueError("Target and prediction must be non-empty with equal shapes.")
        if not np.isfinite(prediction64).all():
            raise FloatingPointError("Non-finite imputation prediction detected.")
        errors = prediction64 - target64
        squared = errors**2
        self.absolute_error_sum += float(np.abs(errors).sum())
        self.squared_error_sum += float(squared.sum())
        self.point_count += len(errors)
        self.negative_prediction_count += int((prediction64 < 0).sum())
        self.sequence_mae.append(float(np.abs(errors).mean()))
        scale = float(np.std(np.asarray(full_valid_sequence, dtype=np.float64)))
        if scale > 1e-8:
            self.sequence_nrmse.append(float(np.sqrt(squared.mean()) / scale))
        else:
            self.unscaled_sequence_count += 1

    def result(self) -> dict[str, float | int]:
        if self.point_count == 0 or not self.sequence_mae:
            raise ValueError("No imputation errors were accumulated.")
        return {
            "mae": self.absolute_error_sum / self.point_count,
            "rmse": float(np.sqrt(self.squared_error_sum / self.point_count)),
            "macro_mae": float(np.mean(self.sequence_mae)),
            "macro_nrmse": (
                float(np.mean(self.sequence_nrmse))
                if self.sequence_nrmse
                else float("nan")
            ),
            "evaluated_points": self.point_count,
            "evaluated_sequences": len(self.sequence_mae),
            "nrmse_sequences": len(self.sequence_nrmse),
            "constant_sequences_excluded_from_nrmse": self.unscaled_sequence_count,
            "negative_prediction_rate": self.negative_prediction_count / self.point_count,
        }


def _build_reconstruction_model(
    config: ExperimentConfig,
    moment_pipeline: Any,
) -> Any:
    if not config.model.config_path.is_file():
        raise FileNotFoundError(f"MOMENT model config not found: {config.model.config_path}")
    with config.model.config_path.open("r", encoding="utf-8") as file:
        pretrained_config = json.load(file)
    return moment_pipeline.from_pretrained(
        config.model.model_id,
        config=pretrained_config,
        model_kwargs={
            "task_name": "reconstruction",
            "seq_len": config.data.max_length,
            "n_channels": config.model.num_channels,
            "freeze_embedder": True,
            "freeze_encoder": True,
            "enable_gradient_checkpointing": False,
        },
    )


def _eligible_test_data(
    bundle: DatasetBundle,
    patch_len: int,
    min_complete_patches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    capped_lengths = np.minimum(bundle.lengths_test, bundle.x_test.shape[1])
    eligible = capped_lengths // patch_len >= min_complete_patches
    if not eligible.any():
        raise ValueError("No test sequences satisfy the minimum complete patch count.")
    return (
        bundle.x_test[eligible],
        bundle.mask_test[eligible],
        capped_lengths[eligible],
        bundle.ids_test[eligible],
        eligible,
    )


def _evaluate_baselines(
    values: np.ndarray,
    input_mask: np.ndarray,
    observation_mask: np.ndarray,
    lengths: np.ndarray,
    example_count: int,
) -> tuple[dict[str, dict[str, float | int]], dict[str, np.ndarray]]:
    accumulators = {method: ErrorAccumulator() for method in BASELINE_METHODS}
    examples = {
        method: np.full((example_count, values.shape[1]), np.nan, dtype=np.float32)
        for method in BASELINE_METHODS
    }
    for row, length in enumerate(lengths):
        valid_length = int(length)
        sequence = values[row, :valid_length]
        visible = observation_mask[row, :valid_length].astype(bool)
        hidden = input_mask[row, :valid_length].astype(bool) & ~visible
        for method, accumulator in accumulators.items():
            prediction = baseline_prediction(sequence, visible, method)
            accumulator.update(sequence[hidden], prediction[hidden], sequence)
            if row < example_count:
                examples[method][row, :valid_length] = prediction.astype(np.float32)
    return (
        {method: accumulator.result() for method, accumulator in accumulators.items()},
        examples,
    )


def _evaluate_moment(
    torch: Any,
    model: Any,
    device: Any,
    values: np.ndarray,
    input_mask: np.ndarray,
    observation_mask: np.ndarray,
    lengths: np.ndarray,
    *,
    batch_size: int,
    amp_enabled: bool,
    example_count: int,
) -> tuple[dict[str, float | int], np.ndarray, float, bool, int | None]:
    accumulator = ErrorAccumulator()
    examples = np.full((example_count, values.shape[1]), np.nan, dtype=np.float32)
    active_amp = amp_enabled
    amp_fallback_batch_start: int | None = None
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            batch_values = values[start:stop].copy()
            batch_observation = observation_mask[start:stop]
            batch_values[batch_observation == 0] = 0.0
            batch_x = torch.from_numpy(batch_values[:, None, :]).to(
                device=device,
                dtype=torch.float32,
            )
            batch_input_mask = torch.from_numpy(input_mask[start:stop]).to(
                device=device,
                dtype=torch.float32,
            )
            batch_mask = torch.from_numpy(batch_observation).to(
                device=device,
                dtype=torch.float32,
            )
            with torch.amp.autocast(device_type=device.type, enabled=active_amp):
                output = model.reconstruct(
                    x_enc=batch_x,
                    input_mask=batch_input_mask,
                    mask=batch_mask,
                )
            reconstruction = output.reconstruction[:, 0, :]
            if active_amp and not torch.isfinite(reconstruction).all():
                print(
                    "Warning: non-finite MOMENT reconstruction under CUDA AMP at "
                    f"batch start {start}; retrying in FP32 and disabling AMP."
                )
                active_amp = False
                amp_fallback_batch_start = start
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    output = model.reconstruct(
                        x_enc=batch_x,
                        input_mask=batch_input_mask,
                        mask=batch_mask,
                    )
                reconstruction = output.reconstruction[:, 0, :]
            if not torch.isfinite(reconstruction).all():
                bad_rows = (
                    ~torch.isfinite(reconstruction).all(dim=1)
                ).nonzero(as_tuple=False).flatten().cpu().tolist()
                raise FloatingPointError(
                    "Non-finite MOMENT reconstruction detected in FP32 for global "
                    f"sample rows {[start + row for row in bad_rows]}."
                )
            predictions = reconstruction.float().cpu().numpy()
            for local_row, prediction in enumerate(predictions):
                row = start + local_row
                valid_length = int(lengths[row])
                hidden = (
                    input_mask[row, :valid_length].astype(bool)
                    & ~observation_mask[row, :valid_length].astype(bool)
                )
                sequence = values[row, :valid_length]
                accumulator.update(
                    sequence[hidden],
                    prediction[:valid_length][hidden],
                    sequence,
                )
                if row < example_count:
                    examples[row, :valid_length] = prediction[:valid_length]
    return (
        accumulator.result(),
        examples,
        time.perf_counter() - started,
        active_amp,
        amp_fallback_batch_start,
    )


def _mask_sha256(sample_ids: np.ndarray, observation_mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    for sample_id, row in zip(sample_ids, observation_mask, strict=True):
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(row, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _summaries(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for pattern in sorted({str(row["pattern"]) for row in condition_rows}):
        for rate in sorted(
            {float(row["mask_rate"]) for row in condition_rows if row["pattern"] == pattern}
        ):
            for method in ALL_METHODS:
                selected = [
                    row
                    for row in condition_rows
                    if row["pattern"] == pattern
                    and float(row["mask_rate"]) == rate
                    and row["method"] == method
                ]
                summary: dict[str, Any] = {
                    "pattern": pattern,
                    "mask_rate": rate,
                    "method": method,
                    "seeds": len(selected),
                }
                for metric in ("mae", "rmse", "macro_mae", "macro_nrmse"):
                    values = [float(row[metric]) for row in selected]
                    summary[f"{metric}_mean"] = statistics.mean(values)
                    summary[f"{metric}_sample_std"] = (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    )
                summaries.append(summary)
    return summaries


def _data_record(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    eligible: np.ndarray,
) -> dict[str, Any]:
    return {
        "dataset_sha256": bundle.dataset_sha256,
        "split_manifest": str(bundle.split_path),
        "source_split": "test",
        "source_test_samples": len(bundle.y_test),
        "evaluated_test_samples": int(eligible.sum()),
        "excluded_for_insufficient_complete_patches": int((~eligible).sum()),
        "labels_used": False,
        "invalid_labels_excluded_by_unified_protocol": bundle.invalid_label_counts,
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
            "Zero-shot imputation requires data.normalize=none so MOMENT can estimate "
            "RevIN statistics from visible values only and metrics remain in raw units."
        )
    bundle = load_dataset(config.data)
    shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
    device = select_device(torch, config.training.device)
    model = _build_reconstruction_model(config, MOMENTPipeline)
    model.init()
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    patch_len = int(getattr(model, "patch_len", 0))
    if patch_len <= 0:
        raise ValueError("Unable to determine MOMENT patch length.")
    values, input_mask, lengths, sample_ids, eligible = _eligible_test_data(
        bundle,
        patch_len,
        config.imputation.min_complete_patches,
    )
    input_mask_u8 = input_mask.astype(np.uint8, copy=False)
    amp_requested = bool(config.training.amp and device.type == "cuda")
    amp_enabled = amp_requested
    amp_fallback_triggered = False
    amp_fallback_condition: dict[str, Any] | None = None
    example_count = min(config.imputation.example_count, len(values))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    condition_rows: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    for pattern in config.imputation.mask_patterns:
        for mask_rate in config.imputation.mask_rates:
            for mask_seed in config.imputation.mask_seeds:
                print(
                    f"Evaluating pattern={pattern} rate={mask_rate:g} seed={mask_seed}"
                )
                observation_mask = generate_observation_mask(
                    input_mask_u8,
                    lengths,
                    sample_ids,
                    mask_rate=mask_rate,
                    pattern=pattern,
                    mask_seed=mask_seed,
                    patch_len=patch_len,
                    min_complete_patches=config.imputation.min_complete_patches,
                )
                hidden = (input_mask_u8 == 1) & (observation_mask == 0)
                baseline_metrics, baseline_examples = _evaluate_baselines(
                    values,
                    input_mask_u8,
                    observation_mask,
                    lengths,
                    example_count,
                )
                (
                    moment_metrics,
                    moment_examples,
                    inference_seconds,
                    condition_amp_enabled,
                    amp_fallback_batch_start,
                ) = _evaluate_moment(
                    torch,
                    model,
                    device,
                    values,
                    input_mask_u8,
                    observation_mask,
                    lengths,
                    batch_size=config.training.feature_extraction_batch_size,
                    amp_enabled=amp_enabled,
                    example_count=example_count,
                )
                if amp_enabled and not condition_amp_enabled:
                    amp_fallback_triggered = True
                    amp_fallback_condition = {
                        "pattern": pattern,
                        "mask_rate": mask_rate,
                        "mask_seed": mask_seed,
                        "batch_start": amp_fallback_batch_start,
                    }
                    amp_enabled = False
                method_metrics = {"moment_zero_shot": moment_metrics, **baseline_metrics}
                condition = {
                    "pattern": pattern,
                    "requested_mask_rate": mask_rate,
                    "mask_seed": mask_seed,
                    "mask_sha256": _mask_sha256(sample_ids, observation_mask),
                    "masked_points": int(hidden.sum()),
                    "complete_patch_points": int(
                        sum((int(length) // patch_len) * patch_len for length in lengths)
                    ),
                    "actual_mask_rate": float(
                        hidden.sum()
                        / sum((int(length) // patch_len) * patch_len for length in lengths)
                    ),
                    "moment_inference_seconds": inference_seconds,
                    "methods": method_metrics,
                }
                conditions.append(condition)
                for method, metrics in method_metrics.items():
                    condition_rows.append(
                        {
                            "pattern": pattern,
                            "mask_rate": mask_rate,
                            "mask_seed": mask_seed,
                            "actual_mask_rate": condition["actual_mask_rate"],
                            "method": method,
                            **metrics,
                            "moment_inference_seconds": (
                                inference_seconds if method == "moment_zero_shot" else 0.0
                            ),
                        }
                    )
                if mask_seed == config.imputation.mask_seeds[0] and example_count:
                    tag = f"{pattern}_rate{mask_rate:g}".replace(".", "p")
                    np.savez_compressed(
                        context.run_dir / f"examples_{tag}.npz",
                        sample_ids=sample_ids[:example_count].astype(str),
                        lengths=lengths[:example_count],
                        raw_values=values[:example_count],
                        input_mask=input_mask_u8[:example_count],
                        observation_mask=observation_mask[:example_count],
                        moment_zero_shot=moment_examples,
                        **baseline_examples,
                    )
                atomic_write_json(
                    context.run_dir / "metrics_partial.json",
                    {
                        "protocol_version": IMPUTATION_PROTOCOL_VERSION,
                        "completed_conditions": conditions,
                    },
                )

    summary = _summaries(condition_rows)
    pd.DataFrame(condition_rows).to_csv(
        context.run_dir / "condition_metrics.csv",
        index=False,
    )
    pd.DataFrame(summary).to_csv(context.run_dir / "summary.csv", index=False)
    execution = {
        "device": str(device),
        "amp_requested": amp_requested,
        "amp_enabled": amp_enabled,
        "amp_fallback_triggered": amp_fallback_triggered,
        "amp_fallback_condition": amp_fallback_condition,
        "batch_size": config.training.feature_extraction_batch_size,
        "total_moment_inference_seconds": float(
            sum(condition["moment_inference_seconds"] for condition in conditions)
        ),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": 0,
    }
    result = {
        "model": "MOMENT_ZERO_SHOT_IMPUTATION",
        "run_name": context.run_name,
        "data": _data_record(bundle, config, eligible),
        "protocol": {
            "imputation_protocol_version": IMPUTATION_PROTOCOL_VERSION,
            "pretrained_reconstruction_head": True,
            "parameter_updates": False,
            "target_values_used_for_normalization": False,
            "normalization": "MOMENT RevIN statistics from visible values only",
            "evaluation_positions": "artificially hidden valid complete-patch points only",
            "patch_len": patch_len,
            "mask_rates": list(config.imputation.mask_rates),
            "mask_patterns": list(config.imputation.mask_patterns),
            "mask_seeds": list(config.imputation.mask_seeds),
            "baselines": list(BASELINE_METHODS),
            "primary_metric": "macro_nrmse",
        },
        "execution": execution,
        "conditions": conditions,
        "summary": summary,
    }
    atomic_write_json(context.run_dir / "metrics.json", result)
    (context.run_dir / "metrics_partial.json").unlink(missing_ok=True)
    print(json.dumps(execution, indent=2, ensure_ascii=False))
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
        model_name="MOMENT_ZERO_SHOT_IMPUTATION",
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
        description="Evaluate MOMENT zero-shot imputation against statistical baselines."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/moment_imputation_zero_shot.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--run-name", help="Unique output directory name.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    evaluate(config, config_path, run_name=args.run_name)


if __name__ == "__main__":
    main()
