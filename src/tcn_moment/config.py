from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    csv_path: Path
    delimiter: str = " "
    id_column: str = "charging_station_id"
    sequence_column: str = "charging_powers_str"
    label_column: str = "InsertedColumn"
    invalid_labels: tuple[str, ...] = ("5",)
    max_length: int = 1024
    min_length: int = 18
    normalize: str = "zscore"
    validation_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 42
    split_path: Path = Path("artifacts/splits/unified_split.json")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "AutonLab/MOMENT-1-large"
    config_path: Path = Path("configs/models/moment-1-large.json")
    num_channels: int = 1
    freeze_backbone: bool = False
    unfreeze_last_n_layers: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path = Path("artifacts/moment")
    epochs: int = 10
    batch_size: int = 16
    evaluation_batch_size: int = 32
    feature_extraction_batch_size: int = 32
    cached_feature_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    num_workers: int = 0
    prefetch_factor: int = 2
    device: str = "auto"
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    gradient_clip_norm: float = 1.0
    amp: bool = True
    fused_optimizer: bool = True
    cache_frozen_features: bool = True
    keep_completed_checkpoint: bool = False


@dataclass(frozen=True)
class TCNModelConfig:
    channels: tuple[int, ...] = (64, 64, 128, 128)
    kernel_size: int = 3
    dropout: float = 0.3


@dataclass(frozen=True)
class TCNTrainingConfig:
    output_dir: Path = Path("artifacts/tcn")
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    gradient_clip_norm: float = 1.0
    amp: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    tcn_model: TCNModelConfig
    tcn_training: TCNTrainingConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(config_path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = config_path.resolve()
    visited = set() if seen is None else seen
    if resolved in visited:
        raise ValueError(f"Circular config inheritance detected at {resolved}")
    visited.add(resolved)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (config_path.parent / str(parent)).resolve()
    return _deep_merge(_load_raw_config(parent_path, visited), raw)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = _load_raw_config(config_path)

    data_raw: dict[str, Any] = raw.get("data", {})
    model_raw: dict[str, Any] = raw.get("model", {})
    training_raw: dict[str, Any] = raw.get("training", {})
    tcn_model_raw: dict[str, Any] = raw.get("tcn_model", {})
    tcn_training_raw: dict[str, Any] = raw.get("tcn_training", {})

    data = DataConfig(
        csv_path=Path(data_raw.get("csv_path", "data/raw/最新多.csv")),
        delimiter=str(data_raw.get("delimiter", " ")),
        id_column=str(data_raw.get("id_column", "charging_station_id")),
        sequence_column=str(data_raw.get("sequence_column", "charging_powers_str")),
        label_column=str(data_raw.get("label_column", "InsertedColumn")),
        invalid_labels=tuple(str(label) for label in data_raw.get("invalid_labels", ["5"])),
        max_length=int(data_raw.get("max_length", 1024)),
        min_length=int(data_raw.get("min_length", 18)),
        normalize=str(data_raw.get("normalize", "zscore")),
        validation_size=float(data_raw.get("validation_size", 0.1)),
        test_size=float(data_raw.get("test_size", 0.2)),
        random_state=int(data_raw.get("random_state", 42)),
        split_path=Path(data_raw.get("split_path", "artifacts/splits/unified_split.json")),
    )
    model = ModelConfig(
        model_id=str(model_raw.get("model_id", "AutonLab/MOMENT-1-large")),
        config_path=Path(
            model_raw.get("config_path", "configs/models/moment-1-large.json")
        ),
        num_channels=int(model_raw.get("num_channels", 1)),
        freeze_backbone=bool(model_raw.get("freeze_backbone", False)),
        unfreeze_last_n_layers=int(model_raw.get("unfreeze_last_n_layers", 0)),
    )
    training = TrainingConfig(
        output_dir=Path(training_raw.get("output_dir", "artifacts/moment")),
        epochs=int(training_raw.get("epochs", 10)),
        batch_size=int(training_raw.get("batch_size", 16)),
        evaluation_batch_size=int(
            training_raw.get("evaluation_batch_size", training_raw.get("batch_size", 16))
        ),
        feature_extraction_batch_size=int(
            training_raw.get(
                "feature_extraction_batch_size",
                training_raw.get("batch_size", 16),
            )
        ),
        cached_feature_batch_size=int(
            training_raw.get(
                "cached_feature_batch_size",
                training_raw.get("batch_size", 16),
            )
        ),
        gradient_accumulation_steps=int(
            training_raw.get("gradient_accumulation_steps", 1)
        ),
        learning_rate=float(training_raw.get("learning_rate", 1e-5)),
        backbone_learning_rate=float(training_raw.get("backbone_learning_rate", 1e-5)),
        weight_decay=float(training_raw.get("weight_decay", 1e-2)),
        num_workers=int(training_raw.get("num_workers", 0)),
        prefetch_factor=int(training_raw.get("prefetch_factor", 2)),
        device=str(training_raw.get("device", "auto")),
        early_stopping_patience=int(training_raw.get("early_stopping_patience", 7)),
        early_stopping_min_delta=float(training_raw.get("early_stopping_min_delta", 1e-4)),
        scheduler_patience=int(training_raw.get("scheduler_patience", 3)),
        scheduler_factor=float(training_raw.get("scheduler_factor", 0.5)),
        gradient_clip_norm=float(training_raw.get("gradient_clip_norm", 1.0)),
        amp=bool(training_raw.get("amp", True)),
        fused_optimizer=bool(training_raw.get("fused_optimizer", True)),
        cache_frozen_features=bool(training_raw.get("cache_frozen_features", True)),
        keep_completed_checkpoint=bool(
            training_raw.get("keep_completed_checkpoint", False)
        ),
    )
    for field_name in (
        "batch_size",
        "evaluation_batch_size",
        "feature_extraction_batch_size",
        "cached_feature_batch_size",
        "gradient_accumulation_steps",
        "prefetch_factor",
    ):
        if getattr(training, field_name) <= 0:
            raise ValueError(f"training.{field_name} must be positive.")
    if training.num_workers < 0:
        raise ValueError("training.num_workers must be non-negative.")
    tcn_model = TCNModelConfig(
        channels=tuple(int(value) for value in tcn_model_raw.get("channels", [64, 64, 128, 128])),
        kernel_size=int(tcn_model_raw.get("kernel_size", 3)),
        dropout=float(tcn_model_raw.get("dropout", 0.3)),
    )
    tcn_training = TCNTrainingConfig(
        output_dir=Path(tcn_training_raw.get("output_dir", "artifacts/tcn")),
        epochs=int(tcn_training_raw.get("epochs", 50)),
        batch_size=int(tcn_training_raw.get("batch_size", 32)),
        learning_rate=float(tcn_training_raw.get("learning_rate", 1e-3)),
        weight_decay=float(tcn_training_raw.get("weight_decay", 1e-4)),
        num_workers=int(tcn_training_raw.get("num_workers", 0)),
        device=str(tcn_training_raw.get("device", "auto")),
        early_stopping_patience=int(tcn_training_raw.get("early_stopping_patience", 7)),
        early_stopping_min_delta=float(tcn_training_raw.get("early_stopping_min_delta", 1e-4)),
        scheduler_patience=int(tcn_training_raw.get("scheduler_patience", 3)),
        scheduler_factor=float(tcn_training_raw.get("scheduler_factor", 0.5)),
        gradient_clip_norm=float(tcn_training_raw.get("gradient_clip_norm", 1.0)),
        amp=bool(tcn_training_raw.get("amp", False)),
    )
    return ExperimentConfig(
        data=data,
        model=model,
        training=training,
        tcn_model=tcn_model,
        tcn_training=tcn_training,
    )


def with_random_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    split_path = config.data.split_path
    seeded_name = f"{split_path.stem}_seed{seed}{split_path.suffix}"
    seeded_data = replace(
        config.data,
        random_state=seed,
        split_path=split_path.with_name(seeded_name),
    )
    return replace(config, data=seeded_data)
