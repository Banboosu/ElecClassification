from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    csv_path: Path
    delimiter: str = " "
    sequence_column: str = "charging_powers_str"
    label_column: str = "InsertedColumn"
    invalid_labels: tuple[str, ...] = ("5",)
    max_length: int = 1024
    min_length: int = 18
    normalize: str = "zscore"
    validation_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "AutonLab/MOMENT-1-large"
    num_channels: int = 1
    freeze_backbone: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path = Path("artifacts/moment")
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    num_workers: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class TCNModelConfig:
    channels: tuple[int, ...] = (64, 64, 128, 128)
    kernel_size: int = 3
    dropout: float = 0.3


@dataclass(frozen=True)
class TCNTrainingConfig:
    output_dir: Path = Path("artifacts/tcn_unified")
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    tcn_model: TCNModelConfig
    tcn_training: TCNTrainingConfig


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    data_raw: dict[str, Any] = raw.get("data", {})
    model_raw: dict[str, Any] = raw.get("model", {})
    training_raw: dict[str, Any] = raw.get("training", {})
    tcn_model_raw: dict[str, Any] = raw.get("tcn_model", {})
    tcn_training_raw: dict[str, Any] = raw.get("tcn_training", {})

    data = DataConfig(
        csv_path=Path(data_raw.get("csv_path", "data/raw/最新多.csv")),
        delimiter=str(data_raw.get("delimiter", " ")),
        sequence_column=str(data_raw.get("sequence_column", "charging_powers_str")),
        label_column=str(data_raw.get("label_column", "InsertedColumn")),
        invalid_labels=tuple(str(label) for label in data_raw.get("invalid_labels", ["5"])),
        max_length=int(data_raw.get("max_length", 1024)),
        min_length=int(data_raw.get("min_length", 18)),
        normalize=str(data_raw.get("normalize", "zscore")),
        validation_size=float(data_raw.get("validation_size", 0.1)),
        test_size=float(data_raw.get("test_size", 0.2)),
        random_state=int(data_raw.get("random_state", 42)),
    )
    model = ModelConfig(
        model_id=str(model_raw.get("model_id", "AutonLab/MOMENT-1-large")),
        num_channels=int(model_raw.get("num_channels", 1)),
        freeze_backbone=bool(model_raw.get("freeze_backbone", False)),
    )
    training = TrainingConfig(
        output_dir=Path(training_raw.get("output_dir", "artifacts/moment")),
        epochs=int(training_raw.get("epochs", 10)),
        batch_size=int(training_raw.get("batch_size", 16)),
        learning_rate=float(training_raw.get("learning_rate", 1e-5)),
        weight_decay=float(training_raw.get("weight_decay", 1e-2)),
        num_workers=int(training_raw.get("num_workers", 0)),
        device=str(training_raw.get("device", "auto")),
    )
    tcn_model = TCNModelConfig(
        channels=tuple(int(value) for value in tcn_model_raw.get("channels", [64, 64, 128, 128])),
        kernel_size=int(tcn_model_raw.get("kernel_size", 3)),
        dropout=float(tcn_model_raw.get("dropout", 0.3)),
    )
    tcn_training = TCNTrainingConfig(
        output_dir=Path(tcn_training_raw.get("output_dir", "artifacts/tcn_unified")),
        epochs=int(tcn_training_raw.get("epochs", 50)),
        batch_size=int(tcn_training_raw.get("batch_size", 32)),
        learning_rate=float(tcn_training_raw.get("learning_rate", 1e-3)),
        weight_decay=float(tcn_training_raw.get("weight_decay", 1e-4)),
        num_workers=int(tcn_training_raw.get("num_workers", 0)),
        device=str(tcn_training_raw.get("device", "auto")),
    )

    return ExperimentConfig(
        data=data,
        model=model,
        training=training,
        tcn_model=tcn_model,
        tcn_training=tcn_training,
    )
