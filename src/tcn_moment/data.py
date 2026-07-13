from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tcn_moment.config import DataConfig, load_config


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    invalid_label_counts: dict[str, int]
    trainable_invalid_label_counts: dict[str, int]
    short_sequence_count: int

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder.classes_)


def parse_power_sequence(value: object) -> list[float]:
    if not isinstance(value, str):
        return []

    powers: list[float] = []
    for item in value.replace('"', "").strip().split(","):
        item = item.strip()
        if not item:
            continue
        try:
            powers.append(float(item))
        except ValueError:
            continue
    return powers


def normalize_sequence(sequence: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return sequence
    if mode == "minmax":
        min_value = float(sequence.min())
        max_value = float(sequence.max())
        scale = max_value - min_value
        if scale < 1e-8:
            return np.zeros_like(sequence, dtype=np.float32)
        return ((sequence - min_value) / scale).astype(np.float32)
    if mode == "zscore":
        mean = float(sequence.mean())
        std = float(sequence.std())
        if std < 1e-8:
            return np.zeros_like(sequence, dtype=np.float32)
        return ((sequence - mean) / std).astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def prepare_sequence(sequence: list[float], max_length: int, normalize: str) -> np.ndarray:
    array = np.asarray(sequence[:max_length], dtype=np.float32)
    array = normalize_sequence(array, normalize)

    if len(array) < max_length:
        padded = np.zeros(max_length, dtype=np.float32)
        padded[: len(array)] = array
        return padded
    return array


def load_dataset(config: DataConfig) -> DatasetBundle:
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < config.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if config.test_size + config.validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    data = pd.read_csv(config.csv_path, delimiter=config.delimiter)
    required = {config.sequence_column, config.label_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    raw_sequences = data[config.sequence_column].map(parse_power_sequence)
    labels_all = data[config.label_column].astype(str)

    long_enough_mask = raw_sequences.map(len) >= config.min_length
    invalid_label_mask = labels_all.isin(config.invalid_labels)
    valid_mask = long_enough_mask & ~invalid_label_mask

    invalid_label_counts = labels_all[invalid_label_mask].value_counts().sort_index().to_dict()
    trainable_invalid_label_counts = (
        labels_all[invalid_label_mask & long_enough_mask].value_counts().sort_index().to_dict()
    )
    short_sequence_count = int((~long_enough_mask).sum())

    valid_sequences = raw_sequences[valid_mask].tolist()
    labels = labels_all[valid_mask].to_numpy()

    if not valid_sequences:
        raise ValueError("No valid sequences after filtering. Check min_length and input columns.")

    x = np.stack(
        [
            prepare_sequence(seq, max_length=config.max_length, normalize=config.normalize)
            for seq in valid_sequences
        ]
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels).astype(np.int64)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    validation_fraction_of_remainder = config.validation_size / (1.0 - config.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_fraction_of_remainder,
        random_state=config.random_state,
        stratify=y_train_val if len(np.unique(y_train_val)) > 1 else None,
    )

    return DatasetBundle(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_encoder=label_encoder,
        invalid_label_counts={str(key): int(value) for key, value in invalid_label_counts.items()},
        trainable_invalid_label_counts={
            str(key): int(value) for key, value in trainable_invalid_label_counts.items()
        },
        short_sequence_count=short_sequence_count,
    )


def save_label_encoder(label_encoder: LabelEncoder, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect charging power classification data.")
    parser.add_argument("--config", default="configs/moment.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_dataset(config.data)
    labels, counts = np.unique(
        np.concatenate([bundle.y_train, bundle.y_val, bundle.y_test]),
        return_counts=True,
    )

    print(f"CSV: {config.data.csv_path}")
    print(f"Train shape: {bundle.x_train.shape}")
    print(f"Validation shape: {bundle.x_val.shape}")
    print(f"Test shape: {bundle.x_test.shape}")
    print(f"Classes: {bundle.label_encoder.classes_.tolist()}")
    print(f"Short sequences excluded: {bundle.short_sequence_count}")
    if bundle.invalid_label_counts:
        print("Invalid labels separated, all rows:")
        for label, count in bundle.invalid_label_counts.items():
            print(f"  {label}: {count}")
        print("Invalid labels separated after length filter:")
        for label, count in bundle.trainable_invalid_label_counts.items():
            print(f"  {label}: {count}")
    print("Class counts:")
    for label_id, count in zip(labels, counts, strict=True):
        label_name = bundle.label_encoder.inverse_transform([label_id])[0]
        print(f"  {label_name}: {count}")


if __name__ == "__main__":
    main()
