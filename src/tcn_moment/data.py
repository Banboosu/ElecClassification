from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tcn_moment.config import DataConfig, load_config
from tcn_moment.io_utils import atomic_write_json, sha256_file


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    mask_train: np.ndarray
    mask_val: np.ndarray
    mask_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    ids_train: np.ndarray
    ids_val: np.ndarray
    ids_test: np.ndarray
    lengths_train: np.ndarray
    lengths_val: np.ndarray
    lengths_test: np.ndarray
    label_encoder: LabelEncoder
    invalid_label_counts: dict[str, int]
    trainable_invalid_label_counts: dict[str, int]
    short_sequence_count: int
    truncated_sequence_count: int
    dataset_sha256: str
    split_path: Path

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder.classes_)

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            "train": len(self.y_train),
            "validation": len(self.y_val),
            "test": len(self.y_test),
        }


def parse_power_sequence(value: object) -> list[float]:
    if not isinstance(value, str):
        return []

    powers: list[float] = []
    for item in value.replace('"', "").strip().split(","):
        item = item.strip()
        if not item:
            continue
        try:
            power = float(item)
            if np.isfinite(power):
                powers.append(power)
        except ValueError:
            continue
    return powers


def normalize_sequence(sequence: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return sequence.astype(np.float32, copy=False)
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
    padded = np.zeros(max_length, dtype=np.float32)
    padded[: len(array)] = array
    return padded


def sequence_mask(length: int, max_length: int) -> np.ndarray:
    valid_length = min(length, max_length)
    mask = np.zeros(max_length, dtype=np.float32)
    mask[:valid_length] = 1.0
    return mask


def _validate_sizes(config: DataConfig) -> None:
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < config.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if config.test_size + config.validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")
    if config.min_length <= 0 or config.max_length < config.min_length:
        raise ValueError("Require 0 < min_length <= max_length.")


def _protocol(config: DataConfig, dataset_sha256: str) -> dict[str, Any]:
    return {
        "dataset_sha256": dataset_sha256,
        "id_column": config.id_column,
        "sequence_column": config.sequence_column,
        "label_column": config.label_column,
        "invalid_labels": list(config.invalid_labels),
        "min_length": config.min_length,
        "validation_size": config.validation_size,
        "test_size": config.test_size,
        "random_state": config.random_state,
    }


def _protocol_hash(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _class_counts(ids: list[str], label_by_id: dict[str, str]) -> dict[str, int]:
    counts = pd.Series([label_by_id[item] for item in ids]).value_counts().sort_index()
    return {
        str(label): int(count)
        for label, count in counts.items()
    }


def _create_split_manifest(
    ids: np.ndarray,
    labels: np.ndarray,
    config: DataConfig,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    ids_train_val, ids_test, labels_train_val, _ = train_test_split(
        ids,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )
    validation_fraction = config.validation_size / (1.0 - config.test_size)
    ids_train, ids_val, _, _ = train_test_split(
        ids_train_val,
        labels_train_val,
        test_size=validation_fraction,
        random_state=config.random_state,
        stratify=labels_train_val,
    )
    split_ids = {
        "train": [str(item) for item in ids_train],
        "validation": [str(item) for item in ids_val],
        "test": [str(item) for item in ids_test],
    }
    label_by_id = dict(zip(ids.astype(str), labels.astype(str), strict=True))
    return {
        "version": 1,
        "protocol": protocol,
        "protocol_sha256": _protocol_hash(protocol),
        "counts": {name: len(values) for name, values in split_ids.items()},
        "class_counts": {
            name: _class_counts(values, label_by_id) for name, values in split_ids.items()
        },
        "splits": split_ids,
    }


def _validate_manifest(
    manifest: dict[str, Any],
    valid_ids: set[str],
    protocol: dict[str, Any],
) -> None:
    if manifest.get("protocol_sha256") != _protocol_hash(protocol):
        raise ValueError(
            "Split manifest does not match the dataset or filtering protocol. "
            "Use a different data.split_path or explicitly rebuild the split."
        )
    splits = manifest.get("splits", {})
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Split manifest must contain train, validation, and test lists.")
    split_sets = {name: set(str(item) for item in values) for name, values in splits.items()}
    if split_sets["train"] & split_sets["validation"]:
        raise ValueError("Train and validation sample IDs overlap.")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("Train and test sample IDs overlap.")
    if split_sets["validation"] & split_sets["test"]:
        raise ValueError("Validation and test sample IDs overlap.")
    if set().union(*split_sets.values()) != valid_ids:
        raise ValueError("Split manifest sample IDs do not exactly match the valid dataset rows.")


def _load_or_create_manifest(
    ids: np.ndarray,
    labels: np.ndarray,
    config: DataConfig,
    protocol: dict[str, Any],
    *,
    rebuild_split: bool,
) -> dict[str, Any]:
    split_path = config.split_path
    if split_path.exists() and not rebuild_split:
        with split_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    else:
        manifest = _create_split_manifest(ids, labels, config, protocol)
        atomic_write_json(split_path, manifest)
    _validate_manifest(manifest, set(ids.astype(str)), protocol)
    return manifest


def load_dataset(config: DataConfig, *, rebuild_split: bool = False) -> DatasetBundle:
    _validate_sizes(config)
    dataset_sha256 = sha256_file(config.csv_path)
    data = pd.read_csv(config.csv_path, delimiter=config.delimiter)
    required = {config.id_column, config.sequence_column, config.label_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    sample_ids_all = data[config.id_column].astype(str)
    if sample_ids_all.duplicated().any():
        duplicates = sample_ids_all[sample_ids_all.duplicated()].head(5).tolist()
        raise ValueError(f"Sample IDs must be unique. Duplicate examples: {duplicates}")

    raw_sequences = data[config.sequence_column].map(parse_power_sequence)
    lengths_all = raw_sequences.map(len)
    labels_all = data[config.label_column].astype(str)
    long_enough_mask = lengths_all >= config.min_length
    invalid_label_mask = labels_all.isin(config.invalid_labels)
    valid_mask = long_enough_mask & ~invalid_label_mask

    invalid_label_counts = labels_all[invalid_label_mask].value_counts().sort_index().to_dict()
    trainable_invalid_label_counts = (
        labels_all[invalid_label_mask & long_enough_mask].value_counts().sort_index().to_dict()
    )
    short_sequence_count = int((~long_enough_mask).sum())
    truncated_sequence_count = int((lengths_all[valid_mask] > config.max_length).sum())

    valid_data = pd.DataFrame(
        {
            "sample_id": sample_ids_all[valid_mask].to_numpy(),
            "sequence": raw_sequences[valid_mask].tolist(),
            "length": lengths_all[valid_mask].to_numpy(dtype=np.int64),
            "label": labels_all[valid_mask].to_numpy(),
        }
    ).set_index("sample_id", drop=False)
    if valid_data.empty:
        raise ValueError("No valid sequences after filtering. Check min_length and input columns.")

    protocol = _protocol(config, dataset_sha256)
    manifest = _load_or_create_manifest(
        valid_data["sample_id"].to_numpy(dtype=str),
        valid_data["label"].to_numpy(dtype=str),
        config,
        protocol,
        rebuild_split=rebuild_split,
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_data["label"].to_numpy(dtype=str))

    prepared: dict[str, tuple[np.ndarray, ...]] = {}
    for split_name in ("train", "validation", "test"):
        split_ids = np.asarray(manifest["splits"][split_name], dtype=str)
        split_rows = valid_data.loc[split_ids]
        sequences = split_rows["sequence"].tolist()
        lengths = split_rows["length"].to_numpy(dtype=np.int64)
        x = np.stack(
            [prepare_sequence(seq, config.max_length, config.normalize) for seq in sequences]
        )
        masks = np.stack([sequence_mask(int(length), config.max_length) for length in lengths])
        y = label_encoder.transform(split_rows["label"].to_numpy(dtype=str)).astype(np.int64)
        prepared[split_name] = (x, masks, y, split_ids, lengths)

    x_train, mask_train, y_train, ids_train, lengths_train = prepared["train"]
    x_val, mask_val, y_val, ids_val, lengths_val = prepared["validation"]
    x_test, mask_test, y_test, ids_test, lengths_test = prepared["test"]
    return DatasetBundle(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        mask_train=mask_train,
        mask_val=mask_val,
        mask_test=mask_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        ids_train=ids_train,
        ids_val=ids_val,
        ids_test=ids_test,
        lengths_train=lengths_train,
        lengths_val=lengths_val,
        lengths_test=lengths_test,
        label_encoder=label_encoder,
        invalid_label_counts={str(key): int(value) for key, value in invalid_label_counts.items()},
        trainable_invalid_label_counts={
            str(key): int(value) for key, value in trainable_invalid_label_counts.items()
        },
        short_sequence_count=short_sequence_count,
        truncated_sequence_count=truncated_sequence_count,
        dataset_sha256=dataset_sha256,
        split_path=config.split_path,
    )


def save_label_encoder(label_encoder: LabelEncoder, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect charging power classification data.")
    parser.add_argument("--config", default="configs/moment.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="Explicitly replace the persisted split manifest.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_dataset(config.data, rebuild_split=args.rebuild_split)
    labels, counts = np.unique(
        np.concatenate([bundle.y_train, bundle.y_val, bundle.y_test]),
        return_counts=True,
    )
    print(f"CSV: {config.data.csv_path}")
    print(f"Dataset SHA-256: {bundle.dataset_sha256}")
    print(f"Split manifest: {bundle.split_path}")
    print(f"Train shape: {bundle.x_train.shape}")
    print(f"Validation shape: {bundle.x_val.shape}")
    print(f"Test shape: {bundle.x_test.shape}")
    print(f"Classes: {bundle.label_encoder.classes_.tolist()}")
    print(f"Short sequences excluded: {bundle.short_sequence_count}")
    print(f"Sequences truncated to max_length: {bundle.truncated_sequence_count}")
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
