from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from tcn_moment.config import load_config
from tcn_moment.data import load_dataset, parse_power_sequence
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.metrics import classification_metrics


def _describe(values: pd.Series) -> dict[str, float]:
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def _sequence_statistics(sequences: pd.Series) -> pd.DataFrame:
    records = []
    for sequence in sequences:
        values = np.asarray(sequence, dtype=np.float64)
        records.append(
            {
                "length": len(values),
                "mean_power": float(values.mean()) if len(values) else np.nan,
                "std_power": float(values.std()) if len(values) else np.nan,
                "min_power": float(values.min()) if len(values) else np.nan,
                "max_power": float(values.max()) if len(values) else np.nan,
                "constant": bool(len(values) > 0 and np.ptp(values) < 1e-8),
            }
        )
    return pd.DataFrame(records)


def _parse_diagnostics(values: pd.Series) -> dict[str, int]:
    invalid_tokens = 0
    non_finite_tokens = 0
    rows_with_invalid_tokens = 0
    for value in values:
        row_has_invalid = False
        if not isinstance(value, str):
            rows_with_invalid_tokens += 1
            continue
        for token in value.replace('"', "").strip().split(","):
            token = token.strip()
            if not token:
                continue
            try:
                number = float(token)
                if not np.isfinite(number):
                    non_finite_tokens += 1
                    row_has_invalid = True
            except ValueError:
                invalid_tokens += 1
                row_has_invalid = True
        rows_with_invalid_tokens += int(row_has_invalid)
    return {
        "invalid_numeric_tokens": invalid_tokens,
        "non_finite_numeric_tokens": non_finite_tokens,
        "rows_with_invalid_tokens": rows_with_invalid_tokens,
    }


def _plot_typical_sequences(
    data: pd.DataFrame,
    sequence_column: str,
    label_column: str,
    output_path: Path,
) -> None:
    labels = sorted(data[label_column].astype(str).unique())
    figure, axes = plt.subplots(len(labels), 1, figsize=(10, 3 * len(labels)), squeeze=False)
    for axis, label in zip(axes[:, 0], labels, strict=True):
        subset = data[data[label_column].astype(str) == label].copy()
        subset["length"] = subset[sequence_column].map(len)
        median_length = float(subset["length"].median())
        selected = subset.iloc[(subset["length"] - median_length).abs().argsort()[:3]]
        for _, row in selected.iterrows():
            axis.plot(row[sequence_column], alpha=0.8)
        axis.set_title(f"Label {label}: examples near median length {median_length:.0f}")
        axis.set_xlabel("Time step")
        axis.set_ylabel("Charging power")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def analyze(config_path: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    data = pd.read_csv(config.data.csv_path, delimiter=config.data.delimiter)
    sequences = data[config.data.sequence_column].map(parse_power_sequence)
    labels = data[config.data.label_column].astype(str)
    stats = _sequence_statistics(sequences)
    stats["label"] = labels.to_numpy()
    stats["sample_id"] = data[config.data.id_column].astype(str).to_numpy()

    valid_mask = (stats["length"] >= config.data.min_length) & ~labels.isin(
        config.data.invalid_labels
    )
    valid_data = data.loc[valid_mask].copy()
    valid_data[config.data.sequence_column] = sequences[valid_mask]
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_typical_sequences(
        valid_data,
        config.data.sequence_column,
        config.data.label_column,
        output_dir / "typical_sequences.png",
    )

    per_label: dict[str, Any] = {}
    for label, group in stats.groupby("label"):
        per_label[str(label)] = {
            "length": _describe(group["length"]),
            "mean_power": _describe(group["mean_power"].dropna()),
            "std_power": _describe(group["std_power"].dropna()),
            "constant_sequences": int(group["constant"].sum()),
            "shorter_than_min_length": int((group["length"] < config.data.min_length).sum()),
        }

    duplicate_groups = data.groupby(config.data.sequence_column, dropna=False).agg(
        rows=(config.data.label_column, "size"),
        labels=(config.data.label_column, "nunique"),
    )
    valid_duplicate_groups = data.loc[valid_mask].groupby(config.data.sequence_column).agg(
        rows=(config.data.label_column, "size"),
        labels=(config.data.label_column, "nunique"),
    )
    bundle = load_dataset(config.data)
    length_model = DecisionTreeClassifier(max_depth=3, random_state=config.data.random_state)
    length_model.fit(bundle.lengths_train.reshape(-1, 1), bundle.y_train)
    class_names = [str(name) for name in bundle.label_encoder.classes_]
    length_shortcut = {
        "validation": classification_metrics(
            bundle.y_val,
            length_model.predict(bundle.lengths_val.reshape(-1, 1)),
            class_names,
            include_details=True,
        ),
        "test": classification_metrics(
            bundle.y_test,
            length_model.predict(bundle.lengths_test.reshape(-1, 1)),
            class_names,
            include_details=True,
        ),
    }

    id_unique_ratio = float(data[config.data.id_column].nunique() / len(data))
    report = {
        "dataset": {
            "path": str(config.data.csv_path),
            "sha256": sha256_file(config.data.csv_path),
            "rows": len(data),
            "columns": data.columns.tolist(),
            "missing_values": {
                str(key): int(value) for key, value in data.isna().sum().items()
            },
            "unique_id_ratio": id_unique_ratio,
        },
        "filtering": {
            "valid_rows": int(valid_mask.sum()),
            "short_rows": int((stats["length"] < config.data.min_length).sum()),
            "invalid_labels": list(config.data.invalid_labels),
            "rows_with_invalid_labels": int(labels.isin(config.data.invalid_labels).sum()),
            "rows_truncated": int((stats.loc[valid_mask, "length"] > config.data.max_length).sum()),
        },
        "quality": {
            "parsing": _parse_diagnostics(data[config.data.sequence_column]),
            "raw_constant_sequences": int(stats["constant"].sum()),
            "valid_constant_sequences": int(stats.loc[valid_mask, "constant"].sum()),
            "raw_duplicate_sequence_groups": int((duplicate_groups["rows"] > 1).sum()),
            "raw_rows_in_duplicate_sequence_groups": int(
                duplicate_groups.loc[duplicate_groups["rows"] > 1, "rows"].sum()
            ),
            "raw_conflicting_label_sequence_groups": int(
                (duplicate_groups["labels"] > 1).sum()
            ),
            "valid_duplicate_sequence_groups": int(
                (valid_duplicate_groups["rows"] > 1).sum()
            ),
            "valid_conflicting_label_sequence_groups": int(
                (valid_duplicate_groups["labels"] > 1).sum()
            ),
        },
        "per_label": per_label,
        "length_only_shortcut_baseline": length_shortcut,
        "group_split_assessment": {
            "available": id_unique_ratio < 1.0,
            "reason": (
                "The configured ID repeats and can be used as a group."
                if id_unique_ratio < 1.0
                else "Every configured ID is unique; no device/group column is available."
            ),
        },
        "time_split_assessment": {
            "available": False,
            "reason": "No timestamp column is present in the current CSV.",
        },
    }
    atomic_write_json(output_dir / "data_quality.json", report)
    print(f"Saved data quality report to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze charging-power data quality.")
    parser.add_argument("--config", default="configs/moment.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_quality"))
    args = parser.parse_args()
    analyze(Path(args.config), args.output_dir)


if __name__ == "__main__":
    main()
