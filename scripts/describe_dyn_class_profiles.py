"""Per-class descriptive profiles of selected DYN features on the training split.

Supports the discussion that relates the dynamics features to known battery and
charger fault signatures (manuscript_iop_en.tex, section "Correspondence with
known fault signatures"). Descriptive only: no model is fitted, the validation
and test splits are not read, and the features are the pre-defined 60-D DYN
features of ``tcn_moment.dynamics_features``.

Run from the project root:
    PYTHONPATH=src python scripts/describe_dyn_class_profiles.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from tcn_moment.dynamics_features import (
    DYNAMICS_FEATURE_NAMES,
    FEATURE_GROUPS,
    extract_dynamics_features,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "artifacts/splits/unified_split.json"
DATA = ROOT / "data/raw/最新多.csv"
OUT = ROOT / "artifacts/analysis/dyn_class_profiles_20261105"
FEATURES = (
    "direction_turn_rate",
    "diff_abs_over_mean",
    "diff_std_over_mean",
    "diff_autocorrelation",
    "increasing_fraction",
    "plateau_fraction",
    "slope_tail",
    "max_drawdown",
    "spectral_band4",
    "length",
)
CLASSES = {0: "normal", 1: "battery_anomaly", 2: "charger_fault"}


def load_training_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_ids = set(split["splits"]["train"])
    csv.field_size_limit(10**8)
    sequences, labels = [], []
    with DATA.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=" "):
            if row["charging_station_id"] not in train_ids:
                continue
            values = [
                float(v)
                for v in row["charging_powers_str"].replace('"', "").split(",")
                if v.strip()
            ]
            sequences.append(values)
            labels.append(int(str(row["InsertedColumn"]).strip()))
    if len(sequences) != len(train_ids):
        raise ValueError("Training split and raw data are out of sync")
    x = np.zeros((len(sequences), 1024))
    mask = np.zeros_like(x, dtype=bool)
    for i, values in enumerate(sequences):
        x[i, : len(values)] = values
        mask[i, : len(values)] = True
    return x, mask, np.asarray(labels)


def group_of(feature: str) -> str:
    return next(group for group, names in FEATURE_GROUPS.items() if feature in names)


def main() -> None:
    x, mask, y = load_training_split()
    features = extract_dynamics_features(x, mask)
    rows = []
    for name in FEATURES:
        values = features[:, DYNAMICS_FEATURE_NAMES.index(name)].astype(np.float64)
        row = {"feature": name, "group": group_of(name)}
        for label, cls in CLASSES.items():
            v = values[y == label]
            row[cls] = {
                "median": float(np.median(v)),
                "q25": float(np.quantile(v, 0.25)),
                "q75": float(np.quantile(v, 0.75)),
            }
        # One-vs-rest AUC of the raw feature; values below 0.5 mean lower values in that class.
        row["auc_battery_vs_rest"] = float(roc_auc_score(y == 1, values))
        row["auc_charger_vs_rest"] = float(roc_auc_score(y == 2, values))
        rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": "unified_split.json train",
        "n": len(y),
        "class_counts": {cls: int((y == label).sum()) for label, cls in CLASSES.items()},
        "test_evaluated": False,
        "validation_used": False,
        "features": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [
        "# DYN feature profiles by class (training split, descriptive)",
        "",
        f"n = {len(y)}; {summary['class_counts']}. Median [IQR]; AUC is one-vs-rest on the raw feature.",
        "",
        "| Feature | Group | Normal | Battery anomaly | Charger fault | AUC battery | AUC charger |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        cells = [
            f"{row[c]['median']:.4g} [{row[c]['q25']:.3g}, {row[c]['q75']:.3g}]"
            for c in CLASSES.values()
        ]
        lines.append(
            f"| {row['feature']} | {row['group']} | "
            + " | ".join(cells)
            + f" | {row['auc_battery_vs_rest']:.3f} | {row['auc_charger_vs_rest']:.3f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
