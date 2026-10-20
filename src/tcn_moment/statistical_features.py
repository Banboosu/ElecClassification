"""Reference 15-feature raw-scale statistics, kept importable without torch."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "length",
    "mean",
    "std",
    "minimum",
    "maximum",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "last_minus_first",
    "slope",
    "diff_mean",
    "diff_std",
    "max_abs_diff",
]


def extract_statistical_features(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for values, valid_mask in zip(x, mask, strict=True):
        valid = values[valid_mask.astype(bool)]
        differences = np.diff(valid)
        slope = float(np.polyfit(np.arange(len(valid)), valid, 1)[0]) if len(valid) > 1 else 0.0
        rows.append(
            [
                float(len(valid)),
                float(valid.mean()),
                float(valid.std()),
                float(valid.min()),
                float(valid.max()),
                *np.quantile(valid, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(float).tolist(),
                float(valid[-1] - valid[0]),
                slope,
                float(differences.mean()) if len(differences) else 0.0,
                float(differences.std()) if len(differences) else 0.0,
                float(np.abs(differences).max()) if len(differences) else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)
