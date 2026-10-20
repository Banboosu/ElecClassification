"""E08: compact raw-scale dynamics features for charging-power curves.

The existing E01 statistical branch uses 15 global features.  This module keeps those
15 features unchanged (so the reference pipeline stays comparable) and appends 45
local-dynamics, spectral, and shape descriptors.  All descriptors are computed from
the *valid* observations only; padding must never influence them.

The feature list, order, and grouping are frozen by
``configs/experiments/improvements/e08_strong_features_protocol.json``.  Do not
reorder or rename entries after a protocol has been frozen.
"""

from __future__ import annotations

import numpy as np

from tcn_moment.statistical_features import FEATURE_NAMES, extract_statistical_features

BASE_FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_NAMES)

EXTRA_FEATURE_NAMES: tuple[str, ...] = (
    "length_over_max",
    "coefficient_of_variation",
    "iqr_over_sum",
    "last_over_first",
    "last_over_mean",
    "max_over_mean",
    "min_over_mean",
    "q05",
    "q95",
    "slope_head",
    "slope_tail",
    "trend_total_change",
    "tail_trend_change",
    "head_mean",
    "tail_mean",
    "head_std",
    "tail_std",
    "tail_minus_head",
    "third1_mean",
    "third2_mean",
    "third3_mean",
    "third1_std",
    "third2_std",
    "third3_std",
    "third3_minus_third1",
    "diff_abs_mean",
    "diff_abs_q90",
    "diff_abs_over_mean",
    "diff_std_over_mean",
    "increasing_fraction",
    "direction_turn_rate",
    "plateau_fraction",
    "longest_plateau_over_length",
    "diff_autocorrelation",
    "spectral_band1",
    "spectral_band2",
    "spectral_band3",
    "spectral_band4",
    "dominant_frequency",
    "level_autocorrelation",
    "histogram_entropy",
    "max_drawdown",
    "max_runup",
    "skewness",
    "kurtosis",
)

DYNAMICS_FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + EXTRA_FEATURE_NAMES

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "length": ("length", "length_over_max"),
    "amplitude": (
        "mean",
        "std",
        "minimum",
        "maximum",
        "q05",
        "q10",
        "q25",
        "median",
        "q75",
        "q90",
        "q95",
        "coefficient_of_variation",
        "iqr_over_sum",
        "last_over_first",
        "last_over_mean",
        "max_over_mean",
        "min_over_mean",
    ),
    "trend": (
        "slope",
        "last_minus_first",
        "slope_head",
        "slope_tail",
        "trend_total_change",
        "tail_trend_change",
        "head_mean",
        "tail_mean",
        "head_std",
        "tail_std",
        "tail_minus_head",
        "third1_mean",
        "third2_mean",
        "third3_mean",
        "third1_std",
        "third2_std",
        "third3_std",
        "third3_minus_third1",
    ),
    "roughness": (
        "diff_mean",
        "diff_std",
        "max_abs_diff",
        "diff_abs_mean",
        "diff_abs_q90",
        "diff_abs_over_mean",
        "diff_std_over_mean",
        "increasing_fraction",
        "direction_turn_rate",
        "plateau_fraction",
        "longest_plateau_over_length",
        "diff_autocorrelation",
    ),
    "spectral": (
        "spectral_band1",
        "spectral_band2",
        "spectral_band3",
        "spectral_band4",
        "dominant_frequency",
        "level_autocorrelation",
        "histogram_entropy",
    ),
    "shape": ("max_drawdown", "max_runup", "skewness", "kurtosis"),
}

BASE_FEATURE_COUNT = len(BASE_FEATURE_NAMES)
DYNAMICS_FEATURE_COUNT = len(DYNAMICS_FEATURE_NAMES)
PLATEAU_EPSILON = 1e-3


def _validate_contract() -> None:
    if len(set(DYNAMICS_FEATURE_NAMES)) != DYNAMICS_FEATURE_COUNT:
        raise RuntimeError("Dynamics feature names must be unique.")
    grouped = [name for names in FEATURE_GROUPS.values() for name in names]
    if len(grouped) != len(set(grouped)):
        raise RuntimeError("Dynamics feature groups must not overlap.")
    if set(grouped) != set(DYNAMICS_FEATURE_NAMES):
        missing = sorted(set(DYNAMICS_FEATURE_NAMES) - set(grouped))
        extra = sorted(set(grouped) - set(DYNAMICS_FEATURE_NAMES))
        raise RuntimeError(f"Feature groups do not partition the feature list: {missing=} {extra=}")


_validate_contract()


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=np.float64), values, 1)[0])


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or left.std() == 0.0 or right.std() == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _extra_features(valid: np.ndarray, *, max_length: int) -> list[float]:
    n = len(valid)
    if n < 2:
        raise ValueError("Dynamics features require at least two valid observations.")
    values = np.asarray(valid, dtype=np.float64)
    differences = np.diff(values)
    absolute_differences = np.abs(differences)
    mean = float(values.mean())
    scale = float(values.std())
    denominator = mean if abs(mean) > 1e-12 else 1e-12

    q05, q10, q25, q50, q75, q90, q95 = np.quantile(
        values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )
    tail_size = max(n // 10, 2)
    head = values[:tail_size]
    tail = values[-tail_size:]
    slope_head = _slope(head)
    slope_tail = _slope(tail)
    slope = _slope(values)

    third = max(n // 3, 1)
    parts = (values[:third], values[third : 2 * third], values[2 * third :])
    part_means = [float(part.mean()) if len(part) else 0.0 for part in parts]
    part_stds = [float(part.std()) if len(part) else 0.0 for part in parts]

    signs = np.sign(differences)
    signs = signs[signs != 0.0]
    turn_rate = float(np.mean(signs[1:] != signs[:-1])) if len(signs) > 1 else 0.0
    increasing_fraction = float(np.mean(differences > 0.0)) if len(differences) else 0.0
    plateau_mask = absolute_differences < PLATEAU_EPSILON
    plateau_fraction = float(np.mean(plateau_mask)) if len(plateau_mask) else 0.0
    longest = current = 1
    for index in range(1, n):
        current = current + 1 if abs(values[index] - values[index - 1]) < PLATEAU_EPSILON else 1
        longest = max(longest, current)

    centered = values - mean
    spectrum = np.abs(np.fft.rfft(centered)) ** 2 if n > 8 else np.zeros(2)
    non_dc = spectrum[1:] if len(spectrum) > 1 else spectrum
    total_power = float(non_dc.sum())
    if total_power > 0.0:
        bands = [float(part.sum()) / total_power for part in np.array_split(non_dc, 4)]
        dominant = float(int(np.argmax(non_dc)) + 1) / n
    else:
        bands = [0.0, 0.0, 0.0, 0.0]
        dominant = 0.0
    level_autocorrelation = _safe_correlation(values[:-1], values[1:])
    diff_autocorrelation = _safe_correlation(differences[:-1], differences[1:])

    histogram, _ = np.histogram(values, bins=10)
    probabilities = histogram / max(int(histogram.sum()), 1)
    positive = probabilities[probabilities > 0.0]
    entropy = float(-(positive * np.log(positive)).sum()) if len(positive) else 0.0

    running_max = np.maximum.accumulate(values)
    running_min = np.minimum.accumulate(values)
    drawdown = float(np.max((running_max - values) / (np.abs(running_max) + 1e-12)))
    runup = float(np.max((values - running_min) / (np.abs(running_min) + 1e-12)))

    if scale > 1e-12:
        standardized = centered / scale
        skewness = float(np.mean(standardized**3))
        kurtosis = float(np.mean(standardized**4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0

    return [
        n / max_length,
        scale / abs(denominator),
        (q75 - q25) / (abs(q75) + abs(q25) + 1e-12),
        values[-1] / (values[0] if abs(values[0]) > 1e-12 else 1e-12),
        values[-1] / denominator,
        values.max() / denominator,
        values.min() / denominator,
        float(q05),
        float(q95),
        slope_head,
        slope_tail,
        slope * (n - 1),
        slope_tail * (len(tail) - 1),
        float(head.mean()),
        float(tail.mean()),
        float(head.std()),
        float(tail.std()),
        float(tail.mean() - head.mean()),
        part_means[0],
        part_means[1],
        part_means[2],
        part_stds[0],
        part_stds[1],
        part_stds[2],
        part_means[2] - part_means[0],
        float(absolute_differences.mean()),
        float(np.quantile(absolute_differences, 0.90)),
        float(absolute_differences.mean()) / abs(denominator),
        float(differences.std()) / abs(denominator),
        increasing_fraction,
        turn_rate,
        plateau_fraction,
        longest / n,
        diff_autocorrelation,
        bands[0],
        bands[1],
        bands[2],
        bands[3],
        dominant,
        level_autocorrelation,
        entropy,
        drawdown,
        runup,
        skewness,
        kurtosis,
    ]


def extract_dynamics_features(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    max_length: int = 1024,
) -> np.ndarray:
    """Return ``[n, 60]`` float32 dynamics features computed on valid observations only."""
    if x.ndim != 2 or mask.shape != x.shape:
        raise ValueError("x and mask must be two-dimensional arrays with the same shape.")
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    base = extract_statistical_features(x, mask)
    rows: list[list[float]] = []
    for values, valid_mask in zip(x, mask, strict=True):
        valid = np.asarray(values, dtype=np.float64)[np.asarray(valid_mask, dtype=bool)]
        rows.append(_extra_features(valid, max_length=max_length))
    result = np.hstack([base, np.asarray(rows, dtype=np.float64)]).astype(np.float32)
    if result.shape[1] != DYNAMICS_FEATURE_COUNT:
        raise RuntimeError("Dynamics feature count changed without updating the contract.")
    if not np.isfinite(result).all():
        raise RuntimeError("Dynamics features contain non-finite values.")
    return result


def select_feature_set(
    features: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    """Return the reference ``f15`` slice or the complete ``dyn`` matrix."""
    if feature_set == "f15":
        return features[:, :BASE_FEATURE_COUNT]
    if feature_set == "dyn":
        return features
    raise ValueError(f"Unknown feature set: {feature_set}")


def group_indices(group: str) -> list[int]:
    """Return column indices for one frozen feature group."""
    try:
        names = FEATURE_GROUPS[group]
    except KeyError as error:
        raise ValueError(f"Unknown feature group: {group}") from error
    index = {name: position for position, name in enumerate(DYNAMICS_FEATURE_NAMES)}
    return [index[name] for name in names]
