from __future__ import annotations

import math

import numpy as np

from tcn_moment.audit_protocol import (
    compare_saved_metrics,
    metrics_from_confusion,
    nonfinite_paths,
    normalized_config,
)


def test_metrics_from_confusion_matches_expected_multiclass_values() -> None:
    confusion = [[8, 2, 0], [1, 7, 2], [0, 1, 9]]
    result = metrics_from_confusion(confusion)

    assert result["accuracy"] == 0.8
    assert np.isclose(result["balanced_accuracy"], 0.8)
    assert np.isclose(result["macro_recall"], 0.8)
    assert 0 < result["macro_f1"] < 1


def test_compare_saved_metrics_accepts_metrics_derived_from_same_confusion() -> None:
    confusion = [[4, 1], [2, 3]]
    payload = {"confusion_matrix": confusion, **metrics_from_confusion(confusion)}

    assert compare_saved_metrics(payload) == 0


def test_nonfinite_paths_reports_nested_nan_and_infinity() -> None:
    value = {"history": [{"loss": 0.2}, {"loss": math.nan}], "score": math.inf}

    assert nonfinite_paths(value) == ["root.history[1].loss", "root.score"]


def test_normalized_config_removes_only_expected_seed_specific_data_fields() -> None:
    config = {
        "data": {
            "random_state": 42,
            "split_path": "artifacts/splits/unified_split_seed42.json",
            "normalize": "zscore",
        },
        "training": {"amp": True},
        "svm": {"max_samples": 10_000},
    }

    result = normalized_config(config, ignored_top_level_sections=("svm",))

    assert result == {"data": {"normalize": "zscore"}, "training": {"amp": True}}
    assert config["data"]["random_state"] == 42
    assert "svm" in config
