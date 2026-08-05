from __future__ import annotations

import numpy as np
import pandas as pd

from tcn_moment.evaluate_error_analysis import (
    add_high_confidence_battery_flags,
    assign_length_bins,
    confusion_tables,
    curve_features,
    error_directions,
    make_length_bins,
    normalize_class_scores,
    select_typical_examples,
)


def test_normalize_class_scores_keeps_probabilities_and_softmaxes_svm_scores() -> None:
    probabilities = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    assert np.allclose(
        normalize_class_scores(probabilities, "softmax_probability"), probabilities
    )

    decisions = np.array([[1.0, 2.0, 4.0], [-2.0, 0.0, 1.0]])
    normalized = normalize_class_scores(decisions, "ovr_decision_function")
    assert np.allclose(normalized.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(normalized, axis=1), np.argmax(decisions, axis=1))


def test_length_bins_are_deterministic_and_cover_values() -> None:
    lengths = np.arange(10, 110, 10)
    edges, labels = make_length_bins(lengths, n_bins=4)
    assigned = assign_length_bins(lengths, edges, labels)

    assert len(labels) == 4
    assert not pd.isna(assigned).any()
    assert list(assigned.categories) == labels


def test_curve_features_use_only_supplied_valid_values() -> None:
    values = np.array([0.0, 1.0, 3.0, 2.0])
    result = curve_features(values)

    assert result["effective_length"] == 4
    assert result["value_range"] == 3
    assert result["mean_abs_step"] == 4 / 3
    assert result["max_abs_step"] == 2
    assert result["zero_fraction"] == 0.25


def test_high_confidence_battery_false_positive_uses_run_local_margin_quantile() -> None:
    frame = pd.DataFrame(
        {
            "model_key": ["TCN_full"] * 5,
            "seed": [42] * 5,
            "true_label": ["0", "0", "1", "2", "2"],
            "predicted_label": ["2", "2", "0", "1", "2"],
            "prediction_margin": [0.2, 0.9, 0.5, 0.8, 0.4],
        }
    )

    flagged, rows = add_high_confidence_battery_flags(frame, "2", 0.5)

    assert flagged["battery_high_confidence_prediction"].tolist() == [False, True, False, False, True]
    assert rows.loc[0, "false_negatives"] == 1
    assert rows.loc[0, "false_positives"] == 2
    assert rows.loc[0, "high_confidence_false_positives"] == 1


def test_error_directions_ranks_largest_direction_per_true_class() -> None:
    frame = pd.DataFrame(
        {
            "model_key": ["TCN_full"] * 5,
            "seed": [42] * 5,
            "true_label": ["0", "0", "0", "1", "1"],
            "predicted_label": ["1", "1", "2", "2", "1"],
            "correct": [False, False, False, False, True],
        }
    )

    per_seed, aggregate = error_directions(frame)

    assert len(per_seed) == 6
    main_zero = aggregate.loc[
        (aggregate["true_label"] == "0") & (aggregate["direction_rank"] == 1)
    ].iloc[0]
    assert main_zero["predicted_label"] == "1"
    assert main_zero["total_error_count"] == 2


def test_confusion_tables_include_zero_cells_and_row_fractions() -> None:
    frame = pd.DataFrame(
        {
            "model_key": ["TCN_full"] * 3,
            "seed": [42] * 3,
            "true_label": ["0", "0", "1"],
            "predicted_label": ["0", "1", "1"],
        }
    )

    per_seed, aggregate = confusion_tables(frame, ["0", "1", "2"])

    assert len(per_seed) == 9
    zero_cell = per_seed.loc[
        (per_seed["true_label"] == "2") & (per_seed["predicted_label"] == "0")
    ].iloc[0]
    assert zero_cell["count"] == 0
    assert zero_cell["row_fraction"] == 0
    class_zero_correct = aggregate.loc[
        (aggregate["true_label"] == "0") & (aggregate["predicted_label"] == "0")
    ].iloc[0]
    assert class_zero_correct["mean_row_fraction"] == 0.5


def test_typical_example_rules_are_ordered_and_do_not_reuse_samples() -> None:
    sample_ids = [f"s{index}" for index in range(1, 7)]
    true_labels = ["2", "0", "0", "0", "0", "0"]
    predictions_by_model = {
        "TCN_full": ["0", "2", "1", "1", "0", "0"],
        "MOMENT_full": ["1", "0", "2", "0", "1", "0"],
        "MOMENT_SVM_10pct": ["2", "0", "0", "0", "0", "1"],
    }
    frames = []
    for model_key, predicted_labels in predictions_by_model.items():
        frames.append(
            pd.DataFrame(
                {
                    "model_key": model_key,
                    "seed": 42,
                    "split": "test",
                    "sample_id": sample_ids,
                    "true_label": true_labels,
                    "predicted_label": predicted_labels,
                    "correct": [
                        true == predicted
                        for true, predicted in zip(true_labels, predicted_labels, strict=True)
                    ],
                    "prediction_margin": np.linspace(0.1, 0.6, 6),
                    "battery_high_confidence_prediction": [
                        model_key == "TCN_full" and sample_id == "s2"
                        for sample_id in sample_ids
                    ],
                }
            )
        )

    selected = select_typical_examples(pd.concat(frames, ignore_index=True), 42, "2")

    assert selected["selection_rule"].tolist() == [
        "battery_false_negative",
        "battery_high_confidence_false_positive",
        "shared_tcn_moment_error",
        "tcn_only_error",
        "moment_only_error",
        "svm10_only_error",
    ]
    assert selected["sample_id"].is_unique
