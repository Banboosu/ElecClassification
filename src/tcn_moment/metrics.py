from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
    *,
    include_details: bool = False,
) -> dict[str, Any]:
    """Calculate the common metric set used by every classifier in the project."""
    labels = list(range(len(class_names)))
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            zero_division=0,
        )
        result[f"{average}_precision"] = float(precision)
        result[f"{average}_recall"] = float(recall)
        result[f"{average}_f1"] = float(f1)

    if include_details:
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        result["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[str(name) for name in class_names],
            zero_division=0,
            output_dict=True,
        )
    return result
