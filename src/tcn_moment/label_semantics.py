"""User-corrected class semantics; never permute the stored numeric labels.

Historical outputs called raw class 2 ``battery``. Those values describe charger
faults, not battery anomalies. Frozen historical protocols are not rewritten.
"""
from __future__ import annotations

from collections.abc import Sequence

SEMANTICS_VERSION = "v2_20261029_user_correction"
CLASS_LABELS = ("0", "1", "2")
LABEL_NAMES_ZH = {"0": "正常", "1": "电池异常", "2": "充电器故障"}
BATTERY_LABEL = "1"
CHARGER_LABEL = "2"


def class_index(class_names: Sequence[object], raw_label: str) -> int:
    """Resolve a raw label to its score column; do not assume encoder order."""
    names = [str(value) for value in class_names]
    if len(set(names)) != len(names) or raw_label not in names:
        raise ValueError(f"Cannot resolve raw label {raw_label!r} in {names!r}")
    return names.index(raw_label)
