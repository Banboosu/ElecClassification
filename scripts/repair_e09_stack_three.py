"""Recompute E09 stack_three with corrected 9-column OOF layout (validation only).

Historical stack_three wrote the moment and diff blocks at 3:6 and 4:7, so the
diff block overwrote part of the moment block and columns 7:9 stayed zero.  This
repair keeps every other E09 condition untouched and writes a new suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tcn_moment import run_e08
from tcn_moment import run_e09_diff_stream as e09
from tcn_moment.data import stratified_train_subset_indices
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.label_semantics import BATTERY_LABEL, class_index


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(protocol_path: Path) -> None:
    e09.freeze(protocol_path)
    p = e09._load(protocol_path)
    root = e09._root(p)
    if (root / "pilot_summary.json").exists():
        print(f"Already complete: {root}", flush=True)
        return
    inputs = e09._load_inputs(protocol_path)
    raw = inputs["raw"]
    class_names = [str(value) for value in raw.label_encoder.classes_]
    battery_index = class_index(class_names, BATTERY_LABEL)
    hgb_params_by_tag = e09._hgb_params(protocol_path)
    seed = int(p["seed"])
    conditions: list[dict[str, Any]] = []
    for fraction in p["fractions"]:
        tag = run_e08._tag(fraction)
        indices = stratified_train_subset_indices(
            raw.y_train, raw.ids_train, fraction, seed
        )
        y_train = raw.y_train[indices]
        y_val = raw.y_val
        prediction, scores, details = e09._stack_three(
            inputs["dyn_train"][indices],
            y_train,
            inputs["dyn_val"],
            inputs["moment_train"][indices],
            inputs["moment_val"],
            inputs["diff_train"][indices],
            inputs["diff_val"],
            hgb_params=hgb_params_by_tag[tag],
            seed=seed,
        )
        metrics = run_e08._evaluate(y_val, prediction, scores, class_names, battery_index)
        prediction_path = run_e08._write_condition(
            root,
            tag=tag,
            method="stack_three",
            ids=raw.ids_val,
            y_true=y_val,
            y_pred=prediction,
            scores=scores,
            battery_index=battery_index,
        )
        record = {
            "seed": seed,
            "tag": tag,
            "fraction": fraction,
            "method": "stack_three",
            "n_train": len(indices),
            "selected_params": details["meta_params"],
            "cv_results": details["meta_cv"],
            "validation": metrics,
            "predictions": prediction_path,
            "repair": "oof blocks 3:6 and 6:9; previous 3:6/4:7 overlap",
        }
        conditions.append(record)
        atomic_write_json(root / "conditions" / tag / "stack_three" / "metrics.json", record)
        print(
            f"E09R stack_three {tag}: macro-F1={100*metrics['macro_f1']:.4f} "
            f"battery_recall={100*metrics['battery']['recall']:.2f} "
            f"battery_fpr={100*metrics['battery']['false_positive_rate']:.2f}",
            flush=True,
        )
    summary = {
        "seed": seed,
        "fractions": list(p["fractions"]),
        "conditions": conditions,
        "test_evaluated": False,
        "repair": "E09 stack_three OOF column layout",
    }
    atomic_write_json(root / "pilot_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 44, 45, 46])
    args = parser.parse_args()
    for seed in args.seeds:
        path = Path(f"configs/experiments/improvements/e09_stack_three_fix_seed{seed}_protocol.json")
        run(path)


if __name__ == "__main__":
    main()
