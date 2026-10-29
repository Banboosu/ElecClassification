"""E09 pilot: frozen MOMENT on the original series and on first differences.

The E08 result shows that raw-scale roughness features carry most of the gain.
This pilot tests whether the frozen MOMENT representation can be made more useful
by adding a derivative stream, without touching the backbone or the test split.

Pilot scope: seed42, 10%/40% labels, validation only.  A positive result is a
gate to a four-seed expansion, not a final claim.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tcn_moment import run_e08
from tcn_moment.config import load_config, with_random_seed
from sklearn.metrics import f1_score

from tcn_moment.data import load_dataset
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.label_semantics import BATTERY_LABEL, class_index
from tcn_moment.train_feature_fusion import frozen_features

DEFAULT_PROTOCOL = Path("configs/experiments/improvements/e09_diff_stream_pilot_protocol.json")
E01_CACHE = Path("artifacts/improvements/e01_20261017_v2_features")
E08_SUITE = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
PILOT_SEED = 42
FRACTIONS = (0.1, 0.4)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root(p: dict[str, Any]) -> Path:
    return Path(p["output_dir"]) / p["suite"]


def _validate(p: dict[str, Any]) -> None:
    if p.get("experiment_id") not in {"E09P", "E09"}:
        raise ValueError("Unexpected experiment id")
    if p.get("seed") not in {42, 44, 45, 46}:
        raise ValueError("E09 supports seeds 42/44/45/46")
    if p.get("fractions") not in ([0.1, 0.4], [0.01, 0.1, 0.4]):
        raise ValueError("E09 supports 10/40% or 1/10/40%")
    if p.get("test_evaluated") or p.get("automatic_expansion"):
        raise ValueError("E09 is validation-only and does not auto-expand")


def _load(path: Path) -> dict[str, Any]:
    p = _read(path)
    _validate(p)
    return p


def freeze(protocol_path: Path) -> None:
    p = _load(protocol_path)
    root = _root(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        return
    sources = [
        Path(__file__).resolve(),
        Path("src/tcn_moment/run_e08.py"),
        Path("src/tcn_moment/label_semantics.py"),
        Path("src/tcn_moment/dynamics_features.py"),
        Path("src/tcn_moment/statistical_features.py"),
        Path("src/tcn_moment/train_feature_fusion.py"),
        Path("src/tcn_moment/train_moment.py"),
        Path("src/tcn_moment/feature_transforms.py"),
        Path("src/tcn_moment/data.py"),
        Path("src/tcn_moment/config.py"),
        Path("src/tcn_moment/metrics.py"),
        Path("src/tcn_moment/experiment.py"),
        Path("src/tcn_moment/io_utils.py"),
        Path("tests/test_e09_diff_stream.py"),
        Path("scripts/run_e09_pilot_remote.sh"),
        protocol_path,
        E08_SUITE / "seed42" / "seed_results.json",
        E08_SUITE / "seed42" / "features.npz",
        E01_CACHE / "seed42" / "train.npz",
        E01_CACHE / "seed42" / "validation.npz",
        E01_CACHE / "seed42" / "manifest.json",
    ]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Missing frozen input: {source}")
    frozen = {
        "frozen_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "protocol": p,
        "protocol_sha256": sha256_file(protocol_path),
        "file_sha256": {str(source): sha256_file(source) for source in sources},
        "test_evaluated": False,
    }
    for source in sources:
        if source.parts[0] in {"src", "configs", "tests", "scripts", "docs"}:
            destination = root / "source_snapshot" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    atomic_write_json(root / "frozen_protocol.json", frozen)
    print(f"Froze E09P: {root / 'frozen_protocol.json'}", flush=True)


def _diff_arrays(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-series first difference on valid observations, z-scored, then padded."""
    output = np.zeros_like(x, dtype=np.float32)
    output_mask = np.zeros_like(mask, dtype=bool)
    for index in range(len(x)):
        values = np.asarray(x[index][mask[index].astype(bool)], dtype=np.float64)
        differences = np.diff(values)
        if len(differences) < 2:
            raise ValueError("A valid sequence is too short for a difference stream.")
        deviations = differences.std()
        if deviations > 0:
            differences = (differences - differences.mean()) / deviations
        length = len(differences)
        output[index, :length] = differences
        output_mask[index, :length] = True
    return output, output_mask


def prepare(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    root = _root(p)
    seed = int(p["seed"])
    freeze(protocol_path)
    config = with_random_seed(load_config(Path(p["config"])), seed)
    raw = load_dataset(replace(config.data, normalize="none"))
    if set(raw.ids_test) & (set(raw.ids_train) | set(raw.ids_val)):
        raise RuntimeError("Split overlap")
    diff_train, diff_train_mask = _diff_arrays(raw.x_train, raw.mask_train)
    diff_val, diff_val_mask = _diff_arrays(raw.x_val, raw.mask_val)
    bundle = replace(
        raw,
        x_train=diff_train,
        mask_train=diff_train_mask,
        x_val=diff_val,
        mask_val=diff_val_mask,
        lengths_train=diff_train_mask.sum(1).astype(np.int64),
        lengths_val=diff_val_mask.sum(1).astype(np.int64),
    )
    diff_config = replace(config, data=replace(config.data, normalize="diff"))
    cache_dir = Path(p["diff_cache_dir"]) / f"seed{seed}"
    features, record = frozen_features(
        diff_config,
        bundle,
        cache_dir,
        evaluate_test=False,
    )
    atomic_write_json(
        root / f"diff_extraction_seed{seed}.json",
        {
            "cache_dir": str(cache_dir),
            "record": record,
            "train_shape": list(features["train"].shape),
            "validation_shape": list(features["validation"].shape),
            "stream": "z-scored first difference on valid observations",
            "test_evaluated": False,
        },
    )
    print(f"Diff features ready: {features['train'].shape}", flush=True)
    return {"config": config, "raw": raw, "diff": features}


def _load_inputs(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    seed = int(p["seed"])
    raw = load_dataset(replace(with_random_seed(load_config(Path(p["config"])), seed).data, normalize="none"))
    e08 = np.load(E08_SUITE / f"seed{seed}" / "features.npz", allow_pickle=False)
    dyn_train = e08["train_features"]
    dyn_val = e08["validation_features"]
    if not np.array_equal(e08["train_ids"].astype(str), raw.ids_train.astype(str)):
        raise ValueError("E08 train IDs do not match")
    if not np.array_equal(e08["validation_ids"].astype(str), raw.ids_val.astype(str)):
        raise ValueError("E08 validation IDs do not match")
    moment_train, moment_val = (
        run_e08._load_moment_split(E01_CACHE, seed=seed, split="train", ids=raw.ids_train)[0],
        run_e08._load_moment_split(E01_CACHE, seed=seed, split="validation", ids=raw.ids_val)[0],
    )
    diff_train, diff_val = (
        run_e08._load_moment_split(
            Path(p["diff_cache_dir"]), seed=seed, split="train", ids=raw.ids_train
        )[0],
        run_e08._load_moment_split(
            Path(p["diff_cache_dir"]), seed=seed, split="validation", ids=raw.ids_val
        )[0],
    )
    return {
        "raw": raw,
        "dyn_train": dyn_train,
        "dyn_val": dyn_val,
        "moment_train": moment_train,
        "moment_val": moment_val,
        "diff_train": diff_train,
        "diff_val": diff_val,
    }


def _hgb_params(protocol_path: Path) -> dict[str, dict[str, Any]]:
    p = _load(protocol_path)
    seed = int(p["seed"])
    seed_results = _read(E08_SUITE / f"seed{seed}" / "seed_results.json")
    result = {
        record["tag"]: record["selected_params"]
        for record in seed_results["conditions"]
        if record["method"] == "dyn_hgb"
    }
    required = {run_e08._tag(fraction) for fraction in p["fractions"]}
    missing = required - set(result)
    if missing:
        raise ValueError(f"E08 dyn_hgb selected params missing: {sorted(missing)}")
    return result


def _stack_three(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    moment_train: np.ndarray,
    moment_val: np.ndarray,
    diff_train: np.ndarray,
    diff_val: np.ndarray,
    *,
    hgb_params: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    folds = 5
    oof = np.zeros((len(y_train), 9), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, valid_index in splitter.split(x_train, y_train):
        hgb = run_e08._hgb_factory(hgb_params, seed).fit(x_train[train_index], y_train[train_index])
        oof[valid_index, :3] = hgb.predict_proba(x_train[valid_index])
        for stream_index, (train_matrix, valid_matrix) in enumerate(
            ((moment_train, moment_val), (diff_train, diff_val))
        ):
            offset = 3 + 3 * stream_index
            model = run_e08._logit_factory({"C": 1.0}, seed).fit(
                train_matrix[train_index], y_train[train_index]
            )
            oof[valid_index, offset : offset + 3] = model.predict_proba(train_matrix[valid_index])
    meta_grid = [{"C": value} for value in (0.01, 0.1, 1.0)]
    meta_params, meta_cv = run_e08._cv_select(
        lambda params: LogisticRegression(C=params["C"], max_iter=5000, random_state=seed),
        meta_grid,
        oof,
        y_train,
        folds=3,
        seed=seed,
    )
    meta = LogisticRegression(C=meta_params["C"], max_iter=5000, random_state=seed).fit(oof, y_train)
    hgb = run_e08._hgb_factory(hgb_params, seed).fit(x_train, y_train)
    moment_model = run_e08._logit_factory({"C": 1.0}, seed).fit(moment_train, y_train)
    diff_model = run_e08._logit_factory({"C": 1.0}, seed).fit(diff_train, y_train)
    stacked = np.hstack(
        [
            hgb.predict_proba(x_val),
            moment_model.predict_proba(moment_val),
            diff_model.predict_proba(diff_val),
        ]
    )
    return meta.predict(stacked), meta.predict_proba(stacked), {
        "hgb_params": hgb_params,
        "meta_params": meta_params,
        "meta_cv": meta_cv,
        "oof_macro_f1": float(f1_score(y_train, meta.predict(oof), average="macro")),
    }


def run(protocol_path: Path) -> None:
    p = _load(protocol_path)
    root = _root(p)
    freeze(protocol_path)
    inputs = _load_inputs(protocol_path)
    raw = inputs["raw"]
    class_names = [str(value) for value in raw.label_encoder.classes_]
    battery_index = class_index(class_names, BATTERY_LABEL)
    hgb_params_by_tag = _hgb_params(protocol_path)
    seed = int(p["seed"])
    conditions = []
    for fraction in p["fractions"]:
        tag = run_e08._tag(fraction)
        indices = __import__("tcn_moment.data", fromlist=["stratified_train_subset_indices"]).stratified_train_subset_indices(
            raw.y_train, raw.ids_train, fraction, seed
        )
        hgb_params = hgb_params_by_tag[tag]
        x_train = inputs["dyn_train"][indices]
        y_train = raw.y_train[indices]
        x_val = inputs["dyn_val"]
        y_val = raw.y_val
        moment_train = inputs["moment_train"][indices]
        moment_val = inputs["moment_val"]
        diff_train = inputs["diff_train"][indices]
        diff_val = inputs["diff_val"]
        moment_diff_train = np.hstack([moment_train, diff_train])
        moment_diff_val = np.hstack([moment_val, diff_val])

        definitions: list[tuple[str, Any]] = [
            (
                "moment_logit",
                lambda: run_e08._run_logit(
                    moment_train, y_train, moment_val, c_values=[0.1, 1.0], folds=3, seed=seed
                ),
            ),
            (
                "diff_logit",
                lambda: run_e08._run_logit(
                    diff_train, y_train, diff_val, c_values=[0.1, 1.0], folds=3, seed=seed
                ),
            ),
            (
                "moment_diff_logit",
                lambda: run_e08._run_logit(
                    moment_diff_train,
                    y_train,
                    moment_diff_val,
                    c_values=[0.1, 1.0],
                    folds=3,
                    seed=seed,
                ),
            ),
            (
                "stack_moment",
                lambda: run_e08._run_stack(
                    x_train,
                    y_train,
                    x_val,
                    moment_train,
                    moment_val,
                    hgb_params=hgb_params,
                    folds=5,
                    moment_c=1.0,
                    meta_c_values=[0.01, 0.1, 1.0],
                    seed=seed,
                ),
            ),
            (
                "stack_moment_diff",
                lambda: run_e08._run_stack(
                    x_train,
                    y_train,
                    x_val,
                    moment_diff_train,
                    moment_diff_val,
                    hgb_params=hgb_params,
                    folds=5,
                    moment_c=1.0,
                    meta_c_values=[0.01, 0.1, 1.0],
                    seed=seed,
                ),
            ),
            (
                "stack_three",
                lambda: _stack_three(
                    x_train,
                    y_train,
                    x_val,
                    moment_train,
                    moment_val,
                    diff_train,
                    diff_val,
                    hgb_params=hgb_params,
                    seed=seed,
                ),
            ),
        ]
        for method, factory in definitions:
            existing = root / "conditions" / tag / method / "metrics.json"
            if existing.exists():
                conditions.append(_read(existing))
                continue
            result = factory()
            if method == "stack_three":
                prediction, scores, details = result
                selected_params, cv_results, seconds = details, details["meta_cv"], None
            elif method.startswith("stack_"):
                prediction, scores, details, meta_cv, seconds = result
                selected_params, cv_results = details, meta_cv
            else:
                _, prediction, scores, selected_params, cv_results = result
                seconds = None
            metrics = run_e08._evaluate(y_val, prediction, scores, class_names, battery_index)
            prediction_path = run_e08._write_condition(
                root,
                tag=tag,
                method=method,
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
                "method": method,
                "n_train": len(indices),
                "selected_params": selected_params,
                "cv_results": cv_results,
                "validation": metrics,
                "predictions": prediction_path,
                "seconds": seconds,
            }
            conditions.append(record)
            atomic_write_json(root / "conditions" / tag / method / "metrics.json", record)
            print(
                f"E09P {method} {tag}: macro-F1={100*metrics['macro_f1']:.4f} "
                f"battery_recall={100*metrics['battery']['recall']:.2f} "
                f"battery_fpr={100*metrics['battery']['false_positive_rate']:.2f}",
                flush=True,
            )
    summary = {
        "seed": seed,
        "fractions": list(p["fractions"]),
        "conditions": conditions,
        "test_evaluated": False,
        "gate": _gate(conditions),
    }
    atomic_write_json(root / "pilot_summary.json", summary)
    if p.get("write_report", True):
        _report(root, summary)


def _gate(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["method"], row["tag"]): row["validation"]["macro_f1"] for row in conditions}
    checks = {}
    for tag in ("10p", "40p"):
        checks[tag] = {
            "moment_diff_minus_moment_pp": 100 * (lookup[("moment_diff_logit", tag)] - lookup[("moment_logit", tag)]),
            "stack_diff_minus_stack_pp": 100
            * (lookup[("stack_moment_diff", tag)] - lookup[("stack_moment", tag)]),
            "stack_three_minus_stack_pp": 100
            * (lookup[("stack_three", tag)] - lookup[("stack_moment", tag)]),
        }
    passed = (
        checks["40p"]["moment_diff_minus_moment_pp"] >= 1.0
        and checks["40p"]["stack_diff_minus_stack_pp"] >= 0.2
    )
    return {"checks": checks, "passed": passed, "rule": "40p: MOM+DIFF - MOM >= 1.0pp and stack +0.2pp"}


def _report(root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# E09 pilot：冻结 MOMENT 原序列 vs 一阶差分双流（validation-only）",
        "",
        "seed42、10%/40% 标签；只改变 MOMENT 输入流，不训练 backbone，不打开 test。",
        "",
        "| 方法 | 10% Macro-F1 | 40% Macro-F1 | 40% 电池 recall/FPR |",
        "|---|---:|---:|---:|",
    ]
    lookup = {(row["method"], row["tag"]): row for row in summary["conditions"]}
    for method in (
        "moment_logit",
        "diff_logit",
        "moment_diff_logit",
        "stack_moment",
        "stack_moment_diff",
        "stack_three",
    ):
        ten = lookup.get((method, "10p"))
        forty = lookup.get((method, "40p"))
        ten_text = "—" if ten is None else f"{100 * ten['validation']['macro_f1']:.2f}"
        forty_text = "—" if forty is None else f"{100 * forty['validation']['macro_f1']:.2f}"
        battery_text = (
            "—"
            if forty is None
            else f"{100 * forty['validation']['battery']['recall']:.2f}/"
            f"{100 * forty['validation']['battery']['false_positive_rate']:.2f}"
        )
        lines.append(f"| {method} | {ten_text} | {forty_text} | {battery_text} |")
    lines += ["", "## Gate", "", "```json", json.dumps(summary["gate"], indent=2), "```", "",
              "若 40% 的 MOM+DIFF 相对 MOM ≥+1.0pp 且 stack 相对 MOM stack ≥+0.2pp，才扩展四 seed；",
              "否则保留为负结果，不再为追正而改输入/门槛。test 不参与本轮。",
              "", f"- 根目录：`{root}`；协议：`{DEFAULT_PROTOCOL}`。"]
    Path("docs/experiment_records/e09_diff_stream_pilot_results_20261020.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare", "run"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol)
    elif args.action == "prepare":
        prepare(args.protocol)
    else:
        run(args.protocol)


if __name__ == "__main__":
    main()
