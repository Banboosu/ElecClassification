"""E14: MultiRocket strong baseline on variable-length charging curves.

Two fixed variable-length treatments: per-series z-score + linear interpolation to
256 points, and raw zero-padding to 1024.  MultiRocket fit/transform is the vendored
official implementation; RidgeClassifierCV selects alpha by training-only CV.
Validation only, no test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path("vendor")))
import multirocket.multirocket as multirocket  # noqa: E402

from tcn_moment import run_e08  # noqa: E402
from tcn_moment.config import load_config, replace, with_random_seed  # noqa: E402
from tcn_moment.data import load_dataset, stratified_train_subset_indices  # noqa: E402
from tcn_moment.e08_analysis import paired_summary  # noqa: E402
from tcn_moment.io_utils import atomic_write_json  # noqa: E402
from tcn_moment.label_semantics import BATTERY_LABEL, class_index  # noqa: E402

PACKAGES = ("numpy", "scikit-learn", "numba", "llvmlite")
VARIANTS = ("interp_z",)
TAGS = ("10p", "40p")
E12B = Path("artifacts/improvements/e12b_fusion_ablation")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root(p: dict[str, Any]) -> Path:
    return Path(p["output_dir"]) / p["suite"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(p: dict[str, Any]) -> None:
    if p.get("experiment_id") != "E14":
        raise ValueError("Unexpected experiment id")
    if p.get("seeds") != [42, 44, 45, 46]:
        raise ValueError("E14 fixes seeds 42/44/45/46")
    if p.get("fractions") not in ([0.1, 0.4], [0.4, 0.1], [0.1], [0.4]):
        raise ValueError("E14 supports 10% and/or 40%")
    if set(p.get("variants", {})) != set(VARIANTS):
        raise ValueError("E14 fixes the interp_z variant")
    if p.get("test_evaluated") or p.get("automatic_expansion"):
        raise ValueError("E14 is validation-only and does not auto-expand")


def _load(path: Path) -> dict[str, Any]:
    p = _read(path)
    _validate(p)
    return p


def _sources(protocol_path: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        Path("src/tcn_moment/label_semantics.py"),
        Path("src/tcn_moment/run_e08.py"),
        Path("vendor/multirocket/multirocket.py"),
        Path("vendor/multirocket/logistic_regression.py"),
        Path("vendor/multirocket/__init__.py"),
        Path("vendor/multirocket/README.md"),
        Path("vendor/multirocket/LICENSE"),
        Path("tests/test_e14_multirocket.py"),
        protocol_path,
    ]


def synthetic_check() -> dict[str, Any]:
    x = np.zeros((3, 12), dtype=np.float32)
    mask = np.zeros((3, 12), dtype=bool)
    x[0, :5] = [1, 2, 3, 4, 5]
    x[1, :4] = [7, 7, 7, 7]
    x[2, :12] = np.linspace(0, 11, 12)
    mask[0, :5] = True
    mask[1, :4] = True
    mask[2, :] = True
    interp = _interp_z(x, mask, 16)
    pad = _pad_raw(x, mask, 16)
    if interp.shape != (3, 16) or pad.shape != (3, 16):
        raise AssertionError("Preprocessing shape check failed")
    if not np.isfinite(interp).all() or not np.isfinite(pad).all():
        raise AssertionError("Preprocessing finite check failed")
    if not np.allclose(pad[0, :5], [1, 2, 3, 4, 5]) or not np.all(pad[0, 5:] == 0):
        raise AssertionError("Pad-raw check failed")
    if not np.allclose(interp[1], 0.0):
        raise AssertionError("Constant-series z-score check failed")
    return {"interp_shape": list(interp.shape), "pad_shape": list(pad.shape)}


def freeze(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    root = _root(p)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen_protocol.json").exists():
        return verify(protocol_path)
    if list(root.glob("seed*")):
        raise ValueError("Freeze must precede running")
    if _sha256(Path(p["dataset"])) != p["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    for seed in p["seeds"]:
        manifest = _read(Path(p["split_manifests"][str(seed)]))
        if (
            manifest["protocol"]["random_state"] != seed
            or manifest["protocol"]["dataset_sha256"] != p["dataset_sha256"]
            or manifest["counts"]["train"] != 24569
            or manifest["counts"]["validation"] != 3510
        ):
            raise ValueError(f"Split mismatch for seed {seed}")
    files = set(_sources(protocol_path))
    files.update(Path(p["split_manifests"][str(seed)]) for seed in p["seeds"])
    files.add(Path(p["dataset"]))
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Missing frozen input: {file}")
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol": p,
        "protocol_sha256": _sha256(protocol_path),
        "file_sha256": {str(file): _sha256(file) for file in sorted(files)},
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "runtime": {"host": platform.node(), "python": platform.python_version()},
        "synthetic_checks": synthetic_check(),
        "test_evaluated": False,
    }
    for file in sorted(files):
        if file.parts[0] in {"src", "configs", "tests", "scripts", "vendor", "docs"}:
            destination = root / "source_snapshot" / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    atomic_write_json(root / "frozen_protocol.json", frozen)
    print(f"Froze E14: {root / 'frozen_protocol.json'}", flush=True)
    return frozen


def verify(protocol_path: Path) -> dict[str, Any]:
    p = _load(protocol_path)
    frozen = _read(_root(p) / "frozen_protocol.json")
    if frozen["protocol"] != p or frozen["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("Frozen protocol changed")
    for file, expected in frozen["file_sha256"].items():
        if _sha256(Path(file)) != expected:
            raise ValueError(f"Frozen input changed: {file}")
    if frozen["packages"] != {name: importlib.metadata.version(name) for name in PACKAGES}:
        raise ValueError("Dependency versions changed")
    return frozen


def _interp_z(x: np.ndarray, mask: np.ndarray, length: int) -> np.ndarray:
    output = np.zeros((len(x), length), dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    for index in range(len(x)):
        values = np.asarray(x[index][mask_bool[index]], dtype=np.float64)
        if len(values) < 2:
            continue
        std = float(values.std())
        normalized = (values - values.mean()) / std if std >= 1e-8 else np.zeros_like(values)
        output[index] = np.interp(
            np.linspace(0.0, 1.0, length),
            np.linspace(0.0, 1.0, len(normalized)),
            normalized,
        )
    return output


def _pad_raw(x: np.ndarray, mask: np.ndarray, length: int) -> np.ndarray:
    output = np.zeros((len(x), length), dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    for index in range(len(x)):
        values = np.asarray(x[index][mask_bool[index]], dtype=np.float64)
        count = min(len(values), length)
        output[index, :count] = values[:count]
    return output


def _preprocess(x: np.ndarray, mask: np.ndarray, variant: str, length: int) -> np.ndarray:
    if variant == "interp_z":
        return _interp_z(x, mask, length)
    if variant == "pad_raw":
        return _pad_raw(x, mask, length)
    raise ValueError(f"Unknown variant: {variant}")


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _fit_mr(
    train: np.ndarray, validation: np.ndarray, *, num_features: int, seed: int, n_features_per_kernel: int
) -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(seed)
    started = time.perf_counter()
    base = multirocket.fit(train, num_features=num_features)
    diff = multirocket.fit(np.diff(train, axis=1), num_features=num_features)
    train_transform = multirocket.transform(
        train, np.diff(train, axis=1), base, diff, n_features_per_kernel
    )
    validation_transform = multirocket.transform(
        validation, np.diff(validation, axis=1), base, diff, n_features_per_kernel
    )
    seconds = time.perf_counter() - started
    return np.nan_to_num(train_transform), np.nan_to_num(validation_transform), seconds


def _write_predictions(
    root: Path,
    tag: str,
    variant: str,
    ids: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
) -> str:
    directory = root / "conditions" / tag / variant
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "split", "y_true", "y_pred", "score_0", "score_1", "score_2"])
        for sample_id, truth, prediction, probability in zip(ids, y, pred, scores, strict=True):
            writer.writerow(
                [sample_id, "validation", int(truth), int(prediction), *[f"{value:.12f}" for value in probability]]
            )
    return str(directory / "predictions.csv")


def run_seed(protocol_path: Path, seed: int) -> dict[str, Any]:
    frozen = verify(protocol_path)
    p = frozen["protocol"]
    root = _root(p) / f"seed{seed}"
    if (root / "seed_summary.json").exists():
        return _read(root / "seed_summary.json")
    config = with_random_seed(load_config(Path("configs/moment.yaml")), seed)
    raw = load_dataset(replace(config.data, normalize="none"))
    class_names = [str(value) for value in raw.label_encoder.classes_]
    battery_index = class_index(class_names, BATTERY_LABEL)
    subset_indices = {
        tag: stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
        for tag, fraction in (("10p", 0.1), ("40p", 0.4))
        if fraction in p["fractions"]
    }
    conditions: list[dict[str, Any]] = []
    for variant, spec in p["variants"].items():
        length = int(spec["length"])
        x_train = _preprocess(raw.x_train, raw.mask_train, variant, length)
        x_val = _preprocess(raw.x_val, raw.mask_val, variant, length)
        for tag, indices in subset_indices.items():
            started = time.perf_counter()
            train_transform, val_transform, mr_seconds = _fit_mr(
                x_train[indices],
                x_val,
                num_features=int(p["num_features"]),
                seed=seed * 100 + VARIANTS.index(variant),
                n_features_per_kernel=int(p["n_features_per_kernel"]),
            )
            classifier = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(
                    alphas=np.asarray(p["ridge_alphas"]),
                    cv=int(p["ridge_cv"]),
                ),
            )
            classifier.fit(train_transform, raw.y_train[indices])
            scores = _softmax(classifier.decision_function(val_transform))
            prediction = scores.argmax(axis=1)
            metrics = run_e08._evaluate(raw.y_val, prediction, scores, class_names, battery_index)
            prediction_path = _write_predictions(
                root, tag, variant, raw.ids_val, raw.y_val, prediction, scores
            )
            record = {
                "seed": seed,
                "tag": tag,
                "variant": variant,
                "variant_spec": spec,
                "method": f"multirocket_{variant}",
                "n_train": len(indices),
                "num_features": int(p["num_features"]),
                "transform_seconds": mr_seconds,
                "total_seconds": time.perf_counter() - started,
                "validation": metrics,
                "predictions": prediction_path,
                "test_evaluated": False,
            }
            conditions.append(record)
            atomic_write_json(root / "conditions" / tag / variant / "metrics.json", record)
            print(
                f"E14 seed={seed} {tag} {variant}: {100*metrics['macro_f1']:.2f} "
                f"battery {100*metrics['battery']['recall']:.2f}/"
                f"{100*metrics['battery']['false_positive_rate']:.2f} "
                f"in {record['total_seconds']:.1f}s",
                flush=True,
            )
    summary = {
        "seed": seed,
        "fractions": list(subset_indices),
        "conditions": conditions,
        "test_evaluated": False,
    }
    atomic_write_json(root / "seed_summary.json", summary)
    return summary


def _read_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = [row["sample_id"] for row in rows]
    y = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    scores = np.asarray(
        [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in rows],
        dtype=np.float64,
    )
    return ids, y, scores


def _align(reference_ids: list[str], ids: list[str], scores: np.ndarray) -> np.ndarray:
    mapping = {sample_id: index for index, sample_id in enumerate(ids)}
    if set(reference_ids) != set(ids):
        raise ValueError("Validation ID sets differ")
    return scores[[mapping[sample_id] for sample_id in reference_ids]]


def _aggregate(summaries: list[dict[str, Any]], p: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tag in [tag for tag in TAGS if tag in {row["tag"] for s in summaries for row in s["conditions"]}]:
        for variant in p["variants"]:
            values = [
                row["validation"]["macro_f1"]
                for s in summaries
                for row in s["conditions"]
                if row["tag"] == tag and row["variant"] == variant
            ]
            recalls = [
                row["validation"]["battery"]["recall"]
                for s in summaries
                for row in s["conditions"]
                if row["tag"] == tag and row["variant"] == variant
            ]
            fprs = [
                row["validation"]["battery"]["false_positive_rate"]
                for s in summaries
                for row in s["conditions"]
                if row["tag"] == tag and row["variant"] == variant
            ]
            rows.append(
                {
                    "tag": tag,
                    "variant": variant,
                    "n": len(values),
                    "mean": 100 * float(np.mean(values)),
                    "sample_sd": 100 * float(np.std(values, ddof=1)),
                    "per_seed": [100 * value for value in values],
                    "battery_recall_mean": 100 * float(np.mean(recalls)),
                    "battery_fpr_mean": 100 * float(np.mean(fprs)),
                }
            )
    return rows


def _paired(
    rows: list[dict[str, Any]], tag: str, variant: str, method: str
) -> dict[str, Any]:
    ours = next(row for row in rows if row["tag"] == tag and row["variant"] == variant)["per_seed"]
    theirs = [
        _read(E12B / tag / f"seed{seed}" / method / "metrics.json")["macro_f1"] * 100
        for seed in (42, 44, 45, 46)
    ]
    return paired_summary([a - b for a, b in zip(ours, theirs, strict=True)])


def _markdown(rows: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    lookup = {(row["tag"], row["variant"]): row for row in rows}
    lines = [
        "# E14：MultiRocket 强基线（validation-only）",
        "",
        f"**状态：{audit['conditions']} 条件 / {audit['rows']:,} 行；仅 validation，0 test。**",
        "",
        "固定使用 `interp_z`：逐序列 z-score 后线性插值到 256 点；这是 MultiRocket 对变长数据的固定预处理，不为它单独调参。",
        "MultiRocket 只使用官方 `fit`/`transform`；RidgeClassifierCV 只在训练子集内用 3 折从 {100,1000} 选择 alpha，validation 不参与。",
        "",
        "## Macro-F1 %",
        "",
        "| 方法 | 10% | 40% |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        cells = []
        for tag in TAGS:
            row = lookup[(tag, variant)]
            cells.append(f"{row['mean']:.2f} ± {row['sample_sd']:.2f}")
        lines.append(f"| MultiRocket ({variant}) | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 与 E12b 参照的逐 seed 配对差（pp）",
        "",
        "| 预算 | 变体 | vs DYN | vs TCN | vs Full |",
        "|---|---|---:|---:|---:|",
    ]
    for tag in TAGS:
        for variant in VARIANTS:
            dyn = _paired(rows, tag, variant, "dyn")
            tcn = _paired(rows, tag, variant, "tcn")
            full = _paired(rows, tag, variant, "full")
            lines.append(
                f"| {tag} | {variant} | {dyn['mean']:+.2f} (SD {dyn['sample_sd']:.2f}) | "
                f"{tcn['mean']:+.2f} (SD {tcn['sample_sd']:.2f}) | "
                f"{full['mean']:+.2f} (SD {full['sample_sd']:.2f}) |"
            )
    lines += [
        "",
        "## 电池异常（40%，默认 argmax）",
        "",
        "| 变体 | Recall % | FPR % |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        row = lookup[("40p", variant)]
        lines.append(f"| {variant} | {row['battery_recall_mean']:.2f} | {row['battery_fpr_mean']:.2f} |")
    lines += [
        "",
        "## 审计",
        "",
        f"- {audit['conditions']} 条件 / {audit['rows']} 行；Macro-F1 复算最大误差 {audit['max_macro_f1_error']:.3g}；0 test。",
        "",
        "## 边界",
        "",
        "- MultiRocket 不支持 mask；两种变长处理都是固定规则，未为它单独调参。",
        "- 使用 vendored 官方实现，只调用 fit/transform；分类头是 RidgeClassifierCV（{100,1000}，训练内 3 折），不是完整原仓库脚本。",
        "- `num_features=2000` 低于论文默认 50k，是为本数据规模/内存固定的预算；不能据此说 MultiRocket 的上限只有这个水平。",
        "- 四 seed split 重叠、validation 反复使用；没有 test、没有外部数据。",
        "- 这是 clean 输入的强基线；没有做噪声/量化鲁棒性。",
    ]
    return "\n".join(lines)


def _audit(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    total = 0
    maximum = 0.0
    for summary in summaries:
        for condition in summary["conditions"]:
            path = Path(condition["predictions"])
            with path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            if len(rows) != 3510 or any(row["split"] != "validation" for row in rows):
                raise ValueError(f"Bad prediction rows: {path}")
            y = np.asarray([int(row["y_true"]) for row in rows])
            pred = np.asarray([int(row["y_pred"]) for row in rows])
            maximum = max(maximum, abs(float(f1_score(y, pred, average="macro")) - condition["validation"]["macro_f1"]))
            total += len(rows)
            count += 1
    return {"conditions": count, "rows": total, "max_macro_f1_error": maximum, "test_rows": 0}


def run(protocol_path: Path) -> None:
    frozen = freeze(protocol_path)
    p = frozen["protocol"]
    root = _root(p)
    if (root / "summary.json").exists():
        print(f"Already complete: {root}", flush=True)
        return
    summaries = [run_seed(protocol_path, seed) for seed in p["seeds"]]
    rows = _aggregate(summaries, p)
    audit = _audit(summaries)
    summary = {"test_evaluated": False, "aggregate": rows, "audit": audit}
    atomic_write_json(root / "summary.json", summary)
    markdown = _markdown(rows, audit)
    (root / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/experiments/improvements/e14_multirocket_protocol.json")
    parser.add_argument("--action", choices=["freeze", "verify", "run", "report"], default="run")
    args = parser.parse_args()
    path = Path(args.protocol)
    if args.action == "freeze":
        freeze(path)
    elif args.action == "verify":
        verify(path)
    elif args.action == "report":
        p = _load(path)
        summary = _read(_root(p) / "summary.json")
        markdown = _markdown(summary["aggregate"], summary["audit"])
        (_root(p) / "summary.md").write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
    else:
        run(path)


if __name__ == "__main__":
    main()
