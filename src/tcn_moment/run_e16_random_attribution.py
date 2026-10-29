"""E16: pretrained vs random-initialized MOMENT attribution.

Same MOMENT architecture and mask-aware pooling; random weights are initialized per
data seed and guarded against any pretrained checkpoint loading.  Original, diff and
dual-stream logistic probes use the same C=1.0 protocol.  Validation only, no test.
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
import torch

sys.path.insert(0, str(Path("vendor")))
from tcn_moment import run_e08  # noqa: E402
from tcn_moment.config import load_config, replace, with_random_seed  # noqa: E402
from tcn_moment.data import load_dataset, stratified_train_subset_indices  # noqa: E402
from tcn_moment.e08_analysis import paired_summary  # noqa: E402
from tcn_moment.io_utils import atomic_write_json  # noqa: E402
from tcn_moment.label_semantics import BATTERY_LABEL, class_index  # noqa: E402
from tcn_moment.run_e09_diff_stream import _diff_arrays  # noqa: E402
from tcn_moment.train_e01c import forbid_pretrained_loading  # noqa: E402
from tcn_moment.train_moment import (  # noqa: E402
    build_model,
    forward_features,
    require_torch_and_moment,
    set_num_classes,
)
from tcn_moment.training_utils import seed_everything  # noqa: E402

PACKAGES = ("torch", "numpy", "pandas", "scikit-learn")
TAGS = ("10p", "40p")
STREAMS = ("orig", "diff", "dual")
INITS = ("pretrained", "random")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _root(p: dict[str, Any]) -> Path:
    return Path(p["output_dir"]) / p["suite"]


def _ids_for(raw: Any, split: str) -> np.ndarray:
    return raw.ids_train if split == "train" else raw.ids_val


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(p: dict[str, Any]) -> None:
    if p.get("experiment_id") != "E16":
        raise ValueError("Unexpected experiment id")
    if p.get("seeds") != [42, 44, 45, 46]:
        raise ValueError("E16 fixes seeds 42/44/45/46")
    allowed = {0.01, 0.05, 0.1, 0.2, 0.4}
    fractions = p.get("fractions", [])
    if not fractions or not set(fractions).issubset(allowed):
        raise ValueError("E16 supports nonempty subsets of 1/5/10/20/40%")
    if tuple(p.get("streams", [])) != STREAMS or tuple(p.get("initializations", [])) != INITS:
        raise ValueError("E16 fixes streams and initializations")
    if p.get("test_evaluated") or p.get("automatic_expansion"):
        raise ValueError("E16 is validation-only and does not auto-expand")


def _load(path: Path) -> dict[str, Any]:
    p = _read(path)
    _validate(p)
    return p


def _sources(protocol_path: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        Path("src/tcn_moment/train_e01c.py"),
        Path("src/tcn_moment/train_moment.py"),
        Path("src/tcn_moment/run_e09_diff_stream.py"),
        Path("src/tcn_moment/run_e08.py"),
        Path("src/tcn_moment/label_semantics.py"),
        Path("src/tcn_moment/data.py"),
        Path("src/tcn_moment/config.py"),
        Path("src/tcn_moment/io_utils.py"),
        Path("tests/test_e16_random_attribution.py"),
        protocol_path,
    ]


def synthetic_check() -> dict[str, Any]:
    x = np.zeros((2, 8), dtype=np.float32)
    mask = np.zeros((2, 8), dtype=bool)
    x[0, :5] = [0.0, 1.0, 2.0, 3.0, 4.0]
    x[1, :4] = [5.0, 5.0, 5.0, 5.0]
    mask[0, :5] = True
    mask[1, :4] = True
    diff, diff_mask = _diff_arrays(x, mask)
    if diff.shape != x.shape or diff_mask.shape != mask.shape:
        raise AssertionError("Diff shape check failed")
    if not np.allclose(diff[1, :3], 0.0):
        raise AssertionError("Constant diff stream should be zero")
    return {"diff_nonzero": int((diff != 0).sum()), "diff_mask": int(diff_mask.sum())}


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
    for seed in p["seeds"]:
        for stream, cache in p["pretrained_caches"].items():
            for split in ("train", "validation"):
                files.add(Path(cache) / f"seed{seed}" / f"{split}.npz")
            files.add(Path(cache) / f"seed{seed}" / "manifest.json")
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Missing frozen input: {file}")
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol": p,
        "protocol_sha256": _sha256(protocol_path),
        "file_sha256": {str(file): _sha256(file) for file in sorted(files)},
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "cuda": torch.version.cuda,
        },
        "synthetic_checks": synthetic_check(),
        "test_evaluated": False,
    }
    for file in sorted(files):
        if file.parts[0] in {"src", "configs", "tests", "scripts", "docs"}:
            destination = root / "source_snapshot" / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
    atomic_write_json(root / "frozen_protocol.json", frozen)
    print(f"Froze E16: {root / 'frozen_protocol.json'}", flush=True)
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


def _normalize_original(x: np.ndarray, mask: np.ndarray, length: int) -> np.ndarray:
    output = np.zeros((len(x), length), dtype=np.float32)
    mask_bool = np.asarray(mask, dtype=bool)
    for index in range(len(x)):
        values = np.asarray(x[index][mask_bool[index]], dtype=np.float64)
        if len(values) == 0:
            continue
        std = float(values.std())
        if std >= 1e-8:
            output[index, : len(values)] = ((values - values.mean()) / std).astype(np.float32)
    return output


def _embed(model: Any, x: np.ndarray, mask: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start : start + batch_size, None, :], dtype=torch.float32, device=device)
            batch_mask = torch.as_tensor(mask[start : start + batch_size], dtype=torch.float32, device=device)
            features = forward_features(model, batch_x, batch_mask).float().cpu().numpy()
            if not np.isfinite(features).all():
                raise FloatingPointError("Nonfinite random MOMENT features")
            chunks.append(features)
    return np.concatenate(chunks, axis=0)


def _save_cache(path: Path, features: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez(file, features=features, sample_ids=ids.astype(str))
    os.replace(temporary, path)
    return {"shape": list(features.shape), "sha256": _sha256(path)}


def _load_cache(path: Path, ids: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float64)
        sample_ids = archive["sample_ids"].astype(str)
    if not np.array_equal(sample_ids, ids.astype(str)):
        raise ValueError(f"Random cache order mismatch: {path}")
    return features


def prepare(protocol_path: Path) -> None:
    frozen = freeze(protocol_path)
    p = frozen["protocol"]
    if not torch.cuda.is_available():
        raise RuntimeError("E16 random extraction requires CUDA")
    device = torch.device("cuda")
    torch_mod, DataLoader, TensorDataset, tqdm, MOMENTPipeline = require_torch_and_moment()
    cache_root = Path(p["random_cache"])
    for seed in p["seeds"]:
        seed_root = cache_root / f"seed{seed}"
        expected = [seed_root / f"{stream}_{split}.npz" for stream in ("orig", "diff") for split in ("train", "validation")]
        if all(path.exists() for path in expected):
            print(f"Random cache exists: {seed_root}", flush=True)
            continue
        seed_root.mkdir(parents=True, exist_ok=True)
        config = with_random_seed(load_config(Path("configs/moment.yaml")), seed)
        random_config = replace(config, model=replace(config.model, initialization="random"))
        seed_everything(torch_mod, seed)
        with forbid_pretrained_loading():
            model = build_model(random_config, MOMENTPipeline, 3)
            set_num_classes(model, 3)
            model.init()
            model.to(device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
        raw = load_dataset(replace(with_random_seed(load_config(Path("configs/moment.yaml")), seed).data, normalize="none"))
        x_orig_train = _normalize_original(raw.x_train, raw.mask_train, p["max_length"])
        x_orig_val = _normalize_original(raw.x_val, raw.mask_val, p["max_length"])
        diff_train, diff_mask_train = _diff_arrays(raw.x_train, raw.mask_train)
        diff_val, diff_mask_val = _diff_arrays(raw.x_val, raw.mask_val)
        started = time.perf_counter()
        features = {
            "orig_train": _embed(model, x_orig_train, raw.mask_train, device),
            "orig_validation": _embed(model, x_orig_val, raw.mask_val, device),
            "diff_train": _embed(model, diff_train, diff_mask_train, device),
            "diff_validation": _embed(model, diff_val, diff_mask_val, device),
        }
        records = {}
        for name, array in features.items():
            stream, split = name.split("_", 1)
            records[name] = _save_cache(seed_root / f"{stream}_{split}.npz", array, _ids_for(raw, split))
        manifest = {
            "seed": seed,
            "initialization": "random",
            "random_initialization_seed": seed,
            "pretrained_checkpoint_loaded": False,
            "architecture": "MOMENT classification backbone, 1024-d pooled features",
            "splits": records,
            "seconds": time.perf_counter() - started,
            "test_evaluated": False,
        }
        atomic_write_json(seed_root / "manifest.json", manifest)
        atomic_write_json(
            seed_root / "pretrained_loading_guard.json",
            {
                "guard_enabled": True,
                "pretrained_checkpoint_loaded": False,
                "random_initialization_seed": seed,
                "guarded_apis": ["MOMENTPipeline.from_pretrained", "T5Model.from_pretrained", "T5EncoderModel.from_pretrained"],
                "source_sha256": _sha256(Path(__file__)),
            },
        )
        del model
        torch.cuda.empty_cache()
        print(f"Random features ready: {seed_root}", flush=True)


def _load_pretrained(p: dict[str, Any], seed: int, stream: str, split: str, ids: np.ndarray) -> np.ndarray:
    if stream == "orig":
        features, _ = run_e08._load_moment_split(Path(p["pretrained_caches"]["orig"]), seed=seed, split=split, ids=ids)
        return features
    features, _ = run_e08._load_moment_split(Path(p["pretrained_caches"]["diff"]), seed=seed, split=split, ids=ids)
    return features


def _load_random(p: dict[str, Any], seed: int, stream: str, split: str, ids: np.ndarray) -> np.ndarray:
    path = Path(p["random_cache"]) / f"seed{seed}" / f"{stream}_{split}.npz"
    return _load_cache(path, ids)


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


def _write_predictions(
    root: Path, tag: str, initialization: str, stream: str, ids: np.ndarray, y: np.ndarray, pred: np.ndarray, scores: np.ndarray
) -> str:
    directory = root / "conditions" / tag / initialization / stream
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "split", "y_true", "y_pred", "score_0", "score_1", "score_2"])
        for sample_id, truth, prediction, probability in zip(ids, y, pred, scores, strict=True):
            writer.writerow([sample_id, "validation", int(truth), int(prediction), *[f"{value:.12f}" for value in probability]])
    return str(directory / "predictions.csv")


def run(protocol_path: Path) -> None:
    frozen = verify(protocol_path)
    p = frozen["protocol"]
    root = _root(p)
    if (root / "summary.json").exists():
        print(f"Already complete: {root}", flush=True)
        return
    class_names = ["0", "1", "2"]
    battery_index = class_index(class_names, BATTERY_LABEL)
    for seed in p["seeds"]:
        seed_root = Path(p["random_cache"]) / f"seed{seed}"
        manifest = _read(seed_root / "manifest.json")
        if manifest.get("initialization") != "random" or manifest.get("pretrained_checkpoint_loaded"):
            raise ValueError(f"Bad random cache provenance: {seed_root}")
        raw = load_dataset(replace(with_random_seed(load_config(Path("configs/moment.yaml")), seed).data, normalize="none"))
        tag_fractions = (("1p", 0.01), ("5p", 0.05), ("10p", 0.1), ("20p", 0.2), ("40p", 0.4))
        indices = {
            tag: stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
            for tag, fraction in tag_fractions
            if fraction in p["fractions"]
        }
        pretrained = {
            stream: {
                split: _load_pretrained(p, seed, stream, split, _ids_for(raw, split))
                for split in ("train", "validation")
            }
            for stream in ("orig", "diff")
        }
        random = {
            stream: {
                split: _load_random(p, seed, stream, split, _ids_for(raw, split))
                for split in ("train", "validation")
            }
            for stream in ("orig", "diff")
        }
        for initialization, features in (("pretrained", pretrained), ("random", random)):
            for stream in ("orig", "diff", "dual"):
                train_matrix = (
                    features["orig"]["train"]
                    if stream == "orig"
                    else features["diff"]["train"]
                    if stream == "diff"
                    else np.hstack([features["orig"]["train"], features["diff"]["train"]])
                )
                val_matrix = (
                    features["orig"]["validation"]
                    if stream == "orig"
                    else features["diff"]["validation"]
                    if stream == "diff"
                    else np.hstack([features["orig"]["validation"], features["diff"]["validation"]])
                )
                for tag, subset in indices.items():
                    classifier = run_e08._logit_factory({"C": float(p["classifier_c"])}, seed).fit(
                        train_matrix[subset], raw.y_train[subset]
                    )
                    scores = classifier.predict_proba(val_matrix)
                    prediction = scores.argmax(axis=1)
                    metrics = run_e08._evaluate(raw.y_val, prediction, scores, class_names, battery_index)
                    prediction_path = _write_predictions(
                        root / f"seed{seed}", tag, initialization, stream, raw.ids_val, raw.y_val, prediction, scores
                    )
                    record = {
                        "seed": seed,
                        "tag": tag,
                        "initialization": initialization,
                        "stream": stream,
                        "n_train": len(subset),
                        "classifier_c": float(p["classifier_c"]),
                        "validation": metrics,
                        "predictions": prediction_path,
                        "test_evaluated": False,
                    }
                    atomic_write_json(
                        root / f"seed{seed}" / "conditions" / tag / initialization / stream / "metrics.json", record
                    )
                    print(
                        f"E16 seed={seed} {tag} {initialization} {stream}: "
                        f"{100*metrics['macro_f1']:.2f} battery "
                        f"{100*metrics['battery']['recall']:.2f}/{100*metrics['battery']['false_positive_rate']:.2f}",
                        flush=True,
                    )
    _summarize(p, root)


def _load_conditions(root: Path, seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        for path in sorted((root / f"seed{seed}" / "conditions").rglob("metrics.json")):
            rows.append(_read(path))
    return rows


def _summarize(p: dict[str, Any], root: Path) -> None:
    conditions = _load_conditions(root, p["seeds"])
    rows = []
    tags = [tag for tag, _ in (("1p", 0.01), ("5p", 0.05), ("10p", 0.1), ("20p", 0.2), ("40p", 0.4)) if tag in {row["tag"] for row in conditions}]
    for tag in tags:
        for initialization in INITS:
            for stream in STREAMS:
                selected = [row for row in conditions if row["tag"] == tag and row["initialization"] == initialization and row["stream"] == stream]
                values = [100 * row["validation"]["macro_f1"] for row in selected]
                recalls = [100 * row["validation"]["battery"]["recall"] for row in selected]
                fprs = [100 * row["validation"]["battery"]["false_positive_rate"] for row in selected]
                rows.append(
                    {
                        "tag": tag,
                        "initialization": initialization,
                        "stream": stream,
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "sample_sd": float(np.std(values, ddof=1)),
                        "per_seed": values,
                        "battery_recall_mean": float(np.mean(recalls)),
                        "battery_fpr_mean": float(np.mean(fprs)),
                    }
                )
    paired = []
    for tag in tags:
        for stream in STREAMS:
            pre = next(row for row in rows if row["tag"] == tag and row["initialization"] == "pretrained" and row["stream"] == stream)
            rnd = next(row for row in rows if row["tag"] == tag and row["initialization"] == "random" and row["stream"] == stream)
            paired.append(
                {
                    "tag": tag,
                    "stream": stream,
                    "comparison": "pretrained - random",
                    **paired_summary([a - b for a, b in zip(pre["per_seed"], rnd["per_seed"], strict=True)]),
                }
            )
    audit = _audit(root, conditions)
    summary = {"test_evaluated": False, "aggregate": rows, "paired": paired, "audit": audit}
    atomic_write_json(root / "summary.json", summary)
    markdown = _markdown(rows, paired, audit)
    (root / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def _audit(root: Path, conditions: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    total = 0
    maximum = 0.0
    for condition in conditions:
        path = Path(condition["predictions"])
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 3510 or any(row["split"] != "validation" for row in rows):
            raise ValueError(f"Bad prediction rows: {path}")
        y = np.asarray([int(row["y_true"]) for row in rows])
        pred = np.asarray([int(row["y_pred"]) for row in rows])
        maximum = max(maximum, abs(float(run_e08.f1_score(y, pred, average="macro")) - condition["validation"]["macro_f1"]))
        total += len(rows)
        count += 1
    return {"conditions": count, "rows": total, "max_macro_f1_error": maximum, "test_rows": 0}


def _markdown(rows: list[dict[str, Any]], paired: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    tags = [tag for tag, _ in (("1p", 0.01), ("5p", 0.05), ("10p", 0.1), ("20p", 0.2), ("40p", 0.4)) if tag in {row["tag"] for row in rows}]
    lookup = {(row["tag"], row["initialization"], row["stream"]): row for row in rows}
    lines = [
        "# E16：预训练 vs 随机初始化归因（validation-only）",
        "",
        f"**状态：{audit['conditions']} 条件 / {audit['rows']:,} 行；仅 validation，0 test。**",
        "",
        "同一 MOMENT 架构与 mask-aware pooling；随机初始化每个数据 seed 一次，原序列/差分/双流共用；预训练加载被 guard 禁止。",
        "Logistic C=1.0；原序列 z-score、差分逐序列 z-score。",
        "",
        "## Macro-F1 %",
        "",
        "| 预算 | 流 | 预训练 | 随机初始化 | 预训练−随机 |",
        "|---|---|---:|---:|---:|",
    ]
    for tag in tags:
        for stream in STREAMS:
            pre = lookup[(tag, "pretrained", stream)]
            rnd = lookup[(tag, "random", stream)]
            diff = next(item for item in paired if item["tag"] == tag and item["stream"] == stream)
            lines.append(
                f"| {tag} | {stream} | {pre['mean']:.2f} ± {pre['sample_sd']:.2f} | "
                f"{rnd['mean']:.2f} ± {rnd['sample_sd']:.2f} | "
                f"{diff['mean']:+.2f} ± {diff['sample_sd']:.2f} ({diff['positive_seeds']}/4) |"
            )
    if "40p" in tags:
        lines += [
            "",
            "## 电池异常（40%，默认 argmax）",
            "",
            "| 流 | 预训练 Recall/FPR | 随机 Recall/FPR |",
            "|---|---:|---:|",
        ]
        for stream in STREAMS:
            pre = lookup[("40p", "pretrained", stream)]
            rnd = lookup[("40p", "random", stream)]
            lines.append(
                f"| {stream} | {pre['battery_recall_mean']:.2f}/{pre['battery_fpr_mean']:.2f} | "
                f"{rnd['battery_recall_mean']:.2f}/{rnd['battery_fpr_mean']:.2f} |"
            )
    lines += [
        "",
        "## 审计",
        "",
        f"- {audit['conditions']} 条件 / {audit['rows']} 行；Macro-F1 复算最大误差 {audit['max_macro_f1_error']:.3g}；0 test。",
        "",
        "## 边界",
        "",
        "- 这是同一架构内的预训练归因；随机初始化只用了一个 draw/seed，不能代表所有随机权重。",
        "- 只改变初始化，不改变池化、特征维度、logistic C、数据与选择预算；backbone 全程冻结。",
        "- 四 seed split 重叠、validation 反复使用；无 test、无外部数据；不是部署保证。",
        "- 没有做随机初始化多次重复；因此“预训练无差异”只能作为本数据上的条件性结论。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/experiments/improvements/e16_random_attribution_protocol.json")
    parser.add_argument("--action", choices=["freeze", "verify", "prepare", "run", "report"], default="run")
    args = parser.parse_args()
    path = Path(args.protocol)
    if args.action == "freeze":
        freeze(path)
    elif args.action == "verify":
        verify(path)
    elif args.action == "prepare":
        prepare(path)
    elif args.action == "report":
        p = _load(path)
        summary = _read(_root(p) / "summary.json")
        markdown = _markdown(summary["aggregate"], summary["paired"], summary["audit"])
        (_root(p) / "summary.md").write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
    else:
        run(path)


if __name__ == "__main__":
    main()
