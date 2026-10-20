"""E12b: two-way fusion ablation on saved E08/E09/E12 base predictions.

Fits only the cheap meta logistic on E12 OOF base probabilities; TCN, DYN-HGB and
MOMENT/ΔP base models are reused from frozen artifacts.  Validation only, no test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tcn_moment import run_e08
from tcn_moment.config import load_config, replace, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.e08_analysis import paired_summary
from tcn_moment.io_utils import atomic_write_json

SEEDS = (42, 44, 45, 46)
TAGS = ("10p", "40p")
E08 = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
E09 = Path("artifacts/improvements/e09_diff_stream")
E01_CACHE = Path("artifacts/improvements/e01_20261017_v2_features")
E09_CACHE = Path("artifacts/improvements/e09_diff_features")
E12 = {
    "10p": Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_10p_20261020_v1"),
    "40p": Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_20261020_v1"),
}
METHODS = ("tcn", "tcn_meta", "dyn", "md", "tcn+dyn", "tcn+md", "dyn+md", "full")
BASE_FILES = {
    "dyn": lambda seed, tag: E08 / f"seed{seed}" / "conditions" / tag / "dyn_hgb" / "predictions.csv",
    "md": lambda seed, tag: E09 / f"e09_diff_stream_20261020_seed{seed}" / "conditions" / tag / "moment_diff_logit" / "predictions.csv",
    "tcn": lambda seed, tag: E12[tag] / f"seed{seed}" / "conditions" / tag / "tcn_full" / "predictions.csv",
}
REFERENCE = {
    "dyn+md": lambda seed, tag: E12[tag] / f"seed{seed}" / "conditions" / tag / "stack_no_tcn" / "predictions.csv",
    "full": lambda seed, tag: E12[tag] / f"seed{seed}" / "conditions" / tag / "stack_tcn" / "predictions.csv",
}


def _read_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = [row["sample_id"] for row in rows]
    y = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    probs = np.asarray(
        [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in rows],
        dtype=np.float64,
    )
    return ids, y, probs, np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)


def _align(
    reference_ids: list[str], ids: list[str], probs: np.ndarray
) -> np.ndarray:
    mapping = {sample_id: index for index, sample_id in enumerate(ids)}
    if set(reference_ids) != set(ids):
        raise ValueError("Validation ID sets differ")
    return probs[[mapping[sample_id] for sample_id in reference_ids]]


def _meta(
    oof: np.ndarray, y: np.ndarray, validation: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    grid = [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    selected, cv_results = run_e08._cv_select(
        lambda params: LogisticRegression(C=params["C"], max_iter=5000, random_state=seed),
        grid,
        oof,
        y,
        folds=3,
        seed=seed,
    )
    model = LogisticRegression(C=selected["C"], max_iter=5000, random_state=seed).fit(oof, y)
    return model.predict_proba(validation), selected, cv_results


def _md_validation(seed: int, tag: str, reference_ids: list[str]) -> np.ndarray:
    """Recompute the MD validation branch with the same C=1.0 used for OOF."""
    fraction = 0.1 if tag == "10p" else 0.4
    config = with_random_seed(load_config(Path("configs/moment.yaml")), seed)
    raw = load_dataset(replace(config.data, normalize="none"))
    moment_train, _ = run_e08._load_moment_split(E01_CACHE, seed=seed, split="train", ids=raw.ids_train)
    moment_val, _ = run_e08._load_moment_split(E01_CACHE, seed=seed, split="validation", ids=raw.ids_val)
    diff_train, _ = run_e08._load_moment_split(E09_CACHE, seed=seed, split="train", ids=raw.ids_train)
    diff_val, _ = run_e08._load_moment_split(E09_CACHE, seed=seed, split="validation", ids=raw.ids_val)
    indices = stratified_train_subset_indices(raw.y_train, raw.ids_train, fraction, seed)
    model = run_e08._logit_factory({"C": 1.0}, seed).fit(
        np.hstack([moment_train[indices], diff_train[indices]]), raw.y_train[indices]
    )
    probabilities = model.predict_proba(np.hstack([moment_val, diff_val]))
    return _align(reference_ids, raw.ids_val.astype(str).tolist(), probabilities)


def _run_seed(tag: str, seed: int, root: Path) -> list[dict[str, Any]]:
    with np.load(E12[tag] / f"seed{seed}" / f"oof_{tag}.npz", allow_pickle=False) as archive:
        oof = {
            "dyn": archive["oof_dyn"],
            "md": archive["oof_md"],
            "tcn": archive["oof_tcn"],
        }
        y_train = archive["y_train"]
    reference_ids, y_val, _, _ = _read_predictions(BASE_FILES["dyn"](seed, tag))
    validation: dict[str, np.ndarray] = {}
    for name, path_fn in BASE_FILES.items():
        path = path_fn(seed, tag)
        ids, y, probs, _ = _read_predictions(path)
        if not np.array_equal(y, y_val):
            raise ValueError(f"Validation labels differ: {path}")
        validation[name] = _align(reference_ids, ids, probs)
    # E09 saved MD validation with its own selected C (usually 0.1); E12 OOF MD
    # used C=1.0.  Recompute validation with C=1.0 so the ablation is not mixed.
    validation["md"] = _md_validation(seed, tag, reference_ids)
    conditions: list[dict[str, Any]] = []
    for method in METHODS:
        if method in REFERENCE:
            ids, y, probs, pred = _read_predictions(REFERENCE[method](seed, tag))
            if not np.array_equal(y, y_val):
                raise ValueError("Reference labels differ")
            probs = _align(reference_ids, ids, probs)
            selected: dict[str, Any] = {"source": str(REFERENCE[method](seed, tag))}
            cv_results: list[dict[str, Any]] = []
        elif method in ("dyn", "md", "tcn"):
            probs = validation[method]
            pred = probs.argmax(1)
            selected = {"source": str(BASE_FILES[method](seed, tag))}
            if method == "md":
                selected = {"C": 1.0, "source": "recomputed clean MD validation (matches E12 OOF)"}
            cv_results = []
        elif method == "tcn_meta":
            probs, selected, cv_results = _meta(oof["tcn"], y_train, validation["tcn"], seed)
            pred = probs.argmax(1)
        else:
            left, right = method.split("+")
            oof_matrix = np.hstack([oof[left], oof[right]])
            validation_matrix = np.hstack([validation[left], validation[right]])
            probs, selected, cv_results = _meta(oof_matrix, y_train, validation_matrix, seed)
            pred = probs.argmax(1)
        metrics = run_e08._evaluate(
            y_val,
            pred,
            probs,
            ["0", "1", "2"],
            2,
        )
        condition_dir = root / tag / f"seed{seed}" / method
        condition_dir.mkdir(parents=True, exist_ok=True)
        with (condition_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["sample_id", "split", "y_true", "y_pred", "score_0", "score_1", "score_2"])
            for sample_id, truth, prediction, probability in zip(
                reference_ids, y_val, pred, probs, strict=True
            ):
                writer.writerow(
                    [sample_id, "validation", int(truth), int(prediction), *[f"{value:.12f}" for value in probability]]
                )
        record = {
            "seed": seed,
            "tag": tag,
            "method": method,
            "macro_f1": float(metrics["macro_f1"]),
            "battery_recall": float(metrics["battery"]["recall"]),
            "battery_fpr": float(metrics["battery"]["false_positive_rate"]),
            "selected_params": selected,
            "cv_results": cv_results,
            "predictions": str(condition_dir / "predictions.csv"),
        }
        conditions.append(record)
        atomic_write_json(condition_dir / "metrics.json", record)
        print(
            f"E12b {tag} seed={seed} {method}: {100*record['macro_f1']:.2f} "
            f"battery {100*record['battery_recall']:.2f}/{100*record['battery_fpr']:.2f}",
            flush=True,
        )
    return conditions


def _aggregate(all_conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for tag in TAGS:
        for method in METHODS:
            values = [row["macro_f1"] for row in all_conditions if row["tag"] == tag and row["method"] == method]
            recalls = [row["battery_recall"] for row in all_conditions if row["tag"] == tag and row["method"] == method]
            fps = [row["battery_fpr"] for row in all_conditions if row["tag"] == tag and row["method"] == method]
            rows.append(
                {
                    "tag": tag,
                    "method": method,
                    "n": len(values),
                    "mean": 100 * float(np.mean(values)),
                    "sample_sd": 100 * float(np.std(values, ddof=1)),
                    "per_seed": [100 * value for value in values],
                    "battery_recall_mean": 100 * float(np.mean(recalls)),
                    "battery_fpr_mean": 100 * float(np.mean(fps)),
                }
            )
    return rows


def _paired(rows: list[dict[str, Any]], left: str, right: str, tag: str) -> dict[str, Any]:
    def get(method: str) -> list[float]:
        return next(row for row in rows if row["tag"] == tag and row["method"] == method)["per_seed"]

    return paired_summary([a - b for a, b in zip(get(left), get(right), strict=True)])


def _markdown(rows: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    lines = [
        "# E12b：两路融合消融（复用 E08/E09/E12 冻结预测，validation-only）",
        "",
        "只重拟合廉价 meta logistic；TCN、DYN-HGB、MOMENT/ΔP 基模型均复用。**无 test。**",
        "",
        "## 主结果（Macro-F1 %，均值 ± 样本 SD）",
        "",
        "| 方法 | 10% | 40% |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for tag in TAGS:
            row = next(row for row in rows if row["tag"] == tag and row["method"] == method)
            cells.append(f"{row['mean']:.2f} ± {row['sample_sd']:.2f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += ["", "## 配对差（pp）", "", "| 比较 | 预算 | 均值 ± SD | 正向 seed |", "|---|---:|---:|---:|"]
    comparisons = [
        ("full", "tcn+dyn"),
        ("full", "tcn+md"),
        ("full", "dyn+md"),
        ("full", "tcn"),
        ("full", "tcn_meta"),
        ("tcn_meta", "tcn"),
        ("tcn+dyn", "tcn"),
        ("tcn+md", "tcn"),
        ("dyn+md", "tcn"),
        ("tcn+dyn", "dyn"),
        ("tcn+md", "md"),
    ]
    for left, right in comparisons:
        for tag in TAGS:
            record = _paired(rows, left, right, tag)
            lines.append(
                f"| {left} − {right} | {tag} | {record['mean']:+.2f} ± {record['sample_sd']:.2f} | "
                f"{record['positive_seeds']}/{record['n']} |"
            )
    lines += [
        "",
        "## 电池异常（40%，默认 argmax）",
        "",
        "| 方法 | Recall % | FPR % |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        row = next(row for row in rows if row["tag"] == "40p" and row["method"] == method)
        lines.append(f"| {method} | {row['battery_recall_mean']:.2f} | {row['battery_fpr_mean']:.2f} |")
    lines += [
        "",
        "## 审计",
        "",
        f"- {audit['conditions']} 条件 / {audit['rows']} 行；最大 Macro-F1 复算误差 {audit['max_macro_f1_error']:.3g}。",
        "- 所有行 split=validation，0 条 test。",
        "",
        "## 边界",
        "",
        "- 这是**同 validation 行、同 OOF 折**的廉价 meta 消融，不是重新训练基模型；10% 的 TCN 端点是固定 50 轮无选模。",
        "- `full` 与 `dyn+md` 直接使用 E12 已保存的 `stack_tcn`/`stack_no_tcn`，其余两路组合重新拟合 meta；C 仍只在 OOF 上选。",
        "- `tcn_meta` 是 TCN 概率经同型 meta 头的控制；`tcn+md` 的 MD validation 已重算为 C=1.0，与 E12 OOF 的 MD 分支一致（E09 保存的 validation 通常用 C=0.1）。",
        "- 四 seed split 重叠；不能把差值写成独立确认。没有 test、校准、鲁棒性。",
    ]
    return "\n".join(lines)


def _audit(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    total = 0
    maximum = 0.0
    for row in rows:
        for seed in SEEDS:
            path = root / row["tag"] / f"seed{seed}" / row["method"] / "predictions.csv"
            with path.open("r", encoding="utf-8", newline="") as file:
                predictions = list(csv.DictReader(file))
            if len(predictions) != 3510 or any(item["split"] != "validation" for item in predictions):
                raise ValueError(f"Bad prediction rows: {path}")
            y = np.asarray([int(item["y_true"]) for item in predictions])
            pred = np.asarray([int(item["y_pred"]) for item in predictions])
            maximum = max(maximum, abs(float(f1_score(y, pred, average="macro")) - row["per_seed"][SEEDS.index(seed)] / 100))
            total += len(predictions)
            count += 1
    return {"conditions": count, "rows": total, "max_macro_f1_error": maximum}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/improvements/e12b_fusion_ablation_corrected")
    args = parser.parse_args()
    root = Path(args.output_dir)
    conditions = []
    for tag in TAGS:
        for seed in SEEDS:
            conditions.extend(_run_seed(tag, seed, root))
    rows = _aggregate(conditions)
    audit = _audit(root, rows)
    summary = {
        "test_evaluated": False,
        "conditions": conditions,
        "aggregate": rows,
        "paired": [
            {"left": left, "right": right, "tag": tag, **_paired(rows, left, right, tag)}
            for left, right in (
                ("full", "tcn+dyn"),
                ("full", "tcn+md"),
                ("full", "dyn+md"),
                ("full", "tcn"),
                ("tcn+dyn", "tcn"),
                ("tcn+md", "tcn"),
            )
            for tag in TAGS
        ],
        "audit": audit,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "summary.json", summary)
    markdown = _markdown(rows, audit)
    (root / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
