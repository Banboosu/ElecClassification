from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tcn_moment.io_utils import atomic_write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _clean_cell(value: str) -> str:
    return value.replace("**", "").strip()


def _table_rows(paper: str, caption: str) -> list[list[str]]:
    section = paper.split(caption, 1)[1]
    rows: list[list[str]] = []
    started = False
    for line in section.splitlines()[1:]:
        if line.startswith("|"):
            started = True
            cells = [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]
            if cells and not all(set(cell) <= {"-", ":"} for cell in cells):
                rows.append(cells)
        elif started:
            break
    return rows[1:]


def _mean_std(values: list[float], scale: float = 100.0) -> str:
    array = np.asarray(values, dtype=np.float64) * scale
    return f"{array.mean():.2f} ± {array.std(ddof=1):.2f}"


def _signed(value: float, suffix: str = " pp") -> str:
    return f"{value:+.2f}{suffix}".replace("-", "−")


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        header = file.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks: list[str] = []
        self.numeric_cells = 0

    def check(self, condition: bool, message: str) -> None:
        if condition:
            self.checks.append(message)
        else:
            self.failures.append(message)

    def cell(self, actual: str, expected: str, context: str) -> None:
        self.numeric_cells += 1
        self.check(actual == expected, f"{context}: expected '{expected}', got '{actual}'")


def _metric_values(
    paths: list[Path],
    metric: str,
    extractor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[float]:
    values = []
    for path in paths:
        record = _read_json(path)
        payload = extractor(record) if extractor else record["test_metrics"]
        values.append(float(payload[metric]))
    return values


def _audit_main_table(root: Path, paper: str, audit: Audit) -> None:
    rows = {row[0]: row for row in _table_rows(paper, "**表 2 ")}
    specs: list[tuple[str, list[Path], Callable[[dict[str, Any]], dict[str, Any]] | None]] = []
    baseline_paths = sorted(root.glob("artifacts/baselines/moment_thesis_baseline_v1_seed*/metrics.json"))
    for label, key in (
        ("多数类", "majority"),
        ("逻辑回归（统计特征）", "logistic_regression"),
        ("随机森林（统计特征）", "random_forest"),
    ):
        specs.append((label, baseline_paths, lambda value, key=key: value["results"][key]["test_metrics"]))
    specs.extend(
        [
            (
                "1D-CNN（逐序列 z-score）",
                sorted(root.glob("artifacts/cnn/cnn_baseline_thesis_cnn_v1_seed*/metrics.json")),
                None,
            ),
            (
                "TCN（逐序列 min-max）",
                sorted(root.glob("artifacts/tcn/normalization_minmax_thesis_tcn_norm_v2_seed*/metrics.json")),
                None,
            ),
            (
                "TCN（逐序列 z-score）",
                sorted(root.glob("artifacts/tcn/normalization_zscore_thesis_tcn_norm_v2_seed*/metrics.json")),
                None,
            ),
            (
                "TCN（原始尺度）",
                sorted(root.glob("artifacts/tcn/normalization_none_thesis_tcn_norm_v2_seed*/metrics.json")),
                None,
            ),
            (
                "MOMENT 线性探测",
                sorted(root.glob("artifacts/moment/moment_linear_probe_thesis_moment_strategy_v2_v100_seed*/metrics.json")),
                None,
            ),
            (
                "MOMENT 最后两层微调",
                sorted(root.glob("artifacts/moment/moment_partial_finetune_thesis_moment_strategy_v2_v100_seed*/metrics.json")),
                None,
            ),
            (
                "MOMENT 冻结表征 + RBF-SVM",
                sorted(root.glob("artifacts/moment_svm/moment_svm_rbf_paper_v1_seed*/metrics.json")),
                None,
            ),
            (
                "MOMENT 完全微调",
                sorted(root.glob("artifacts/moment/moment_full_finetune_thesis_moment_strategy_v2_v100_seed*/metrics.json")),
                None,
            ),
        ]
    )
    for label, paths, extractor in specs:
        audit.check(len(paths) == 5, f"Table 2 source count for {label}")
        row = rows.get(label)
        audit.check(row is not None, f"Table 2 row exists for {label}")
        if row is None or len(paths) != 5:
            continue
        for column, metric in enumerate(("accuracy", "balanced_accuracy", "macro_f1"), start=1):
            expected = _mean_std(_metric_values(paths, metric, extractor))
            audit.cell(row[column], expected, f"Table 2 {label} {metric}")


def _audit_few_shot(root: Path, paper: str, audit: Audit) -> None:
    rows = {row[0].split("（", 1)[0]: row for row in _table_rows(paper, "**表 5 ")}
    moment_paths = sorted(
        root.glob("artifacts/moment_svm_few_shot/moment_svm_thesis_few_shot_v1_seed*/metrics.json")
    )
    for tag, fraction in (("1%", 0.01), ("5%", 0.05), ("10%", 0.1), ("20%", 0.2), ("40%", 0.4)):
        tcn_paths = sorted(
            root.glob(f"artifacts/tcn_few_shot/tcn_{int(fraction * 100):02d}_percent_thesis_few_shot_v1_seed*/metrics.json")
        )
        tcn = _metric_values(tcn_paths, "macro_f1")
        moment = []
        for path in moment_paths:
            record = _read_json(path)
            match = next(item for item in record["fractions"] if np.isclose(item["train_fraction"], fraction))
            moment.append(float(match["test_metrics"]["macro_f1"]))
        row = rows[tag]
        audit.cell(row[1], _mean_std(tcn), f"Table 5 {tag} TCN")
        audit.cell(row[2], _mean_std(moment), f"Table 5 {tag} MOMENT")
        audit.cell(row[3], _signed(100 * (np.mean(moment) - np.mean(tcn))), f"Table 5 {tag} difference")


def _audit_m01(root: Path, paper: str, audit: Audit) -> None:
    condition = _read_csv(root / "artifacts/analysis/m01_pretraining_ablation/condition_summary.csv")
    paired = _read_csv(root / "artifacts/analysis/m01_pretraining_ablation/paired_summary.csv")
    rows = {row[0]: row for row in _table_rows(paper, "**表 6 ")}
    for tag, fraction in (("1%", 0.01), ("5%", 0.05), ("10%", 0.1)):
        selected = [row for row in condition if np.isclose(float(row["train_fraction"]), fraction)]
        values = {row["condition"]: row for row in selected}
        difference = next(row for row in paired if np.isclose(float(row["train_fraction"]), fraction))
        row = rows[tag]
        pretrained = values["pretrained"]
        random = values["random"]
        audit.cell(
            row[1],
            f"{100 * float(pretrained['mean']):.2f} ± {100 * float(pretrained['standard_deviation']):.2f}",
            f"Table 6 {tag} pretrained",
        )
        audit.cell(
            row[2],
            f"{100 * float(random['mean']):.2f} ± {100 * float(random['standard_deviation']):.2f}",
            f"Table 6 {tag} random",
        )
        audit.cell(row[3], _signed(100 * float(difference["mean"]), ""), f"Table 6 {tag} difference")
        interval = f"[{100 * float(difference['ci_95_lower']):+.2f}, {100 * float(difference['ci_95_upper']):+.2f}]"
        audit.cell(row[4], interval, f"Table 6 {tag} interval")


def _audit_m02(root: Path, paper: str, audit: Audit) -> None:
    length_rows = _read_csv(root / "artifacts/m02_error_analysis_20260805/length_macro_f1_aggregate.csv")
    paper_rows = {row[0]: row for row in _table_rows(paper, "**表 7 ")}
    models = {
        "TCN（100%）": "TCN_full",
        "MOMENT 完全微调（100%）": "MOMENT_full",
        "MOMENT-SVM（1%）": "MOMENT_SVM_1pct",
        "MOMENT-SVM（5%）": "MOMENT_SVM_5pct",
        "MOMENT-SVM（10%）": "MOMENT_SVM_10pct",
    }
    bins = ("<= 237", "238-332", "333-439", "> 439")
    for paper_name, model_key in models.items():
        source = {row["length_bin"]: row for row in length_rows if row["model_key"] == model_key}
        row = paper_rows[paper_name]
        for index, length_bin in enumerate(bins, start=1):
            item = source[length_bin]
            expected = (
                f"{100 * float(item['macro_f1_mean']):.2f} ± "
                f"{100 * float(item['macro_f1_std']):.2f}"
            )
            audit.cell(row[index], expected, f"Table 7 {paper_name} {length_bin}")

    confusion = _read_csv(root / "artifacts/m02_error_analysis_20260805/confusion_matrix_aggregate.csv")
    table = _table_rows(paper, "**表 8 ")
    model_names = {"TCN_full": "TCN", "MOMENT_full": "MOMENT 完全微调"}
    for model_key, paper_name in model_names.items():
        for true_label in ("0", "1", "2"):
            row = next(item for item in table if item[0] == paper_name and item[1] == true_label)
            for predicted in ("0", "1", "2"):
                item = next(
                    source
                    for source in confusion
                    if source["model_key"] == model_key
                    and source["true_label"] == true_label
                    and source["predicted_label"] == predicted
                )
                audit.cell(
                    row[2 + int(predicted)],
                    f"{100 * float(item['mean_row_fraction']):.2f}",
                    f"Table 8 {paper_name} {true_label}->{predicted}",
                )


def _audit_battery(root: Path, paper: str, audit: Audit) -> None:
    source_rows = _read_csv(root / "artifacts/battery_safety_thesis_v1/full_label_battery_summary.csv")
    table = _table_rows(paper, "**表 9 ")
    point_names = {"Argmax": "argmax", "Max-F2": "max_f2", "99% Recall 目标": "recall_0.99"}
    model_names = {"TCN": "TCN", "MOMENT 完全微调": "MOMENT_FULL_FINETUNE"}
    metrics = (
        "test_battery_recall",
        "test_battery_precision",
        "test_false_positive_rate",
        "test_f2",
    )
    for paper_point, source_point in point_names.items():
        for paper_model, source_model in model_names.items():
            row = next(item for item in table if item[0] == paper_point and item[1] == paper_model)
            source = next(
                item
                for item in source_rows
                if item["model"] == source_model and item["operating_point"] == source_point
            )
            for index, metric in enumerate(metrics, start=2):
                expected = (
                    f"{100 * float(source[f'{metric}_mean']):.2f} ± "
                    f"{100 * float(source[f'{metric}_std']):.2f}"
                )
                audit.cell(row[index], expected, f"Table 9 {paper_point} {paper_model} {metric}")
            counts = (
                f"{float(source['test_false_negatives_mean']):,.1f} / "
                f"{float(source['test_false_positives_mean']):,.1f}"
            )
            audit.cell(row[6], counts, f"Table 9 {paper_point} {paper_model} counts")


def _audit_transfer_tables(root: Path, paper: str, audit: Audit) -> None:
    retrieval = _read_csv(
        root / "artifacts/moment_retrieval/moment_retrieval_zero_shot_thesis_v1/summary.csv"
    )
    rows = {row[0]: row for row in _table_rows(paper, "**表 10 ")}
    methods = {
        "MOMENT": "moment",
        "原始曲线重采样": "raw_resampled",
        "人工统计特征": "statistical",
    }
    columns = (
        "macro_precision_at_1_mean",
        "macro_precision_at_10_mean",
        "map_at_10_mean",
        "mean_length_relative_error_at_10_mean",
    )
    for paper_name, method in methods.items():
        source = next(item for item in retrieval if item["condition"] == "clean" and item["method"] == method)
        for index, column in enumerate(columns, start=1):
            audit.cell(rows[paper_name][index], f"{100 * float(source[column]):.2f}", f"Table 10 {paper_name} {column}")

    length = _read_csv(
        root
        / "artifacts/moment_retrieval/moment_retrieval_zero_shot_thesis_v1/length_only_condition_metrics.csv"
    )
    for index, column in enumerate(("macro_precision_at_1", "macro_precision_at_10", "map_at_10"), start=1):
        values = [float(item[column]) for item in length]
        audit.cell(rows["仅长度"][index], _mean_std(values), f"Table 10 length only {column}")
    relative_error = 100 * np.mean([float(item["mean_length_relative_error_at_10"]) for item in length])
    audit.cell(rows["仅长度"][4], f"{relative_error:.2f}", "Table 10 length only relative error")

    imputation = _read_csv(
        root / "artifacts/moment_imputation/moment_imputation_zero_shot_thesis_v2/summary.csv"
    )
    table = _table_rows(paper, "**表 11 ")
    pattern_names = {"随机 patch": "random_patches", "连续区块": "contiguous_block"}
    method_order = ("linear", "forward_fill", "moment_zero_shot", "mean")
    for row in table:
        pattern = pattern_names[row[0]]
        rate = float(row[1].rstrip("%")) / 100
        for index, method in enumerate(method_order, start=2):
            source = next(
                item
                for item in imputation
                if item["pattern"] == pattern
                and item["method"] == method
                and np.isclose(float(item["mask_rate"]), rate)
            )
            audit.cell(row[index], f"{float(source['macro_nrmse_mean']):.4f}", f"Table 11 {pattern} {rate} {method}")


def _audit_structure(root: Path, paper: str, audit: Audit) -> dict[str, Any]:
    table_numbers = [int(value) for value in re.findall(r"\*\*表 (\d+) ", paper)]
    figure_matches = re.findall(r"!\[图 (\d+) [^]]*]\(([^)]+)\)", paper)
    figure_numbers = [int(number) for number, _ in figure_matches]
    audit.check(table_numbers == list(range(1, 13)), "Tables are numbered sequentially 1-12")
    audit.check(figure_numbers == list(range(1, 8)), "Figures are numbered sequentially 1-7")

    figure_inventory = []
    for number, relative in figure_matches:
        png_path = root / "docs" / relative
        svg_path = png_path.with_suffix(".svg")
        audit.check(png_path.is_file(), f"Figure {number} PNG exists")
        audit.check(svg_path.is_file(), f"Figure {number} SVG exists")
        if png_path.is_file():
            width, height = _png_dimensions(png_path)
            audit.check(width >= 1600 and height >= 900, f"Figure {number} has publication-scale raster resolution")
            figure_inventory.append(
                {
                    "number": int(number),
                    "png": str(png_path.relative_to(root)),
                    "svg": str(svg_path.relative_to(root)),
                    "width": width,
                    "height": height,
                }
            )

    forbidden = (
        "实验 E06 将",
        "必须完成 E02",
        "必须完成 E03",
        "必须完成 E04",
        "生产回放后才能成立",
        "当前投稿硬缺口",
    )
    for phrase in forbidden:
        audit.check(phrase not in paper, f"Manuscript excludes hard-gap phrase: {phrase}")
    for alias in ("TCN（none）", "TCN（minmax）", "TCN（z-score）"):
        audit.check(alias not in paper, f"Manuscript excludes non-canonical model alias: {alias}")

    paper_lines = paper.splitlines()
    placeholders = [line for line in paper_lines if "【待补" in line]
    metadata_labels = ("作者", "单位", "通信作者", "基金项目", "CRediT", "利益冲突", "致谢")

    def is_metadata_placeholder(index: int, line: str) -> bool:
        if any(label in line for label in metadata_labels):
            return True
        preceding_context = "\n".join(paper_lines[max(0, index - 2) : index])
        return line.strip() == "【待补】" and "## 致谢" in preceding_context

    audit.check(
        all(
            is_metadata_placeholder(index, line)
            for index, line in enumerate(paper_lines)
            if "【待补" in line
        ),
        "Remaining placeholders are submission metadata only",
    )
    audit.check("### 5.6 未来工作" in paper, "Future work is consolidated in section 5.6")
    audit.check("M04 已完成" in (root / "docs/paper_todo_student_20260805.md").read_text(encoding="utf-8"), "M04 completion is recorded in the student TODO")
    return {"table_numbers": table_numbers, "figure_inventory": figure_inventory, "metadata_placeholders": placeholders}


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# M04 manuscript audit",
        "",
        f"Overall status: **{result['status'].upper()}**.",
        "",
        f"- Verified numeric table cells: {result['numeric_cells']}.",
        f"- Total passed checks: {result['passed_check_count']}.",
        f"- Tables: {result['structure']['table_numbers']}.",
        f"- Figures with PNG/SVG pairs: {len(result['structure']['figure_inventory'])}.",
        "- Remaining manuscript placeholders are limited to author/submission metadata.",
        "- Unexecuted extensions are consolidated as future work and are not current evidence gaps.",
        "",
    ]
    if result["failures"]:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in result["failures"])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the final M04 manuscript against artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/m04_manuscript_audit_20260805"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    paper = (root / "docs/paper_draft_20260805.md").read_text(encoding="utf-8")
    audit = Audit()
    _audit_main_table(root, paper, audit)
    _audit_few_shot(root, paper, audit)
    _audit_m01(root, paper, audit)
    _audit_m02(root, paper, audit)
    _audit_battery(root, paper, audit)
    _audit_transfer_tables(root, paper, audit)
    structure = _audit_structure(root, paper, audit)
    result = {
        "audit": "M04_manuscript_finalization",
        "status": "passed" if not audit.failures else "failed",
        "numeric_cells": audit.numeric_cells,
        "passed_check_count": len(audit.checks),
        "structure": structure,
        "formatting_policy": {
            "classification_metrics": "percent with two decimals; mean +/- sample standard deviation",
            "figures": "sequential numbering; PNG and SVG; shared DejaVu Sans and color-blind-safe palette",
            "model_names": "canonical names in the M03/M04 records",
        },
        "claim_boundary": {
            "supported": [
                "low-label MOMENT label efficiency and stability under the tested pipelines",
                "pretrained-vs-random contribution under the fixed architecture and RBF-SVM",
                "full-label TCN training-side resource advantage",
            ],
            "future_work_only": [
                "additional strong representation baselines",
                "input-degradation robustness and probability calibration",
                "uniform online inference benchmarks",
                "production-compatible offline replay",
            ],
        },
        "failures": audit.failures,
    }
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "audit.json", result)
    (output_dir / "summary.md").write_text(_summary(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if audit.failures:
        raise SystemExit(f"M04 audit failed with {len(audit.failures)} issue(s).")


if __name__ == "__main__":
    main()
