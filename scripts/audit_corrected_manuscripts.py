"""Check corrected manuscript tables against audited validation metrics (no models)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tcn_moment.io_utils import atomic_write_json, sha256_file

PAPERS = (Path("docs/manuscript_journal_zh.md"), Path("docs/manuscript_journal_zh_full.md"))
NAMES = {
    "MOMENT 原序列": "moment_logit", "MOMENT 双流": "moment_diff_logit",
    "F15 + HGB": "f15_hgb", "DYN + HGB": "dyn_hgb", "Masked TCN": "tcn",
    "TCN + DYN": "tcn_dyn", "DYN + MOMENT/ΔP（分开概率支路）": "dyn_moment_diff",
    "完整融合（E18）": "full",
}
ABLATION = {
    "TCN": "tcn", "TCN + 同型元学习器": "tcn_meta", "DYN": "dyn", "MOMENT/ΔP": "md",
    "DYN + MOMENT/ΔP": "dyn+md", "TCN + MOMENT/ΔP": "tcn+md",
    "TCN + DYN": "tcn+dyn", "完整三路（本文）": "full",
}
STREAMS = {"原序列 $P$": "moment_logit", "差分 $\\Delta P$": "diff_logit",
           "双流 $P+\\Delta P$（本文）": "moment_diff_logit", "双流 + DYN 堆叠": "stack_moment_diff"}


def tables(text: str):
    block = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            block.append([v.strip().replace("**", "") for v in line.strip().strip("|").split("|")])
        elif block:
            if len(block) >= 3:
                yield block[0], block[2:]
            block = []


def check_cell(cell: str, expected: list[float], where: str) -> int:
    actual = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", cell)]
    if len(actual) != len(expected) or any(abs(a - b) > .005001 for a, b in zip(actual, expected, strict=True)):
        raise AssertionError(f"{where}: displayed {cell!r}, audited {expected}")
    return len(expected)


def run(source: Path, output: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["battery_label"] != "1" or data["charger_label"] != "2":
        raise ValueError("Corrected class-1 battery / class-2 charger summary required")
    lookup = {(r["suite"], r["tag"], r["method"]): r for r in data["aggregate"]}
    result = {"test_evaluated": False, "source_summary_sha256": sha256_file(source), "papers": []}
    for path in PAPERS:
        text = path.read_text(encoding="utf-8")
        checked = 0
        categories = set()
        for header, rows in tables(text):
            if header[0] == "方法" and "100%" in header:
                categories.add("E18_1_to_100_percent")
                for row in rows:
                    method = NAMES.get(row[0])
                    if row[0] == "TCN + MOMENT/ΔP（E18b联合MD对照）":
                        if row[-1] != "—":
                            raise AssertionError("100% E18b TCN+MD was not run")
                        continue
                    if method is None:
                        raise AssertionError(f"Unmapped E18 row: {row[0]}")
                    for tag, cell in zip(header[1:], row[1:], strict=True):
                        suite = "e18_100p" if tag == "100%" else "e18"
                        rec = lookup[(suite, tag.replace("%", "p"), method)]["macro_f1"]
                        checked += check_cell(cell, [100 * rec["mean"], 100 * rec["sd"]], f"{path}:{row[0]}:{tag}")
            elif header[0] == "支路组合" and "40% Macro-F1" in header:
                categories.add("E12b_class1")
                for row in rows:
                    method = ABLATION[row[0]]
                    for index, tag in ((1, "10p"), (2, "40p")):
                        rec = lookup[("e12b_corrected", tag, method)]["macro_f1"]
                        checked += check_cell(row[index], [100 * rec["mean"], 100 * rec["sd"]], f"{path}:E12b:{method}:{tag}")
                    for index, column in enumerate(header):
                        if "电池" not in column:
                            continue
                        tag = "10p" if column.startswith("10%") else "40p"
                        rec = lookup[("e12b_corrected", tag, method)]["battery"]
                        keys = ["recall", "fpr"] if "/" in column else (["recall"] if "Recall" in column else ["fpr"])
                        checked += check_cell(row[index], [100 * rec[k]["mean"] for k in keys], f"{path}:E12b:{method}:{column}")
            elif header[0] == "输入流" and any("电池" in h for h in header):
                categories.add("E09_class1_and_class2")
                for row in rows:
                    method = STREAMS[row[0]]
                    for index, column in enumerate(header):
                        cls = "battery" if "电池" in column else "charger" if "充电器" in column else None
                        if cls:
                            rec = lookup[("e09", "40p", method)][cls]
                            checked += check_cell(row[index], [100 * rec[k]["mean"] for k in ("recall", "fpr")], f"{path}:E09:{method}:{cls}")
        if categories != {"E18_1_to_100_percent", "E12b_class1", "E09_class1_and_class2"}:
            raise AssertionError(f"Expected corrected tables missing: {path}: {categories}")
        if "电池异常（1）" not in text or "充电器故障（2）" not in text:
            raise AssertionError(f"Explicit numeric class mapping missing: {path}")
        images = re.findall(r'<img src="([^"]+)"', text)
        for image in images:
            if not (path.parent / image).exists():
                raise AssertionError(f"Missing manuscript image: {path}: {image}")
        result["papers"].append({"path": str(path), "sha256": sha256_file(path),
                                 "checked_numeric_values": checked, "checked_table_groups": sorted(categories),
                                 "image_links_exist": len(images)})
    atomic_write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/analysis/label_semantics_correction_20261029_v2/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/label_semantics_manuscript_check_20261029.json"))
    args = parser.parse_args()
    run(args.input, args.output)
