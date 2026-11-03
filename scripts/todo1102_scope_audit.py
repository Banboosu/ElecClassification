"""TODO1102-6C: scope audit of Macro-F1 numbers in the IOP LaTeX manuscript.

Every percentage that looks like a Macro-F1-scale result (>= 60%) is matched
against a canonical scope ledger built from audited artifacts.  The audit reports
which numbers are unambiguous (single scope), which are reused across scopes and
therefore need an explicit budget/partition label at every mention, and which
numbers cannot be traced to a frozen source at all.

The ledger is generated once into ``configs/audits/todo1102_claim_scopes.json``
and reused, so the audit is reproducible and reviewable.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MANUSCRIPT = Path("docs/ioplatextemplate/manuscript_iop_zh.tex")
DEFAULT_LEDGER = Path("configs/audits/todo1102_claim_scopes.json")
DEFAULT_OUTPUT = Path("artifacts/analysis/todo1102_20261102/scope_audit")
E12B_SUMMARY = Path("artifacts/analysis/label_semantics_correction_20261029_v2/summary.json")
E05C_LEGACY = {
    "93.68": "E05C validation-selected TCN at 10% labels",
    "97.78": "E05C validation-selected TCN at 40% labels",
    "98.40": "E05C validation-selected TCN at 100% labels",
}
E05C_PROVENANCE = "docs/experiment_records/e05c_tcn_replication_results_20261019.md"

# Phase-1 numbers live in a record document rather than a single summary file;
# they are frozen here with their provenance so the audit can still trace them.
PHASE1_TEST = {
    "69.46": "frozen MOMENT + default pooling (test, 5 seeds)",
    "69.53": "frozen MOMENT + default pooling accuracy (test, 5 seeds)",
    "77.41": "frozen MOMENT + masked pooling linear probe (test, 5 seeds)",
    "77.02": "frozen MOMENT + masked pooling accuracy (test, 5 seeds)",
    "81.50": "MOMENT last-2-layer unfreeze (test, 5 seeds)",
    "81.08": "MOMENT last-2-layer unfreeze accuracy (test, 5 seeds)",
    "85.17": "frozen MOMENT + RBF-SVM (test, 5 seeds)",
    "84.77": "frozen MOMENT + RBF-SVM accuracy (test, 5 seeds)",
    "95.37": "MOMENT full fine-tuning (test, 5 seeds, 100% labels)",
    "95.26": "MOMENT full fine-tuning accuracy (test, 5 seeds)",
    "95.99": "raw-scale TCN baseline (test, 5 seeds, 100% labels)",
    "95.94": "raw-scale TCN baseline accuracy (test, 5 seeds)",
    "92.00": "TCN with per-series min-max (test)",
    "88.89": "F15 + random forest (test)",
    "88.85": "F15 + random forest accuracy (test)",
}
PHASE1_PROVENANCE = "docs/experiment_records/v100_results_20260720.md"


def _round_key(value: float) -> str:
    return f"{round(value, 2):.2f}"


def build_ledger() -> dict[str, Any]:
    ledger: dict[str, list[dict[str, str]]] = {}

    def add(value: str, scope: str, source: str) -> None:
        ledger.setdefault(value, [])
        entry = {"scope": scope, "source": source}
        if entry not in ledger[value]:
            ledger[value].append(entry)

    for value, label in PHASE1_TEST.items():
        add(value, "phase1_test", f"{PHASE1_PROVENANCE}: {label}")
    for value, label in E05C_LEGACY.items():
        add(value, "e05c_validation_selected", f"{E05C_PROVENANCE}: {label}")

    e12b = json.loads(E12B_SUMMARY.read_text(encoding="utf-8"))
    for record in e12b["aggregate"]:
        suite = record["suite"]
        tag = record["tag"]
        method = record["method"]
        macro = record["macro_f1"]["mean"] * 100.0
        add(_round_key(macro), f"{suite}_{tag}", f"{E12B_SUMMARY}: {method}/{tag}/macro_f1")
        for cls_key in ("battery", "charger"):
            block = record.get(cls_key)
            if not isinstance(block, dict):
                continue
            for metric in ("recall", "fpr"):
                if metric in block:
                    add(
                        _round_key(block[metric]["mean"] * 100.0),
                        f"{suite}_{tag}_{cls_key}",
                        f"{E12B_SUMMARY}: {method}/{tag}/{cls_key}/{metric}",
                    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase1_provenance": PHASE1_PROVENANCE,
        "sources": [str(E12B_SUMMARY), E05C_PROVENANCE],
        "scopes": ledger,
    }


def _scan(path: Path) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    pattern = re.compile(r"(?<![\d.])(\d{2,3}\.\d{2})(?![\d])")
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        for match in pattern.finditer(line):
            value = float(match.group(1))
            if value < 60.0 or value > 100.0:
                continue
            occurrences.append(
                {
                    "line": number,
                    "value": match.group(1),
                    "context": line[max(0, match.start() - 60) : match.end() + 60],
                }
            )
    return occurrences


def run(manuscript: Path, ledger_path: Path, output: Path, rebuild_ledger: bool) -> None:
    if rebuild_ledger or not ledger_path.exists():
        ledger = build_ledger()
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    else:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    occurrences = _scan(manuscript)
    scopes = ledger["scopes"]
    unmatched: dict[str, list[dict[str, Any]]] = {}
    ambiguous: dict[str, list[dict[str, Any]]] = {}
    by_value: dict[str, list[dict[str, Any]]] = {}
    for occurrence in occurrences:
        by_value.setdefault(occurrence["value"], []).append(occurrence)
        if occurrence["value"] not in scopes:
            unmatched.setdefault(occurrence["value"], []).append(occurrence)
        elif len(scopes[occurrence["value"]]) > 1:
            ambiguous.setdefault(occurrence["value"], []).append(occurrence)

    output.mkdir(parents=True, exist_ok=True)
    (output / "ledger.json").write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manuscript": str(manuscript),
        "ledger": str(ledger_path),
        "n_occurrences": len(occurrences),
        "n_distinct_values": len(by_value),
        "unmatched": {
            value: {"occurrences": rows, "note": "no traced source value"}
            for value, rows in sorted(unmatched.items())
        },
        "ambiguous": {
            value: {"scopes": scopes[value], "occurrences": rows}
            for value, rows in sorted(ambiguous.items())
        },
        "values": {value: rows for value, rows in sorted(by_value.items())},
    }
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# TODO1102-6C Macro-F1 scope audit",
        "",
        f"- Manuscript: `{manuscript}`",
        f"- Occurrences (>=60%): {len(occurrences)} across {len(by_value)} distinct values",
        f"- Untraceable values: {len(unmatched)}",
        f"- Values reused across scopes: {len(ambiguous)}",
        "",
        "## Values reused across more than one scope",
        "",
        "| Value | Scopes | Lines |",
        "|---|---|---|",
    ]
    for value, rows in sorted(ambiguous.items()):
        scope_text = "; ".join(entry["scope"] for entry in scopes[value])
        line_text = ", ".join(str(row["line"]) for row in rows)
        lines.append(f"| {value} | {scope_text} | {line_text} |")
    lines += ["", "## Untraceable values", "", "| Value | Lines |", "|---|---|"]
    for value, rows in sorted(unmatched.items()):
        lines.append(f"| {value} | {', '.join(str(row['line']) for row in rows)} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"occurrences={len(occurrences)} distinct={len(by_value)}")
    print(f"unmatched={sorted(unmatched)}")
    print(f"ambiguous={sorted(ambiguous)}")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-ledger", action="store_true")
    args = parser.parse_args()
    run(args.manuscript, args.ledger, args.output, args.rebuild_ledger)
