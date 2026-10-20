"""Render the E08 validation summary, paired comparisons, and audit.

The script never reads the test split. It only consumes saved validation
predictions and metrics produced by ``tcn_moment.run_e08``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tcn_moment.e08_analysis import (
    aggregate_conditions,
    audit_condition,
    evaluate_gates,
    paired_differences,
)

DISPLAY = {
    "f15_hgb": "F15 + HGB",
    "dyn_hgb": "DYN + HGB",
    "f15_svm": "F15 + RBF-SVM",
    "dyn_svm": "DYN + RBF-SVM",
    "moment_logit": "MOMENT + logistic",
    "stack": "DYN + MOMENT stack",
}
BUDGET_ORDER = ["1p", "5p", "10p", "40p", "100p"]
METHOD_ORDER = ["f15_hgb", "dyn_hgb", "f15_svm", "dyn_svm", "moment_logit", "stack"]
PAIR_ORDER = [
    ("dyn_hgb", "f15_hgb"),
    ("dyn_svm", "f15_svm"),
    ("stack", "dyn_hgb"),
    ("stack", "f15_hgb"),
    ("f15_svm", "f15_hgb"),
]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _find_seed_results(suite_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    results = []
    for path in sorted(suite_dir.glob("seed*/seed_results.json")):
        with path.open("r", encoding="utf-8") as file:
            results.append((path.parent, json.load(file)))
    if not results:
        raise FileNotFoundError(f"No seed_results.json under {suite_dir}")
    return results


def _aggregate_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["method"], row["tag"]): row for row in rows}


def _format(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _markdown(
    *,
    suite_dir: Path,
    seeds: list[int],
    aggregate: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    ablation: list[dict[str, Any]],
    importance: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    tcn_context: dict[str, float] | None,
) -> str:
    lookup = _aggregate_lookup(aggregate)
    lines: list[str] = []
    lines.append("# E08 strong dynamics features and MOMENT stacking")
    lines.append("")
    lines.append(f"- Suite: `{suite_dir}`")
    lines.append(f"- Seeds: {', '.join(str(seed) for seed in seeds)}")
    lines.append("- Split: validation only; no test split was constructed or evaluated.")
    lines.append("- Uncertainty: mean and sample SD across seeds; paired intervals are descriptive only.")
    lines.append("")

    lines.append("## Main validation table (Macro-F1 %)")
    lines.append("")
    header = "| Method | " + " | ".join(
        {"1p": "1%", "5p": "5%", "10p": "10%", "40p": "40%", "100p": "100%"}[tag]
        for tag in BUDGET_ORDER
    ) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(BUDGET_ORDER))
    for method in METHOD_ORDER:
        cells = []
        for tag in BUDGET_ORDER:
            row = lookup.get((method, tag))
            cells.append("—" if row is None else f"{_format(row['macro_f1_mean'])} ± {_format(row['macro_f1_sd'])}")
        lines.append(f"| {DISPLAY[method]} | " + " | ".join(cells) + " |")
    if tcn_context is not None:
        cells = ["—"] * len(BUDGET_ORDER)
        cells[3] = f"{_format(tcn_context['macro_f1_mean'])} ± {_format(tcn_context['macro_f1_sd'])}"
        lines.append("| E05C legacy masked TCN (context) | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Paired validation differences (pp)")
    lines.append("")
    lines.append("| Comparison | Budget | Mean ± SD | Positive seeds | Descriptive 95% interval |")
    lines.append("|---|---:|---:|---:|---|")
    ordered_pairs = sorted(
        paired,
        key=lambda row: (
            PAIR_ORDER.index((row["left"], row["right"]))
            if (row["left"], row["right"]) in PAIR_ORDER
            else 99,
            BUDGET_ORDER.index(row["tag"]) if row["tag"] in BUDGET_ORDER else 99,
        ),
    )
    for row in ordered_pairs:
        if (row["left"], row["right"]) not in PAIR_ORDER:
            continue
        interval = row["descriptive_ci95"]
        interval_text = "—" if interval is None else f"[{_format(interval[0])}, {_format(interval[1])}]"
        lines.append(
            f"| {DISPLAY[row['left']]} − {DISPLAY[row['right']]} | {row['tag']} | "
            f"{_format(row['mean'])} ± {_format(row['sample_sd'])} | "
            f"{row['positive_seeds']}/{row['n']} | {interval_text} |"
        )
    lines.append("")

    lines.append("## Frozen gates")
    lines.append("")
    lines.append("| Gate | Comparison | Mean gain (pp) | Required | Positive seeds | Status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for gate in gates:
        mean_gain = gate.get("mean_gain_pp")
        required = gate.get("required_mean_pp")
        positive = gate.get("positive_seeds", "—")
        required_positive = gate.get("required_positive_seeds", "—")
        lines.append(
            f"| {gate['id']} | {gate.get('comparison', '—')} | "
            f"{'—' if mean_gain is None else _format(mean_gain)} | "
            f"{'—' if required is None else _format(required, 1)} | "
            f"{positive}/{required_positive} | "
            f"**{gate['status']}** |"
        )
    lines.append("")

    lines.append("## Battery anomaly at 40% labels (default argmax)")
    lines.append("")
    lines.append("| Method | Recall % | FPR % | Average precision % |")
    lines.append("|---|---:|---:|---:|")
    for method in METHOD_ORDER:
        row = lookup.get((method, "40p"))
        if row is None:
            continue
        lines.append(
            f"| {DISPLAY[method]} | {_format(row['battery_recall_mean'])} | "
            f"{_format(row['battery_fpr_mean'])} | {_format(row['battery_ap_mean'])} |"
        )
    lines.append("")

    lines.append("## Fit cost (median seconds per condition)")
    lines.append("")
    lines.append("| Method | 10% | 40% | 100% |")
    lines.append("|---|---:|---:|---:|")
    for method in METHOD_ORDER:
        cells = []
        for tag in ("10p", "40p", "100p"):
            row = lookup.get((method, tag))
            cells.append("—" if row is None else _format(row["fit_seconds_median"], 1))
        lines.append(f"| {DISPLAY[method]} | " + " | ".join(cells) + " |")
    lines.append("")

    if ablation:
        lines.append("## Feature-group ablation at 40% labels (HGB, mean over seeds)")
        lines.append("")
        lines.append("| Group | Features | Drop-group F1 % | Drop delta pp | Group-only F1 % |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in ablation:
            lines.append(
                f"| {row['group']} | {row['n_features']} | {_format(row['drop_group_f1'])} | "
                f"{_format(row['drop_delta_pp'])} | {_format(row['only_group_f1'])} |"
            )
        lines.append("")

    if importance:
        lines.append("## Permutation importance at 40% labels (top 15, mean over seeds)")
        lines.append("")
        lines.append("| Feature | Mean Macro-F1 drop |")
        lines.append("|---|---:|")
        for row in importance[:15]:
            lines.append(f"| `{row['feature']}` | {_format(row['mean_drop'])} |")
        lines.append("")

    if audits:
        maximum = max(row["absolute_error"] for row in audits)
        lines.append("## Prediction audit")
        lines.append("")
        lines.append(f"- Conditions audited: {len(audits)}")
        lines.append(f"- Maximum recomputed-vs-recorded Macro-F1 error: {maximum:.3e}")
        lines.append("")

    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append(
        "All comparisons use one reused validation split per seed. The seeds share overlapping "
        "training data; the paired intervals are descriptive, not independent external inference. "
        "Test was not evaluated. The feature set was frozen from an earlier exploratory probe on "
        "seed42; this run is a pre-registered confirmation on the remaining seeds and budgets."
    )
    lines.append("")
    return "\n".join(lines)


def _plot(suite_dir: Path, aggregate: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    from tcn_moment.paper_style import apply_paper_style

    apply_paper_style(plt)
    lookup = _aggregate_lookup(aggregate)
    colors = {
        "f15_hgb": "#999999",
        "dyn_hgb": "#009E73",
        "dyn_svm": "#56B4E9",
        "moment_logit": "#CC79A7",
        "stack": "#D55E00",
    }
    fractions = [1, 5, 10, 40, 100]
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for method in ("f15_hgb", "dyn_hgb", "dyn_svm", "moment_logit", "stack"):
        tags = ["1p", "5p", "10p", "40p", "100p"]
        means, errors, xs = [], [], []
        for x, tag in zip(fractions, tags):
            row = lookup.get((method, tag))
            if row is None:
                continue
            xs.append(x)
            means.append(row["macro_f1_mean"])
            errors.append(row["macro_f1_sd"])
        if not xs:
            continue
        axis.errorbar(
            xs,
            means,
            yerr=errors,
            marker="o",
            capsize=3,
            label=DISPLAY[method],
            color=colors[method],
        )
    axis.set_xscale("log")
    axis.set_xticks(fractions, [f"{value}%" for value in fractions])
    axis.set_xlabel("Training labels (% of training pool)")
    axis.set_ylabel("Validation Macro-F1 (%)")
    axis.set_title("E08: dynamics features and frozen MOMENT")
    axis.legend(loc="lower right", frameon=False)
    outputs = []
    for suffix in ("png", "svg"):
        path = suite_dir / f"e08_macro_f1_vs_budget.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-dir",
        default="artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1",
    )
    parser.add_argument("--protocol", default="configs/experiments/improvements/e08_strong_features_protocol.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tcn-mean", type=float, default=None)
    parser.add_argument("--tcn-sd", type=float, default=None)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir)
    output_dir = Path(args.output_dir) if args.output_dir else suite_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.protocol).open("r", encoding="utf-8") as file:
        protocol = json.load(file)

    seed_results = _find_seed_results(suite_dir)
    results = [result for _, result in seed_results]
    seeds = [int(result["seed"]) for result in results]
    aggregate = aggregate_conditions(results)
    paired = paired_differences(results)
    gates = evaluate_gates(protocol, paired)

    audits: list[dict[str, Any]] = []
    for seed_dir, result in seed_results:
        with np.load(seed_dir / "features.npz", allow_pickle=False) as archive:
            expected_ids = set(archive["validation_ids"].astype(str))
        for record in result["conditions"]:
            condition_dir = seed_dir / "conditions" / record["tag"] / record["method"]
            audits.append(
                audit_condition(
                    predictions_path=condition_dir / "predictions.csv",
                    metrics_path=condition_dir / "metrics.json",
                    expected_ids=expected_ids,
                )
            )

    ablation_rows: list[dict[str, Any]] = []
    group_names = sorted({row["group"] for result in results for row in result.get("group_ablation_40p", [])})
    for group in group_names:
        rows = [
            row
            for result in results
            for row in result.get("group_ablation_40p", [])
            if row["group"] == group
        ]
        if not rows:
            continue
        ablation_rows.append(
            {
                "group": group,
                "n_features": rows[0]["n_features"],
                "drop_group_f1": float(np.mean([row["drop_group_macro_f1"] * 100 for row in rows])),
                "drop_delta_pp": float(np.mean([row["drop_delta_pp"] for row in rows])),
                "only_group_f1": float(np.mean([row["only_group_macro_f1"] * 100 for row in rows])),
            }
        )

    importance_rows: list[dict[str, Any]] = []
    feature_names = sorted(
        {row["feature"] for result in results for row in result.get("permutation_importance_40p", {}).get("top20", [])}
    )
    for feature in feature_names:
        values = [
            row["mean_drop"]
            for result in results
            for row in result.get("permutation_importance_40p", {}).get("top20", [])
            if row["feature"] == feature
        ]
        if values:
            importance_rows.append({"feature": feature, "mean_drop": float(np.mean(values))})
    importance_rows.sort(key=lambda row: -row["mean_drop"])

    tcn_context = None
    if args.tcn_mean is not None and args.tcn_sd is not None:
        tcn_context = {"macro_f1_mean": args.tcn_mean, "macro_f1_sd": args.tcn_sd}

    aggregate_columns = [
        "method",
        "tag",
        "fraction",
        "n_seeds",
        "n_train",
        "macro_f1_mean",
        "macro_f1_sd",
        "accuracy_mean",
        "battery_recall_mean",
        "battery_fpr_mean",
        "battery_ap_mean",
        "fit_seconds_median",
    ]
    _write_csv(output_dir / "aggregate.csv", aggregate, aggregate_columns)
    _write_csv(
        output_dir / "paired.csv",
        paired,
        ["left", "right", "tag", "n", "mean", "sample_sd", "positive_seeds", "per_seed", "descriptive_ci95"],
    )

    summary = {
        "suite_dir": str(suite_dir),
        "seeds": seeds,
        "protocol_sha256": next((result.get("protocol_sha256") for result in results if result), None),
        "test_evaluated": False,
        "aggregate": aggregate,
        "paired": paired,
        "gates": gates,
        "group_ablation_40p": ablation_rows,
        "permutation_importance_40p": importance_rows,
        "audit": {
            "conditions": len(audits),
            "max_absolute_macro_f1_error": max(row["absolute_error"] for row in audits),
        },
        "tcn_context": tcn_context,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")
    markdown = _markdown(
        suite_dir=suite_dir,
        seeds=seeds,
        aggregate=aggregate,
        paired=paired,
        gates=gates,
        ablation=ablation_rows,
        importance=importance_rows,
        audits=audits,
        tcn_context=tcn_context,
    )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    figures = [] if args.no_figure else _plot(suite_dir, aggregate)
    print(markdown)
    if figures:
        print("Figures:")
        for figure in figures:
            print(f"- {figure}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
