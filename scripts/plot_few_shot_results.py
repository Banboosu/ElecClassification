from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tcn_moment.paper_style import PAPER_COLORS, apply_paper_style


def _load_results(root: Path) -> dict[str, dict[float, list[float]]]:
    results: dict[str, dict[float, list[float]]] = {
        "TCN": {},
        "Frozen MOMENT + RBF-SVM": {},
    }
    for path in sorted((root / "tcn_few_shot").glob("*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        fraction = float(record["data"]["train_subset"]["requested_fraction"])
        results["TCN"].setdefault(fraction, []).append(
            float(record["test_metrics"]["macro_f1"])
        )
    for path in sorted((root / "moment_svm_few_shot").glob("*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for fraction_record in record["fractions"]:
            fraction = float(fraction_record["train_fraction"])
            results["Frozen MOMENT + RBF-SVM"].setdefault(fraction, []).append(
                float(fraction_record["test_metrics"]["macro_f1"])
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the five-seed few-shot results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "artifacts/imports/few_shot_metrics_20260728/artifacts"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/few_shot_macro_f1_20260728"),
    )
    args = parser.parse_args()

    results = _load_results(args.input)
    fractions = sorted(results["TCN"])
    percentages = np.asarray(fractions) * 100

    apply_paper_style(plt)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    styles = {
        "TCN": {"color": PAPER_COLORS["tcn"], "marker": "o"},
        "Frozen MOMENT + RBF-SVM": {
            "color": PAPER_COLORS["moment_rbf"],
            "marker": "s",
        },
    }
    for model, values_by_fraction in results.items():
        values = [values_by_fraction[fraction] for fraction in fractions]
        means = np.asarray([np.mean(values_at_fraction) for values_at_fraction in values])
        standard_deviations = np.asarray(
            [np.std(values_at_fraction, ddof=1) for values_at_fraction in values]
        )
        axis.errorbar(
            percentages,
            means * 100,
            yerr=standard_deviations * 100,
            linewidth=2,
            capsize=4,
            markersize=6,
            label=model,
            **styles[model],
        )

    axis.set_xscale("log")
    axis.set_xticks(percentages, [f"{value:g}" for value in percentages])
    axis.set_xlabel("Labeled training data (%)")
    axis.set_ylabel("Test Macro-F1 (%)")
    axis.set_title("Label efficiency on charging-power classification (5 seeds)")
    axis.legend(frameon=True)
    axis.set_ylim(55, 97)
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved {args.output.with_suffix('.png')}")
    print(f"Saved {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
