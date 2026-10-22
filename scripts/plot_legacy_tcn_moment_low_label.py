"""Redraw the MOMENT/TCN curve from matched saved validation predictions.

No training, inference, test predictions, or remote access. The historical test
figure is left unchanged. E01 supplies the original frozen MOMENT + RBF-SVM
pipeline; N2/E05C supply the legacy-structure masked TCN's selected checkpoints.
There is no accepted local 20% TCN result in these suites: leave that point blank.

Run from the repository root, with numpy and matplotlib installed:
    PYTHONPATH=src python scripts/plot_legacy_tcn_moment_low_label.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

SEEDS = (42, 44, 45, 46)
FRACTIONS = (0.01, 0.05, 0.1, 0.2, 0.4)
TCN_FRACTIONS = (0.01, 0.05, 0.1, 0.4)
E01 = Path("artifacts/imports/e01_validation_20261017/runs")
N2 = Path("artifacts/imports/n2_tcn_low_label_20261020/n2_tcn_low_label_20261020_v1")
E05C = Path("artifacts/imports/e05c_tcn_replication_20261019/terminal_artifacts")
STEM = "few_shot_legacy_tcn_validation_20261021"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_validation(record: dict[str, Any]) -> None:
    if record.get("test_evaluated") is not False:
        raise ValueError("Expected an explicitly validation-only result")


def score_predictions(
    path: Path, expected_ids: list[str], truth: dict[str, int]
) -> tuple[float, int]:
    """Independently rebuild Macro-F1, checking ordered IDs and raw CSV labels."""
    matrix = [[0] * 3 for _ in range(3)]
    ids = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        moment = "true_index" in (reader.fieldnames or [])
        for row in reader:
            sample_id = row["sample_id"]
            y = int(row["true_index" if moment else "y_true"])
            pred = int(row["predicted_index" if moment else "y_pred"])
            if y not in range(3) or pred not in range(3) or truth[sample_id] != y:
                raise ValueError(f"Invalid class / raw truth mismatch: {path}")
            if moment:
                if int(row["true_label"]) != y or int(row["predicted_label"]) != pred:
                    raise ValueError(f"Label encoding mismatch: {path}")
            else:
                probs = [float(row[f"p{c}"]) for c in range(3)]
                if (
                    not all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
                    or abs(sum(probs) - 1) > 2e-7
                    or max(range(3), key=probs.__getitem__) != pred
                ):
                    raise ValueError(f"Invalid probability / argmax: {path}")
            ids.append(sample_id)
            matrix[y][pred] += 1
    if len(set(ids)) != len(ids) or ids != expected_ids:
        raise ValueError(f"Duplicate or mismatched validation IDs: {path}")
    f1 = []
    for c in range(3):
        denominator = sum(matrix[c]) + sum(row[c] for row in matrix)
        f1.append(2 * matrix[c][c] / denominator if denominator else 0.0)
    return statistics.mean(f1), len(ids)


def summarize(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float], dict[int, float]] = defaultdict(dict)
    for row in records:
        key = (row["model"], row["fraction"])
        if row["seed"] in grouped[key]:
            raise ValueError(f"Duplicate seed: {key}")
        grouped[key][row["seed"]] = row["macro_f1_percent"]
    aggregates = []
    for (model, fraction), scores in sorted(grouped.items()):
        if set(scores) != set(SEEDS):
            raise ValueError(f"Require the same four seeds: {model}/{fraction}")
        values = [scores[seed] for seed in SEEDS]
        aggregates.append({
            "model": model, "fraction": fraction, "n_seeds": len(values),
            "mean_percent": statistics.mean(values),
            "sample_sd_percent": statistics.stdev(values), "per_seed": scores,
        })
    paired = []
    for fraction in FRACTIONS:
        moment, tcn = grouped.get(("moment", fraction)), grouped.get(("tcn", fraction))
        if moment is None or tcn is None:
            continue
        differences = [moment[seed] - tcn[seed] for seed in SEEDS]
        paired.append({
            "fraction": fraction, "comparison": "MOMENT - legacy masked TCN",
            "mean_pp": statistics.mean(differences),
            "sample_sd_pp": statistics.stdev(differences),
            "moment_wins": sum(d > 0 for d in differences),
            "n_seeds": len(differences),
            "per_seed_pp": dict(zip(SEEDS, differences, strict=True)),
        })
    return aggregates, paired


def collect(root: Path) -> dict[str, Any]:
    import numpy as np

    hashes: dict[str, str] = {}

    def fingerprint(path: Path) -> str:
        key = str(path.relative_to(root))
        if key not in hashes:
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes[key]

    dataset = root / "data/raw/最新多.csv"
    dataset_sha = fingerprint(dataset)
    with dataset.open(encoding="utf-8-sig", newline="") as file:
        truth = {r["charging_station_id"]: int(r["InsertedColumn"])
                 for r in csv.DictReader(file, delimiter=" ")}
    records = []
    prediction_rows = 0
    max_error = 0.0
    for seed in SEEDS:
        role = "pilot" if seed == 42 else "expansion"
        run = root / E01 / f"e01_20261017_v2_{role}_seed{seed}"
        manifest_path = run / "split_manifest.json"
        manifest = read_json(manifest_path)
        fingerprint(manifest_path)
        expected_ids = manifest["splits"]["validation"]
        if len(expected_ids) != 3510 or manifest["protocol"]["dataset_sha256"] != dataset_sha:
            raise ValueError("Unexpected validation split or raw dataset")
        train_pool = set(manifest["splits"]["train"])
        if train_pool & set(expected_ids):
            raise ValueError("Train/validation overlap")
        for fraction in FRACTIONS:
            percent = round(fraction * 100)
            moment_dir = run / f"moment_paper_svm_fraction_{percent}"
            metadata_path = moment_dir / "metrics.json"
            metadata = read_json(metadata_path)
            require_validation(metadata)
            if (metadata["seed"] != seed or metadata["train_fraction"] != fraction
                    or metadata["model"] != "moment_paper_svm"
                    or metadata["dataset_sha256"] != dataset_sha):
                raise ValueError("MOMENT metadata mismatch")
            ids_path = run / f"train_ids_{percent}.npy"
            train_ids = np.load(ids_path, allow_pickle=False).astype(str).tolist()
            if (len(set(train_ids)) != len(train_ids)
                    or len(train_ids) != metadata["train_samples"]
                    or not set(train_ids) <= train_pool):
                raise ValueError("Invalid training subset")
            cases = [("moment", moment_dir / "validation_predictions.csv",
                      metadata_path, metadata["validation_metrics"]["macro_f1"], None)]
            fingerprint(ids_path)
            if fraction in TCN_FRACTIONS:
                suite = N2 if fraction < 0.1 else E05C
                tcn_dir = root / suite / f"seed{seed}" / f"fraction{percent:03d}_legacy_masked50"
                tcn_metadata_path = tcn_dir / "result.json"
                tcn = read_json(tcn_metadata_path)
                require_validation(tcn)
                if (tcn["condition"] != "legacy_masked50" or tcn["fraction"] != fraction
                        or tcn["epochs"] != 50
                        or tcn["train_count"] != len(train_ids)
                        or tcn["train_ids_sha256"] != metadata["train_ids_sha256"]):
                    raise ValueError("TCN condition / training subset fingerprint mismatch")
                tcn_ids = tcn_dir / "train_ids.npy"
                if np.load(tcn_ids, allow_pickle=False).astype(str).tolist() != train_ids:
                    raise ValueError("The two methods use different ordered training IDs")
                prediction = tcn_dir / "validation_selected.csv"
                if fingerprint(prediction) != tcn["output_sha256"][prediction.name]:
                    raise ValueError("TCN saved prediction hash mismatch")
                if fingerprint(tcn_ids) != tcn["output_sha256"][tcn_ids.name]:
                    raise ValueError("TCN saved training ID hash mismatch")
                cases.append(("tcn", prediction, tcn_metadata_path,
                              tcn["selected_macro_f1"], tcn["selected_epoch"]))
            for model, prediction, result_path, saved_f1, epoch in cases:
                f1, count = score_predictions(prediction, expected_ids, truth)
                error = abs(f1 - saved_f1)
                if error > 1e-12:
                    raise ValueError(f"Saved Macro-F1 mismatch: {prediction}")
                max_error = max(max_error, error)
                prediction_rows += count
                fingerprint(prediction)
                fingerprint(result_path)
                records.append({
                    "model": model, "seed": seed, "fraction": fraction,
                    "train_samples": len(train_ids), "validation_samples": count,
                    "macro_f1_percent": 100 * f1, "selected_epoch": epoch,
                    "predictions": str(prediction.relative_to(root)),
                    "metadata": str(result_path.relative_to(root)),
                })
    aggregates, paired = summarize(records)
    return {
        "split": "validation", "test_evaluated": False, "new_inference": False,
        "seeds": list(SEEDS), "fractions": list(FRACTIONS),
        "conditions_checked": len(records), "prediction_rows_checked": prediction_rows,
        "raw_truth_and_ordered_validation_ids_checked": True,
        "paired_ordered_training_ids_checked": True,
        "max_saved_macro_f1_abs_error": max_error,
        "missing_results": [{"model": "tcn", "fraction": 0.2, "seeds": list(SEEDS)}],
        "per_seed": records, "aggregate": aggregates, "paired": paired,
        "source_sha256": hashes,
        "limitations": [
            "Reused validation; overlapping random splits; not an independent test.",
            "Training-label budget only; validation and historical development labels are excluded.",
            "TCN is validation-selected within 50 epochs; MOMENT-SVM uses training-only CV.",
            "Legacy-structure PyTorch port, not a bitwise reproduction of the original Keras training.",
            "MOMENT is the original frozen representation + paper-aligned SVM, not the later fusion.",
            "E01 FP32 validation results are not the historical five-seed AMP test results.",
            "No 20% TCN score is invented or interpolated; no E18 training is launched.",
        ],
    }


def plot(report: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tcn_moment.paper_style import PAPER_COLORS, apply_paper_style

    apply_paper_style(plt)
    figure, axis = plt.subplots(figsize=(8.8, 5.8))
    x = [100 * f for f in FRACTIONS]
    styles = {
        "tcn": ("Legacy-structure TCN (masked, selected)", PAPER_COLORS["tcn"], "o"),
        "moment": ("Frozen MOMENT + RBF-SVM", PAPER_COLORS["moment_rbf"], "s"),
    }
    for model, (label, color, marker) in styles.items():
        by_fraction = {r["fraction"]: r for r in report["aggregate"] if r["model"] == model}
        means = [by_fraction[f]["mean_percent"] if f in by_fraction else math.nan for f in FRACTIONS]
        sds = [by_fraction[f]["sample_sd_percent"] if f in by_fraction else math.nan for f in FRACTIONS]
        # NaN intentionally breaks the TCN line at the missing 20% point.
        axis.errorbar(x, means, yerr=sds, label=label, color=color, marker=marker,
                      capsize=4, markersize=6, linewidth=2)
        for fraction, row in by_fraction.items():
            offset = (8, -3) if model == "tcn" and fraction == 0.01 else (0, 11)
            if model == "moment":
                offset = (0, -17) if fraction >= 0.05 else (0, 11)
            anchor_y = row["mean_percent"]
            if offset[1] > 0:
                anchor_y += row["sample_sd_percent"]
            elif model == "moment":
                anchor_y -= row["sample_sd_percent"]
            axis.annotate(f"{row['mean_percent']:.2f}", (100 * fraction, anchor_y),
                          textcoords="offset points", xytext=offset,
                          ha="left" if offset[0] else "center", fontsize=9, color=color)
    axis.set_xscale("log")
    axis.set_xticks(x, [f"{v:g}" for v in x])
    axis.set_xlim(0.8, 50)
    axis.set_ylim(30, 104)
    axis.set_yticks(range(30, 101, 10))
    axis.set_xlabel("Labeled training data (%)")
    axis.set_ylabel("Validation Macro-F1 (%)")
    axis.set_title("MOMENT vs legacy-structure TCN: training-label budget", pad=12)
    axis.text(20, 95, "20% TCN: no archived result", color="#777777", fontsize=8,
              ha="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    axis.legend(loc="lower right", frameon=True)
    axis.grid(True, which="major", alpha=0.25)
    axis.grid(False, which="minor")
    figure.text(0.5, 0.065,
                "Mean ± sample SD; matched seeds 42/44/45/46. Reused validation, not a new test.",
                ha="center", fontsize=8)
    figure.text(0.5, 0.039,
                "TCN: validation-selected within 50 epochs; MOMENT-SVM: training-only CV. "
                "Validation labels excluded from x-axis budget.", ha="center", fontsize=7.5)
    figure.text(0.5, 0.013,
                "The 20% TCN point is unavailable; the curve is intentionally broken. "
                "No interpolation or new training.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        figure.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight")
        print(f"Saved {output.with_suffix(suffix)}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("docs/figures") / STEM)
    parser.add_argument("--analysis", type=Path,
                        default=Path("artifacts/analysis/legacy_tcn_moment_curve_20261021"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    report = collect(root)
    analysis = root / args.analysis
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    with (analysis / "per_seed.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report["per_seed"][0]))
        writer.writeheader()
        writer.writerows(report["per_seed"])
    with (analysis / "aggregate.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ("model", "fraction", "n_seeds", "mean_percent", "sample_sd_percent")
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["aggregate"])
    plot(report, root / args.output)
    print(f"Verified {report['conditions_checked']} conditions / "
          f"{report['prediction_rows_checked']} prediction rows; max F1 error "
          f"{report['max_saved_macro_f1_abs_error']:.3g}")
    for row in report["paired"]:
        print(f"{100 * row['fraction']:g}%: MOMENT - TCN {row['mean_pp']:+.4f} pp; "
              f"MOMENT higher in {row['moment_wins']}/{row['n_seeds']} seeds")


if __name__ == "__main__":
    main()
