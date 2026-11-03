"""Verify every number cited in todo1102-report.md against the saved artifacts.

The report quotes ~200 numbers.  Each entry below names the artifact field it
should come from, so a re-run or a hand edit that drifts out of sync fails loudly
instead of being noticed by a reader.  Arithmetic identities (deltas, gaps) are
checked too, because those are the ones most easily mis-stated.

Usage:  PYTHONPATH=src .venv/bin/python scripts/todo1102_report_check.py
Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

A = Path("artifacts/analysis/todo1102_20261102")
H = Path("artifacts/analysis/todo1102_holdout")
TOL = 0.005

FAILS: list[str] = []
CHECKS = 0


def _load(path: Path) -> Any:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError(path)


def check(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    global CHECKS
    CHECKS += 1
    if abs(float(actual) - float(expected)) > tol:
        FAILS.append(f"{label}: artifact={actual} report={expected}")


def check_int(label: str, actual: int, expected: int) -> None:
    global CHECKS
    CHECKS += 1
    if int(actual) != int(expected):
        FAILS.append(f"{label}: artifact={actual} report={expected}")


def holdout() -> None:
    s = _load(H / "summary.json")
    b = s["by_tag"]
    table = {
        ("10p", "validation", "tcn"): 88.55,
        ("10p", "sealed", "tcn"): 89.41,
        ("10p", "validation", "dyn"): 92.49,
        ("10p", "sealed", "dyn"): 92.39,
        ("10p", "validation", "md"): 90.81,
        ("10p", "sealed", "md"): 89.92,
        ("10p", "validation", "tcn_dyn"): 93.73,
        ("10p", "sealed", "tcn_dyn"): 93.96,
        ("10p", "validation", "full"): 94.65,
        ("10p", "sealed", "full"): 94.81,
        ("40p", "validation", "tcn"): 97.41,
        ("40p", "sealed", "tcn"): 97.32,
        ("40p", "validation", "dyn"): 94.78,
        ("40p", "sealed", "dyn"): 94.95,
        ("40p", "validation", "md"): 92.18,
        ("40p", "sealed", "md"): 92.04,
        ("40p", "validation", "tcn_dyn"): 97.84,
        ("40p", "sealed", "tcn_dyn"): 97.68,
        ("40p", "validation", "full"): 97.86,
        ("40p", "sealed", "full"): 97.76,
    }
    for (tag, split, method), expected in table.items():
        check(f"holdout {tag} {split} {method}", b[tag][split][method]["macro_f1_mean"], expected)
    check("holdout 40p sealed full std", b["40p"]["sealed"]["full"]["macro_f1_std"], 0.10)
    check("holdout 10p sealed full std", b["10p"]["sealed"]["full"]["macro_f1_std"], 0.29)

    split = _load(H / "sealed_split.json")
    check_int("sealed rows", split["counts"]["sealed"], 3677)
    check_int("sealed clusters", split["sealed_clusters"], 618)
    for index, expected in zip(("0", "1", "2"), (1353, 925, 1399)):
        check_int(f"sealed class {index}", split["sealed_class_counts"][index], expected)

    unified = _load(Path("artifacts/splits/unified_split.json"))
    sealed_ids = set(split["sealed_ids"])
    hist_train_val = set(unified["splits"]["train"]) | set(unified["splits"]["validation"])
    hist_test = set(unified["splits"]["test"])
    check_int("sealed overlap train+val", len(sealed_ids & hist_train_val), 2924)
    check_int("sealed overlap test", len(sealed_ids & hist_test), 753)

    # TCN deltas quoted in 6.3
    check(
        "tcn 10p sealed-minus-val",
        b["10p"]["sealed"]["tcn"]["macro_f1_mean"] - b["10p"]["validation"]["tcn"]["macro_f1_mean"],
        0.87,
    )
    check(
        "tcn 40p sealed-minus-val",
        b["40p"]["sealed"]["tcn"]["macro_f1_mean"] - b["40p"]["validation"]["tcn"]["macro_f1_mean"],
        -0.09,
    )
    # full - tcn_dyn on sealed
    for tag, expected in (("10p", 0.84), ("40p", 0.07)):
        check(
            f"sealed full-minus-tcn_dyn {tag}",
            b[tag]["sealed"]["full"]["macro_f1_mean"] - b[tag]["sealed"]["tcn_dyn"]["macro_f1_mean"],
            expected,
        )


def cluster() -> None:
    c = _load(A / "cluster_audit/summary.json")
    for tag, locv, minimum in (("10p", 95.18, 93.60), ("40p", 98.14, 96.89)):
        check(f"cluster {tag} locv", c["by_tag"][tag]["leave-cluster-out"]["stack_mean"], locv)
        check(f"cluster {tag} locv min", c["by_tag"][tag]["leave-cluster-out"]["stack_min"], minimum)
    check("cluster 10p in-dist", c["by_tag"]["10p"]["in-distribution"]["stack_mean"], 95.14)
    check("cluster 40p in-dist", c["by_tag"]["40p"]["in-distribution"]["stack_mean"], 98.12)

    rows = _load(A / "cluster_audit/clusters.csv")
    for tag, distinct_below, row_below in (("10p", 20, 24), ("40p", 11, 12)):
        selected = [row for row in rows if row["tag"] == tag]
        check_int(f"cluster {tag} rows", len(selected), 121)
        check_int(
            f"cluster {tag} distinct clusters",
            len({row["cluster"] for row in selected}),
            71,
        )
        check_int(
            f"cluster {tag} below-90 distinct",
            len({row["cluster"] for row in selected if float(row["stack_acc"]) < 0.90}),
            distinct_below,
        )
        check_int(
            f"cluster {tag} below-90 rows",
            sum(1 for row in selected if float(row["stack_acc"]) < 0.90),
            row_below,
        )
        sizes = sorted(int(row["n_val"]) for row in selected)
        check_int(f"cluster {tag} median size", sizes[len(sizes) // 2], 13)


def feature_reduction() -> None:
    for path, tag, dim36, delta36, dim34, delta34 in (
        (A / "feature_reduction/summary.json", "40p", 94.98, -0.03, 94.95, -0.06),
        (A / "feature_reduction_10p/summary.json", "10p", 92.76, -0.05, 92.43, -0.38),
    ):
        s = _load(path)
        by_groups = {row["groups"]: row for row in s["aggregates"]}
        check(f"{tag} 36D mean", by_groups["length+trend+roughness+shape"]["mean_macro_f1"] * 100, dim36)
        check(f"{tag} 36D delta", by_groups["length+trend+roughness+shape"]["delta_vs_full_pp"], delta36)
        check(f"{tag} 34D mean", by_groups["trend+roughness+shape"]["mean_macro_f1"] * 100, dim34)
        check(f"{tag} 34D delta", by_groups["trend+roughness+shape"]["delta_vs_full_pp"], delta34)
    check("40p 60D mean", _load(A / "feature_reduction/summary.json")["baseline_full"]["mean_macro_f1"] * 100, 95.00)
    check("10p 60D mean", _load(A / "feature_reduction_10p/summary.json")["baseline_full"]["mean_macro_f1"] * 100, 92.81)


def dyn_augmentation() -> None:
    d = _load(A / "dyn_augmentation/summary.json")
    agg = {(row["tag"], row["arm"], row["noise_fraction"]): row for row in d["aggregates"]}
    table = {
        ("10p", "clean_trained", 0.0): 92.81,
        ("10p", "clean_trained", 0.005): 38.21,
        ("10p", "aug_trained", 0.005): 91.01,
        ("40p", "clean_trained", 0.0): 95.00,
        ("40p", "clean_trained", 0.005): 42.30,
        ("40p", "aug_trained", 0.005): 93.58,
    }
    for key, expected in table.items():
        check(f"4a {key}", agg[key]["macro_f1_mean_pct"], expected)
    paired = {(row["tag"], row["noise_fraction"]): row for row in d["paired"]}
    check("4a delta 10p 0.5", paired[("10p", 0.005)]["delta_mean_pp"], 52.79)
    check("4a delta 40p 0.5", paired[("40p", 0.005)]["delta_mean_pp"], 51.28)
    check("4a clean cost 10p", paired[("10p", 0.0)]["delta_mean_pp"], -1.68)
    check("4a clean cost 40p", paired[("40p", 0.0)]["delta_mean_pp"], -0.70)

    d36 = _load(A / "dyn_augmentation_36d/summary.json")
    paired36 = {(row["tag"], row["noise_fraction"]): row for row in d36["paired"]}
    check("4a36 delta 10p", paired36[("10p", 0.005)]["delta_mean_pp"], 55.10)
    check("4a36 delta 40p", paired36[("40p", 0.005)]["delta_mean_pp"], 50.91)


def fusion_augmentation() -> None:
    s = _load(A / "fusion_augmentation_merged/summary.json")
    agg = {
        (row["tag"], row["arm"], row["method"], row["noise_fraction"]): row["macro_f1_mean"]
        for row in s["aggregates"]
    }
    table = {
        ("10p", "clean", "full", 0.0): 94.84,
        ("10p", "clean", "full", 0.0025): 80.98,
        ("10p", "clean", "full", 0.005): 55.39,
        ("10p", "clean", "full", 0.01): 25.61,
        ("10p", "aug", "full", 0.0): 95.58,
        ("10p", "aug", "full", 0.0025): 94.03,
        ("10p", "aug", "full", 0.005): 94.46,
        ("10p", "aug", "full", 0.01): 45.97,
        ("40p", "clean", "full", 0.0): 97.94,
        ("40p", "clean", "full", 0.0025): 92.29,
        ("40p", "clean", "full", 0.005): 64.72,
        ("40p", "clean", "full", 0.01): 28.93,
        ("40p", "aug", "full", 0.0): 97.98,
        ("40p", "aug", "full", 0.0025): 97.00,
        ("40p", "aug", "full", 0.005): 97.47,
        ("40p", "aug", "full", 0.01): 49.30,
        ("10p", "clean", "dyn", 0.005): 34.66,
        ("10p", "aug", "dyn", 0.005): 91.25,
        ("40p", "clean", "dyn", 0.005): 41.18,
        ("40p", "aug", "dyn", 0.005): 93.77,
    }
    for key, expected in table.items():
        check(f"4b {key}", agg[key], expected)
    paired = {(row["tag"], row["method"], row["noise_fraction"]): row for row in s["paired"]}
    for tag, level, expected, seeds in (
        ("10p", 0.0, 0.74, 4),
        ("10p", 0.005, 39.07, 4),
        ("40p", 0.0, 0.04, 1),
        ("40p", 0.005, 32.76, 4),
        ("10p", 0.005, 56.59, 4),
    ):
        method = "dyn" if (tag, level) == ("10p", 0.005) and expected > 50 else "full"
        row = paired[(tag, method, level)]
        check(f"4b delta {tag} {method} {level}", row["delta_mean_pp"], expected, tol=0.02)
    check_int("4b 40p clean 0.5 seeds", paired[("40p", "full", 0.005)]["positive_seeds"], 4)


def fusion_dyn36() -> None:
    s = _load(A / "fusion_dyn36/summary.json")
    table = {
        ("10p", "full_60"): 95.09,
        ("10p", "full_36"): 95.29,
        ("10p", "tcn_dyn_60"): 94.26,
        ("10p", "tcn_dyn_36"): 94.33,
        ("40p", "full_60"): 98.11,
        ("40p", "full_36"): 98.12,
        ("40p", "tcn_dyn_60"): 97.93,
        ("40p", "tcn_dyn_36"): 97.94,
    }
    for (tag, variant), expected in table.items():
        check(f"6D {tag} {variant}", s["by_tag"][tag][variant]["macro_f1_mean"], expected)
    # 36 - 60, the convention used in the report
    for tag, full_delta, tcn_delta in (("10p", 0.19, 0.07), ("40p", 0.01, 0.01)):
        check(
            f"6D {tag} full delta 36-60",
            s["by_tag"][tag]["full_36"]["macro_f1_mean"] - s["by_tag"][tag]["full_60"]["macro_f1_mean"],
            full_delta,
        )
        check(
            f"6D {tag} tcn_dyn delta 36-60",
            s["by_tag"][tag]["tcn_dyn_36"]["macro_f1_mean"]
            - s["by_tag"][tag]["tcn_dyn_60"]["macro_f1_mean"],
            tcn_delta,
        )


def rocket() -> None:
    s = _load(A / "rocket_noise/summary.json")
    agg = {(row["method"], row["tag"], row["arm"], row["noise_fraction"]): row for row in s["aggregates"]}
    table = {
        ("multirocket", "10p", "clean", 0.0): 96.57,
        ("multirocket", "10p", "clean", 0.005): 62.29,
        ("multirocket", "40p", "clean", 0.0): 97.30,
        ("multirocket", "40p", "clean", 0.005): 64.96,
        ("minirocket", "10p", "clean", 0.005): 56.19,
        ("minirocket", "40p", "clean", 0.0): 93.45,
        ("minirocket", "40p", "clean", 0.005): 61.73,
        ("multirocket", "10p", "aug", 0.0): 92.84,
        ("multirocket", "10p", "aug", 0.005): 90.85,
    }
    for key, expected in table.items():
        check(f"2c {key}", agg[key]["macro_f1_mean"], expected)
    check(
        "2c multirocket 40p drop",
        agg[("multirocket", "40p", "clean", 0.0)]["macro_f1_mean"]
        - agg[("multirocket", "40p", "clean", 0.005)]["macro_f1_mean"],
        32.33,
        tol=0.02,
    )


def quantization() -> None:
    q = _load(A / "quantization/summary.json")
    check("3d tcn fp32", q["accuracy"]["fp32"]["tcn_macro_f1_mean"], 97.53)
    check("3d tcn onnx fp32", q["accuracy"]["onnx_fp32"]["tcn_macro_f1_mean"], 97.53)
    check("3d tcn int8", q["accuracy"]["onnx_int8"]["tcn_macro_f1_mean"], 49.34)
    check("3d fused int8", q["accuracy"]["onnx_int8"]["fused_macro_f1_mean"], 80.08)
    check("3d delta int8", q["accuracy"]["onnx_int8_minus_fp32_pp"]["mean"], -18.03)
    check("3d onnx drift", q["accuracy"]["onnx_fp32"]["max_abs_logit_drift"] * 1e4, 2.4, tol=0.1)

    rows = _load(A / "quantization/accuracy.csv")
    peaks = {
        row["seed"]: (float(row["fp32_max_abs_logit"]), float(row["int8_max_abs_logit"]))
        for row in rows
        if row["backend"] == "fp32"
    }
    for seed, (fp32_peak, int8_peak) in {
        "46": (300.74, 117.15),
        "42": (819.81, 153.79),
        "44": (1285.7, 778.98),
        "45": (1597.8, 522.9),
    }.items():
        check(f"3d peak fp32 seed{seed}", peaks[seed][0], fp32_peak, tol=0.02)
        check(f"3d peak int8 seed{seed}", peaks[seed][1], int8_peak, tol=0.02)

    latency = {
        (row["backend"], int(row["threads"]), int(row["batch"])): float(row["ms_per_sample"])
        for row in _load(A / "quantization/latency.csv")
    }
    table = {
        ("torch_cuda_fp32", 0, 32): 0.02,
        ("torch_cpu_fp32", 8, 32): 0.88,
        ("ort_cpu_fp32", 8, 32): 1.23,
        ("ort_cpu_int8", 8, 32): 0.99,
        ("torch_cpu_fp32", 1, 1): 4.02,
        ("ort_cpu_int8", 1, 1): 1.20,
        ("torch_cuda_fp32", 0, 1): 0.22,
        ("torch_cpu_fp32", 8, 1): 2.11,
    }
    for key, expected in table.items():
        check(f"3d latency {key}", latency[key], expected, tol=0.006)

    variants = _load(A / "quantization/int8_variants.json")
    found = {row["variant"]: row["tcn_macro_f1"] for row in variants["variants"]}
    for name, expected in {
        "quint8_entropy": 53.20,
        "quint8_percentile": 58.02,
        "qint8_symmetric": 53.20,
        "quint8_per_tensor": 53.87,
    }.items():
        check(f"3d variant {name}", found[name], expected)


def main() -> None:
    holdout()
    cluster()
    feature_reduction()
    dyn_augmentation()
    fusion_augmentation()
    fusion_dyn36()
    rocket()
    quantization()
    print(f"checked {CHECKS} values")
    if FAILS:
        print(f"{len(FAILS)} mismatch(es):")
        for line in FAILS:
            print("  -", line)
        raise SystemExit(1)
    print("all report numbers match the saved artifacts")


if __name__ == "__main__":
    main()
