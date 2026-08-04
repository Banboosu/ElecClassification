from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcn_moment.config import load_config
from tcn_moment.data import load_dataset
from tcn_moment.evaluate_retrieval import retrieval_metrics
from tcn_moment.io_utils import atomic_write_json


def _query_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def length_only_neighbors(
    gallery_lengths: np.ndarray,
    query_lengths: np.ndarray,
    query_ids: np.ndarray,
    *,
    k: int,
    tie_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank by length distance; break equal-distance ties without using labels."""
    groups = {
        int(length): np.flatnonzero(gallery_lengths == length)
        for length in np.unique(gallery_lengths)
    }
    available_lengths = np.asarray(sorted(groups), dtype=np.int64)
    indices = np.empty((len(query_lengths), k), dtype=np.int32)
    similarities = np.empty((len(query_lengths), k), dtype=np.float32)
    for row, (query_length, query_id) in enumerate(
        zip(query_lengths, query_ids, strict=True)
    ):
        distances = np.abs(available_lengths - int(query_length))
        selected: list[int] = []
        generator = np.random.default_rng(_query_seed(tie_seed, str(query_id)))
        for distance in np.unique(distances):
            tied_lengths = available_lengths[distances == distance]
            candidates = np.concatenate([groups[int(length)] for length in tied_lengths])
            candidates = generator.permutation(candidates)
            remaining = k - len(selected)
            selected.extend(candidates[:remaining].tolist())
            if len(selected) == k:
                break
        row_indices = np.asarray(selected, dtype=np.int32)
        indices[row] = row_indices
        relative_distance = np.abs(
            gallery_lengths[row_indices].astype(np.float64) - int(query_length)
        ) / max(int(query_length), 1)
        similarities[row] = (1.0 / (1.0 + relative_distance)).astype(np.float32)
    return indices, similarities


def _length_summary(
    split: str,
    lengths: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for label, class_name in enumerate(class_names):
        selected = lengths[labels == label].astype(np.float64)
        rows.append(
            {
                "split": split,
                "class_index": label,
                "class_label": class_name,
                "count": len(selected),
                "mean": float(selected.mean()),
                "sample_std": float(selected.std(ddof=1)),
                "minimum": int(selected.min()),
                "q25": float(np.quantile(selected, 0.25)),
                "median": float(np.median(selected)),
                "q75": float(np.quantile(selected, 0.75)),
                "maximum": int(selected.max()),
            }
        )
    return rows


def _relevance_by_query(
    neighbor_indices: np.ndarray,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    return (
        gallery_labels[neighbor_indices[:, :k]] == query_labels[:, None]
    ).mean(axis=1)


def stratified_bootstrap_macro_difference(
    first: np.ndarray,
    second: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 42,
    repeats: int = 2000,
) -> tuple[float, float, float]:
    differences = first - second
    classes = np.unique(labels)
    observed = float(np.mean([differences[labels == label].mean() for label in classes]))
    generator = np.random.default_rng(seed)
    bootstrap = np.zeros(repeats, dtype=np.float64)
    for label in classes:
        class_values = differences[labels == label]
        sampled = generator.integers(0, len(class_values), size=(repeats, len(class_values)))
        bootstrap += class_values[sampled].mean(axis=1) / len(classes)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return observed, float(lower), float(upper)


def _paired_comparisons(
    run_dir: Path,
    condition_metrics: pd.DataFrame,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    *,
    k: int,
) -> list[dict[str, Any]]:
    comparisons = []
    clean = np.load(run_dir / "neighbors_clean.npz")
    for baseline in ("raw_resampled", "statistical"):
        moment_values = _relevance_by_query(
            clean["moment_indices"], gallery_labels, query_labels, k
        )
        baseline_values = _relevance_by_query(
            clean[f"{baseline}_indices"], gallery_labels, query_labels, k
        )
        difference, lower, upper = stratified_bootstrap_macro_difference(
            moment_values, baseline_values, query_labels
        )
        comparisons.append(
            {
                "condition": "clean",
                "comparison": f"moment_minus_{baseline}",
                "metric": f"macro_precision_at_{k}",
                "paired_units": len(query_labels),
                "difference": difference,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "ci_method": "2000-repeat class-stratified query bootstrap",
            }
        )

    t_critical_df4 = 2.7764451051977987
    for condition in ("random_patches_rate0.4", "contiguous_block_rate0.4"):
        selected = condition_metrics[condition_metrics["condition"] == condition]
        for baseline in ("raw_resampled", "statistical"):
            moment = (
                selected[selected["method"] == "moment"]
                .sort_values("mask_seed")[f"macro_precision_at_{k}"]
                .to_numpy()
            )
            other = (
                selected[selected["method"] == baseline]
                .sort_values("mask_seed")[f"macro_precision_at_{k}"]
                .to_numpy()
            )
            differences = moment - other
            mean = float(differences.mean())
            standard_error = float(differences.std(ddof=1) / np.sqrt(len(differences)))
            comparisons.append(
                {
                    "condition": condition,
                    "comparison": f"moment_minus_{baseline}",
                    "metric": f"macro_precision_at_{k}",
                    "paired_units": len(differences),
                    "difference": mean,
                    "ci95_lower": mean - t_critical_df4 * standard_error,
                    "ci95_upper": mean + t_critical_df4 * standard_error,
                    "ci_method": "paired t interval across five mask seeds",
                }
            )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retrieval confounds and CIs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "artifacts/moment_retrieval/moment_retrieval_zero_shot_thesis_v1"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/moment_retrieval_zero_shot.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_dataset(config.data)
    run_dir = args.run_dir
    class_names = [str(value) for value in bundle.label_encoder.classes_.tolist()]
    gallery_lengths = np.minimum(bundle.lengths_train, config.data.max_length)
    query_lengths = np.minimum(bundle.lengths_test, config.data.max_length)
    k_values = config.retrieval.k_values
    max_k = max(k_values)

    length_rows = []
    for tie_seed in config.retrieval.mask_seeds:
        indices, scores = length_only_neighbors(
            gallery_lengths,
            query_lengths,
            bundle.ids_test,
            k=max_k,
            tie_seed=tie_seed,
        )
        metrics = retrieval_metrics(
            indices,
            scores,
            bundle.y_train,
            bundle.y_test,
            gallery_lengths,
            query_lengths,
            k_values,
        )
        length_rows.append({"tie_seed": tie_seed, **metrics})
    pd.DataFrame(length_rows).to_csv(
        run_dir / "length_only_condition_metrics.csv", index=False
    )

    length_summary = _length_summary(
        "gallery_train", gallery_lengths, bundle.y_train, class_names
    ) + _length_summary("query_test", query_lengths, bundle.y_test, class_names)
    pd.DataFrame(length_summary).to_csv(
        run_dir / "length_summary_by_class.csv", index=False
    )

    condition_metrics = pd.read_csv(run_dir / "condition_metrics.csv")
    comparisons = _paired_comparisons(
        run_dir,
        condition_metrics,
        bundle.y_train,
        bundle.y_test,
        k=max_k,
    )
    pd.DataFrame(comparisons).to_csv(run_dir / "paired_comparisons.csv", index=False)

    numeric_names = list(length_rows[0])
    length_aggregate = {}
    for name in numeric_names:
        if name == "tie_seed":
            continue
        values = [float(row[name]) for row in length_rows]
        length_aggregate[f"{name}_mean"] = statistics.mean(values)
        length_aggregate[f"{name}_sample_std"] = statistics.stdev(values)
    output = {
        "analysis_type": "post_hoc_length_confound_and_paired_uncertainty",
        "length_only_ranking_uses_labels": False,
        "length_only_tie_break_seeds": list(config.retrieval.mask_seeds),
        "length_only_summary": length_aggregate,
        "paired_comparisons": comparisons,
        "length_summary_by_class": length_summary,
    }
    atomic_write_json(run_dir / "derived_analysis.json", output)
    print(pd.DataFrame(length_rows)[
        ["tie_seed", f"macro_precision_at_{max_k}", f"precision_at_{max_k}"]
    ].to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"Saved derived analysis to {run_dir}")


if __name__ == "__main__":
    main()
