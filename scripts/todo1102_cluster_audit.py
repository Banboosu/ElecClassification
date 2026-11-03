"""TODO1102-1B: leave-cluster-out stability of the stacking layer.

Station IDs carry a deployment prefix (first six characters).  This audit asks
whether the fitted meta-learner still behaves when it is evaluated on deployment
clusters it has never been fitted on, and how much performance varies across
clusters.  Branch models are the frozen E08/E09/E12 artifacts, so only the
stacking layer is refitted; the encoders are not retrained (as specified in
`todo1102.md` §1b).

Limitation stated up front: the TCN/DYN/MOMENT branch models were trained on all
clusters, so this isolates the stacking layer's cross-cluster behaviour, not the
encoders'.  Results are descriptive; test is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

SEEDS = (42, 44, 45, 46)
TAGS = ("10p", "40p")
E08 = Path("artifacts/improvements/e08_strong_features/e08_strong_features_20261020_v1")
E09 = Path("artifacts/improvements/e09_diff_stream")
E12 = {
    "10p": Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_10p_20261020_v1"),
    "40p": Path("artifacts/improvements/e12_tcn_stack/e12_tcn_stack_20261020_v1"),
}
DEFAULT_OUTPUT = Path("artifacts/analysis/todo1102_20261102/cluster_audit")
BRANCHES = ("dyn", "md", "tcn")
CLUSTER_PREFIX = 6


def _read_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = [row["sample_id"] for row in rows]
    y = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    probs = np.asarray(
        [[float(row["score_0"]), float(row["score_1"]), float(row["score_2"])] for row in rows],
        dtype=np.float64,
    )
    return ids, y, probs


def _align(reference: list[str], ids: list[str], probs: np.ndarray) -> np.ndarray:
    mapping = {sample_id: index for index, sample_id in enumerate(ids)}
    if set(reference) != set(ids):
        raise ValueError("Prediction ID sets differ")
    return probs[[mapping[sample_id] for sample_id in reference]]


def _branch_files(seed: int, tag: str) -> dict[str, Path]:
    return {
        "dyn": E08 / f"seed{seed}" / "conditions" / tag / "dyn_hgb" / "predictions.csv",
        "md": E09 / f"e09_diff_stream_20261020_seed{seed}" / "conditions" / tag
        / "moment_diff_logit" / "predictions.csv",
        "tcn": E12[tag] / f"seed{seed}" / "conditions" / tag / "tcn_full" / "predictions.csv",
    }


def _cluster_groups(clusters: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    """Greedy balanced partition of clusters into ``n_groups`` groups."""
    unique, counts = np.unique(clusters, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    generator = np.random.default_rng(seed)
    tie_break = generator.permutation(len(order))
    order = order[np.lexsort((tie_break, -counts[order]))]
    load = np.zeros(n_groups, dtype=np.int64)
    assignment: dict[str, int] = {}
    for position in order:
        target = int(np.argmin(load))
        assignment[str(unique[position])] = target
        load[target] += int(counts[position])
    return np.asarray([assignment[str(value)] for value in clusters], dtype=np.int64)


def _meta(oof: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    model = LogisticRegression(C=1.0, max_iter=5000, random_state=seed)
    model.fit(oof, y)
    return model


def main() -> None:
    global E08, E09, E12
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--min-cluster-rows", type=int, default=12)
    parser.add_argument("--e08", type=Path, default=E08)
    parser.add_argument("--e09", type=Path, default=E09)
    parser.add_argument(
        "--e12-root",
        type=Path,
        default=None,
        help="Directory containing e12_tcn_stack_* suites (default: artifacts/improvements/e12_tcn_stack).",
    )
    args = parser.parse_args()
    E08 = args.e08
    E09 = args.e09
    if args.e12_root is not None:
        E12 = {
            "10p": args.e12_root / "e12_tcn_stack_10p_20261020_v1",
            "40p": args.e12_root / "e12_tcn_stack_20261020_v1",
        }
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    pooled: dict[tuple[str, str], list[float]] = {}

    for seed in SEEDS:
        archive = np.load(E08 / f"seed{seed}" / "features.npz", allow_pickle=True)
        train_ids = archive["train_ids"].astype(str)
        train_clusters = np.asarray([value[:CLUSTER_PREFIX] for value in train_ids])
        reference_val, y_val, val_dyn = _read_predictions(_branch_files(seed, "40p")["dyn"])
        val_clusters = np.asarray([value[:CLUSTER_PREFIX] for value in reference_val])
        group_of_val_cluster = _cluster_groups(val_clusters, args.groups, seed)

        for tag in TAGS:
            oof_path = E12[tag] / f"seed{seed}" / f"oof_{tag}.npz"
            oof = np.load(oof_path, allow_pickle=False)
            indices = oof["indices"]
            oof_matrix = np.hstack([oof[f"oof_{branch}"] for branch in BRANCHES])
            y_train = oof["y_train"]
            train_cluster_rows = train_clusters[indices]

            branch_probs: dict[str, np.ndarray] = {}
            for branch in BRANCHES:
                ids, _, probs = _read_predictions(_branch_files(seed, tag)[branch])
                branch_probs[branch] = _align(reference_val, ids, probs)
            val_matrix = np.hstack([branch_probs[branch] for branch in BRANCHES])

            full_meta = _meta(oof_matrix, y_train, seed)
            full_pred = full_meta.predict(val_matrix)
            rows.append(
                {
                    "seed": seed,
                    "tag": tag,
                    "scheme": "in-distribution",
                    "fold": -1,
                    "n_val": int(len(y_val)),
                    "stack": float(f1_score(y_val, full_pred, average="macro")),
                    "tcn": float(f1_score(y_val, branch_probs["tcn"].argmax(1), average="macro")),
                    "dyn": float(f1_score(y_val, branch_probs["dyn"].argmax(1), average="macro")),
                    "md": float(f1_score(y_val, branch_probs["md"].argmax(1), average="macro")),
                    "tcn_dyn": float(
                        f1_score(
                            y_val,
                            _meta(
                                np.hstack([oof["oof_dyn"], oof["oof_tcn"]]), y_train, seed
                            ).predict(np.hstack([branch_probs["dyn"], branch_probs["tcn"]])),
                            average="macro",
                        )
                    ),
                }
            )
            pooled.setdefault((tag, "in-distribution"), []).append(rows[-1]["stack"])

            for group in range(args.groups):
                held_clusters = {
                    str(value)
                    for value, assignment in zip(val_clusters, group_of_val_cluster, strict=True)
                    if assignment == group
                }
                fit_mask = ~np.isin(train_cluster_rows, list(held_clusters))
                eval_mask = group_of_val_cluster == group
                if fit_mask.sum() < 100 or eval_mask.sum() < 50:
                    raise ValueError(f"Fold too small: seed={seed} tag={tag} group={group}")
                meta = _meta(oof_matrix[fit_mask], y_train[fit_mask], seed)
                pred = meta.predict(val_matrix[eval_mask])
                tcn_dyn_meta = _meta(
                    np.hstack([oof["oof_dyn"], oof["oof_tcn"]])[fit_mask],
                    y_train[fit_mask],
                    seed,
                )
                tcn_dyn_pred = tcn_dyn_meta.predict(
                    np.hstack([branch_probs["dyn"], branch_probs["tcn"]])[eval_mask]
                )
                rows.append(
                    {
                        "seed": seed,
                        "tag": tag,
                        "scheme": "leave-cluster-out",
                        "fold": group,
                        "n_val": int(eval_mask.sum()),
                        "n_fit": int(fit_mask.sum()),
                        "stack": float(f1_score(y_val[eval_mask], pred, average="macro")),
                        "tcn": float(
                            f1_score(
                                y_val[eval_mask],
                                branch_probs["tcn"][eval_mask].argmax(1),
                                average="macro",
                            )
                        ),
                        "dyn": float(
                            f1_score(
                                y_val[eval_mask],
                                branch_probs["dyn"][eval_mask].argmax(1),
                                average="macro",
                            )
                        ),
                        "md": float(
                            f1_score(
                                y_val[eval_mask],
                                branch_probs["md"][eval_mask].argmax(1),
                                average="macro",
                            )
                        ),
                        "tcn_dyn": float(f1_score(y_val[eval_mask], tcn_dyn_pred, average="macro")),
                    }
                )
                pooled.setdefault((tag, "leave-cluster-out"), []).append(rows[-1]["stack"])

            # per-cluster accuracy of the in-distribution stack
            unique, counts = np.unique(val_clusters, return_counts=True)
            for cluster in unique[counts >= args.min_cluster_rows]:
                mask = val_clusters == cluster
                cluster_rows.append(
                    {
                        "seed": seed,
                        "tag": tag,
                        "cluster": str(cluster),
                        "n_val": int(mask.sum()),
                        "stack_acc": float(accuracy_score(y_val[mask], full_pred[mask])),
                        "tcn_acc": float(
                            accuracy_score(y_val[mask], branch_probs["tcn"][mask].argmax(1))
                        ),
                        "dyn_acc": float(
                            accuracy_score(y_val[mask], branch_probs["dyn"][mask].argmax(1))
                        ),
                    }
                )

    fieldnames = [
        "seed", "tag", "scheme", "fold", "n_val", "n_fit", "stack", "tcn_dyn", "tcn", "dyn", "md",
    ]
    with (args.output / "conditions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with (args.output / "clusters.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["seed", "tag", "cluster", "n_val", "stack_acc", "tcn_acc", "dyn_acc"]
        )
        writer.writeheader()
        for row in cluster_rows:
            writer.writerow(row)

    summary: dict[str, Any] = {"test_evaluated": False, "groups": args.groups, "by_tag": {}}
    for tag in TAGS:
        block: dict[str, Any] = {}
        for scheme in ("in-distribution", "leave-cluster-out"):
            values = [
                row["stack"]
                for row in rows
                if row["tag"] == tag and row["scheme"] == scheme and row["fold"] >= 0
            ]
            if scheme == "in-distribution":
                values = [
                    row["stack"]
                    for row in rows
                    if row["tag"] == tag and row["scheme"] == scheme
                ]
            block[scheme] = {
                "n": len(values),
                "stack_mean": float(np.mean(values) * 100.0),
                "stack_std": float(np.std(values, ddof=0) * 100.0),
                "stack_min": float(np.min(values) * 100.0),
            }
        locv = [
            row
            for row in rows
            if row["tag"] == tag and row["scheme"] == "leave-cluster-out"
        ]
        for method in ("tcn_dyn", "tcn", "dyn", "md"):
            values = [row[method] for row in locv]
            block[f"locv_{method}"] = {
                "mean": float(np.mean(values) * 100.0),
                "std": float(np.std(values, ddof=0) * 100.0),
            }
        clusters = [row for row in cluster_rows if row["tag"] == tag]
        acc = np.asarray([row["stack_acc"] for row in clusters]) * 100.0
        block["per_cluster"] = {
            "n_clusters": len(clusters),
            "min_rows": args.min_cluster_rows,
            "stack_acc_median": float(np.median(acc)) if len(acc) else float("nan"),
            "stack_acc_p10": float(np.percentile(acc, 10)) if len(acc) else float("nan"),
            "stack_acc_min": float(acc.min()) if len(acc) else float("nan"),
            "n_below_90": int((acc < 90.0).sum()),
        }
        summary["by_tag"][tag] = block

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["# TODO1102-1B leave-cluster-out audit", ""]
    for tag, block in summary["by_tag"].items():
        lines += [
            f"## {tag}",
            "",
            f"- In-distribution stack: {block['in-distribution']['stack_mean']:.2f}% "
            f"± {block['in-distribution']['stack_std']:.2f} (n={block['in-distribution']['n']})",
            f"- Leave-cluster-out stack: {block['leave-cluster-out']['stack_mean']:.2f}% "
            f"± {block['leave-cluster-out']['stack_std']:.2f} "
            f"(min {block['leave-cluster-out']['stack_min']:.2f}, n={block['leave-cluster-out']['n']})",
            f"- LOCV branches: TCN+DYN {block['locv_tcn_dyn']['mean']:.2f}%, "
            f"TCN {block['locv_tcn']['mean']:.2f}%, DYN {block['locv_dyn']['mean']:.2f}%, "
            f"MD {block['locv_md']['mean']:.2f}%",
            f"- Per-cluster stack accuracy: median {block['per_cluster']['stack_acc_median']:.2f}%, "
            f"p10 {block['per_cluster']['stack_acc_p10']:.2f}%, "
            f"min {block['per_cluster']['stack_acc_min']:.2f}%, "
            f"below 90%: {block['per_cluster']['n_below_90']}/{block['per_cluster']['n_clusters']}",
            "",
        ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
