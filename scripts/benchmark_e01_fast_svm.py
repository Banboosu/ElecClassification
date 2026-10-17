"""Shadow-check fast CV against an already completed pilot, without test access."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tcn_moment.config import load_config, with_random_seed
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.fast_svm import FastRBFGridSearchCV
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.train_feature_fusion import (
    assemble_features,
    build_search,
    frozen_features,
    raw_statistics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-run", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", default="fusion_stats_svm")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.pilot_run)
    plan = json.loads((run / "experiment_plan.json").read_text())
    if plan["evaluate_test"]:
        raise ValueError("This benchmark is validation-only.")
    config = with_random_seed(
        load_config("configs/experiments/improvements/feature_fusion.yaml"),
        plan["seed"],
    )
    bundle = load_dataset(config.data)
    stats = raw_statistics(config, bundle)
    features, _ = frozen_features(config, bundle, Path(args.cache_dir), evaluate_test=False)
    train = assemble_features(args.model, features["train"], stats["train"])
    validation = assemble_features(args.model, features["validation"], stats["validation"])
    indices = stratified_train_subset_indices(
        bundle.y_train, bundle.ids_train, args.fraction, plan["seed"]
    )
    tag = f"{args.fraction * 100:g}".replace(".", "p")
    condition = run / f"{args.model}_fraction_{tag}"
    np.testing.assert_array_equal(bundle.ids_train[indices], np.load(run / f"train_ids_{tag}.npy"))
    old = json.loads((condition / "metrics.json").read_text())
    expected = pd.read_csv(condition / "cv_results.csv")
    reference = build_search(
        args.model,
        train.shape[1],
        features["train"].shape[1],
        config,
        n_jobs=args.n_jobs,
        seed=plan["seed"],
    )
    fast = FastRBFGridSearchCV(**reference.get_params(deep=False))
    started = time.perf_counter()
    fast.fit(train[indices], bundle.y_train[indices])
    elapsed = time.perf_counter() - started
    maximum_difference = 0.0
    for column in ["mean_test_score"] + [
        f"split{i}_test_score" for i in range(config.svm.cv_folds)
    ]:
        actual = fast.cv_results_[column]
        np.testing.assert_allclose(actual, expected[column].to_numpy(), atol=1e-12, rtol=0)
        maximum_difference = max(
            maximum_difference, float(np.max(np.abs(actual - expected[column])))
        )
    if fast.best_params_ != old["best_params"]:
        raise AssertionError("Fast and reference CV chose different hyperparameters.")
    original_classifier = joblib.load(condition / "classifier.joblib")
    np.testing.assert_array_equal(fast.predict(validation), original_classifier.predict(validation))
    report = {
        "status": "passed",
        "model": args.model,
        "fraction": args.fraction,
        "train_samples": len(indices),
        "candidate_count": len(fast.cv_results_["params"]),
        "folds": config.svm.cv_folds,
        "max_cv_score_difference": maximum_difference,
        "best_params_identical": True,
        "validation_predictions_identical": True,
        "reference_fit_seconds": old["fit_seconds"],
        "fast_fit_seconds": elapsed,
        "observed_speedup": old["fit_seconds"] / elapsed,
        "timings": fast.backend_timings_,
        "test_evaluated": False,
        "source_sha256": sha256_file(Path("src/tcn_moment/fast_svm.py")),
        "note": "Reference timing is historical; benchmark may share CPU with the running pilot.",
    }
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
