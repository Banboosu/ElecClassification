"""Check that controlled-reference CV reproduces the existing E01B pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tcn_moment.config import load_config, with_random_seed
from tcn_moment.controlled_svm import fit_controlled_rbf, reference_scaler_digest
from tcn_moment.data import load_dataset, stratified_train_subset_indices
from tcn_moment.io_utils import atomic_write_json, sha256_file
from tcn_moment.train_feature_fusion import cache_signature, raw_statistics, validate_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument(
        "--output", default="artifacts/analysis/e01c_reference_equivalence_20261017.json"
    )
    args = parser.parse_args()
    config = with_random_seed(
        load_config("configs/experiments/improvements/feature_fusion.yaml"), args.seed
    )
    bundle = load_dataset(config.data)
    cached = validate_cache(
        Path("artifacts/improvements/e01_20261017_v2_features") / f"seed{args.seed}",
        cache_signature(config, bundle),
        {"train": bundle.ids_train, "validation": bundle.ids_val},
    )
    if cached is None:
        raise ValueError("Verified parent reference cache is required.")
    stats = raw_statistics(config, bundle)
    x = np.concatenate([cached["train"], stats["train"]], axis=1)
    validation = np.concatenate([cached["validation"], stats["validation"]], axis=1)
    indices = stratified_train_subset_indices(
        bundle.y_train, bundle.ids_train, args.fraction, args.seed
    )
    tag = f"{args.fraction * 100:g}"
    parent = (
        Path("artifacts/improvements/e01b_high_label")
        / f"e01b_high_label_20261017_v1_seed{args.seed}"
    )
    np.testing.assert_array_equal(
        bundle.ids_train[indices], np.load(parent / f"train_ids_{tag}.npy")
    )
    condition = parent / f"fusion_stats_svm_fraction_{tag}"
    old = json.loads((condition / "metrics.json").read_text())
    cv = pd.read_csv(condition / "cv_results.csv")
    result = fit_controlled_rbf(
        x[indices],
        bundle.y_train[indices],
        x[indices],
        kind="fusion",
        reference_dim=1024,
        c_values=[0.1, 1, 10, 100, 1000],
        gamma_tokens=["reference_scale", 0.05, 5.0],
    )
    max_error = 0.0
    for key in ["mean_test_score"] + [f"split{i}_test_score" for i in range(5)]:
        np.testing.assert_allclose(result.cv_results_[key], cv[key].to_numpy(), rtol=0, atol=1e-12)
        max_error = max(
            max_error, float(np.max(np.abs(result.cv_results_[key] - cv[key].to_numpy())))
        )
    expected = {"C": old["best_params"]["svc__C"], "gamma_token": old["best_params"]["svc__gamma"]}
    if expected["gamma_token"] == "scale":
        expected["gamma_token"] = "reference_scale"
    if result.best_params_ != expected:
        raise AssertionError("Controlled reference selected different parameters.")
    model = joblib.load(condition / "classifier.joblib")
    np.testing.assert_array_equal(result.predict(validation), model.predict(validation))
    if result.geometry_["full_reference_scaler_sha256"] != reference_scaler_digest(
        model.named_steps["scale"]
    ):
        raise AssertionError("Reference preprocessing differs from the original fitted scaler.")
    report = {
        "status": "passed",
        "seed": args.seed,
        "fraction": args.fraction,
        "cv_candidates": 15,
        "cv_folds": 5,
        "max_cv_score_difference": max_error,
        "best_params_identical": True,
        "validation_predictions_identical": True,
        "reference_scaler_identical": True,
        "test_evaluated": False,
        "controlled_backend_sha256": sha256_file(Path("src/tcn_moment/controlled_svm.py")),
        "parent_metrics_sha256": sha256_file(condition / "metrics.json"),
    }
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
