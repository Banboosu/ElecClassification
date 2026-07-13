from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tcn_moment.config import ExperimentConfig, load_config, with_random_seed
from tcn_moment.data import DatasetBundle, load_dataset, save_label_encoder
from tcn_moment.experiment import prepare_run
from tcn_moment.io_utils import atomic_write_json
from tcn_moment.metrics import classification_metrics


FEATURE_NAMES = [
    "length",
    "mean",
    "std",
    "minimum",
    "maximum",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "last_minus_first",
    "slope",
    "diff_mean",
    "diff_std",
    "max_abs_diff",
]


def extract_statistical_features(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows: list[list[float]] = []
    for values, valid_mask in zip(x, mask, strict=True):
        valid = values[valid_mask.astype(bool)]
        differences = np.diff(valid)
        slope = float(np.polyfit(np.arange(len(valid)), valid, 1)[0]) if len(valid) > 1 else 0.0
        rows.append(
            [
                float(len(valid)),
                float(valid.mean()),
                float(valid.std()),
                float(valid.min()),
                float(valid.max()),
                *np.quantile(valid, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(float).tolist(),
                float(valid[-1] - valid[0]),
                slope,
                float(differences.mean()) if len(differences) else 0.0,
                float(differences.std()) if len(differences) else 0.0,
                float(np.abs(differences).max()) if len(differences) else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".joblib.tmp")
    os.close(descriptor)
    try:
        joblib.dump(value, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _evaluate_model(
    model: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    return {
        "validation_metrics": classification_metrics(
            y_val,
            model.predict(x_val),
            class_names,
            include_details=True,
        ),
        "test_metrics": classification_metrics(
            y_test,
            model.predict(x_test),
            class_names,
            include_details=True,
        ),
    }


def _data_record(bundle: DatasetBundle) -> dict[str, Any]:
    return {
        "dataset_sha256": bundle.dataset_sha256,
        "split_manifest": str(bundle.split_path),
        "split": bundle.split_counts,
        "classes": bundle.label_encoder.classes_.tolist(),
    }


def train_baselines(
    config: ExperimentConfig,
    config_path: Path,
    *,
    run_name: str | None = None,
) -> None:
    raw_config = replace(config, data=replace(config.data, normalize="none"))
    context = prepare_run(
        model_name="BASELINES",
        base_output_dir=Path("artifacts/baselines"),
        config=raw_config,
        config_path=config_path,
        torch=torch,
        run_name=run_name,
        resume_dir=None,
    )
    try:
        bundle = load_dataset(raw_config.data)
        shutil.copy2(bundle.split_path, context.run_dir / "split_manifest.json")
        save_label_encoder(bundle.label_encoder, context.run_dir)
        features_train = extract_statistical_features(bundle.x_train, bundle.mask_train)
        features_val = extract_statistical_features(bundle.x_val, bundle.mask_val)
        features_test = extract_statistical_features(bundle.x_test, bundle.mask_test)
        class_names = [str(name) for name in bundle.label_encoder.classes_]
        results: dict[str, Any] = {}

        majority_class = int(np.bincount(bundle.y_train).argmax())
        majority_started = time.perf_counter()
        majority_val = np.full_like(bundle.y_val, majority_class)
        majority_test = np.full_like(bundle.y_test, majority_class)
        results["majority"] = {
            "fit_seconds": time.perf_counter() - majority_started,
            "validation_metrics": classification_metrics(
                bundle.y_val, majority_val, class_names, include_details=True
            ),
            "test_metrics": classification_metrics(
                bundle.y_test, majority_test, class_names, include_details=True
            ),
        }

        models = {
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=config.data.random_state),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=config.data.random_state,
                n_jobs=-1,
            ),
        }
        for name, model in models.items():
            started = time.perf_counter()
            model.fit(features_train, bundle.y_train)
            result = _evaluate_model(
                model,
                features_val,
                bundle.y_val,
                features_test,
                bundle.y_test,
                class_names,
            )
            result["fit_seconds"] = time.perf_counter() - started
            results[name] = result
            _atomic_joblib_dump(model, context.run_dir / f"{name}.joblib")

        atomic_write_json(
            context.run_dir / "metrics.json",
            {
                "model": "STATISTICAL_BASELINES",
                "run_name": context.run_name,
                "data": _data_record(bundle),
                "feature_names": FEATURE_NAMES,
                "results": results,
            },
        )
    except BaseException as exc:
        context.set_status("failed", error_type=type(exc).__name__, message=str(exc))
        raise
    else:
        context.set_status("completed")
        print(f"Saved baseline results to {context.run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train statistical classification baselines.")
    parser.add_argument("--config", default="configs/moment.yaml", help="Path to YAML config.")
    parser.add_argument("--seed", type=int, help="Override random seed and use its split manifest.")
    parser.add_argument("--run-name", help="Unique output directory name.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.seed is not None:
        config = with_random_seed(config, args.seed)
    train_baselines(config, config_path, run_name=args.run_name)


if __name__ == "__main__":
    main()
