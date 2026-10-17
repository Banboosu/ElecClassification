"""RBF controls with C and numeric bandwidths shared with a pretrained reference.

Reference gamma='scale' is resolved from a fold-local reference transform, NOT
from the control's variance. Final estimators are ordinary raw-input pipelines.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tcn_moment.fast_svm import _fit_kernel, gamma_value, squared_distances
from tcn_moment.feature_transforms import BranchScaler


class StatisticalBranchScaler(TransformerMixin, BaseEstimator):
    """The same statistics branch geometry as BranchScaler, without extra inputs."""

    def fit(self, x: np.ndarray, y: Any = None) -> StatisticalBranchScaler:
        del y
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError("Statistics must be a nonempty 2D array.")
        self.n_features_in_ = x.shape[1]
        self.scaler_ = StandardScaler().fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Statistics feature dimension changed.")
        return self.scaler_.transform(x) / np.sqrt(self.n_features_in_)


def reference_scaler_digest(scaler: BranchScaler) -> str:
    digest = hashlib.sha256()
    for branch in (scaler.moment_scaler_, scaler.stats_scaler_):
        for values in (branch.mean_, branch.var_, branch.scale_):
            digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def resolve_gamma(token: str | float, reference_gamma: float) -> float:
    return reference_gamma if token == "reference_scale" else float(token)


@dataclass
class ControlledSearchResult:
    best_estimator_: Pipeline
    best_params_: dict[str, Any]
    best_score_: float
    cv_results_: dict[str, Any]
    geometry_: dict[str, Any]
    backend_timings_: dict[str, Any]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.best_estimator_.predict(x)


def fit_controlled_rbf(
    x: np.ndarray,
    y: np.ndarray,
    reference: np.ndarray,
    *,
    kind: str,
    reference_dim: int,
    c_values: list[float],
    gamma_tokens: list[str | float],
    cv_folds: int = 5,
    n_jobs: int = 5,
    device: str = "cuda",
    cache_size_mb: float = 512,
    max_iter: int = 10000000,
) -> ControlledSearchResult:
    if (
        kind not in {"statistics", "fusion"}
        or x.ndim != 2
        or reference.ndim != 2
        or len(x) != len(y)
        or len(reference) != len(y)
        or not 0 < reference_dim < reference.shape[1]
        or not np.isfinite(x).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Invalid control/reference data.")
    expected_dim = (
        reference.shape[1] - reference_dim if kind == "statistics" else reference.shape[1]
    )
    if x.shape[1] != expected_dim:
        raise ValueError("Candidate features differ from the declared control.")
    transformer = StatisticalBranchScaler() if kind == "statistics" else BranchScaler(reference_dim)
    candidates = list(ParameterGrid({"C": c_values, "gamma_token": gamma_tokens}))
    folds = list(StratifiedKFold(cv_folds, shuffle=False).split(x, y))
    scores = np.empty((len(candidates), len(folds)))
    fit_times, score_times = np.empty_like(scores), np.empty_like(scores)
    scorer = make_scorer(f1_score, average="macro", zero_division=0)
    geometry = {
        "gamma_scope": "pretrained reference, CV training fold only",
        "reference_dim": reference_dim,
        "candidate_kind": kind,
        "folds": [],
    }
    started_all = time.perf_counter()
    kernel_seconds = 0.0
    for fold, (train, held_out) in enumerate(folds):
        started = time.perf_counter()
        reference_transform = BranchScaler(reference_dim)
        reference_train = np.asarray(
            reference_transform.fit_transform(reference[train]),
            dtype=np.float64,
            order="C",
        )
        shared_gamma = gamma_value("scale", reference_train)
        transform = clone(transformer)
        train_x = np.asarray(transform.fit_transform(x[train]), dtype=np.float64, order="C")
        held_out_x = np.asarray(transform.transform(x[held_out]), dtype=np.float64, order="C")
        distances = squared_distances(train_x, None, device)
        validation_distances = squared_distances(held_out_x, train_x, device)
        kernels, jobs = {}, []
        for index, candidate in enumerate(candidates):
            gamma = resolve_gamma(candidate["gamma_token"], shared_gamma)
            if gamma not in kernels:
                kernels[gamma] = (np.exp(-gamma * distances), np.exp(-gamma * validation_distances))
            estimator = SVC(
                C=candidate["C"],
                kernel="precomputed",
                gamma=1.0,
                cache_size=cache_size_mb,
                max_iter=max_iter,
            )
            jobs.append(
                delayed(_fit_kernel)(
                    index,
                    estimator,
                    *kernels[gamma],
                    y[train],
                    y[held_out],
                    scorer,
                )
            )
        geometry["folds"].append(
            {
                "fold": fold,
                "training_count": len(train),
                "held_out_count": len(held_out),
                "reference_scale_gamma": shared_gamma,
                "reference_scaler_sha256": reference_scaler_digest(reference_transform),
                "resolved_gamma_choices": [
                    resolve_gamma(token, shared_gamma) for token in gamma_tokens
                ],
            }
        )
        kernel_seconds += time.perf_counter() - started
        with tempfile.TemporaryDirectory(prefix="e01c_kernel_") as temporary:
            results = Parallel(
                n_jobs=n_jobs,
                pre_dispatch=n_jobs,
                max_nbytes="16M",
                mmap_mode="r",
                temp_folder=temporary,
            )(jobs)
        for index, score, fit_seconds, score_seconds in results:
            scores[index, fold], fit_times[index, fold], score_times[index, fold] = (
                score,
                fit_seconds,
                score_seconds,
            )
        print(
            f"Controlled RBF {kind} fold={fold + 1}/{cv_folds} "
            f"shared_scale_gamma={shared_gamma:.12g} elapsed={time.perf_counter() - started_all:.1f}s",
            flush=True,
        )
        del kernels, jobs, distances, validation_distances
    means = scores.mean(axis=1)
    best_index = int(np.argmax(means))
    best = candidates[best_index]
    full_reference_transform = BranchScaler(reference_dim)
    full_reference = np.asarray(
        full_reference_transform.fit_transform(reference), dtype=np.float64, order="C"
    )
    full_gamma = gamma_value("scale", full_reference)
    resolved = resolve_gamma(best["gamma_token"], full_gamma)
    estimator = Pipeline(
        [
            ("scale", clone(transformer)),
            (
                "svc",
                SVC(
                    C=best["C"],
                    gamma=resolved,
                    kernel="rbf",
                    cache_size=cache_size_mb,
                    max_iter=max_iter,
                ),
            ),
        ]
    )
    print(f"Refitting ordinary {kind} pipeline with shared gamma={resolved:.12g}", flush=True)
    started = time.perf_counter()
    estimator.fit(x, y)
    if estimator.named_steps["svc"].fit_status_ != 0:
        raise RuntimeError("Final control SVM did not converge.")
    refit_seconds = time.perf_counter() - started
    geometry.update(
        {
            "full_reference_scale_gamma": full_gamma,
            "resolved_refit_gamma": resolved,
            "full_reference_scaler_sha256": reference_scaler_digest(full_reference_transform),
        }
    )
    cv_results = {
        "params": candidates,
        "mean_test_score": means,
        "std_test_score": scores.std(axis=1),
        "rank_test_score": rankdata(-means, method="min").astype(int),
        "mean_fit_time": fit_times.mean(axis=1),
        "std_fit_time": fit_times.std(axis=1),
        "mean_score_time": score_times.mean(axis=1),
        "std_score_time": score_times.std(axis=1),
    }
    for fold in range(cv_folds):
        cv_results[f"split{fold}_test_score"] = scores[:, fold]
    return ControlledSearchResult(
        estimator,
        best,
        float(means[best_index]),
        cv_results,
        geometry,
        {
            "backend": "fold_local_reference_bandwidth_float64",
            "device": device,
            "n_jobs": n_jobs,
            "kernel_seconds": kernel_seconds,
            "refit_seconds": refit_seconds,
            "total_seconds": time.perf_counter() - started_all,
        },
    )
