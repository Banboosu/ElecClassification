"""Equivalent RBF CV with fold-local transforms and reusable float64 kernels.

Only execution changes: the parameter grid, folds, scorer and final ordinary
RBF estimator are unchanged. GPU acceleration is confined to squared distances.
"""

from __future__ import annotations

import tempfile
import time
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import GridSearchCV, ParameterGrid, check_cv
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def squared_distances(x: np.ndarray, y: np.ndarray | None, device: str) -> np.ndarray:
    """Use double precision on CPU or CUDA; never TF32/FP16 for CV kernels."""
    import torch

    left = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float64), device=device)
    right = (
        left
        if y is None
        else torch.as_tensor(
            np.ascontiguousarray(y, dtype=np.float64),
            device=device,
        )
    )
    with torch.inference_mode():
        distances = left.square().sum(1)[:, None] + right.square().sum(1)[None, :]
        distances.addmm_(left, right.T, beta=1, alpha=-2)
        distances.clamp_min_(0)
        if y is None:
            distances.fill_diagonal_(0)
        result = distances.cpu().numpy().copy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite float64 RBF distances.")
    return result


def gamma_value(value: str | float, x: np.ndarray) -> float:
    if value == "scale":
        variance = x.var()
        return 1.0 / (x.shape[1] * variance) if variance else 1.0
    if value == "auto":
        return 1.0 / x.shape[1]
    return float(value)


def _fit_kernel(
    candidate: int,
    estimator: SVC,
    train_kernel: np.ndarray,
    validation_kernel: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    scorer: Any,
) -> tuple[int, float, float, float]:
    started = time.perf_counter()
    estimator.fit(train_kernel, train_y)
    if estimator.fit_status_ != 0:
        raise RuntimeError("Precomputed SVM did not converge.")
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    score = float(scorer(estimator, validation_kernel, validation_y))
    return candidate, score, fit_seconds, time.perf_counter() - started


class FastRBFGridSearchCV(GridSearchCV):
    """GridSearchCV-compatible single-scorer RBF backend; RF delegates unchanged."""

    # Class attributes keep sklearn's inherited constructor/get_params contract.
    kernel_device = "cuda"
    temporary_root: str | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, **params: Any) -> FastRBFGridSearchCV:
        base_svc = (
            self.estimator.steps[-1][1] if isinstance(self.estimator, Pipeline) else self.estimator
        )
        if not isinstance(base_svc, SVC):
            return super().fit(x, y, **params)
        if params or base_svc.kernel != "rbf" or self.refit is not True:
            raise ValueError("Fast RBF backend requires RBF, refit=True, and no fit metadata.")
        if isinstance(self.scoring, (dict, list, tuple)):
            raise TypeError("Fast RBF backend supports one scorer only.")
        prefix = f"{self.estimator.steps[-1][0]}__" if isinstance(self.estimator, Pipeline) else ""
        candidates = list(ParameterGrid(self.param_grid))
        if any(set(candidate) - {prefix + "C", prefix + "gamma"} for candidate in candidates):
            raise ValueError(
                "Only C/gamma may vary: transforms must remain identical within a fold."
            )
        cv = check_cv(self.cv, y, classifier=True)
        folds = list(cv.split(x, y))
        scores = np.empty((len(candidates), len(folds)), dtype=np.float64)
        fit_times = np.empty_like(scores)
        score_times = np.empty_like(scores)
        scorer = check_scoring(base_svc, scoring=self.scoring)
        kernel_seconds = 0.0
        started_all = time.perf_counter()
        # Process one fold at a time: no need to store 5 folds * all gamma kernels.
        for fold, (train, validation) in enumerate(folds):
            started = time.perf_counter()
            if isinstance(self.estimator, Pipeline):
                transform = clone(self.estimator[:-1])
                train_x = transform.fit_transform(x[train], y[train])
                validation_x = transform.transform(x[validation])
            else:
                train_x, validation_x = x[train], x[validation]
            # SVC converts to float64 after preprocessing, before resolving gamma='scale'.
            train_x = np.asarray(train_x, dtype=np.float64, order="C")
            validation_x = np.asarray(validation_x, dtype=np.float64, order="C")
            train_distances = squared_distances(train_x, None, self.kernel_device)
            validation_distances = squared_distances(validation_x, train_x, self.kernel_device)
            kernels: dict[float, tuple[np.ndarray, np.ndarray]] = {}
            jobs = []
            for index, candidate in enumerate(candidates):
                estimator = clone(self.estimator).set_params(**candidate)
                svc = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
                gamma = gamma_value(svc.gamma, train_x)
                if gamma not in kernels:
                    kernels[gamma] = (
                        np.exp(-gamma * train_distances),
                        np.exp(-gamma * validation_distances),
                    )
                precomputed = clone(svc).set_params(kernel="precomputed", gamma=1.0)
                train_kernel, validation_kernel = kernels[gamma]
                jobs.append(
                    delayed(_fit_kernel)(
                        index,
                        precomputed,
                        train_kernel,
                        validation_kernel,
                        y[train],
                        y[validation],
                        scorer,
                    )
                )
            kernel_seconds += time.perf_counter() - started
            # joblib memmaps share the large, read-only kernels between CPU workers.
            with tempfile.TemporaryDirectory(
                prefix="e01_rbf_", dir=self.temporary_root
            ) as temporary:
                computed = Parallel(
                    n_jobs=self.n_jobs,
                    pre_dispatch=self.n_jobs,
                    temp_folder=temporary,
                    max_nbytes="16M",
                    mmap_mode="r",
                )(jobs)
            for index, score, fit_seconds, score_seconds in computed:
                scores[index, fold] = score
                fit_times[index, fold] = fit_seconds
                score_times[index, fold] = score_seconds
            print(
                f"Fast RBF CV fold={fold + 1}/{len(folds)} "
                f"candidates={len(candidates)} elapsed={time.perf_counter() - started_all:.1f}s",
                flush=True,
            )
            del jobs, kernels, train_distances, validation_distances
        means = scores.mean(axis=1)
        self.best_index_ = int(np.argmax(means))  # first candidate wins ties, as in GridSearchCV
        self.best_params_ = candidates[self.best_index_]
        self.best_score_ = float(means[self.best_index_])
        self.n_splits_ = len(folds)
        self.multimetric_ = False
        self.scorer_ = scorer
        self.cv_results_ = {
            "params": candidates,
            "mean_test_score": means,
            "std_test_score": scores.std(axis=1),
            "rank_test_score": rankdata(-means, method="min").astype(int),
            "mean_fit_time": fit_times.mean(axis=1),
            "std_fit_time": fit_times.std(axis=1),
            "mean_score_time": score_times.mean(axis=1),
            "std_score_time": score_times.std(axis=1),
        }
        for key in sorted(set().union(*(candidate.keys() for candidate in candidates))):
            self.cv_results_[f"param_{key}"] = np.asarray(
                [candidate.get(key) for candidate in candidates],
                dtype=object,
            )
        for fold in range(len(folds)):
            self.cv_results_[f"split{fold}_test_score"] = scores[:, fold]
        # Save a normal raw-feature RBF model, NOT a precomputed-kernel-only estimator.
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        started = time.perf_counter()
        print("Fast RBF CV complete; refitting the selected ordinary RBF model", flush=True)
        self.best_estimator_.fit(x, y)
        self.refit_time_ = time.perf_counter() - started
        self.backend_timings_ = {
            "backend": "fold_local_precomputed_rbf_float64",
            "kernel_device": self.kernel_device,
            "n_jobs": self.n_jobs,
            "kernel_seconds": kernel_seconds,
            "refit_seconds": self.refit_time_,
            "total_seconds": time.perf_counter() - started_all,
        }
        return self
